"""``/test <role>``: a small trial search that reports yield, never relevance, and never writes to the database.

At most ``TEST_MAX_CALLS`` calls of ``TEST_MAX_RESULTS`` rows each run on one worker thread, through the same
scraper (and so the same ``ScrapeCoordinator`` lock and LinkedIn pacing) as the scheduled cycle. The worker
only reads the database (to tell new postings from known ones) and sends its own reply.
"""
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.bot.commands import BotContext
from src.bot.render import chunk_html, esc
from src.bot.updates import OutgoingMessage, Update
from src.filters.rules import CompiledRule
from src.profile.compiler import compile_profile, parse_expanded_query_id
from src.profile.ops import AddRole, apply_operation, normalise_snapshot, validate_snapshot
from src.profile.snapshot import ProfileFields, ProfileSnapshot
from src.profile.store import ProfileStore
from src.scraper.jobspy_scraper import COUNTRY_BOUND_BOARDS
from src.scraper.query_catalog import SearchQuery
from src.storage.db import Database

logger = logging.getLogger(__name__)

TEST_COOLDOWN_SECONDS = 300
TEST_MAX_RESULTS = 5
TEST_MAX_CALLS = 3

_USAGE = "Usage: /test role [linkedin]"
_NO_MARKET = "Add a market first (/add_market)."
_NOTHING = "Nothing to search for that role; is the role enabled?"
_DISCLAIMER = "This does not measure relevance."
_SKIP_REASONS = {"disabled": "disabled in SCRAPER_ENABLED_BOARDS", "no_country": "no country"}
MAX_REASON_PROBLEMS = 2


@dataclass(frozen=True)
class ProbeResult:
    rows: int
    new_jobs: int
    prefiltered: int
    titles: Tuple[str, ...]
    calls: int
    failed: int
    postings: int = 0  # distinct postings seen (the denominator of the pre-filtered share)


def plan_probe(
    snapshot: ProfileSnapshot,
    role: str,
    include_linkedin: bool,
    board_skip_reason: Optional[Callable[[str, SearchQuery], Optional[str]]] = None,
) -> List[Tuple[SearchQuery, str]]:
    """The (query, board) calls of a trial search for ``role``; an unknown role is added in memory only.

    Pairs the scraper would refuse (``board_skip_reason`` returns a reason) are dropped before the call budget
    is applied, so the budget is spent on calls that really happen.
    """
    wanted = role.strip().casefold()
    names = {r.name.strip().casefold() for r in snapshot.roles}
    if wanted not in names:
        snapshot = apply_operation(snapshot, AddRole(role.strip()))
    compiled = compile_profile(snapshot)
    plan: List[Tuple[SearchQuery, str]] = []
    for query in compiled.queries:
        parsed = parse_expanded_query_id(query.query_id)
        if parsed is None or parsed["role"].strip().casefold() != wanted:
            continue
        for site in query.site_names:
            board = site.lower()
            if board == "linkedin" and not include_linkedin:
                continue
            if board in COUNTRY_BOUND_BOARDS and not query.country_code:
                continue
            if board_skip_reason is not None and board_skip_reason(site, query) is not None:
                continue
            plan.append((query, site))
    return plan[:TEST_MAX_CALLS]


def run_probe(
    scraper: Any, db: Database, calls: Sequence[Tuple[SearchQuery, str]], rules: Optional[Sequence[CompiledRule]]
) -> ProbeResult:
    """Run ``calls`` until ``TEST_MAX_RESULTS`` rows are collected; nothing is written."""
    rows = made = failed = 0
    seen: Dict[str, Any] = {}
    for query, site in calls:
        if rows >= TEST_MAX_RESULTS:
            break
        outcome = scraper.scrape_board_outcome(
            site, query, track_cycle=False, rules=rules, results_wanted=TEST_MAX_RESULTS
        )
        made += 1
        rows += outcome.rows
        if outcome.status == "failed":
            failed += 1
        for posting in outcome.postings:
            seen.setdefault(posting.dedup_hash, posting)
    postings = list(seen.values())
    return ProbeResult(
        rows=rows,
        new_jobs=sum(1 for p in postings if db.get_job_by_hash(p.dedup_hash) is None),
        prefiltered=sum(1 for p in postings if p.pre_filtered),
        titles=tuple(p.title for p in postings[:TEST_MAX_RESULTS]),
        calls=made,
        failed=failed,
        postings=len(postings),
    )


def render_probe(result: ProbeResult) -> str:
    share = round(100 * result.prefiltered / result.postings) if result.postings else 0
    lines = [
        f"rows {result.rows}, new {result.new_jobs}, "
        f"pre-filtered {result.prefiltered} of {result.postings} ({share}%)"
    ]
    if result.failed:
        lines.append(f"{result.failed} of {result.calls} calls failed.")
    lines.extend(f"- {esc(' '.join(title.split())[:120])}" for title in result.titles)
    lines.append(_DISCLAIMER)
    return "\n".join(lines)


def role_name_problems(role: str) -> List[str]:
    """Why ``role`` is not a valid role name, by the same validation ``ops`` applies (empty list: valid)."""
    empty = ProfileSnapshot(ProfileFields(paused=True))
    return validate_snapshot(normalise_snapshot(apply_operation(empty, AddRole(role))), None)


class ProbeRunner:
    RETRY_WAIT_SECONDS = 3.0  # one retry of an undelivered result

    def __init__(self, ctx: BotContext, client: Any, *, clock: Callable[[], float] = time.time):
        self._ctx = ctx
        self._client = client
        self._clock = clock
        self._db = Database(ctx.db_path)  # read-only use: get_job_by_hash
        self._store = ProfileStore(self._db)
        self._lock = threading.Lock()
        self._last_started: Optional[float] = None
        self._thread: Optional[threading.Thread] = None

    def request(self, update: Update) -> List[OutgoingMessage]:
        words = update.args.split()
        include_linkedin = len(words) > 1 and words[-1].lower() == "linkedin"
        if include_linkedin:
            words.pop()
        role = " ".join(words)
        if not role:
            return [OutgoingMessage(_USAGE)]
        problems = role_name_problems(role)
        if problems:
            shown = "; ".join(esc(" ".join(p.split())[:120]) for p in problems[:MAX_REASON_PROBLEMS])
            return [OutgoingMessage(f"That role name is not valid: {shown}")]
        snapshot = self._store.read_snapshot()
        if snapshot is None or not any(m.enabled for m in snapshot.markets):
            return [OutgoingMessage(_NO_MARKET)]
        skipped: Dict[str, str] = {}
        skip_reason = getattr(self._ctx.scraper, "board_skip_reason", None)

        def scraper_skips(site: str, query: SearchQuery) -> Optional[str]:
            reason = skip_reason(site, query) if callable(skip_reason) else None
            if reason:
                skipped.setdefault(site.lower(), _SKIP_REASONS.get(reason, reason))
            return reason

        plan = plan_probe(snapshot, role, include_linkedin, scraper_skips)
        skipped_text = ", ".join(f"{esc(board)} ({esc(why)})" for board, why in skipped.items())
        if not plan:
            if skipped_text:
                return [OutgoingMessage(f"Nothing to search: every board is skipped: {skipped_text}.")]
            return [OutgoingMessage(_NOTHING)]
        rules = compile_profile(snapshot).rules
        with self._lock:
            now = self._clock()
            running = self._thread is not None and self._thread.is_alive()
            if self._last_started is not None and (running or now - self._last_started < TEST_COOLDOWN_SECONDS):
                wait = max(1, int(TEST_COOLDOWN_SECONDS - (now - self._last_started) + 0.999))
                return [OutgoingMessage(f"One /test per 5 minutes; try again in {wait} s.")]
            self._last_started = now
            self._thread = threading.Thread(
                target=self._work, args=(update.chat_id, plan, rules, skipped_text), name="bot-test", daemon=True
            )
            self._thread.start()
        return [OutgoingMessage("Searching...")]

    def _work(self, chat_id: Optional[int], plan, rules, skipped_text: str = "") -> None:
        try:
            text = render_probe(run_probe(self._ctx.scraper, self._db, plan, rules))
        except Exception as exc:  # never the message: it can hold a request URL
            logger.warning("test search failed: %s", type(exc).__name__)
            text = f"The test search failed ({type(exc).__name__})."
        if skipped_text:
            text += f"\nSkipped boards: {skipped_text}"
        if chat_id is None:
            return
        try:
            for piece in chunk_html(text):
                if self._client.send_message(chat_id, piece) is None and not self._retry(chat_id, piece):
                    logger.warning("could not send the test result")
                    return
        except Exception as exc:
            logger.warning("could not send the test result: %s", type(exc).__name__)

    def _retry(self, chat_id: int, piece: str) -> bool:
        """One more attempt after a short wait that ends early on shutdown."""
        stop = self._ctx.stop_event
        if stop is not None:
            if stop.wait(self.RETRY_WAIT_SECONDS):
                return False
        else:
            time.sleep(self.RETRY_WAIT_SECONDS)
        return self._client.send_message(chat_id, piece) is not None
