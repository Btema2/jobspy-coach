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
from src.profile.ops import AddRole, apply_operation
from src.profile.snapshot import ProfileSnapshot
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
_ID_SEPARATORS = (";", "=")


@dataclass(frozen=True)
class ProbeResult:
    rows: int
    new_jobs: int
    prefiltered: int
    titles: Tuple[str, ...]
    calls: int
    failed: int
    postings: int = 0  # distinct postings seen (the denominator of the pre-filtered share)


def plan_probe(snapshot: ProfileSnapshot, role: str, include_linkedin: bool) -> List[Tuple[SearchQuery, str]]:
    """The (query, board) calls of a trial search for ``role``; an unknown role is added in memory only."""
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


class ProbeRunner:
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
        if any(sep in role for sep in _ID_SEPARATORS):
            return [OutgoingMessage("A role name cannot contain ';' or '='.")]
        snapshot = self._store.read_snapshot()
        if snapshot is None or not any(m.enabled for m in snapshot.markets):
            return [OutgoingMessage(_NO_MARKET)]
        plan = plan_probe(snapshot, role, include_linkedin)
        if not plan:
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
                target=self._work, args=(update.chat_id, plan, rules), name="bot-test", daemon=True
            )
            self._thread.start()
        return [OutgoingMessage("Searching...")]

    def _work(self, chat_id: Optional[int], plan, rules) -> None:
        try:
            text = render_probe(run_probe(self._ctx.scraper, self._db, plan, rules))
        except Exception as exc:  # never the message: it can hold a request URL
            logger.warning("test search failed: %s", type(exc).__name__)
            text = f"The test search failed ({type(exc).__name__})."
        try:
            if chat_id is not None:
                for piece in chunk_html(text):
                    self._client.send_message(chat_id, piece)
        except Exception as exc:
            logger.warning("could not send the test result: %s", type(exc).__name__)
