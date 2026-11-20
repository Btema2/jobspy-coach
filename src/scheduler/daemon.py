import html
import inspect
import logging
import math
import random
import re
import signal
import sqlite3
import sys
import threading
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from concurrent.futures import FIRST_COMPLETED, CancelledError, Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import httpx
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from src.bot.auth import Authorizer, adopt_chat_id
from src.bot.chat_llm import ChatLLM
from src.bot.client import TelegramBotClient
from src.bot.commands import (
    COMMANDS,
    TRIGGER_BUSY,
    TRIGGER_NOT_READY,
    TRIGGER_STARTED,
    BotContext,
    CommandRouter,
)
from src.bot.conversation_store import ConversationStore
from src.bot.poller import BotPoller
from src.bot.render import esc
from src.bot.state import BotStateStore
from src.config import get_settings
from src.evaluator.chain import ProviderChainEvaluator, build_chain_from_settings
from src.evaluator.errors import ConfigurationError, EvaluationRejected, ProvidersUnavailable, StopRequested
from src.evaluator.scoring import ScoringPolicy, decide_wow, no_profile_policy
from src.filters.rules import CompiledRule
from src.logging_config import redact_secrets
from src.profile.compiler import CompiledProfile, compile_profile
from src.profile.errors import ProfileError
from src.profile.store import ProfileStore
from src.notifier.telegram_notifier import TelegramNotifier
from src.notifier.templates import format_version_update_notice
from src.scraper.jobspy_scraper import JobSpyScraper
from src.scraper.outcome import STATUS_EMPTY, STATUS_FAILED, STATUS_OK, ScrapeOutcome
from src.scraper.query_catalog import SearchQuery, query_key
from src.scraper.version_checker import check_jobspy_update
from src.storage.db import STALE_PENDING_DAYS, Database
from src.storage.models import JobPosting

logger = logging.getLogger(__name__)


_MAX_STORED_ERROR_CHARS = 300
CHAT_UNCONFIGURED_TTL_SECONDS = 60.0

# Default of ``evaluate_unprocessed(profile=...)``: "resolve the profile now". An explicit ``None`` means
# "this cycle has no profile" and is never re-resolved (a profile created mid-cycle waits for the next one).
UNSET: Any = object()


def _is_busy_error(exc: sqlite3.OperationalError) -> bool:
    """A lock or busy timeout (transient), as opposed to a missing table or an I/O error."""
    text = str(exc).lower()
    return "locked" in text or "busy" in text


def sanitize_error_message(message: Optional[str]) -> Optional[str]:
    """Redact secrets (proxy URL credentials, tokens) and truncate an error before it is persisted."""
    if message is None:
        return None
    return redact_secrets(str(message))[:_MAX_STORED_ERROR_CHARS]


@dataclass(frozen=True)
class LatencyStats:
    mean: float
    p95: float
    max: float


def latency_stats(latencies: List[float]) -> LatencyStats:
    """Mean, nearest-rank p95 and max of per-job latencies in seconds (zeros when empty)."""
    if not latencies:
        return LatencyStats(0.0, 0.0, 0.0)
    ordered = sorted(latencies)
    rank = max(1, math.ceil(0.95 * len(ordered)))
    return LatencyStats(sum(ordered) / len(ordered), ordered[rank - 1], ordered[-1])


STOPPED_PROVIDERS_UNAVAILABLE = "providers_unavailable"
STOPPED_SHUTDOWN = "shutdown"
STOPPED_PROFILE_BUSY = "profile_db_busy"


@dataclass(frozen=True)
class EvalSummary:
    """Result of one ``evaluate_unprocessed`` batch.

    ``not_scored_in_batch``: jobs of this batch the batch never handled (provider outage, shutdown, cancelled):
    they stay ``unprocessed`` with attempts unchanged. Job-specific failures are handled, not counted here.
    ``beyond_limit``: unprocessed jobs that did not fit into the batch (count read before the batch, >= 0).
    ``stopped_reason``: ``None`` or one of the ``STOPPED_*`` constants.
    """

    evaluated: int
    wow_sent: int
    not_scored_in_batch: int
    beyond_limit: int
    stopped_reason: Optional[str]

    def as_tuple(self) -> Tuple[int, int]:
        return self.evaluated, self.wow_sent


_EMPTY_SUMMARY = EvalSummary(0, 0, 0, 0, None)

REASON_BLOCKED_429 = "blocked (429)"
REASON_DETAIL_PAGES = "detail pages blocked"
REASON_NO_DATA = "no data returned"
_MAX_REASON_CHARS = 60
# 429 as a standalone status number (never inside a word or a decimal such as 0.429), or its reason phrase.
_RE_429 = re.compile(r"(?<![\w.])429(?!\w)(?!\.\d)|too many requests", re.IGNORECASE)


def _error_reason(board: str, error: Optional[str]) -> str:
    """Short, redacted reason for a failed (query, board) call."""
    text = error or ""
    if board == "linkedin" and _RE_429.search(text):
        return REASON_BLOCKED_429
    if board == "google" and "no job data returned" in text.lower():
        return REASON_NO_DATA
    # Redact first, then collapse whitespace, then cut: a secret is never split by the cut.
    return " ".join(redact_secrets(text).split())[:_MAX_REASON_CHARS] or "unknown error"


def classify_scrape_problems(board: str, outcome: ScrapeOutcome) -> List[str]:
    """The problem reasons of one scrape outcome (empty when the call went fine)."""
    board = board.lower()
    reasons: List[str] = []
    if outcome.status == STATUS_FAILED:
        reasons.append(_error_reason(board, outcome.error))
    if outcome.skipped_empty:
        reasons.append(REASON_DETAIL_PAGES)
    return reasons


@dataclass
class ScrapeStats:
    """Counters of one scrape cycle (for the run report); ``problem_details`` is for logs only (query ids)."""

    ok: int = 0
    empty: int = 0
    failed: int = 0
    skipped_boards: int = 0  # distinct boards skipped for a missing country (scraper end_cycle)
    new_to_evaluate: int = 0
    prefiltered: int = 0
    problems: Counter = field(default_factory=Counter)
    problem_details: List[Tuple[str, str, str]] = field(default_factory=list)

    @property
    def total_new_saved(self) -> int:
        return self.new_to_evaluate + self.prefiltered

    def _add_problem(self, query_id: str, board: str, reason: str) -> None:
        self.problems[(board, reason)] += 1
        self.problem_details.append((query_id, board, reason))

    def record(self, query_id: str, board: str, outcome: ScrapeOutcome) -> None:
        board = board.lower()
        if outcome.status == STATUS_OK:
            self.ok += 1
        elif outcome.status == STATUS_EMPTY:
            self.empty += 1
        elif outcome.status == STATUS_FAILED:
            self.failed += 1
        for reason in classify_scrape_problems(board, outcome):
            self._add_problem(query_id, board, reason)

    def record_exception(self, query_id: str, board: str, exc: BaseException) -> None:
        board = board.lower()
        self.failed += 1
        self._add_problem(query_id, board, _error_reason(board, str(exc) or type(exc).__name__))

    def record_new(self, posting: JobPosting) -> None:
        if posting.pre_filtered:
            self.prefiltered += 1
        else:
            self.new_to_evaluate += 1

    def problem_list(self) -> List[Tuple[str, str, int]]:
        """Distinct ``(board, reason, count)`` in first-seen order."""
        return [(board, reason, n) for (board, reason), n in self.problems.items()]


_STOP_TEXT = {
    STOPPED_PROVIDERS_UNAVAILABLE: "no AI provider was available",
    STOPPED_SHUTDOWN: "the daemon was stopping",
    STOPPED_PROFILE_BUSY: "the profile was busy",
}
NO_LLM_KEY_ERROR = "no LLM key is configured"
DIGEST_HINT = "Send /digest to receive them now."
_MAX_REPORT_PROBLEMS = 3  # problem lines in a report (the last one says how many more there are)


@dataclass(frozen=True)
class CycleReport:
    """What one cycle did, for the run report after ``/run_now`` / ``/resume`` (see ``format_cycle_report``).

    ``unscored_batch_limit``: unprocessed jobs that did not fit into this cycle's batch; ``unscored_outage``: jobs
    of the batch left unscored because the batch stopped (``stopped_reason``, one of the ``STOPPED_*`` values).
    ``error``: set when the cycle stopped with an exception (exception class name or the no-LLM-key hint).
    """

    new_to_evaluate: int
    prefiltered: int
    calls_ok: int
    calls_empty: int
    calls_failed: int
    skipped_boards: int
    problems: Tuple[Tuple[str, str, int], ...]
    evaluated: int
    unscored_batch_limit: int
    unscored_outage: int
    wow_sent: int
    digest_waiting: int
    error: Optional[str]
    stopped_reason: Optional[str] = None


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def format_cycle_report(report: CycleReport) -> str:
    """The run report as one short Telegram HTML message (plain English, every value escaped, <= 12 lines)."""
    lines = ["<b>Run stopped</b>" if report.error else "<b>Run finished</b>"]
    calls = report.calls_ok + report.calls_empty + report.calls_failed
    if calls:
        scrape = (
            f"Searches: {report.calls_ok} with results, {report.calls_empty} empty, {report.calls_failed} failed"
        )
        if report.skipped_boards:
            scrape += f"; {_plural(report.skipped_boards, 'board')} skipped (no country)"
    elif report.skipped_boards:
        scrape = f"Searches: none ran ({_plural(report.skipped_boards, 'board')} skipped: no country set)"
    else:
        scrape = "Searches: none ran (paused, no queries or no profile)"
    lines.append(scrape)
    lines.append(
        f"New jobs: {report.new_to_evaluate} to evaluate, {report.prefiltered} dropped by the title rules"
    )
    problems = list(report.problems)
    if len(problems) > _MAX_REPORT_PROBLEMS:
        shown = problems[: _MAX_REPORT_PROBLEMS - 1]
        rest = len(problems) - len(shown)
    else:
        shown, rest = problems, 0
    for board, reason, count in shown:
        lines.append(f"Problem: {esc(board)}: {esc(reason)} ({_plural(count, 'search')})")
    if rest:
        lines.append(f"... and {rest} more problems (see the log)")
    lines.append(f"Evaluated: {report.evaluated}; WOW alerts sent: {report.wow_sent}")
    if report.unscored_batch_limit:
        lines.append(f"{report.unscored_batch_limit} left for the next cycle (batch limit)")
    if report.unscored_outage:
        why = _STOP_TEXT.get(report.stopped_reason or "", "the batch stopped")
        lines.append(f"{report.unscored_outage} not scored ({esc(why)})")
    if report.error:
        lines.append(f"The cycle stopped: {esc(report.error)}")
    if report.digest_waiting > 0:
        lines.append(f"{_plural(report.digest_waiting, 'job')} waiting for the digest. {DIGEST_HINT}")
    return "\n".join(lines)


def _fallback_report(error: str) -> CycleReport:
    """A report when building the real one failed: only the error is known."""
    return CycleReport(0, 0, 0, 0, 0, 0, (), 0, 0, 0, 0, 0, error)


@dataclass(frozen=True)
class DigestSend:
    """What the last digest run sent: ``sent`` of ``total`` planned messages for ``jobs`` rows.

    ``more``: qualifying rows left out because ``digest_max_items`` truncated the digest (0 otherwise).
    Recorded by the digest body on ``_last_digest_send`` and read by ``_digest_now_job`` under ``_digest_lock``.
    """

    sent: int
    total: int
    jobs: int = 0
    more: int = 0


DIGEST_ALREADY_RUNNING = "A digest is already running"
DIGEST_NOTHING_WAITING = "Nothing is waiting"


def format_digest_result(result: Optional[DigestSend]) -> str:
    """The one result message of an on-demand digest (``/digest``), plain English."""
    if result is None or result.jobs == 0:
        return DIGEST_NOTHING_WAITING
    if result.sent <= 0:  # before the total check: a notifier double without counts has total == sent == 0
        line = "Sending failed; the jobs stay waiting"
    elif result.sent >= result.total:
        line = f"Sent {_plural(result.jobs, 'job')}"
    else:
        line = f"Sent {result.sent} of {result.total} messages; some jobs may be missing"
    if result.more > 0:
        line += f"\n{result.more} more waiting, send /digest again"
    return line


@dataclass
class _CycleResults:
    """Filled by ``_run_pipeline_cycle`` as it goes, so a cycle that raises still reports what it did."""

    profile: Optional[CompiledProfile] = None
    scrape: ScrapeStats = field(default_factory=ScrapeStats)
    summary: EvalSummary = _EMPTY_SUMMARY


def _iter_completed(futures: Dict[Future, Any], poll_seconds: float = 0.2) -> Iterator[Future]:
    """Yield futures as they finish, including ones cancelled by ``shutdown(cancel_futures=True)``.

    ``concurrent.futures.as_completed`` never reports a future that was cancelled while still
    pending (the executor's shutdown does not notify its waiters), so it would block forever
    after a batch abort or a shutdown signal. Cancelled futures are therefore polled for.
    """
    pending = set(futures)
    while pending:
        done, pending = wait(pending, timeout=poll_seconds, return_when=FIRST_COMPLETED)
        yield from done
        cancelled = {f for f in pending if f.cancelled()}
        pending -= cancelled
        yield from cancelled


class DaemonService:
    def __init__(
        self,
        db_path: Optional[Path] = None,
        scraper: Optional[JobSpyScraper] = None,
        evaluator: Optional[ProviderChainEvaluator] = None,
        notifier: Optional[TelegramNotifier] = None,
    ):
        settings = get_settings()
        self.settings = settings
        self.db = Database(db_path or settings.db_path)
        self.db.init_db()

        self.scraper = scraper or JobSpyScraper(
            jitter_min=settings.scraper_jitter_min_seconds,
            jitter_max=settings.scraper_jitter_max_seconds,
            results_per_query=settings.scraper_results_per_query,
            hours_old=settings.hours_old_window,
            enabled_boards=settings.scraper_enabled_boards,
            proxies=settings.scraper_proxies,
        )
        # The provider chain is built lazily (first evaluation): constructing the daemon must
        # never require API keys. ``_stop_event`` is shared with the chain once it is built.
        self.evaluator: Optional[ProviderChainEvaluator] = evaluator
        self._stop_event = threading.Event()
        self._shutdown_requested = threading.Event()
        # One scrape cycle at a time: a scheduled cycle that finds it held is skipped, /run_now is refused.
        self._cycle_lock = threading.Lock()
        # Held while a digest runs: it reads and marks pending rows, so /requeue must wait for it as well.
        self._digest_lock = threading.Lock()
        # Set by trigger_cycle(notify=True) when it answers "started"; read and cleared at the end of the next
        # cycle that runs (inside _cycle_lock), which then sends a run report.
        self._notify_requested = False
        # Guards the flag and ``_guard_active``; cycle_guard takes/drops _cycle_lock under it (non-blocking only).
        self._notify_lock = threading.Lock()
        self._guard_active = False  # cycle_guard (/requeue) holds _cycle_lock: no cycle runs, nobody reads the flag
        # Written by the digest body (under _digest_lock); read by the /digest job under the same lock.
        self._last_digest_send: Optional[DigestSend] = None
        self._bot_poller: Optional[BotPoller] = None
        self._bot_thread: Optional[threading.Thread] = None
        self._bot_router: Optional[CommandRouter] = None
        self._conversation_thread: Optional[threading.Thread] = None
        self.chat_llm: Optional[ChatLLM] = None
        self._chat_unconfigured_until = 0.0  # time.monotonic() until which the chat path skips the chain build
        self._chain_lock = threading.Lock()
        # Futures of the batch in flight (snapshot-cancelled by request_shutdown).
        self._futures: List[Future] = []
        self.notifier = notifier or TelegramNotifier(
            token=settings.telegram_bot_token,
            chat_id=settings.telegram_chat_id,
        )
        self.scheduler: Optional[BlockingScheduler] = None
        # The daemon only reads the profile; it never auto-imports legacy files (the CLI does).
        self.profile_store = ProfileStore(self.db)
        self._last_good: Optional[CompiledProfile] = None
        self._rules_dropped_warned = False
        self._alerted: set = set()
        self._alert_lock = threading.Lock()

    # ----- profile -----

    def _alert_once(self, key: str, text: str) -> None:
        """Telegram alert the first time ``key`` is delivered in this process; failures are logged only.

        The key is recorded only after a successful send, so a transient Telegram failure (a False
        return or an exception) is retried on the next cycle.
        """
        with self._alert_lock:
            if key in self._alerted:
                return
            try:
                delivered = bool(self.notifier.send_message(html.escape(text, quote=False)))
            except Exception as exc:  # an alert must never break a cycle
                logger.warning("Could not send profile alert (%s): %s", type(exc).__name__, redact_secrets(str(exc)))
                return
            if delivered:
                self._alerted.add(key)
            else:
                logger.warning("Profile alert was not delivered; it will be retried on the next cycle.")

    def _rearm_alerts(self, prefix: str) -> None:
        with self._alert_lock:
            self._alerted = {k for k in self._alerted if not k.startswith(prefix)}

    def _rebuild_from_versions(self) -> Optional[CompiledProfile]:
        """Newest ``profile_versions`` snapshot that still compiles (None when there is none)."""
        try:
            versions = self.profile_store.list_versions()
        except sqlite3.Error as exc:
            logger.error("Could not list profile versions: %s", exc)
            return None
        for info in versions:
            try:
                compiled = compile_profile(
                    self.profile_store.load_version(info.version),
                    scrape_interval_hours=self.settings.scrape_interval_hours,
                )
            except Exception as exc:  # a stored snapshot that cannot be read or compiled is skipped
                logger.warning("Profile version %s is not usable: %s", info.version, sanitize_error_message(str(exc)))
                continue
            logger.warning("Using profile version %s as the last good profile.", info.version)
            return compiled
        return None

    def _profile_failure(self, message: str, alert: bool) -> Tuple[Optional[CompiledProfile], str]:
        """A profile that cannot be read or compiled: log, alert once per distinct text, use the last good one."""
        logger.error(message)
        if alert:
            self._alert_once(f"compile:{message}", message)
        if self._last_good is None:
            self._last_good = self._rebuild_from_versions()
        self._rearm_alerts("no_profile")  # a profile exists again, even if it does not compile
        return self._last_good, "compile_error"

    def current_profile(self, alert: bool = True) -> Tuple[Optional[CompiledProfile], Optional[str]]:
        """Compile the stored profile; ``(profile, problem)`` with problem ``None``, ``no_profile``, ``compile_error`` or ``db_busy``.

        On a compile or snapshot-read error the fallback is the last good profile (in memory), else the
        newest stored version that compiles, else None; the error is alerted once per distinct text.
        """
        try:
            snapshot = self.profile_store.read_snapshot()
            if snapshot is None:
                return None, "no_profile"
            compiled = compile_profile(snapshot, scrape_interval_hours=self.settings.scrape_interval_hours)
        except sqlite3.OperationalError as exc:
            if _is_busy_error(exc):
                # A locked database says nothing about the profile: no alert, no scraping this cycle,
                # and evaluation keeps the last good profile (and is skipped when there is none).
                logger.warning(
                    "Profile database unavailable (%s); skipping scraping this cycle.", sanitize_error_message(str(exc))
                )
                return self._last_good, "db_busy"
            # Anything else (no such table, disk I/O error, ...) will not go away by itself: alert once.
            return self._profile_failure(f"profile cannot be read: {sanitize_error_message(str(exc))}", alert)
        except ProfileError as exc:
            return self._profile_failure(f"profile cannot be compiled: {sanitize_error_message(str(exc))}", alert)
        self._last_good = compiled
        self._rearm_alerts("compile:")
        self._rearm_alerts("no_profile")
        return compiled, None

    # ----- evaluation engine -----

    def _get_evaluator(self) -> Any:
        """Return the provider chain, building it on first use (raises ConfigurationError).

        The chain is built once and holds no profile state: prompt, thresholds, bonus and cap arrive per
        call as a ``ScoringPolicy``, so a profile change takes effect without a restart.
        """
        with self._chain_lock:
            if self.evaluator is None:
                self.evaluator = build_chain_from_settings(self.settings, stop_event=self._stop_event)
            return self.evaluator

    def _chat_providers(self) -> Any:
        """The chain's providers for the chat path; an unconfigured install is not re-checked for a while.

        Without a key every free-text message would rebuild the chain (and log its warnings) just to fail again.
        """
        if time.monotonic() < self._chat_unconfigured_until:
            raise ConfigurationError("no LLM provider is configured")
        try:
            return self._get_evaluator().providers
        except ConfigurationError:
            self._chat_unconfigured_until = time.monotonic() + CHAT_UNCONFIGURED_TTL_SECONDS
            raise

    def _policy(self, profile: Optional[CompiledProfile]) -> ScoringPolicy:
        """The scoring policy of a cycle: the profile's, else the ``.env`` thresholds (bonus off, legacy rules)."""
        if profile is not None:
            return profile.scoring
        return no_profile_policy(
            "", self.settings.wow_score_threshold, self.settings.digest_score_threshold
        )

    def _stop_chain(self) -> None:
        stop = getattr(self.evaluator, "stop", None)
        if callable(stop):
            stop()

    def request_shutdown(self) -> None:
        """Signal-handler hook: wake blocked workers and drop queued evaluations.

        Safe to call at any time (before the chain/batch exist, or repeatedly). It never
        calls ``executor.shutdown()``: that takes the executor's internal lock, which the
        interrupted main thread may already hold (self-deadlock). Queued futures are
        cancelled instead; the batch's own caller thread shuts the executor down. Workers
        mid-HTTP finish or time out on their own; nothing joins them.
        """
        self._shutdown_requested.set()
        self._stop_event.set()
        self._stop_chain()
        for future in list(self._futures):
            future.cancel()

    def _rearm_stop_event(self, chain: Any) -> bool:
        """Undo a previous batch abort so a new batch can run.

        Returns False (leaving the stop events set) if a shutdown was requested, including
        one that landed between the caller's earlier check and the clear below.
        """
        self._stop_event.clear()
        event = getattr(chain, "stop_event", None)
        if isinstance(event, threading.Event):
            event.clear()
        if self._shutdown_requested.is_set():
            self._stop_event.set()
            if isinstance(event, threading.Event):
                event.set()
            return False
        return True

    @staticmethod
    def _provider_for_model(chain: Any, model: Optional[str]) -> str:
        for provider in getattr(chain, "providers", None) or []:
            models = [getattr(provider, "model", None), *(getattr(provider, "models", None) or [])]
            if model and model in models:
                return str(getattr(provider, "name", "unknown"))
        return "unknown"

    def evaluate_unprocessed(
        self, limit: Optional[int] = None, dry_run: bool = False, profile: Any = UNSET
    ) -> EvalSummary:
        """Evaluate queued jobs concurrently; return an ``EvalSummary`` (``.as_tuple()`` = ``(evaluated, wow_sent)``).

        ``profile`` is the cycle's compiled profile; only the ``UNSET`` default resolves it here, an explicit
        ``None`` means no profile. One policy is taken at the start and used for the whole batch.

        Invariant: the thread that calls this method is the only one that touches the
        database or Telegram. Worker threads only run ``chain.evaluate_job`` (LLM calls).

        Database or Telegram errors raised while handling a result abort the whole batch
        (the exception propagates); jobs not yet handled stay ``unprocessed`` and are
        retried next cycle.
        """
        if self._shutdown_requested.is_set():
            return EvalSummary(0, 0, 0, 0, STOPPED_SHUTDOWN)
        if limit is None:
            limit = self.settings.eval_batch_limit
        backlog = self.db.count_unprocessed_jobs()
        jobs = [j for j in self.db.get_unprocessed_jobs(limit=limit) if j.id is not None]
        if not jobs:
            return _EMPTY_SUMMARY
        beyond_limit = max(0, backlog - len(jobs))

        if profile is UNSET:
            profile, problem = self.current_profile(alert=not dry_run)
            if problem == "db_busy" and profile is None:
                self._skip_evaluation_without_profile()
                return EvalSummary(0, 0, len(jobs), beyond_limit, STOPPED_PROFILE_BUSY)
        policy = self._policy(profile)
        has_profile = profile is not None

        chain = self._get_evaluator()
        if not self._rearm_stop_event(chain) or self._shutdown_requested.is_set():
            return EvalSummary(0, 0, len(jobs), beyond_limit, STOPPED_SHUTDOWN)
        total = len(jobs)
        settings = self.settings
        logger.info(
            "Evaluating %d unprocessed postings with %d workers...", total, settings.eval_max_workers
        )

        evaluated_count = 0
        wow_sent = 0
        failures = 0
        done = 0
        per_provider: Counter = Counter()
        batch_stopped = False
        t0 = time.monotonic()

        stats_lock = threading.Lock()
        call_latencies: List[float] = []
        in_flight = 0
        in_flight_peak = 0

        def timed_eval(job: JobPosting) -> Tuple[Any, float]:
            nonlocal in_flight, in_flight_peak
            started = time.monotonic()
            with stats_lock:
                in_flight += 1
                in_flight_peak = max(in_flight_peak, in_flight)
            try:
                if not has_profile:  # no profile: the chain's own default policy, legacy one-argument call
                    return chain.evaluate_job(job), time.monotonic() - started
                return chain.evaluate_job(job, policy=policy), time.monotonic() - started
            finally:
                with stats_lock:
                    in_flight -= 1
                    call_latencies.append(time.monotonic() - started)

        # No ``with``: its __exit__ would shutdown(wait=True) and block shutdown for minutes.
        executor = ThreadPoolExecutor(max_workers=max(1, settings.eval_max_workers), thread_name_prefix="eval")
        futures: Dict[Future, JobPosting] = {}
        self._futures = []
        try:
            for job in jobs:
                if self._shutdown_requested.is_set():
                    break
                future = executor.submit(timed_eval, job)
                futures[future] = job
                self._futures.append(future)
            if self._shutdown_requested.is_set():  # a signal landed while submitting
                for future in futures:
                    future.cancel()
            for future in _iter_completed(futures):
                job = futures[future]
                try:
                    eval_res, latency = future.result()
                except (CancelledError, StopRequested):
                    continue
                except ProvidersUnavailable as e:
                    # Provider-wide outage: job stays unprocessed with attempts unchanged.
                    if not batch_stopped:
                        batch_stopped = True
                        logger.error(
                            "All evaluation providers unavailable (%s); stopping this batch. "
                            "Remaining jobs stay 'unprocessed' for the next cycle.",
                            e,
                        )
                        executor.shutdown(wait=False, cancel_futures=True)
                        self._stop_chain()
                    continue
                except EvaluationRejected as e:
                    failures += 1
                    done += 1
                    logger.warning(
                        "[%d/%d] job %s ('%s') rejected by every provider: %s", done, total, job.id, job.title, e
                    )
                    if not dry_run:
                        self.db.record_eval_attempt(job.id, str(e), settings.eval_max_attempts)
                    continue
                except Exception as e:  # keep the pre-existing exception -> outcome mapping
                    failures += 1
                    done += 1
                    self._handle_legacy_failure(job, e, dry_run)
                    continue

                done += 1
                model = eval_res.eval_model if isinstance(eval_res.eval_model, str) else None
                provider_name = self._provider_for_model(chain, model)
                per_provider[provider_name] += 1
                logger.info(
                    "[%d/%d] %s %s %.1fs %s -> %.1f",
                    done, total, provider_name, model or "unknown-model", latency, job.title, eval_res.score,
                )

                # The alert decision lives here, from the cycle's policy, once. WOW additionally requires a
                # core role (while the cap is on), so a lowered threshold can never promote capped roles.
                eval_res.is_wow = decide_wow(
                    eval_res.score,
                    hard_reject=eval_res.hard_reject,
                    is_core_role=eval_res.is_core_role,
                    policy=policy,
                )
                if dry_run:
                    evaluated_count += 1
                    if eval_res.is_wow:
                        wow_sent += 1
                    continue

                notification_status = None
                if eval_res.score < policy.digest_threshold and not eval_res.is_wow:
                    notification_status = "dismissed"

                self.db.update_evaluation(
                    job_id=job.id, eval_result=eval_res, notification_status=notification_status
                )
                evaluated_count += 1
                if notification_status:
                    job.notification_status = notification_status

                if eval_res.is_wow:
                    job.eval_score = eval_res.score
                    job.eval_is_wow = eval_res.is_wow
                    job.eval_summary_uk = eval_res.summary
                    job.eval_reasoning = eval_res.reasoning
                    job.eval_languages = eval_res.language_requirements
                    job.eval_pros = eval_res.pros
                    job.eval_cons = eval_res.cons
                    sent = self.notifier.send_wow_alert(
                        job, lang=profile.output_language if profile else None
                    )
                    if sent:
                        self.db.mark_instant_sent(job.id)
                        wow_sent += 1
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
            self._futures = []

        wall = time.monotonic() - t0
        rpm = (done / wall * 60.0) if wall > 0 else 0.0
        provider_counts = ", ".join(f"{k}={v}" for k, v in sorted(per_provider.items())) or "none"
        with stats_lock:
            lat = latency_stats(list(call_latencies))
            peak = in_flight_peak
        logger.info(
            "Evaluation batch finished: processed=%d/%d evaluated=%d wow=%d failures=%d providers[%s] "
            "wall=%.1fs effective=%.1f rpm latency[mean=%.1fs p95=%.1fs max=%.1fs] inflight_peak=%d%s",
            done, total, evaluated_count, wow_sent, failures, provider_counts, wall, rpm,
            lat.mean, lat.p95, lat.max, peak,
            " (batch aborted: providers unavailable)" if batch_stopped else "",
        )
        not_scored = total - done
        if batch_stopped:
            stopped_reason: Optional[str] = STOPPED_PROVIDERS_UNAVAILABLE
        elif not_scored and self._shutdown_requested.is_set():
            stopped_reason = STOPPED_SHUTDOWN
        else:
            stopped_reason = None
        return EvalSummary(evaluated_count, wow_sent, not_scored, beyond_limit, stopped_reason)

    def _handle_legacy_failure(self, job: JobPosting, e: Exception, dry_run: bool) -> None:
        """Pre-existing mapping: transient errors leave the job unprocessed, others fail it."""
        if isinstance(e, (httpx.TimeoutException, httpx.NetworkError)):
            logger.warning(
                f"Transient network/timeout error evaluating job {job.id} ('{job.title}'): {e}. "
                "Leaving in 'unprocessed' status for subsequent cycle retry."
            )
        elif isinstance(e, httpx.HTTPStatusError):
            if e.response.status_code in (429, 500, 502, 503, 504):
                logger.warning(
                    f"Transient HTTP {e.response.status_code} evaluating job {job.id} ('{job.title}'): {e}. "
                    "Leaving in 'unprocessed' status for subsequent cycle retry."
                )
            else:
                logger.error(f"Non-recoverable HTTP {e.response.status_code} evaluating job {job.id}: {e}")
                if not dry_run:
                    self.db.update_eval_failure(job.id, str(e))
        elif any(k in str(e).lower() for k in ["503", "502", "504", "429", "timeout", "connection", "network"]):
            logger.warning(
                f"Transient error evaluating job {job.id} ('{job.title}'): {e}. "
                "Leaving in 'unprocessed' status for subsequent cycle retry."
            )
        else:
            logger.error(f"Permanent evaluation failure for job {job.id}: {e}")
            if not dry_run:
                self.db.update_eval_failure(job.id, str(e))

    def _begin_scrape_progress(self, queries: List[SearchQuery]) -> List[Tuple[SearchQuery, str]]:
        """Starts the scraper's progress tracking and returns the (query, board) calls to make."""
        begin = getattr(self.scraper, "begin_cycle", None)
        plan = begin(queries) if callable(begin) else None
        if isinstance(plan, list):
            return plan
        # Scraper without a call plan (e.g. a test double): run every configured board.
        return [(q, site) for q in queries for site in q.site_names]

    @staticmethod
    def _skip_evaluation_without_profile() -> None:
        # A score computed against the no-profile prompt would dismiss rows for good; the jobs stay queued.
        logger.warning(
            "Profile database is busy and there is no last good profile: evaluation skipped this cycle "
            "(jobs stay unprocessed)."
        )

    def _scrape_call(
        self, site: str, query: SearchQuery, rules: Optional[Sequence[CompiledRule]] = None
    ) -> ScrapeOutcome:
        """One (query, board) scrape as a ScrapeOutcome; wraps the plain list of a legacy test double.

        ``rules`` (the cycle profile's compiled rules) go to the scraper only when it accepts the keyword,
        so a double with the older ``(site, query)`` signature still works.
        """
        outcome_fn = getattr(self.scraper, "scrape_board_outcome", None)
        if callable(outcome_fn):
            if rules is not None and self._accepts_rules(outcome_fn):
                result = outcome_fn(site, query, rules=rules)
            else:
                if rules is not None and not self._rules_dropped_warned:
                    self._rules_dropped_warned = True
                    logger.warning(
                        "The scraper's scrape_board_outcome does not accept `rules`: the profile's rules are "
                        "not applied while scraping (only the title safety net at evaluation uses them)."
                    )
                result = outcome_fn(site, query)
            if isinstance(result, ScrapeOutcome):
                return result
        postings = self.scraper.scrape_board(site, query)
        return ScrapeOutcome(
            postings=postings, rows=len(postings), status=STATUS_OK if postings else STATUS_EMPTY
        )

    @staticmethod
    def _accepts_rules(fn: Any) -> bool:
        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            return False
        return "rules" in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())

    def is_cycle_running(self) -> bool:
        """A scrape/evaluate cycle or a digest is running (the bot refuses /requeue meanwhile)."""
        return self._cycle_lock.locked() or self._digest_lock.locked()

    @contextmanager
    def cycle_guard(self) -> Iterator[bool]:
        """Hold the cycle and digest locks (non-blocking, atomic) while the body runs; yields False if one is held.

        Used by ``/requeue``: its writes must not interleave with a cycle or a digest, and a scheduled cycle that
        wants to start meanwhile is skipped (like any cycle that finds the lock held).

        ``_cycle_lock`` is taken and released together with ``_guard_active`` under ``_notify_lock``, so a
        ``run_now`` job that finds the lock held knows atomically whether a guard (not a cycle) holds it.
        """
        with self._notify_lock:
            if not self._cycle_lock.acquire(blocking=False):
                acquired = False
            else:
                acquired = True
                self._guard_active = True
        if not acquired:
            yield False
            return
        try:
            if not self._digest_lock.acquire(blocking=False):
                yield False
                return
            try:
                yield True
            finally:
                self._digest_lock.release()
        finally:
            with self._notify_lock:
                self._guard_active = False
                self._cycle_lock.release()

    def trigger_cycle(self, notify: bool = False) -> str:
        """Schedule an immediate cycle (``/run_now``, ``/resume``); the scheduler thread runs it.

        The answer is "started" when the job was queued while no cycle ran. If the interval job takes the lock
        first, the queued ``run_now`` cycle is skipped with a WARNING; a cycle is running either way, which is
        what the caller wanted. ``notify``: the next cycle that ends sends a run report (the flag is set only
        on "started", before the job is added).
        """
        if self._shutdown_requested.is_set() or self.scheduler is None:
            return TRIGGER_NOT_READY
        if self._cycle_lock.locked():
            return TRIGGER_BUSY
        if notify:
            with self._notify_lock:
                self._notify_requested = True
        self.scheduler.add_job(
            self.run_pipeline_cycle_report,
            kwargs={"run_now": True},
            id="run_now",
            replace_existing=True,
            misfire_grace_time=None,
        )
        return TRIGGER_STARTED

    def trigger_digest(self) -> str:
        """Schedule an immediate digest (``/digest``); an APScheduler executor thread runs it.

        "not ready" until the scheduler runs (the startup cycle runs before ``scheduler.start()``, so a queued
        job would wait for it), "busy" while a digest runs. A running scrape cycle does not refuse it (like
        the cron digest). Never blocks: the lock is only inspected here, the job takes it non-blocking.
        """
        scheduler = self.scheduler
        if self._shutdown_requested.is_set() or scheduler is None or not scheduler.running:
            return TRIGGER_NOT_READY
        if self._digest_lock.locked():
            return TRIGGER_BUSY
        # max_instances=2: a previous job may have released the lock but still be sending its result line;
        # APScheduler would skip this one (default 1) after "started". The lock serialises the digest bodies.
        scheduler.add_job(
            self._digest_now_job,
            id="digest_now",
            replace_existing=True,
            misfire_grace_time=None,
            max_instances=2,
        )
        return TRIGGER_STARTED

    def _digest_now_job(self) -> None:
        """The ``/digest`` job: the shared digest body without the update notice, then one result message.

        If the cron digest holds the lock, it says so and does nothing else. Rows are marked as the cron
        marks them (sent when at least one message was delivered), which is why a partial send warns.
        Failures are logged by exception type name only (no job titles or companies).
        """
        if not self._digest_lock.acquire(blocking=False):
            self._send_digest_line(DIGEST_ALREADY_RUNNING)
            return
        try:
            try:
                self._run_daily_digest_cycle(False, send_update_notice=False)
                text = format_digest_result(self._last_digest_send)
            except Exception as exc:
                logger.error("On-demand digest failed: %s", type(exc).__name__)
                text = f"The digest stopped: {esc(type(exc).__name__)}"
        finally:
            self._digest_lock.release()
        self._send_digest_line(text)

    def _send_digest_line(self, text: str) -> None:
        try:
            delivered = bool(self.notifier.send_message(text))
        except Exception as exc:
            logger.warning("Could not send the digest result: %s", type(exc).__name__)
            return
        if not delivered:
            logger.warning("Could not send the digest result.")

    def _take_notify(self) -> bool:
        with self._notify_lock:
            requested, self._notify_requested = self._notify_requested, False
        return requested

    def run_pipeline_cycle(
        self, queries: Optional[List[SearchQuery]] = None, dry_run: bool = False
    ) -> Tuple[int, int]:
        """One cycle without a report (the CLI ``scrape``); returns ``(new jobs saved, WOW alerts sent)``."""
        if not self._cycle_lock.acquire(blocking=False):
            logger.warning("A pipeline cycle is already running; this one is skipped.")
            return 0, 0
        try:
            results = self._run_pipeline_cycle(queries, dry_run)
            return results.scrape.total_new_saved, results.summary.wow_sent
        finally:
            self._cycle_lock.release()

    def run_pipeline_cycle_report(self, run_now: bool = False) -> Optional[CycleReport]:
        """Every daemon cycle (startup, interval, ``run_now``): run it, then send a report when one was asked for.

        Returns the report that was sent, or ``None`` when no report was asked for or another holder had the
        lock (then nothing else is touched). The notify flag is read and cleared in a ``finally`` inside
        ``_cycle_lock`` and the report is built there from this cycle's own results; it is sent after the lock
        is released. An exception is logged (type name only) and becomes the report's error in a flagged cycle;
        in an unflagged cycle it propagates unchanged (APScheduler logs it; ``start()`` logs a
        ``ConfigurationError``).

        ``run_now``: this is the job queued by ``trigger_cycle``. If it finds the lock held by ``cycle_guard``
        (``/requeue``), the requested run is dropped and nobody would read the flag, so it is cleared here (a
        later interval cycle must not send an unrequested report). If a cycle holds the lock, that cycle reads
        the flag at its end and reports instead.
        """
        with self._notify_lock:
            acquired = self._cycle_lock.acquire(blocking=False)
            if not acquired and run_now and self._guard_active and self._notify_requested:
                self._notify_requested = False
                logger.info("The requested run was skipped while /requeue ran; no report will be sent.")
        if not acquired:
            logger.warning("A pipeline cycle is already running; this one is skipped.")
            return None
        notify = False
        failure: Optional[Exception] = None
        report: Optional[CycleReport] = None
        results = _CycleResults()
        try:
            try:
                self._run_pipeline_cycle(None, False, results)
            except Exception as exc:
                failure = exc
            finally:
                notify = self._take_notify()
            if notify:
                if failure is not None:
                    logger.error("Pipeline cycle failed: %s", type(failure).__name__)
                try:
                    report = self._build_report(results, failure)
                except Exception as exc:  # the requested report is still sent, with what is known
                    logger.error("Could not build the run report: %s", type(exc).__name__)
                    report = _fallback_report(type(exc).__name__)
        finally:
            self._cycle_lock.release()
        if notify:
            self._send_report(report)
        if failure is not None and not notify:
            raise failure
        return report

    def _build_report(self, results: _CycleResults, failure: Optional[Exception]) -> CycleReport:
        stats, summary = results.scrape, results.summary
        if failure is None:
            error = None
        elif isinstance(failure, ConfigurationError):
            error = NO_LLM_KEY_ERROR
        else:
            error = type(failure).__name__
        threshold = (
            results.profile.scoring.digest_threshold
            if results.profile is not None
            else self.settings.digest_score_threshold
        )
        try:
            digest_waiting = self.db.count_pending_digest_jobs(threshold)
        except Exception as exc:  # the report must not fail on a busy database
            logger.warning("Could not count the jobs waiting for the digest: %s", type(exc).__name__)
            digest_waiting = 0
        return CycleReport(
            new_to_evaluate=stats.new_to_evaluate,
            prefiltered=stats.prefiltered,
            calls_ok=stats.ok,
            calls_empty=stats.empty,
            calls_failed=stats.failed,
            skipped_boards=stats.skipped_boards,
            problems=tuple(stats.problem_list()),
            evaluated=summary.evaluated,
            unscored_batch_limit=summary.beyond_limit,
            unscored_outage=summary.not_scored_in_batch if summary.stopped_reason else 0,
            wow_sent=summary.wow_sent,
            digest_waiting=digest_waiting,
            error=error,
            stopped_reason=summary.stopped_reason,
        )

    def _send_report(self, report: Optional[CycleReport]) -> None:
        """Send the run report once; a failure is logged (type name only) and not retried.

        One message, no chunking: the report is bounded by construction (<= 11 lines, problem reasons <= 60
        characters, at most 3 problem lines), far below Telegram's 3,800-character chunk size.
        """
        if report is None:
            return
        try:
            delivered = bool(self.notifier.send_message(format_cycle_report(report)))
        except Exception as exc:
            logger.warning("Could not send the run report: %s", type(exc).__name__)
            return
        if not delivered:
            logger.warning("The run report was not delivered; it is not retried.")

    def _busy_summary(self) -> EvalSummary:
        """Evaluation skipped because the profile database was busy: everything queued stays unscored."""
        limit = self.settings.eval_batch_limit
        try:
            backlog = self.db.count_unprocessed_jobs()
        except Exception as exc:  # the database is busy already; the cycle goes on without the count
            logger.warning("Could not count the unprocessed jobs: %s", type(exc).__name__)
            backlog = 0
        in_batch = min(backlog, limit) if limit else backlog
        return EvalSummary(0, 0, in_batch, backlog - in_batch, STOPPED_PROFILE_BUSY)

    def _run_pipeline_cycle(
        self,
        queries: Optional[List[SearchQuery]],
        dry_run: bool,
        results: Optional[_CycleResults] = None,
    ) -> _CycleResults:
        results = results if results is not None else _CycleResults()
        # One snapshot per cycle: the profile is compiled once here and drives both scraping and
        # evaluation, so a change committed mid-cycle only takes effect in the next cycle.
        profile, problem = self.current_profile(alert=not dry_run)
        results.profile = profile
        if queries is None:
            queries = self._profile_queries(profile, problem, dry_run)

        rules = profile.rules if profile is not None else None
        if queries:
            results.scrape = self._scrape_queries(queries, dry_run, rules)
        total_new_saved = results.scrape.total_new_saved

        if dry_run:
            results.scrape = ScrapeStats()  # a dry run reports (0, 0), as before
            return results

        if problem == "db_busy" and profile is None:
            self._skip_evaluation_without_profile()
            results.summary = self._busy_summary()
            return results

        # Process evaluations
        results.summary = self.evaluate_unprocessed(
            limit=self.settings.eval_batch_limit, dry_run=False, profile=profile
        )
        logger.info(f"Pipeline cycle completed. New: {total_new_saved}, WOW alerts: {results.summary.wow_sent}")
        return results

    def _profile_queries(
        self, profile: Optional[CompiledProfile], problem: Optional[str], dry_run: bool
    ) -> List[SearchQuery]:
        """The queries a profile-driven cycle scrapes (empty = skip scraping, with the reason logged/alerted)."""
        if problem == "no_profile":
            logger.warning("No profile: scraping skipped. Run `jobspy-coach init`.")
            if not dry_run:
                self._alert_once("no_profile", "no profile: run jobspy-coach init")
            return []
        if problem == "db_busy":
            return []  # already logged by current_profile; nothing to alert about
        if problem == "compile_error":
            logger.warning("Profile does not compile: scraping skipped this cycle.")
            return []
        assert profile is not None
        if profile.paused:
            logger.info("Profile is paused: scraping skipped, evaluation continues.")
            return []
        if not profile.queries:
            logger.warning("The profile compiles to no search queries: scraping skipped.")
            if not dry_run:
                self._alert_once(
                    "empty_queries", "the profile has no search queries to run; add roles and markets"
                )
            return []
        self._rearm_alerts("empty_queries")
        return list(profile.queries)

    def _scrape_queries(
        self,
        active_queries: List[SearchQuery],
        dry_run: bool,
        rules: Optional[Sequence[CompiledRule]] = None,
    ) -> ScrapeStats:
        logger.info("Starting pipeline scrape cycle across queries...")
        stats = ScrapeStats()

        # The scraper owns the progress lines ([i/N] before/after every call, per-board summary);
        # the call plan holds only the calls that will really be requested (enabled boards, country set).
        plan = self._begin_scrape_progress(active_queries)

        try:
            for idx, (q, site) in enumerate(plan, start=1):
                jitter_max = getattr(self.scraper, "jitter_max", 0.0)
                if idx > 1 and isinstance(jitter_max, (int, float)) and jitter_max > 0:
                    jitter_min = getattr(self.scraper, "jitter_min", 0.0)
                    min_val = jitter_min if isinstance(jitter_min, (int, float)) else 0.0
                    time.sleep(random.uniform(min_val, jitter_max))

                t0 = time.time()
                recorded = False
                try:
                    outcome = self._scrape_call(site, q, rules)
                    stats.record(q.query_id, site, outcome)
                    recorded = True
                    board_postings = outcome.postings
                    duration = time.time() - t0
                    board_new_saved = 0
                    hit_key = query_key(q, site)

                    for posting in board_postings:
                        if dry_run:
                            logger.info(
                                f"[DRY-RUN] Discovered: '{posting.title}' at '{posting.company}' (Pre-filtered: {posting.pre_filtered})"
                            )
                            continue

                        # save_job returns the id for known jobs too; source_query is only stored for new rows.
                        job_id, is_new = self.db.save_job(posting)
                        self.db.record_job_hit(job_id, hit_key)
                        if is_new:
                            board_new_saved += 1
                            stats.record_new(posting)

                    if not dry_run:
                        logger.info(
                            "[%d/%d] %s %s saved %d new (%d already known)",
                            idx, len(plan), q.query_id, site, board_new_saved, len(board_postings) - board_new_saved,
                        )
                        failed = outcome.status == STATUS_FAILED
                        self.db.record_scrape_run(
                            board=site,
                            query=q.search_term,
                            jobs_found=len(board_postings),
                            new_jobs_saved=board_new_saved,
                            duration_sec=duration,
                            status="failed" if failed else "completed",
                            error_message=sanitize_error_message(outcome.error) if failed else None,
                            rows_returned=outcome.rows,
                            prefiltered=outcome.prefiltered,
                            outcome=outcome.status,
                            query_key=hit_key,
                        )
                except Exception as e:
                    duration = time.time() - t0
                    if not recorded:  # a failure after the call (e.g. saving) is not a scrape problem
                        stats.record_exception(q.query_id, site, e)
                    safe_error = sanitize_error_message(str(e))
                    logger.error(
                        f"Failed scrape cycle for board '{site}' query '{q.query_id}': {redact_secrets(str(e))}"
                    )
                    if not dry_run:
                        self.db.record_scrape_run(
                            board=site,
                            query=q.search_term,
                            jobs_found=0,
                            new_jobs_saved=0,
                            duration_sec=duration,
                            status="failed",
                            error_message=safe_error,
                            outcome="failed",
                            query_key=query_key(q, site),
                        )
        finally:
            end_cycle = getattr(self.scraper, "end_cycle", None)
            if callable(end_cycle):
                skipped = end_cycle()
                # A test double may return anything; only a real count is taken.
                if isinstance(skipped, int) and not isinstance(skipped, bool):
                    stats.skipped_boards = skipped
        return stats

    def run_daily_digest_cycle(self, dry_run: bool = False, *, send_update_notice: bool = True) -> int:
        with self._digest_lock:
            return self._run_daily_digest_cycle(dry_run, send_update_notice=send_update_notice)

    def _run_daily_digest_cycle(self, dry_run: bool, *, send_update_notice: bool = True) -> int:
        """The digest body (caller holds ``_digest_lock``); returns the number of qualifying rows taken.

        ``send_update_notice``: check PyPI for a python-jobspy update and append the notice (the cron digest;
        ``/digest`` skips it). Records a ``DigestSend`` on ``_last_digest_send`` (not on a dry run).
        """
        self._last_digest_send = None
        if not dry_run:
            expired = self.db.expire_stale_pending_jobs(days=STALE_PENDING_DAYS)
            if expired > 0:
                logger.info(
                    f"Transitioned {expired} stale pending jobs (>{STALE_PENDING_DAYS} days) to 'expired_dismissed'."
                )

        profile = self.current_profile(alert=False)[0]
        lang = profile.output_language if profile else None
        digest_threshold = (
            profile.scoring.digest_threshold if profile is not None else self.settings.digest_score_threshold
        )
        pending = self.db.get_pending_digest_jobs(
            min_score=digest_threshold,
            limit=self.settings.digest_max_items,
        )
        if not pending:
            logger.info("No pending qualifying jobs for daily digest.")
            if not dry_run:
                self._last_digest_send = DigestSend(sent=0, total=0)
            return 0

        date_str = datetime.now(ZoneInfo(self.settings.timezone)).strftime("%d.%m.%Y")
        if dry_run:
            logger.info(f"[DRY-RUN] Daily digest would contain {len(pending)} jobs.")
            return len(pending)

        update_notice = None
        if send_update_notice:
            try:
                version_info = check_jobspy_update()
                if version_info and version_info.has_update:
                    update_notice = format_version_update_notice(version_info, lang=lang)
            except Exception as e:
                logger.debug("Failed checking JobSpy update for daily digest: %s", e)

        # Rows left out by ``digest_max_items`` (counted before marking; only when the limit was reached).
        more = 0
        if len(pending) >= self.settings.digest_max_items:
            try:
                more = max(0, self.db.count_pending_digest_jobs(digest_threshold) - len(pending))
            except Exception as exc:  # the digest itself must still go out
                logger.warning("Could not count the jobs left for the next digest: %s", type(exc).__name__)

        sent_count = self.notifier.send_daily_digest(
            pending, date_str=date_str, update_notice=update_notice, lang=lang
        )
        if sent_count > 0:
            self.db.mark_digest_sent([j.id for j in pending if j.id is not None])
        self._last_digest_send = self._digest_send(sent_count, len(pending), more)
        return len(pending)

    def _digest_send(self, sent_count: Any, jobs: int, more: int) -> DigestSend:
        """The ``DigestSend`` of one digest; a notifier double without counts reads as fully delivered."""
        sent = sent_count if isinstance(sent_count, int) and not isinstance(sent_count, bool) else 0
        total = getattr(self.notifier, "last_digest_total", None)
        if not isinstance(total, int) or isinstance(total, bool):
            total = sent
        return DigestSend(sent=sent, total=total, jobs=jobs, more=more)

    # ----- bot -----

    def _on_paired(self, chat_id: int) -> None:
        """A new pairing: alerts and digests go to the paired chat from now on, without a restart.

        A new account never inherits the previous one's proposals, consent or clarification context: the
        conversation service forgets its memory state first (a request still in flight loses its result),
        then the three conversation tables are cleared.
        """
        self.notifier.chat_id = str(chat_id)
        try:
            if self._bot_router is not None:
                self._bot_router.conversation.reset_memory()
            ConversationStore(self.db).clear_all()
        except Exception as exc:  # pairing itself must go through
            logger.warning("could not clear the conversation state: %s", type(exc).__name__)

    def _start_bot(self) -> Optional[BotPoller]:
        """Start the Telegram bot poller thread; any failure is logged and the daemon runs without the bot.

        The poller stops on ``_shutdown_requested`` (set once, for good), never on ``_stop_event``, which is
        re-armed for every evaluation batch. It owns its ``Database`` instance (a second DB and Telegram
        thread by design; it never runs evaluation).
        """
        settings = self.settings
        if not settings.telegram_bot_token or not settings.telegram_bot_polling:
            return None
        try:
            client = TelegramBotClient(settings.telegram_bot_token, stop_event=self._shutdown_requested)
            state = BotStateStore(Database(self.db.db_path))
            adopt_chat_id(state, settings.telegram_chat_id)
            pairing = state.get_pairing()
            if not self.notifier.chat_id and pairing is not None:
                self.notifier.chat_id = str(pairing.chat_id)
            env_path = Path(".env")
            # Always built: without a key ``available()`` is False and free text answers "not configured".
            # It drives the chain's own providers (shared limiters and budget), stops only at process shutdown.
            self.chat_llm = ChatLLM(self._chat_providers, stop_event=self._shutdown_requested)
            ctx = BotContext(
                db_path=self.db.db_path,
                env_path=env_path,
                settings=settings,
                trigger_cycle=self.trigger_cycle,
                trigger_digest=self.trigger_digest,
                is_cycle_running=self.is_cycle_running,
                scraper=self.scraper,
                cycle_guard=self.cycle_guard,
                stop_event=self._shutdown_requested,
                chat_llm=self.chat_llm,
            )
            router = CommandRouter(ctx, client=client)
            poller = BotPoller(
                client,
                state,
                Authorizer(state),
                router,
                stop_event=self._shutdown_requested,
                env_path=env_path,
                on_paired=self._on_paired,
                commands=COMMANDS,
            )
            self._bot_router = router
            self._bot_thread = poller.start()
            self._bot_poller = poller
            # The free-text worker (daemon thread, stops on the shutdown event; never joined without a timeout).
            try:
                self._conversation_thread = router.conversation.start()
            except Exception as exc:  # free text is off; commands and the poller keep running
                logger.warning("Telegram conversation worker could not start: %s", type(exc).__name__)
            logger.info("Telegram bot polling started.")
            return poller
        except Exception as exc:  # never the message: it can hold a request URL
            logger.warning("Telegram bot could not start: %s", type(exc).__name__)
            return None

    def start(self) -> None:
        self.scheduler = BlockingScheduler(timezone=self.settings.timezone)

        # Scrape interval
        self.scheduler.add_job(
            self.run_pipeline_cycle_report,
            trigger=IntervalTrigger(hours=self.settings.scrape_interval_hours),
            id="scrape_pipeline",
            name="Scrape & Evaluate Pipeline",
        )

        # Daily digest cron at the configured local time (settings.timezone)
        hour, minute = self.settings.digest_time_cet.split(":")
        self.scheduler.add_job(
            self.run_daily_digest_cycle,
            trigger=CronTrigger(hour=int(hour), minute=int(minute)),
            id="daily_digest",
            name="Morning Digest",
        )

        def shutdown(signum, frame):
            logger.info("Received termination signal. Shutting down daemon gracefully...")
            self.request_shutdown()
            if self.scheduler and self.scheduler.running:
                self.scheduler.shutdown(wait=False)
            sys.exit(0)

        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGTERM, shutdown)

        try:
            version_info = check_jobspy_update()
            if version_info and version_info.has_update:
                logger.info(
                    "New JobSpy version available: v%s (installed: v%s). Run 'pip install -U python-jobspy' to update.",
                    version_info.latest_version,
                    version_info.installed_version,
                )
        except Exception as e:
            logger.debug("Failed startup JobSpy version check: %s", e)

        if self.evaluator is None and not (self.settings.nim_enabled or self.settings.openrouter_enabled):
            logger.warning(
                "No LLM provider is configured (set NVIDIA_API_KEY and/or OPENROUTER_API_KEY with "
                "OPENROUTER_MODELS): scraping will run but evaluation will fail until one is set."
            )

        self._start_bot()  # before the (long) startup cycle, so the bot answers while it runs

        logger.info("Running initial startup scraping cycle...")
        try:
            self.run_pipeline_cycle_report()
        except ConfigurationError as exc:
            logger.error(
                "Startup cycle could not evaluate jobs: %s. Scraping continues on schedule; "
                "queued jobs stay unprocessed until an LLM provider is configured.",
                exc,
            )

        logger.info(
            f"Daemon started. Scrape every {self.settings.scrape_interval_hours}h, "
            f"Digest at {self.settings.digest_time_cet} {self.settings.timezone}."
        )
        self.scheduler.start()
