"""Run report after a manual cycle (/run_now, /resume): CycleReport, notify flag and delivery (no network)."""
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import pytest

from src.bot.commands import TRIGGER_BUSY, TRIGGER_STARTED
from src.config import Settings
from src.evaluator.errors import ConfigurationError
from src.scheduler.daemon import (
    STOPPED_PROFILE_BUSY,
    STOPPED_PROVIDERS_UNAVAILABLE,
    STOPPED_SHUTDOWN,
    CycleReport,
    DaemonService,
    EvalSummary,
    format_cycle_report,
)
from src.scraper.outcome import STATUS_EMPTY, STATUS_FAILED, STATUS_OK, ScrapeOutcome
from src.scraper.query_catalog import SearchQuery
from src.storage.models import JobPosting

WAIT = 5.0


def _settings(**kw) -> Settings:
    base = dict(EVAL_BATCH_LIMIT=50, DIGEST_SCORE_THRESHOLD=6.5, WOW_SCORE_THRESHOLD=8.5)
    base.update(kw)
    return Settings(_env_file=None, **base)


def _posting(n: int, pre_filtered: bool = False) -> JobPosting:
    return JobPosting(
        dedup_hash=f"report_{n}",
        site="indeed",
        job_url=f"https://example.test/{n}",
        title=f"Role {n}",
        company="Example Co",
        location="Remote",
        description="A role description that is long enough to be meaningful for the tests here.",
        pre_filtered=pre_filtered,
    )


class FakeScraper:
    """Scripted outcomes per (query id, board); ``end_cycle`` returns the skipped-board count."""

    def __init__(self, outcomes: Optional[Dict[Tuple[str, str], ScrapeOutcome]] = None, skipped: int = 0):
        self.outcomes = outcomes or {}
        self.skipped = skipped
        self.entered = threading.Event()
        self.release: Optional[threading.Event] = None

    def scrape_board_outcome(self, site, query):
        self.entered.set()
        if self.release is not None:
            assert self.release.wait(WAIT)
        return self.outcomes.get((query.query_id, site), ScrapeOutcome(postings=[], rows=0, status=STATUS_EMPTY))

    def end_cycle(self):
        return self.skipped


class FakeNotifier:
    def __init__(self, daemon_ref: List[DaemonService], result=True):
        self.daemon_ref = daemon_ref
        self.result = result
        self.sent: List[str] = []
        self.running_at_send: List[bool] = []
        self.chat_id = "1"

    def send_message(self, html_text, reply_markup=None):
        self.sent.append(html_text)
        self.running_at_send.append(self.daemon_ref[0].is_cycle_running())
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


class StubScheduler:
    def __init__(self, running: bool = True):
        self.running = running
        self.jobs: List[Tuple[object, dict]] = []

    def add_job(self, func, **kwargs):
        self.jobs.append((func, kwargs))


def _make(tmp_path: Path, scraper=None, queries=None, summary=None, notifier_result=True, **settings):
    ref: List[DaemonService] = []
    notifier = FakeNotifier(ref, notifier_result)
    daemon = DaemonService(db_path=tmp_path / "report.db", scraper=scraper or FakeScraper(), notifier=notifier)
    ref.append(daemon)
    daemon.settings = _settings(**settings)
    # No stored profile is needed: the cycle scrapes the given queries with no profile.
    daemon.current_profile = lambda alert=True: (None, None)  # type: ignore[assignment]
    daemon._profile_queries = lambda profile, problem, dry_run: list(queries or [])  # type: ignore[assignment]
    if summary is not None:
        daemon.evaluate_unprocessed = MagicMock(return_value=summary)
    daemon.scheduler = StubScheduler()
    return daemon, notifier


def _pending_for_digest(daemon: DaemonService, scores: List[float]) -> None:
    for i, score in enumerate(scores):
        job_id, _ = daemon.db.save_job(_posting(1000 + i))
        with daemon.db._get_connection() as conn:
            conn.execute(
                "UPDATE jobs SET eval_status='evaluated', notification_status='pending', eval_score=? WHERE id=?",
                (score, job_id),
            )


def _q(qid: str, sites: List[str]) -> SearchQuery:
    return SearchQuery(qid, "term", "loc", sites, country_code="germany")


# --- report fields ---------------------------------------------------------------------------------


def test_report_fields_from_scrape_outcomes_and_eval_summary(tmp_path):
    outcomes = {
        ("Q1", "linkedin"): ScrapeOutcome(postings=[], rows=0, status=STATUS_FAILED, error="HTTP 429 from guest api"),
        ("Q1", "google"): ScrapeOutcome(postings=[], rows=0, status=STATUS_FAILED, error="Google: no job data returned"),
        ("Q1", "indeed"): ScrapeOutcome(
            postings=[_posting(1), _posting(2, pre_filtered=True), _posting(3)], rows=3, status=STATUS_OK, prefiltered=1
        ),
        ("Q2", "linkedin"): ScrapeOutcome(postings=[_posting(4)], rows=3, status=STATUS_OK, skipped_empty=2),
        ("Q2", "indeed"): ScrapeOutcome(postings=[], rows=0, status=STATUS_FAILED, error="x" * 200),
        ("Q2", "glassdoor"): ScrapeOutcome(postings=[], rows=0, status=STATUS_EMPTY),
    }
    queries = [_q("Q1", ["linkedin", "google", "indeed"]), _q("Q2", ["linkedin", "indeed", "glassdoor"])]
    daemon, notifier = _make(
        tmp_path, FakeScraper(outcomes, skipped=1), queries, summary=EvalSummary(3, 1, 0, 7, None)
    )
    _pending_for_digest(daemon, [7.0, 9.0, 5.0])
    daemon.trigger_cycle(notify=True)

    report = daemon.run_pipeline_cycle_report()

    assert report == CycleReport(
        new_to_evaluate=3,
        prefiltered=1,
        calls_ok=2,
        calls_empty=1,
        calls_failed=3,
        skipped_boards=1,
        problems=(
            ("linkedin", "blocked (429)", 1),
            ("google", "no data returned", 1),
            ("linkedin", "detail pages blocked", 1),
            ("indeed", "x" * 60, 1),
        ),
        evaluated=3,
        unscored_batch_limit=7,
        unscored_outage=0,
        wow_sent=1,
        digest_waiting=2,
        error=None,
    )
    assert len(notifier.sent) == 1


def test_unflagged_cycle_builds_no_report(tmp_path):
    daemon, notifier = _make(tmp_path, summary=EvalSummary(1, 0, 0, 0, None))
    daemon.db.count_pending_digest_jobs = MagicMock(side_effect=AssertionError("no count without a report"))
    assert daemon.run_pipeline_cycle_report() is None
    assert notifier.sent == []


@pytest.mark.parametrize(
    "summary, outage, phrase",
    [
        (EvalSummary(2, 0, 4, 10, STOPPED_PROVIDERS_UNAVAILABLE), 4, "4 not scored (no AI provider was available)"),
        (EvalSummary(0, 0, 5, 0, STOPPED_SHUTDOWN), 5, "5 not scored (the daemon was stopping)"),
        (EvalSummary(0, 0, 6, 0, STOPPED_PROFILE_BUSY), 6, "6 not scored (the profile was busy)"),
    ],
)
def test_unscored_split_batch_limit_vs_outage(tmp_path, summary, outage, phrase):
    daemon, _ = _make(tmp_path, summary=summary)
    daemon.trigger_cycle(notify=True)
    report = daemon.run_pipeline_cycle_report()
    assert report.unscored_outage == outage
    assert report.unscored_batch_limit == summary.beyond_limit
    text = format_cycle_report(report)
    assert phrase in text
    if summary.beyond_limit:
        assert "10 left for the next cycle (batch limit)" in text


def test_no_stop_reason_means_no_outage(tmp_path):
    daemon, _ = _make(tmp_path, summary=EvalSummary(5, 0, 0, 12, None))
    daemon.trigger_cycle(notify=True)
    report = daemon.run_pipeline_cycle_report()
    assert (report.unscored_outage, report.unscored_batch_limit) == (0, 12)
    text = format_cycle_report(report)
    assert "12 left for the next cycle (batch limit)" in text
    assert "not scored" not in text


# --- flag semantics and delivery ---------------------------------------------------------------------


def test_run_now_flagged_cycle_sends_exactly_one_report(tmp_path):
    daemon, notifier = _make(tmp_path, summary=EvalSummary(2, 1, 0, 0, None))
    assert daemon.trigger_cycle(notify=True) == TRIGGER_STARTED
    ((func, kwargs),) = daemon.scheduler.jobs
    assert func == daemon.run_pipeline_cycle_report
    assert kwargs == {"kwargs": {"run_now": True}, "id": "run_now", "replace_existing": True, "misfire_grace_time": None}
    func(**kwargs["kwargs"])
    assert len(notifier.sent) == 1
    assert "Evaluated: 2" in notifier.sent[0]
    func(**kwargs["kwargs"])  # the flag was cleared: a later cycle reports nothing
    assert len(notifier.sent) == 1


def test_trigger_without_notify_sets_no_flag(tmp_path):
    daemon, notifier = _make(tmp_path, summary=EvalSummary(0, 0, 0, 0, None))
    assert daemon.trigger_cycle() == TRIGGER_STARTED
    daemon.run_pipeline_cycle_report()
    assert notifier.sent == []


def test_busy_answer_sets_no_flag(tmp_path):
    daemon, notifier = _make(tmp_path, summary=EvalSummary(0, 0, 0, 0, None))
    assert daemon._cycle_lock.acquire(blocking=False)
    try:
        assert daemon.trigger_cycle(notify=True) == TRIGGER_BUSY
    finally:
        daemon._cycle_lock.release()
    assert daemon.scheduler.jobs == []
    daemon.run_pipeline_cycle_report()
    assert notifier.sent == []


def test_scheduled_cycle_holding_the_lock_reports_and_skipped_run_now_sends_nothing(tmp_path):
    scraper = FakeScraper()
    scraper.release = threading.Event()
    daemon, notifier = _make(
        tmp_path, scraper, [_q("Q1", ["google"])], summary=EvalSummary(1, 0, 0, 0, None)
    )
    # /run_now queued its job (flag set), but the interval cycle takes the lock first.
    assert daemon.trigger_cycle(notify=True) == TRIGGER_STARTED
    ((run_now_job, job_kwargs),) = daemon.scheduler.jobs
    results = {}
    scheduled = threading.Thread(target=lambda: results.setdefault("r", daemon.run_pipeline_cycle_report()))
    scheduled.start()
    assert scraper.entered.wait(WAIT)
    assert run_now_job(**job_kwargs["kwargs"]) is None  # lock held by a cycle: skipped, the flag stays
    assert notifier.sent == []
    scraper.release.set()
    scheduled.join(WAIT)
    assert not scheduled.is_alive()
    assert results["r"] is not None
    assert len(notifier.sent) == 1  # the scheduled cycle read the flag at its end


def test_report_is_sent_after_the_lock_is_released(tmp_path):
    daemon, notifier = _make(tmp_path, summary=EvalSummary(0, 0, 0, 0, None))
    daemon.trigger_cycle(notify=True)
    daemon.run_pipeline_cycle_report()
    assert notifier.running_at_send == [False]


def test_configuration_error_flagged_reports(tmp_path):
    daemon, notifier = _make(tmp_path)
    daemon.evaluate_unprocessed = MagicMock(side_effect=ConfigurationError("No LLM provider is configured"))
    daemon.trigger_cycle(notify=True)
    report = daemon.run_pipeline_cycle_report()
    assert report is not None and report.error is not None
    assert len(notifier.sent) == 1
    assert "The cycle stopped: no LLM key is configured" in notifier.sent[0]
    assert daemon.is_cycle_running() is False


def test_configuration_error_unflagged_only_raises(tmp_path):
    daemon, notifier = _make(tmp_path)
    daemon.evaluate_unprocessed = MagicMock(side_effect=ConfigurationError("No LLM provider is configured"))
    with pytest.raises(ConfigurationError):
        daemon.run_pipeline_cycle_report()
    assert notifier.sent == []
    assert daemon.is_cycle_running() is False


def test_exception_in_flagged_cycle_clears_flag_and_reports(tmp_path, caplog):
    outcomes = {("Q1", "indeed"): ScrapeOutcome(postings=[_posting(1)], rows=1, status=STATUS_OK)}
    daemon, notifier = _make(tmp_path, FakeScraper(outcomes), [_q("Q1", ["indeed"])])
    daemon.evaluate_unprocessed = MagicMock(side_effect=RuntimeError("secret-detail"))
    daemon.trigger_cycle(notify=True)
    with caplog.at_level(logging.ERROR, logger="src.scheduler.daemon"):
        report = daemon.run_pipeline_cycle_report()
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors == ["Pipeline cycle failed: RuntimeError"]
    assert not any("secret-detail" in r.getMessage() for r in caplog.records)
    assert report.error == "RuntimeError"
    assert report.new_to_evaluate == 1  # what the cycle did before it stopped is kept
    assert len(notifier.sent) == 1
    assert "The cycle stopped: RuntimeError" in notifier.sent[0]
    assert "secret-detail" not in notifier.sent[0]
    # the flag was cleared: the next (failing, unflagged) cycle raises and sends nothing
    with pytest.raises(RuntimeError):
        daemon.run_pipeline_cycle_report()
    assert len(notifier.sent) == 1


@pytest.mark.parametrize("result", [False, RuntimeError("telegram down")])
def test_failed_send_is_logged_and_not_retried(tmp_path, caplog, result):
    daemon, notifier = _make(tmp_path, summary=EvalSummary(0, 0, 0, 0, None), notifier_result=result)
    daemon.trigger_cycle(notify=True)
    with caplog.at_level(logging.WARNING, logger="src.scheduler.daemon"):
        report = daemon.run_pipeline_cycle_report()
    assert report is not None
    assert len(notifier.sent) == 1
    assert any("run report" in r.getMessage().lower() for r in caplog.records)
    assert not any("telegram down" in r.getMessage() for r in caplog.records)
    daemon.run_pipeline_cycle_report()
    assert len(notifier.sent) == 1


def test_startup_and_interval_cycles_without_flag_send_no_report(tmp_path, monkeypatch):
    scheduler = MagicMock()
    monkeypatch.setattr("src.scheduler.daemon.BlockingScheduler", MagicMock(return_value=scheduler))
    monkeypatch.setattr("src.scheduler.daemon.check_jobspy_update", MagicMock(return_value=None))
    monkeypatch.setattr("signal.signal", MagicMock())
    daemon, notifier = _make(tmp_path, summary=EvalSummary(0, 0, 0, 0, None))
    daemon.start()  # runs the startup cycle
    funcs = {kw["id"]: call.args[0] for call in scheduler.add_job.call_args_list for kw in [call.kwargs]}
    assert funcs["scrape_pipeline"] == daemon.run_pipeline_cycle_report
    funcs["scrape_pipeline"]()  # an interval cycle
    assert notifier.sent == []


def test_start_keeps_its_configuration_error_log(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr("src.scheduler.daemon.BlockingScheduler", MagicMock())
    monkeypatch.setattr("src.scheduler.daemon.check_jobspy_update", MagicMock(return_value=None))
    monkeypatch.setattr("signal.signal", MagicMock())
    daemon, notifier = _make(tmp_path)
    daemon.evaluate_unprocessed = MagicMock(side_effect=ConfigurationError("No LLM provider is configured"))
    with caplog.at_level(logging.ERROR, logger="src.scheduler.daemon"):
        daemon.start()
    assert any("Startup cycle could not evaluate jobs" in r.getMessage() for r in caplog.records)
    assert notifier.sent == []


def test_lock_held_returns_none(tmp_path):
    daemon, notifier = _make(tmp_path, summary=EvalSummary(0, 0, 0, 0, None))
    assert daemon._cycle_lock.acquire(blocking=False)
    try:
        assert daemon.run_pipeline_cycle_report() is None
    finally:
        daemon._cycle_lock.release()
    assert notifier.sent == []


def test_run_pipeline_cycle_still_returns_counts(tmp_path):
    outcomes = {("Q1", "indeed"): ScrapeOutcome(postings=[_posting(1)], rows=1, status=STATUS_OK)}
    daemon, notifier = _make(tmp_path, FakeScraper(outcomes), [_q("Q1", ["indeed"])], summary=EvalSummary(1, 1, 0, 0, None))
    daemon.trigger_cycle(notify=True)
    assert daemon.run_pipeline_cycle() == (1, 1)
    assert notifier.sent == []  # the CLI path never reports


# --- formatting --------------------------------------------------------------------------------------


def _report(**kw) -> CycleReport:
    base = dict(
        new_to_evaluate=0, prefiltered=0, calls_ok=0, calls_empty=0, calls_failed=0, skipped_boards=0,
        problems=(), evaluated=0, unscored_batch_limit=0, unscored_outage=0, wow_sent=0, digest_waiting=0,
        error=None,
    )
    base.update(kw)
    return CycleReport(**base)


def test_format_escapes_and_stays_short():
    problems = tuple((f"board{i}", f"<b>reason & {i}</b>", i + 1) for i in range(8))
    report = _report(
        new_to_evaluate=5, prefiltered=2, calls_ok=3, calls_empty=1, calls_failed=8, skipped_boards=2,
        problems=problems, evaluated=4, unscored_batch_limit=9, unscored_outage=3, wow_sent=1, digest_waiting=6,
        error="Weird<Error>", stopped_reason=STOPPED_PROVIDERS_UNAVAILABLE,
    )
    text = format_cycle_report(report)
    assert len(text.splitlines()) <= 12
    assert "<b>reason" not in text and "&lt;b&gt;reason &amp; 0&lt;/b&gt;" in text
    assert "Weird&lt;Error&gt;" in text
    assert "more" in text  # the problem list is cut
    assert text.rstrip().endswith("Send /digest to receive them now.")
    assert "6" in text.splitlines()[-1]


def test_format_without_digest_waiting_has_no_hint():
    text = format_cycle_report(_report(calls_ok=1, evaluated=1))
    assert "/digest" not in text
    assert len(text) < 1000


# --- /requeue (cycle_guard) and a pending run_now job ----------------------------------------------


def test_requeue_before_the_run_now_job_keeps_the_report(tmp_path):
    """The guard ran and ended before the run_now job started: the job runs its own cycle and reports once."""
    daemon, notifier = _make(tmp_path, summary=EvalSummary(1, 0, 0, 0, None))
    assert daemon.trigger_cycle(notify=True) == TRIGGER_STARTED
    ((job, kwargs),) = daemon.scheduler.jobs
    with daemon.cycle_guard() as ok:
        assert ok
    assert daemon._notify_requested is True
    assert job(**kwargs["kwargs"]) is not None
    assert len(notifier.sent) == 1
    daemon.run_pipeline_cycle_report()  # a later interval cycle
    assert len(notifier.sent) == 1


def test_run_now_job_skipped_by_the_guard_clears_the_flag(tmp_path):
    """The run_now job finds /requeue holding the lock: the run is dropped, so is its report."""
    daemon, notifier = _make(tmp_path, summary=EvalSummary(1, 0, 0, 0, None))
    assert daemon.trigger_cycle(notify=True) == TRIGGER_STARTED
    ((job, kwargs),) = daemon.scheduler.jobs
    with daemon.cycle_guard() as ok:
        assert ok
        assert job(**kwargs["kwargs"]) is None
        assert daemon.run_pipeline_cycle_report() is None  # an interval job skipped meanwhile changes nothing
    assert daemon._notify_requested is False
    assert daemon.is_cycle_running() is False
    daemon.run_pipeline_cycle_report()  # the next interval cycle
    assert notifier.sent == []


def test_interval_job_skipped_by_the_guard_keeps_a_pending_run_now_report(tmp_path):
    daemon, notifier = _make(tmp_path, summary=EvalSummary(1, 0, 0, 0, None))
    assert daemon.trigger_cycle(notify=True) == TRIGGER_STARTED
    ((job, kwargs),) = daemon.scheduler.jobs
    with daemon.cycle_guard():
        assert daemon.run_pipeline_cycle_report() is None  # the interval job, not the run_now job
    assert job(**kwargs["kwargs"]) is not None
    assert len(notifier.sent) == 1


def test_build_failure_in_a_flagged_cycle_still_reports(tmp_path, caplog, monkeypatch):
    daemon, notifier = _make(tmp_path)
    daemon.evaluate_unprocessed = MagicMock(side_effect=RuntimeError("secret-detail"))
    monkeypatch.setattr(daemon, "_build_report", MagicMock(side_effect=KeyError("x")))
    daemon.trigger_cycle(notify=True)
    with caplog.at_level(logging.ERROR, logger="src.scheduler.daemon"):
        report = daemon.run_pipeline_cycle_report()
    assert report.error == "KeyError"
    assert len(notifier.sent) == 1 and "The cycle stopped: KeyError" in notifier.sent[0]
    messages = [r.getMessage() for r in caplog.records]
    assert "Pipeline cycle failed: RuntimeError" in messages
    assert "Could not build the run report: KeyError" in messages
    assert daemon.is_cycle_running() is False


def test_busy_summary_logs_a_count_failure(tmp_path, caplog):
    daemon, _ = _make(tmp_path)
    daemon.db.count_unprocessed_jobs = MagicMock(side_effect=RuntimeError("secret-detail"))
    with caplog.at_level(logging.WARNING, logger="src.scheduler.daemon"):
        summary = daemon._busy_summary()
    assert summary == EvalSummary(0, 0, 0, 0, STOPPED_PROFILE_BUSY)
    assert any("RuntimeError" in r.getMessage() for r in caplog.records)
    assert not any("secret-detail" in r.getMessage() for r in caplog.records)


def test_format_all_boards_skipped_names_the_cause():
    text = format_cycle_report(_report(skipped_boards=2))
    assert "Searches: none ran (2 boards skipped: no country set)" in text
    assert "paused" not in text


# --- /digest: trigger_digest and the digest_now job ----------------------------------------------------

from src.bot.commands import TRIGGER_NOT_READY  # noqa: E402
from src.notifier.telegram_notifier import TelegramNotifier  # noqa: E402
from src.scheduler.daemon import DigestSend  # noqa: E402


class DigestNotifier(TelegramNotifier):
    """The real ``send_daily_digest`` (chunking, counting) over scripted ``send_message`` results."""

    def __init__(self, results=None):
        super().__init__(token="<TOKEN>", chat_id="1", client=MagicMock())
        self.results = list(results or [])
        self.sent: List[str] = []

    def send_message(self, html_text, reply_markup=None):
        self.sent.append(html_text)
        result = self.results.pop(0) if self.results else True
        if isinstance(result, BaseException):
            raise result
        return result


def _digest_daemon(tmp_path, monkeypatch, results=None, **settings):
    monkeypatch.setattr("src.notifier.telegram_notifier.time.sleep", lambda seconds: None)
    daemon, _ = _make(tmp_path, **settings)
    notifier = DigestNotifier(results)
    daemon.notifier = notifier
    return daemon, notifier


def _pending_with_summaries(daemon: DaemonService, count: int, chars: int = 590) -> None:
    _pending_for_digest(daemon, [7.0 + (i % 10) * 0.1 for i in range(count)])
    with daemon.db._get_connection() as conn:
        conn.execute("UPDATE jobs SET eval_summary_uk=?", ("s" * chars,))


def _status_counts(daemon: DaemonService) -> Dict[str, int]:
    with daemon.db._get_connection() as conn:
        rows = conn.execute("SELECT notification_status, COUNT(*) FROM jobs GROUP BY notification_status").fetchall()
    return {status: n for status, n in rows}


def test_trigger_digest_not_ready_without_a_running_scheduler(tmp_path):
    daemon, _ = _make(tmp_path)
    daemon.scheduler = None
    assert daemon.trigger_digest() == TRIGGER_NOT_READY
    daemon.scheduler = StubScheduler(running=False)  # the startup cycle runs before scheduler.start()
    assert daemon.trigger_digest() == TRIGGER_NOT_READY
    assert daemon.scheduler.jobs == []


def test_trigger_digest_busy_while_a_digest_runs(tmp_path):
    daemon, _ = _make(tmp_path)
    assert daemon._digest_lock.acquire(blocking=False)
    try:
        assert daemon.trigger_digest() == TRIGGER_BUSY
    finally:
        daemon._digest_lock.release()
    assert daemon.scheduler.jobs == []


def test_trigger_digest_is_not_refused_by_a_running_cycle(tmp_path):
    daemon, _ = _make(tmp_path)
    assert daemon._cycle_lock.acquire(blocking=False)
    try:
        assert daemon.trigger_digest() == TRIGGER_STARTED
    finally:
        daemon._cycle_lock.release()


def test_trigger_digest_queues_exactly_one_job(tmp_path):
    daemon, _ = _make(tmp_path)
    assert daemon.trigger_digest() == TRIGGER_STARTED
    ((func, kwargs),) = daemon.scheduler.jobs
    assert func == daemon._digest_now_job
    assert kwargs == {"id": "digest_now", "replace_existing": True, "misfire_grace_time": None, "max_instances": 2}


def test_trigger_digest_returns_at_once_while_another_thread_holds_the_lock(tmp_path):
    daemon, _ = _make(tmp_path)
    held, release = threading.Event(), threading.Event()

    def holder():
        with daemon._digest_lock:
            held.set()
            release.wait(WAIT)

    thread = threading.Thread(target=holder, daemon=True)
    thread.start()
    assert held.wait(WAIT)
    try:
        done: List[str] = []
        caller = threading.Thread(target=lambda: done.append(daemon.trigger_digest()), daemon=True)
        caller.start()
        caller.join(1.0)
        assert done == [TRIGGER_BUSY]
    finally:
        release.set()
        thread.join(WAIT)


def test_digest_now_sends_rows_once_and_they_leave_the_queue(tmp_path, monkeypatch):
    daemon, notifier = _digest_daemon(tmp_path, monkeypatch)
    _pending_for_digest(daemon, [7.0, 9.0])
    daemon.trigger_digest()
    ((job, _),) = daemon.scheduler.jobs
    job()
    assert len(notifier.sent) == 2  # one digest chunk, one result line
    assert "Role 1000" in notifier.sent[0] and "Role 1001" in notifier.sent[0]
    assert notifier.sent[-1] == "Sent 2 jobs"
    assert daemon._last_digest_send == DigestSend(sent=1, total=1, jobs=2, more=0)
    assert _status_counts(daemon) == {"digest_sent": 2}
    job()
    assert notifier.sent[-1] == "Nothing is waiting"
    assert daemon.run_daily_digest_cycle() == 0  # the next (cron) digest has nothing left


def test_digest_now_with_nothing_waiting(tmp_path, monkeypatch):
    daemon, notifier = _digest_daemon(tmp_path, monkeypatch)
    _pending_for_digest(daemon, [3.0])  # below the digest threshold
    daemon._digest_now_job()
    assert notifier.sent == ["Nothing is waiting"]


def test_digest_now_partial_send_warns(tmp_path, monkeypatch):
    daemon, notifier = _digest_daemon(tmp_path, monkeypatch, results=[True, False, False, False])
    _pending_with_summaries(daemon, 10)
    daemon._digest_now_job()
    total = notifier.last_digest_total
    assert total >= 2
    assert notifier.sent[-1] == f"Sent 1 of {total} messages; some jobs may be missing"
    assert _status_counts(daemon) == {"digest_sent": 10}  # marking rule unchanged: one delivered message


def test_digest_now_failed_send_keeps_rows_waiting(tmp_path, monkeypatch):
    daemon, notifier = _digest_daemon(tmp_path, monkeypatch, results=[False, False, False, False, False])
    _pending_with_summaries(daemon, 10)
    daemon._digest_now_job()
    assert notifier.sent[-1] == "Sending failed; the jobs stay waiting"
    assert _status_counts(daemon) == {"pending": 10}


def test_digest_now_truncation_line(tmp_path, monkeypatch):
    daemon, notifier = _digest_daemon(tmp_path, monkeypatch, DIGEST_MAX_ITEMS=2)
    _pending_for_digest(daemon, [7.0, 7.5, 8.0, 8.1, 6.9])
    daemon._digest_now_job()
    assert notifier.sent[-1] == "Sent 2 jobs\n3 more waiting, send /digest again"
    daemon._digest_now_job()
    assert notifier.sent[-1] == "Sent 2 jobs\n1 more waiting, send /digest again"
    daemon._digest_now_job()
    assert notifier.sent[-1] == "Sent 1 job"


def test_digest_now_truncation_line_after_a_failed_send(tmp_path, monkeypatch):
    daemon, notifier = _digest_daemon(tmp_path, monkeypatch, results=[False], DIGEST_MAX_ITEMS=2)
    _pending_for_digest(daemon, [7.0, 7.5, 8.0])
    daemon._digest_now_job()
    assert notifier.sent[-1] == "Sending failed; the jobs stay waiting\n1 more waiting, send /digest again"


def test_digest_now_skips_the_update_notice_but_the_cron_does_not(tmp_path, monkeypatch):
    check = MagicMock(return_value=None)
    monkeypatch.setattr("src.scheduler.daemon.check_jobspy_update", check)
    daemon, notifier = _digest_daemon(tmp_path, monkeypatch)
    _pending_for_digest(daemon, [7.0])
    daemon._digest_now_job()
    assert check.call_count == 0
    assert notifier.sent[-1] == "Sent 1 job"
    with daemon.db._get_connection() as conn:  # make the row wait again for the cron digest
        conn.execute("UPDATE jobs SET notification_status='pending'")
    assert daemon.run_daily_digest_cycle() == 1
    assert check.call_count == 1


def test_digest_now_loses_the_race_with_the_cron_digest(tmp_path, monkeypatch):
    daemon, notifier = _digest_daemon(tmp_path, monkeypatch)
    _pending_for_digest(daemon, [7.0])
    held, release = threading.Event(), threading.Event()

    def cron():
        with daemon._digest_lock:
            held.set()
            release.wait(WAIT)

    thread = threading.Thread(target=cron, daemon=True)
    thread.start()
    assert held.wait(WAIT)
    try:
        daemon._digest_now_job()
    finally:
        release.set()
        thread.join(WAIT)
    assert notifier.sent == ["A digest is already running"]
    assert _status_counts(daemon) == {"pending": 1}
    assert not daemon._digest_lock.locked()


def test_digest_now_uses_env_thresholds_without_a_profile(tmp_path, monkeypatch):
    daemon, notifier = _digest_daemon(tmp_path, monkeypatch, DIGEST_SCORE_THRESHOLD=7.2)
    _pending_for_digest(daemon, [7.0, 7.5])
    daemon._digest_now_job()
    assert notifier.sent[-1] == "Sent 1 job"
    assert _status_counts(daemon) == {"pending": 1, "digest_sent": 1}


def test_digest_now_error_is_reported_by_type_name_only(tmp_path, monkeypatch, caplog):
    daemon, notifier = _digest_daemon(tmp_path, monkeypatch)
    daemon.db.get_pending_digest_jobs = MagicMock(side_effect=RuntimeError("<SECRET> title"))
    with caplog.at_level(logging.DEBUG):
        daemon._digest_now_job()
    assert notifier.sent == ["The digest stopped: RuntimeError"]
    assert not daemon._digest_lock.locked()
    assert any("RuntimeError" in r.getMessage() for r in caplog.records)
    assert not any("<SECRET>" in r.getMessage() for r in caplog.records)


def test_digest_now_failed_result_line_is_logged_by_type_name(tmp_path, monkeypatch, caplog):
    daemon, notifier = _digest_daemon(tmp_path, monkeypatch, results=[True, RuntimeError("<SECRET> Role")])
    _pending_for_digest(daemon, [7.0])
    with caplog.at_level(logging.DEBUG):
        daemon._digest_now_job()
    assert _status_counts(daemon) == {"digest_sent": 1}
    assert any("RuntimeError" in r.getMessage() for r in caplog.records)
    assert not any("<SECRET>" in r.getMessage() or "Role 1000" in r.getMessage() for r in caplog.records)


def test_digest_now_right_after_a_job_that_is_still_sending_its_line(tmp_path, monkeypatch):
    # The first job released the lock but still sends its result line: a new /digest is "started", and its
    # job (a second instance, max_instances=2) runs and reports what is left instead of being skipped.
    daemon, notifier = _digest_daemon(tmp_path, monkeypatch, DIGEST_MAX_ITEMS=2)
    _pending_for_digest(daemon, [7.0, 7.5, 8.0])
    sending, release = threading.Event(), threading.Event()
    original = notifier.send_message

    def send_message(html_text, reply_markup=None):
        if html_text.startswith("Sent 2 jobs") and not release.is_set():
            sending.set()
            assert release.wait(WAIT)
        return original(html_text, reply_markup)

    notifier.send_message = send_message
    first = threading.Thread(target=daemon._digest_now_job, daemon=True)
    first.start()
    try:
        assert sending.wait(WAIT)
        assert not daemon._digest_lock.locked()
        assert daemon.trigger_digest() == TRIGGER_STARTED
        ((job, kwargs),) = daemon.scheduler.jobs
        assert kwargs["max_instances"] == 2
        job()
        assert notifier.sent[-1] == "Sent 1 job"
        job()
        assert notifier.sent[-1] == "Nothing is waiting"
    finally:
        release.set()
        first.join(WAIT)
    assert "Sent 2 jobs\n1 more waiting, send /digest again" in notifier.sent


def test_format_digest_result_nothing_sent_is_a_failure():
    from src.scheduler.daemon import format_digest_result

    assert format_digest_result(DigestSend(sent=0, total=0, jobs=2)) == "Sending failed; the jobs stay waiting"


def test_digest_now_with_a_notifier_without_counts_and_nothing_sent(tmp_path, monkeypatch):
    daemon, _ = _make(tmp_path)

    class CountlessNotifier:
        chat_id = "1"

        def __init__(self):
            self.sent: List[str] = []

        def send_daily_digest(self, jobs, date_str, update_notice=None, lang=None):
            return 0

        def send_message(self, html_text, reply_markup=None):
            self.sent.append(html_text)
            return True

    notifier = CountlessNotifier()
    daemon.notifier = notifier
    _pending_for_digest(daemon, [7.0])
    daemon._digest_now_job()
    assert notifier.sent == ["Sending failed; the jobs stay waiting"]
    assert _status_counts(daemon) == {"pending": 1}
