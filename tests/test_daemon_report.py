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
    assert notifier.sent == []  # unflagged: no report sent


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
    report = daemon.run_pipeline_cycle_report()
    assert report.unscored_outage == outage
    assert report.unscored_batch_limit == summary.beyond_limit
    text = format_cycle_report(report)
    assert phrase in text
    if summary.beyond_limit:
        assert "10 left for the next cycle (batch limit)" in text


def test_no_stop_reason_means_no_outage(tmp_path):
    daemon, _ = _make(tmp_path, summary=EvalSummary(5, 0, 0, 12, None))
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
    assert kwargs == {"id": "run_now", "replace_existing": True, "misfire_grace_time": None}
    func()
    assert len(notifier.sent) == 1
    assert "Evaluated: 2" in notifier.sent[0]
    func()  # the flag was cleared: a later cycle reports nothing
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
    ((run_now_job, _),) = daemon.scheduler.jobs
    results = {}
    scheduled = threading.Thread(target=lambda: results.setdefault("r", daemon.run_pipeline_cycle_report()))
    scheduled.start()
    assert scraper.entered.wait(WAIT)
    assert run_now_job() is None  # lock held: skipped, touches nothing
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


def test_exception_in_flagged_cycle_clears_flag_and_reports(tmp_path):
    outcomes = {("Q1", "indeed"): ScrapeOutcome(postings=[_posting(1)], rows=1, status=STATUS_OK)}
    daemon, notifier = _make(tmp_path, FakeScraper(outcomes), [_q("Q1", ["indeed"])])
    daemon.evaluate_unprocessed = MagicMock(side_effect=RuntimeError("secret-detail"))
    daemon.trigger_cycle(notify=True)
    report = daemon.run_pipeline_cycle_report()
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
