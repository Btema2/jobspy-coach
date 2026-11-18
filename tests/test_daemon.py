import pytest
from unittest.mock import MagicMock
from src.scheduler.daemon import DaemonService
from src.storage.models import JobPosting
from src.scraper.query_catalog import SearchQuery


def test_daemon_pipeline_cycle(monkeypatch, tmp_path):
    mock_scraper = MagicMock()
    mock_job = JobPosting(
        dedup_hash="hash_test",
        site="indeed",
        job_url="https://indeed.com/1",
        title="Executive Coach",
        company="Global Leadership",
        location="Remote Poland",
        description=(
            "A corporate leadership coaching opportunity with senior executives to develop organizational strategy "
            "and drive executive performance in high-growth companies across European and global markets. "
            "Requires proven leadership coaching background and executive mentoring."
        ),
        is_remote=True,
    )
    mock_scraper.scrape_board.return_value = [mock_job]

    mock_evaluator = MagicMock()
    mock_eval_result = MagicMock()
    mock_eval_result.score = 9.5
    mock_eval_result.is_wow = True
    mock_eval_result.summary_uk = "Чудова роль executive коуча."
    mock_eval_result.reasoning = "Top fit"
    mock_eval_result.language_requirements = ["English"]
    mock_eval_result.pros = []
    mock_eval_result.cons = []
    mock_eval_result.hard_reject = False
    mock_eval_result.eval_model = None
    mock_evaluator.evaluate_job.return_value = mock_eval_result

    mock_notifier = MagicMock()
    mock_notifier.send_wow_alert.return_value = True

    daemon = DaemonService(
        db_path=tmp_path / "daemon.db",
        scraper=mock_scraper,
        evaluator=mock_evaluator,
        notifier=mock_notifier,
    )

    test_query = SearchQuery(
        query_id="Q-TEST",
        search_term="Executive Coach",
        location="Remote Poland",
        site_names=["indeed"],
    )
    new_jobs, wow_sent = daemon.run_pipeline_cycle(queries=[test_query], dry_run=False)
    assert new_jobs == 1
    assert wow_sent == 1
    assert mock_notifier.send_wow_alert.called

    with daemon.db._get_connection() as conn:
        runs = conn.execute("SELECT * FROM scrape_runs").fetchall()
        assert len(runs) == 1
        assert runs[0]["board"] == "indeed"
        assert runs[0]["jobs_found"] == 1
        assert runs[0]["new_jobs_saved"] == 1
        assert runs[0]["status"] == "completed"


def test_daemon_evaluate_transient_error_leaves_unprocessed(tmp_path):
    db_file = tmp_path / "daemon_transient.db"
    mock_evaluator = MagicMock()
    mock_evaluator.evaluate_job.side_effect = RuntimeError("HTTP 503 Service Unavailable")

    daemon = DaemonService(
        db_path=db_file,
        evaluator=mock_evaluator,
    )

    job = JobPosting(
        dedup_hash="hash_transient",
        site="indeed",
        job_url="https://indeed.com/transient",
        title="Career Coach",
        company="Company",
        location="Warsaw",
        description="Career coaching role in Warsaw.",
    )
    job_id, _ = daemon.db.save_job(job)

    # Evaluate unprocessed
    evaluated_count, wow_sent = daemon.evaluate_unprocessed(limit=10)
    assert evaluated_count == 0
    assert wow_sent == 0

    # Job must remain 'unprocessed' to retry next cycle
    reloaded = daemon.db.get_job_by_hash("hash_transient")
    assert reloaded is not None
    assert reloaded.eval_status == "unprocessed"


def test_daemon_daily_digest_expires_stale_jobs(tmp_path):
    db_file = tmp_path / "daemon_digest_expire.db"
    daemon = DaemonService(db_path=db_file)

    job = JobPosting(
        dedup_hash="hash_stale",
        site="indeed",
        job_url="https://indeed.com/stale",
        title="Stale Coach",
        company="Old Corp",
        location="Warsaw",
        description="Stale coaching vacancy in Warsaw.",
        eval_status="evaluated",
        eval_score=7.5,
        notification_status="pending",
    )
    job_id, _ = daemon.db.save_job(job)
    with daemon.db._get_connection() as conn:
        conn.execute("UPDATE jobs SET created_at = DATETIME('now', '-8 days') WHERE id = ?", (job_id,))
        conn.commit()

    count = daemon.run_daily_digest_cycle(dry_run=False)
    # The job was expired (>7 days) so it was not included in digest
    assert count == 0
    reloaded = daemon.db.get_job_by_hash("hash_stale")
    assert reloaded.notification_status == "expired_dismissed"


def test_daemon_evaluate_permanent_failure_marks_failed(tmp_path):
    db_file = tmp_path / "daemon_perm_fail.db"
    mock_evaluator = MagicMock()
    mock_evaluator.evaluate_job.side_effect = ValueError("Invalid prompt format syntax")

    daemon = DaemonService(
        db_path=db_file,
        evaluator=mock_evaluator,
    )

    job = JobPosting(
        dedup_hash="hash_fail",
        site="linkedin",
        job_url="https://linkedin.com/fail",
        title="Leadership Trainer",
        company="Company X",
        location="Warsaw",
        description="Leadership training role.",
    )
    daemon.db.save_job(job)

    evaluated_count, wow_sent = daemon.evaluate_unprocessed(limit=10)
    assert evaluated_count == 0
    assert wow_sent == 0

    reloaded = daemon.db.get_job_by_hash("hash_fail")
    assert reloaded is not None
    assert reloaded.eval_status == "failed"
    assert "Invalid prompt format" in (reloaded.eval_error or "")


def test_daemon_daily_digest_sends_and_marks_sent(monkeypatch, tmp_path):
    monkeypatch.setattr("src.scheduler.daemon.check_jobspy_update", MagicMock(return_value=None))
    db_file = tmp_path / "daemon_digest_send.db"
    mock_notifier = MagicMock()
    mock_notifier.send_daily_digest.return_value = 1

    daemon = DaemonService(
        db_path=db_file,
        notifier=mock_notifier,
    )

    job = JobPosting(
        dedup_hash="hash_digest_ok",
        site="indeed",
        job_url="https://indeed.com/ok",
        title="Agile Coach",
        company="Agile Corp",
        location="Warsaw",
        description="Agile coaching role description.",
        eval_status="evaluated",
        eval_score=8.0,
        notification_status="pending",
    )
    daemon.db.save_job(job)

    count = daemon.run_daily_digest_cycle(dry_run=False)
    assert count == 1
    assert mock_notifier.send_daily_digest.called

    from datetime import datetime
    from zoneinfo import ZoneInfo
    expected_date = datetime.now(ZoneInfo(daemon.settings.timezone)).strftime("%d.%m.%Y")
    call_kwargs = mock_notifier.send_daily_digest.call_args[1]
    assert call_kwargs["date_str"] == expected_date

    reloaded = daemon.db.get_job_by_hash("hash_digest_ok")
    assert reloaded.notification_status == "digest_sent"


def test_daemon_pipeline_cycle_dry_run(tmp_path):
    mock_scraper = MagicMock()
    mock_job = JobPosting(
        dedup_hash="hash_dry",
        site="indeed",
        job_url="https://indeed.com/dry",
        title="Executive Coach",
        company="Global Leadership",
        location="Remote Poland",
        description="Corporate coaching role description for executive leadership.",
        is_remote=True,
    )
    mock_scraper.scrape_board.return_value = [mock_job]

    daemon = DaemonService(
        db_path=tmp_path / "daemon_dry.db",
        scraper=mock_scraper,
    )

    test_query = SearchQuery(
        query_id="Q-DRY",
        search_term="Executive Coach",
        location="Remote Poland",
        site_names=["indeed"],
    )
    new_jobs, wow_sent = daemon.run_pipeline_cycle(queries=[test_query], dry_run=True)
    assert new_jobs == 0
    assert wow_sent == 0

    with daemon.db._get_connection() as conn:
        runs = conn.execute("SELECT * FROM scrape_runs").fetchall()
        assert len(runs) == 0
        jobs = conn.execute("SELECT * FROM jobs").fetchall()
        assert len(jobs) == 0


def test_daemon_pipeline_scrape_board_failure_recorded(tmp_path):
    mock_scraper = MagicMock()
    mock_scraper.scrape_board.side_effect = RuntimeError("Scraper network error")

    daemon = DaemonService(
        db_path=tmp_path / "daemon_err.db",
        scraper=mock_scraper,
    )

    test_query = SearchQuery(
        query_id="Q-ERR",
        search_term="Career Coach",
        location="Warsaw",
        site_names=["linkedin"],
    )
    new_jobs, wow_sent = daemon.run_pipeline_cycle(queries=[test_query], dry_run=False)
    assert new_jobs == 0

    with daemon.db._get_connection() as conn:
        runs = conn.execute("SELECT * FROM scrape_runs").fetchall()
        assert len(runs) == 1
        assert runs[0]["status"] == "failed"
        assert "Scraper network error" in runs[0]["error_message"]


def test_daemon_evaluate_unprocessed_dry_run(tmp_path):
    db_file = tmp_path / "daemon_eval_dry.db"
    mock_evaluator = MagicMock()
    mock_eval_res = MagicMock()
    mock_eval_res.score = 9.8
    mock_eval_res.is_wow = True
    mock_eval_res.hard_reject = False
    mock_eval_res.eval_model = None
    mock_evaluator.evaluate_job.return_value = mock_eval_res

    mock_notifier = MagicMock()

    daemon = DaemonService(
        db_path=db_file,
        evaluator=mock_evaluator,
        notifier=mock_notifier,
    )

    job = JobPosting(
        dedup_hash="hash_eval_dry",
        site="indeed",
        job_url="https://indeed.com/eval_dry",
        title="Senior Coach",
        company="Company A",
        location="Warsaw",
        description="Senior coaching role description.",
    )
    daemon.db.save_job(job)

    evaluated_count, wow_sent = daemon.evaluate_unprocessed(limit=10, dry_run=True)
    assert evaluated_count == 1
    assert wow_sent == 1
    assert not mock_notifier.send_wow_alert.called

    reloaded = daemon.db.get_job_by_hash("hash_eval_dry")
    assert reloaded.eval_status == "unprocessed"


def test_daemon_evaluate_unprocessed_httpx_status_errors(tmp_path):
    import httpx

    db_file = tmp_path / "daemon_httpx_err.db"
    mock_evaluator = MagicMock()

    # 429 Too Many Requests -> transient, leaves unprocessed
    req = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    resp_429 = httpx.Response(status_code=429, request=req)
    mock_evaluator.evaluate_job.side_effect = httpx.HTTPStatusError("Rate limited", request=req, response=resp_429)

    daemon = DaemonService(db_path=db_file, evaluator=mock_evaluator)
    job1 = JobPosting(
        dedup_hash="hash_429",
        site="indeed",
        job_url="https://indeed.com/429",
        title="Coach 429",
        company="Company 429",
        location="Warsaw",
        description="Coaching role description.",
    )
    daemon.db.save_job(job1)

    daemon.evaluate_unprocessed(limit=10)
    reloaded1 = daemon.db.get_job_by_hash("hash_429")
    assert reloaded1.eval_status == "unprocessed"

    # 400 Bad Request -> permanent failure, marks failed
    resp_400 = httpx.Response(status_code=400, request=req)
    mock_evaluator.evaluate_job.side_effect = httpx.HTTPStatusError("Bad Request", request=req, response=resp_400)
    daemon.evaluate_unprocessed(limit=10)
    reloaded1_failed = daemon.db.get_job_by_hash("hash_429")
    assert reloaded1_failed.eval_status == "failed"


def test_daemon_start_lifecycle_and_shutdown(monkeypatch, tmp_path):
    import signal

    mock_scheduler = MagicMock()
    monkeypatch.setattr("src.scheduler.daemon.BlockingScheduler", MagicMock(return_value=mock_scheduler))
    monkeypatch.setattr("src.scheduler.daemon.check_jobspy_update", MagicMock(return_value=None))

    daemon = DaemonService(db_path=tmp_path / "daemon_start.db")
    mock_run = MagicMock(return_value=(0, 0))
    daemon.run_pipeline_cycle = mock_run

    handlers = {}

    def mock_signal(sig, handler):
        handlers[sig] = handler

    monkeypatch.setattr("signal.signal", mock_signal)

    daemon.start()

    assert mock_scheduler.add_job.call_count == 2
    assert mock_run.called
    assert mock_scheduler.start.called

    # Test shutdown handler
    mock_sys_exit = MagicMock()
    monkeypatch.setattr("sys.exit", mock_sys_exit)
    mock_scheduler.running = True
    handlers[signal.SIGINT](signal.SIGINT, None)
    assert mock_scheduler.shutdown.called
    assert mock_sys_exit.called


def test_daemon_pipeline_cycle_jitter_delay(monkeypatch, tmp_path):
    mock_scraper = MagicMock()
    mock_scraper.jitter_min = 2.0
    mock_scraper.jitter_max = 5.0
    mock_scraper.scrape_board.return_value = []

    mock_sleep = MagicMock()
    monkeypatch.setattr("time.sleep", mock_sleep)

    daemon = DaemonService(
        db_path=tmp_path / "daemon_jitter.db",
        scraper=mock_scraper,
    )

    q1 = SearchQuery(query_id="Q1", search_term="Coach", location="Warsaw", site_names=["indeed", "linkedin"])
    q2 = SearchQuery(query_id="Q2", search_term="Trainer", location="Warsaw", site_names=["indeed"])

    daemon.run_pipeline_cycle(queries=[q1, q2], dry_run=True)

    # Boards:
    # 1. q1, indeed: (q_idx=0, s_idx=0) -> no jitter
    # 2. q1, linkedin: (q_idx=0, s_idx=1) -> jitter 1
    # 3. q2, indeed: (q_idx=1, s_idx=0) -> jitter 2
    assert mock_sleep.call_count == 2
    for call in mock_sleep.call_args_list:
        delay = call[0][0]
        assert 2.0 <= delay <= 5.0


def test_daemon_evaluate_unprocessed_immediate_dismissal(tmp_path):
    db_file = tmp_path / "daemon_dismissal.db"
    mock_evaluator = MagicMock()

    # Low score, not wow
    eval_low = MagicMock()
    eval_low.score = 4.5
    eval_low.is_wow = False
    eval_low.summary_uk = "Low match"
    eval_low.reasoning = "Not relevant"
    eval_low.language_requirements = []
    eval_low.pros = []
    eval_low.cons = []
    eval_low.hard_reject = False
    eval_low.eval_model = None

    # High score, not wow
    eval_high = MagicMock()
    eval_high.score = 7.5
    eval_high.is_wow = False
    eval_high.summary_uk = "High match"
    eval_high.reasoning = "Good fit"
    eval_high.language_requirements = []
    eval_high.pros = []
    eval_high.cons = []
    eval_high.hard_reject = False
    eval_high.eval_model = None

    # Workers finish in arbitrary order, so key the scripted result by job title.
    results_by_title = {"Junior Sourcer": eval_low, "Executive Coach": eval_high}
    mock_evaluator.evaluate_job.side_effect = lambda job: results_by_title[job.title]

    daemon = DaemonService(db_path=db_file, evaluator=mock_evaluator)

    job_low = JobPosting(
        dedup_hash="hash_low",
        site="indeed",
        job_url="https://indeed.com/low",
        title="Junior Sourcer",
        company="Co",
        location="Warsaw",
        description="Coaching sourcers.",
    )
    job_high = JobPosting(
        dedup_hash="hash_high",
        site="indeed",
        job_url="https://indeed.com/high",
        title="Executive Coach",
        company="Co",
        location="Warsaw",
        description="Executive coaching.",
    )

    daemon.db.save_job(job_low)
    daemon.db.save_job(job_high)

    evaluated_count, wow_sent = daemon.evaluate_unprocessed(limit=10)
    assert evaluated_count == 2
    assert wow_sent == 0

    reloaded_low = daemon.db.get_job_by_hash("hash_low")
    assert reloaded_low is not None
    assert reloaded_low.eval_status == "evaluated"
    assert reloaded_low.eval_score == 4.5
    assert reloaded_low.notification_status == "dismissed"

    reloaded_high = daemon.db.get_job_by_hash("hash_high")
    assert reloaded_high is not None
    assert reloaded_high.eval_status == "evaluated"
    assert reloaded_high.eval_score == 7.5
    assert reloaded_high.notification_status == "pending"


def test_daemon_daily_digest_includes_update_notice(monkeypatch, tmp_path):
    from src.scraper.version_checker import VersionCheckResult

    db_file = tmp_path / "daemon_digest_update.db"
    mock_notifier = MagicMock()
    mock_notifier.send_daily_digest.return_value = 1

    mock_check = MagicMock(return_value=VersionCheckResult(
        has_update=True,
        installed_version="1.3.0",
        latest_version="1.4.0",
    ))
    monkeypatch.setattr("src.scheduler.daemon.check_jobspy_update", mock_check)

    daemon = DaemonService(
        db_path=db_file,
        notifier=mock_notifier,
    )

    job = JobPosting(
        dedup_hash="hash_digest_update",
        site="indeed",
        job_url="https://indeed.com/update",
        title="Agile Coach",
        company="Agile Corp",
        location="Warsaw",
        description="Agile coaching role description.",
        eval_status="evaluated",
        eval_score=8.0,
        notification_status="pending",
    )
    daemon.db.save_job(job)

    count = daemon.run_daily_digest_cycle(dry_run=False)
    assert count == 1
    assert mock_notifier.send_daily_digest.called

    call_kwargs = mock_notifier.send_daily_digest.call_args[1]
    assert "update_notice" in call_kwargs
    assert call_kwargs["update_notice"] is not None
    assert "Доступне оновлення python-jobspy: v1.4.0" in call_kwargs["update_notice"]
    assert "v1.3.0" in call_kwargs["update_notice"]


def test_daemon_daily_digest_no_update_notice_when_latest(monkeypatch, tmp_path):
    from src.scraper.version_checker import VersionCheckResult

    db_file = tmp_path / "daemon_digest_latest.db"
    mock_notifier = MagicMock()
    mock_notifier.send_daily_digest.return_value = 1

    mock_check = MagicMock(return_value=VersionCheckResult(
        has_update=False,
        installed_version="1.4.0",
        latest_version="1.4.0",
    ))
    monkeypatch.setattr("src.scheduler.daemon.check_jobspy_update", mock_check)

    daemon = DaemonService(
        db_path=db_file,
        notifier=mock_notifier,
    )

    job = JobPosting(
        dedup_hash="hash_digest_latest",
        site="indeed",
        job_url="https://indeed.com/latest",
        title="Agile Coach",
        company="Agile Corp",
        location="Warsaw",
        description="Agile coaching role description.",
        eval_status="evaluated",
        eval_score=8.0,
        notification_status="pending",
    )
    daemon.db.save_job(job)

    count = daemon.run_daily_digest_cycle(dry_run=False)
    assert count == 1
    assert mock_notifier.send_daily_digest.called

    call_kwargs = mock_notifier.send_daily_digest.call_args[1]
    assert call_kwargs.get("update_notice") is None


def test_daemon_start_logs_update_notice(monkeypatch, tmp_path, caplog):
    from src.scraper.version_checker import VersionCheckResult
    import logging

    mock_scheduler = MagicMock()
    monkeypatch.setattr("src.scheduler.daemon.BlockingScheduler", MagicMock(return_value=mock_scheduler))

    mock_check = MagicMock(return_value=VersionCheckResult(
        has_update=True,
        installed_version="1.3.0",
        latest_version="1.4.0",
    ))
    monkeypatch.setattr("src.scheduler.daemon.check_jobspy_update", mock_check)

    daemon = DaemonService(db_path=tmp_path / "daemon_start_log.db")
    daemon.run_pipeline_cycle = MagicMock(return_value=(0, 0))

    with caplog.at_level(logging.INFO):
        daemon.start()

    assert "New JobSpy version available: v1.4.0 (installed: v1.3.0)" in caplog.text





# --- Task 9: truthful scrape outcomes and per-query job hits -------------------------------------

from src.scraper.outcome import ScrapeOutcome
from src.scraper.query_catalog import query_key

_LONG_DESC = (
    "A corporate leadership coaching opportunity with senior executives to develop organizational "
    "strategy and drive executive performance in high-growth companies across many global markets."
)


def _posting(hash_: str, source_query=None) -> JobPosting:
    return JobPosting(
        dedup_hash=hash_,
        site="indeed",
        job_url=f"https://indeed.com/{hash_}",
        title="Executive Coach",
        company="Global Leadership",
        location="Remote",
        description=_LONG_DESC,
        is_remote=True,
        source_query=source_query,
    )


class _OutcomeScraper:
    """Test double exposing only scrape_board_outcome, returning queued outcomes in call order."""

    def __init__(self, *outcomes):
        self._outcomes = list(outcomes)

    def scrape_board_outcome(self, site, query):
        return self._outcomes.pop(0)


def _query(query_id="Q-OUT", site="indeed"):
    return SearchQuery(query_id=query_id, search_term="Executive Coach", location="Remote", site_names=[site])


def _daemon(tmp_path, scraper):
    """Daemon whose evaluation step is stubbed out: these tests only exercise scrape recording."""
    daemon = DaemonService(db_path=tmp_path / "d.db", scraper=scraper, evaluator=MagicMock())
    daemon.evaluate_unprocessed = MagicMock(return_value=(0, 0))
    return daemon


def _runs(daemon):
    with daemon.db._get_connection() as conn:
        return conn.execute("SELECT * FROM scrape_runs ORDER BY id").fetchall()


def test_cycle_records_failed_outcome_from_blocked_board(tmp_path):
    scraper = _OutcomeScraper(ScrapeOutcome(postings=[], rows=0, status="failed", error="blocked"))
    daemon = _daemon(tmp_path, scraper)
    daemon.run_pipeline_cycle(queries=[_query()], dry_run=False)
    runs = _runs(daemon)
    assert len(runs) == 1
    assert runs[0]["status"] == "failed"
    assert runs[0]["outcome"] == "failed"
    assert runs[0]["error_message"] == "blocked"
    assert runs[0]["query_key"] == "Q-OUT|indeed"


def test_cycle_records_empty_outcome_as_completed(tmp_path):
    scraper = _OutcomeScraper(ScrapeOutcome(postings=[], rows=0, status="empty"))
    daemon = _daemon(tmp_path, scraper)
    daemon.run_pipeline_cycle(queries=[_query()], dry_run=False)
    runs = _runs(daemon)
    assert len(runs) == 1
    assert runs[0]["status"] == "completed"
    assert runs[0]["outcome"] == "empty"
    assert runs[0]["rows_returned"] == 0
    assert runs[0]["error_message"] is None


def test_cycle_records_skipped_outcome_as_completed(tmp_path):
    scraper = _OutcomeScraper(
        ScrapeOutcome(postings=[], rows=0, status="skipped", skip_reason="no_country")
    )
    daemon = _daemon(tmp_path, scraper)
    daemon.run_pipeline_cycle(queries=[_query()], dry_run=False)
    runs = _runs(daemon)
    assert len(runs) == 1
    assert runs[0]["status"] == "completed"
    assert runs[0]["outcome"] == "skipped"


def test_cycle_records_rows_and_prefiltered_from_outcome(tmp_path):
    scraper = _OutcomeScraper(
        ScrapeOutcome(postings=[_posting("h-rows")], rows=7, status="ok", prefiltered=3)
    )
    daemon = _daemon(tmp_path, scraper)
    daemon.run_pipeline_cycle(queries=[_query()], dry_run=False)
    runs = _runs(daemon)
    assert runs[0]["outcome"] == "ok"
    assert runs[0]["rows_returned"] == 7
    assert runs[0]["prefiltered"] == 3
    assert runs[0]["jobs_found"] == 1
    assert runs[0]["new_jobs_saved"] == 1


def test_same_job_from_two_queries_gets_two_hits_and_first_source(tmp_path):
    q1, q2 = _query("Q-ONE"), _query("Q-TWO")
    k1, k2 = query_key(q1, "indeed"), query_key(q2, "indeed")
    scraper = _OutcomeScraper(
        ScrapeOutcome(postings=[_posting("h-same", source_query=k1)], rows=1, status="ok"),
        ScrapeOutcome(postings=[_posting("h-same", source_query=k2)], rows=1, status="ok"),
    )
    daemon = _daemon(tmp_path, scraper)
    new_jobs, _ = daemon.run_pipeline_cycle(queries=[q1, q2], dry_run=False)
    assert new_jobs == 1
    with daemon.db._get_connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1
    job = daemon.db.get_job_by_hash("h-same")
    assert job.source_query == k1
    hits = daemon.db.get_job_hits(job.id)
    assert {h["query_key"] for h in hits} == {k1, k2}
    runs = _runs(daemon)
    assert [r["new_jobs_saved"] for r in runs] == [1, 0]


def test_cycle_works_with_double_lacking_scrape_board_outcome(tmp_path):
    class LegacyScraper:
        def scrape_board(self, site, query):
            return [_posting("h-legacy")]

    daemon = _daemon(tmp_path, LegacyScraper())
    new_jobs, _ = daemon.run_pipeline_cycle(queries=[_query()], dry_run=False)
    assert new_jobs == 1
    runs = _runs(daemon)
    assert runs[0]["status"] == "completed"
    assert runs[0]["outcome"] == "ok"
    assert runs[0]["rows_returned"] == 1
    job = daemon.db.get_job_by_hash("h-legacy")
    assert [h["query_key"] for h in daemon.db.get_job_hits(job.id)] == ["Q-OUT|indeed"]


def test_cycle_works_with_magicmock_scraper(tmp_path):
    mock_scraper = MagicMock()
    mock_scraper.scrape_board.return_value = [_posting("h-mm")]
    daemon = _daemon(tmp_path, mock_scraper)
    new_jobs, _ = daemon.run_pipeline_cycle(queries=[_query()], dry_run=False)
    assert new_jobs == 1
    assert _runs(daemon)[0]["outcome"] == "ok"


def test_dry_run_writes_no_hits_or_runs(tmp_path):
    scraper = _OutcomeScraper(ScrapeOutcome(postings=[_posting("h-dry")], rows=1, status="ok"))
    daemon = DaemonService(db_path=tmp_path / "d.db", scraper=scraper)
    daemon.run_pipeline_cycle(queries=[_query()], dry_run=True)
    with daemon.db._get_connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM scrape_runs").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM job_hits").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0


def test_exception_path_records_failed_outcome_and_query_key(tmp_path):
    mock_scraper = MagicMock()
    mock_scraper.scrape_board.side_effect = RuntimeError("boom")
    daemon = DaemonService(db_path=tmp_path / "d.db", scraper=mock_scraper)
    daemon.run_pipeline_cycle(queries=[_query()], dry_run=False)
    runs = _runs(daemon)
    assert runs[0]["status"] == "failed"
    assert runs[0]["outcome"] == "failed"
    assert runs[0]["query_key"] == "Q-OUT|indeed"


def test_outcome_error_with_proxy_url_is_redacted_before_storing(tmp_path):
    err = "ProxyError: cannot connect to http://scrapeuser:s3cretpw@proxy.example.com:8080 " + "x" * 500
    scraper = _OutcomeScraper(ScrapeOutcome(postings=[], rows=0, status="failed", error=err))
    daemon = _daemon(tmp_path, scraper)
    daemon.run_pipeline_cycle(queries=[_query()], dry_run=False)
    stored = _runs(daemon)[0]["error_message"]
    assert "s3cretpw" not in stored
    assert "scrapeuser" not in stored
    assert "proxy.example.com" in stored
    assert len(stored) <= 300


def test_exception_message_with_proxy_url_is_redacted_before_storing(tmp_path):
    mock_scraper = MagicMock()
    mock_scraper.scrape_board.side_effect = RuntimeError(
        "tunnel failed via socks5://scrapeuser:s3cretpw@proxy.example.com:1080"
    )
    daemon = DaemonService(db_path=tmp_path / "d.db", scraper=mock_scraper)
    daemon.run_pipeline_cycle(queries=[_query()], dry_run=False)
    stored = _runs(daemon)[0]["error_message"]
    assert "s3cretpw" not in stored
    assert "scrapeuser" not in stored
    assert "tunnel failed" in stored


def test_exception_log_keeps_full_redacted_text_while_stored_value_is_truncated(tmp_path, caplog):
    import logging

    tail = "DIAGNOSTIC-TAIL-MARKER"
    mock_scraper = MagicMock()
    mock_scraper.scrape_board.side_effect = RuntimeError(
        "tunnel failed via socks5://scrapeuser:s3cretpw@proxy.example.com:1080 " + "x" * 400 + " " + tail
    )
    daemon = DaemonService(db_path=tmp_path / "d.db", scraper=mock_scraper)
    with caplog.at_level(logging.ERROR):
        daemon.run_pipeline_cycle(queries=[_query()], dry_run=False)
    assert "Failed scrape cycle" in caplog.text
    assert tail in caplog.text
    assert "s3cretpw" not in caplog.text
    assert "scrapeuser" not in caplog.text
    stored = _runs(daemon)[0]["error_message"]
    assert len(stored) <= 300
    assert tail not in stored
    assert "s3cretpw" not in stored
