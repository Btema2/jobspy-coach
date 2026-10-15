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


def test_daemon_daily_digest_sends_and_marks_sent(tmp_path):
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

