import pytest
from typer.testing import CliRunner
from unittest.mock import MagicMock
from src.cli import app


runner = CliRunner()


@pytest.fixture(autouse=True)
def isolate_cli_test_db(tmp_path, monkeypatch):
    """Isolate CLI test database to a temporary path to avoid touching data/jobs.db."""
    test_db = tmp_path / "test_cli.db"
    monkeypatch.setenv("DB_PATH", str(test_db))
    monkeypatch.setenv("DATABASE_PATH", str(test_db))
    from src.config import get_settings

    get_settings.cache_clear()
    yield test_db
    get_settings.cache_clear()


def test_cli_stats_command(monkeypatch):
    mock_stats = MagicMock(return_value={
        "total_jobs": 42,
        "prefiltered_jobs": 15,
        "evaluated_jobs": 20,
        "failed_jobs": 1,
        "instant_sent": 3,
        "digest_sent": 12,
        "pending_digest": 5,
    })
    monkeypatch.setattr("src.storage.db.Database.get_stats", mock_stats)
    from src.scraper.version_checker import VersionCheckResult
    mock_check = MagicMock(return_value=VersionCheckResult(
        has_update=True,
        installed_version="1.3.0",
        latest_version="1.4.0",
    ))
    monkeypatch.setattr("src.cli.check_jobspy_update", mock_check)

    result = runner.invoke(app, ["stats"])
    assert result.exit_code == 0
    assert "42" in result.stdout
    assert "15" in result.stdout
    assert "Jobspy Version" in result.stdout
    assert "1.3.0 (Update: v1.4.0)" in result.stdout


def test_cli_stats_shows_jobspy_version_latest(monkeypatch):
    mock_stats = MagicMock(return_value={"total_jobs": 42})
    monkeypatch.setattr("src.storage.db.Database.get_stats", mock_stats)
    from src.scraper.version_checker import VersionCheckResult
    mock_check = MagicMock(return_value=VersionCheckResult(
        has_update=False,
        installed_version="1.4.0",
        latest_version="1.4.0",
    ))
    monkeypatch.setattr("src.cli.check_jobspy_update", mock_check)

    result = runner.invoke(app, ["stats"])
    assert result.exit_code == 0
    assert "Jobspy Version" in result.stdout
    assert "1.4.0 (Latest)" in result.stdout


def test_cli_stats_shows_jobspy_version_fallback(monkeypatch):
    mock_stats = MagicMock(return_value={"total_jobs": 42})
    monkeypatch.setattr("src.storage.db.Database.get_stats", mock_stats)
    monkeypatch.setattr("src.cli.check_jobspy_update", MagicMock(return_value=None))
    monkeypatch.setattr("importlib.metadata.version", MagicMock(return_value="1.3.5"))

    result = runner.invoke(app, ["stats"])
    assert result.exit_code == 0
    assert "Jobspy Version" in result.stdout
    assert "1.3.5" in result.stdout


def test_cli_test_telegram_command(monkeypatch):
    mock_send = MagicMock(return_value=True)
    monkeypatch.setattr("src.notifier.telegram_notifier.TelegramNotifier.send_message", mock_send)

    result = runner.invoke(app, ["test-telegram"])
    assert result.exit_code == 0
    assert "Telegram test message sent successfully" in result.stdout


def test_cli_test_telegram_command_failure(monkeypatch):
    mock_send = MagicMock(return_value=False)
    monkeypatch.setattr("src.notifier.telegram_notifier.TelegramNotifier.send_message", mock_send)

    result = runner.invoke(app, ["test-telegram"])
    assert result.exit_code == 1
    assert "Failed to send Telegram test message" in result.stdout


def test_cli_evaluate_command(monkeypatch):
    mock_eval = MagicMock(return_value=(5, 1))
    monkeypatch.setattr("src.scheduler.daemon.DaemonService.evaluate_unprocessed", mock_eval)

    result = runner.invoke(app, ["evaluate", "--limit", "10"])
    assert result.exit_code == 0
    assert "Evaluated: 5" in result.stdout
    assert "Instant WOW alerts: 1" in result.stdout
    mock_eval.assert_called_once_with(limit=10, dry_run=False)


def test_cli_scrape_command(monkeypatch):
    mock_scrape = MagicMock(return_value=(8, 2))
    monkeypatch.setattr("src.scheduler.daemon.DaemonService.run_pipeline_cycle", mock_scrape)

    result = runner.invoke(app, ["scrape", "--dry-run"])
    assert result.exit_code == 0
    assert "New jobs: 8" in result.stdout
    assert "Instant WOW alerts: 2" in result.stdout
    mock_scrape.assert_called_once_with(dry_run=True)


def test_cli_digest_command(monkeypatch):
    mock_digest = MagicMock(return_value=3)
    monkeypatch.setattr("src.scheduler.daemon.DaemonService.run_daily_digest_cycle", mock_digest)

    result = runner.invoke(app, ["digest"])
    assert result.exit_code == 0
    assert "Processed 3 jobs" in result.stdout
    mock_digest.assert_called_once_with(dry_run=False)


def test_cli_run_command(monkeypatch):
    mock_start = MagicMock()
    monkeypatch.setattr("src.scheduler.daemon.DaemonService.start", mock_start)

    result = runner.invoke(app, ["run"])
    assert result.exit_code == 0
    assert "Starting JobSpy Coach Daemon" in result.stdout
    assert mock_start.called


def test_main_entrypoint(monkeypatch):
    import runpy
    mock_start = MagicMock()
    monkeypatch.setattr("src.scheduler.daemon.DaemonService.start", mock_start)
    monkeypatch.setattr("sys.argv", ["jobspy-coach", "run"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("src.main", run_name="__main__")
    assert exc.value.code == 0
    assert mock_start.called

