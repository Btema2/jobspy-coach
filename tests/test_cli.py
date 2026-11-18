from pathlib import Path
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
    monkeypatch.chdir(tmp_path)  # the CLI auto-imports legacy files from the cwd: never the real ones
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


def test_cli_evaluate_default_limit_comes_from_settings(monkeypatch):
    monkeypatch.setenv("EVAL_BATCH_LIMIT", "7")
    from src.config import get_settings

    get_settings.cache_clear()
    mock_eval = MagicMock(return_value=(0, 0))
    monkeypatch.setattr("src.scheduler.daemon.DaemonService.evaluate_unprocessed", mock_eval)

    result = runner.invoke(app, ["evaluate"])
    assert result.exit_code == 0
    mock_eval.assert_called_once_with(limit=7, dry_run=False)


def test_cli_evaluate_reports_missing_provider_configuration(monkeypatch):
    from src.evaluator.errors import ConfigurationError

    monkeypatch.setattr(
        "src.scheduler.daemon.DaemonService.evaluate_unprocessed",
        MagicMock(side_effect=ConfigurationError("No LLM provider is configured.")),
    )
    result = runner.invoke(app, ["evaluate"])
    assert result.exit_code == 1
    assert "No LLM provider is configured" in result.stdout
    assert "Traceback" not in result.stdout


def test_cli_stats_uses_digest_threshold_setting(monkeypatch):
    monkeypatch.setenv("DIGEST_SCORE_THRESHOLD", "7.25")
    from src.config import get_settings

    get_settings.cache_clear()
    mock_stats = MagicMock(return_value={"total_jobs": 1})
    monkeypatch.setattr("src.storage.db.Database.get_stats", mock_stats)
    monkeypatch.setattr("src.cli.check_jobspy_update", MagicMock(return_value=None))

    result = runner.invoke(app, ["stats"])
    assert result.exit_code == 0
    mock_stats.assert_called_once_with(pending_threshold=7.25)


def test_cli_scrape_reports_missing_provider_configuration(monkeypatch):
    from src.evaluator.errors import ConfigurationError

    monkeypatch.setattr(
        "src.scheduler.daemon.DaemonService.run_pipeline_cycle",
        MagicMock(side_effect=ConfigurationError("No LLM provider is configured.")),
    )
    result = runner.invoke(app, ["scrape"])
    assert result.exit_code == 1
    assert "No LLM provider is configured" in result.stdout


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


def _create_profile():
    """`run` refuses to start without a stored profile."""
    assert runner.invoke(app, ["init", "--skip-credentials", "--no-start"]).exit_code == 0


def test_cli_run_command(monkeypatch):
    _create_profile()
    mock_start = MagicMock()
    monkeypatch.setattr("src.scheduler.daemon.DaemonService.start", mock_start)

    result = runner.invoke(app, ["run"])
    assert result.exit_code == 0
    assert "Starting JobSpy Coach Daemon" in result.stdout
    assert mock_start.called


def test_main_entrypoint(monkeypatch):
    import runpy
    _create_profile()
    mock_start = MagicMock()
    monkeypatch.setattr("src.scheduler.daemon.DaemonService.start", mock_start)
    monkeypatch.setattr("sys.argv", ["jobspy-coach", "run"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("src.main", run_name="__main__")
    assert exc.value.code == 0
    assert mock_start.called


def test_cli_rehash_command(monkeypatch):
    mock_rehash = MagicMock(return_value={"total": 5, "rehashed": 3, "merged": 1, "kept_twins": 0})
    monkeypatch.setattr("src.storage.db.Database.rehash_all", mock_rehash)

    result = runner.invoke(app, ["rehash"])

    assert result.exit_code == 0
    mock_rehash.assert_called_once_with(dry_run=False)
    assert "merged" in result.stdout.lower()


def test_cli_rehash_dry_run_flag(monkeypatch):
    mock_rehash = MagicMock(return_value={"total": 5, "rehashed": 3, "merged": 1, "kept_twins": 0})
    monkeypatch.setattr("src.storage.db.Database.rehash_all", mock_rehash)

    result = runner.invoke(app, ["rehash", "--dry-run"])

    assert result.exit_code == 0
    mock_rehash.assert_called_once_with(dry_run=True)


def test_cli_cleanup_empty_deletes_empty_description_stubs(monkeypatch):
    mock_delete = MagicMock(return_value=4)
    monkeypatch.setattr("src.storage.db.Database.delete_empty_description_jobs", mock_delete)

    result = runner.invoke(app, ["cleanup-empty"])

    assert result.exit_code == 0
    mock_delete.assert_called_once_with("linkedin")
    assert "4" in result.stdout


def _stats_threshold(monkeypatch) -> float:
    mock_stats = MagicMock(return_value={"total_jobs": 1})
    monkeypatch.setattr("src.storage.db.Database.get_stats", mock_stats)
    monkeypatch.setattr("src.cli.check_jobspy_update", MagicMock(return_value=None))
    assert runner.invoke(app, ["stats"]).exit_code == 0
    return mock_stats.call_args.kwargs["pending_threshold"]


def test_cli_stats_uses_the_profile_digest_threshold(monkeypatch):
    from src.profile.ops import ReplaceProfile, apply_changes
    from src.profile.snapshot import PinnedQuery, ProfileFields, ProfileSnapshot
    from src.config import get_settings
    from src.profile.store import ProfileStore
    from src.storage.db import Database

    db = Database(get_settings().db_path)
    db.init_db()
    snapshot = ProfileSnapshot(
        ProfileFields(persona="<PERSONA>", digest_threshold=6.0, wow_threshold=8.0),
        pinned_queries=(PinnedQuery("Q-1", "<ROLE>", "<MARKET>", ("linkedin",)),),
    )
    apply_changes(ProfileStore(db), [ReplaceProfile(snapshot)], label="p", expected_version=0)

    assert _stats_threshold(monkeypatch) == 6.0


def test_cli_stats_without_profile_uses_env_threshold(monkeypatch):
    monkeypatch.setenv("DIGEST_SCORE_THRESHOLD", "6.8")

    assert _stats_threshold(monkeypatch) == 6.8


def test_cli_test_telegram_uses_profile_language(monkeypatch, isolate_cli_test_db):
    from src.i18n import t
    from src.profile.ops import ReplaceProfile, apply_changes
    from src.profile.snapshot import ProfileFields, ProfileSnapshot
    from src.profile.store import ProfileStore
    from src.storage.db import Database

    sent = []

    def fake_send(self, text, reply_markup=None):
        sent.append(text)
        return True

    monkeypatch.setattr("src.notifier.telegram_notifier.TelegramNotifier.send_message", fake_send)

    assert runner.invoke(app, ["test-telegram"]).exit_code == 0
    assert sent[-1] == t("test_telegram", None)
    assert sent[-1] == "👋 <b>JobSpy Coach Daemon</b> | Підключення успішне! Тестове сповіщення доставлено."

    db = Database(isolate_cli_test_db)
    db.init_db()
    snapshot = ProfileSnapshot(ProfileFields(paused=True, output_language="en"))
    apply_changes(ProfileStore(db), [ReplaceProfile(snapshot)], label="test", expected_version=0)

    assert runner.invoke(app, ["test-telegram"]).exit_code == 0
    assert sent[-1] == t("test_telegram", "en")
    assert sent[-1] != sent[0]


def test_init_never_accepts_secrets_as_arguments():
    for args in (["init", "--token", "x"], ["init", "--api-key", "x"], ["init", "--nvidia-api-key", "x"]):
        result = runner.invoke(app, args)
        assert result.exit_code == 2, result.output


def test_init_options_and_run_init_wiring(monkeypatch, tmp_path):
    captured = {}

    def fake_run_init(**kwargs):
        captured.update(kwargs)
        return 3

    monkeypatch.setattr("src.cli.run_init", fake_run_init)

    result = runner.invoke(app, ["init", "--skip-credentials", "--no-start", "--language", "pl"])

    assert result.exit_code == 3
    options = captured["options"]
    assert (options.skip_credentials, options.start, options.language) == (True, False, "pl")
    assert captured["env_path"] == Path(".env")
    result = runner.invoke(app, ["init"])
    assert captured["options"].start is None and captured["options"].language == "en"


def test_init_start_daemon_clears_settings_cache_then_runs_the_run_path(monkeypatch):
    from src.config import get_settings

    calls = []

    def fake_run_init(**kwargs):
        get_settings()  # fill the cache
        kwargs["start_daemon"]()
        return 0

    monkeypatch.setattr("src.cli.run_init", fake_run_init)
    monkeypatch.setattr("src.cli._run_daemon", lambda: calls.append(get_settings.cache_info().currsize))
    assert runner.invoke(app, ["init", "--skip-credentials", "--start"]).exit_code == 0

    assert calls == [0]


def test_bot_unpair_clears_pairing_and_chat_id(isolate_cli_test_db, tmp_path):
    from src.bot.state import BotStateStore
    from src.storage.db import Database

    db = Database(isolate_cli_test_db)
    db.init_db()
    state = BotStateStore(db)
    state.set_pairing(5, 5)
    env = tmp_path / ".env"
    env.write_text("# keep\nTELEGRAM_BOT_TOKEN=123:<TOKEN>\nTELEGRAM_CHAT_ID=5\nOTHER=1\n")

    result = runner.invoke(app, ["bot", "unpair"])

    assert result.exit_code == 0, result.output
    assert state.get_pairing() is None
    assert env.read_text() == "# keep\nTELEGRAM_BOT_TOKEN=123:<TOKEN>\nTELEGRAM_CHAT_ID=\nOTHER=1\n"
    assert result.output.strip().startswith("Unpaired. Alerts stop until you pair again")
    assert "jobspy-coach init" in result.output and "Restart the daemon" in result.output


def test_bot_unpair_without_env_file(isolate_cli_test_db, tmp_path):
    result = runner.invoke(app, ["bot", "unpair"])

    assert result.exit_code == 0, result.output
    assert not (tmp_path / ".env").exists()


def test_run_and_init_start_share_one_code_path(monkeypatch):
    assert runner.invoke(app, ["init", "--skip-credentials", "--no-start"]).exit_code == 0
    started = []
    monkeypatch.setattr("src.scheduler.daemon.DaemonService.start", lambda self: started.append(1))

    assert runner.invoke(app, ["run"]).exit_code == 0
    assert runner.invoke(app, ["init", "--skip-credentials", "--start"]).exit_code == 0
    assert started == [1, 1]


def test_init_interactive_flow_writes_env_and_never_echoes_secrets(monkeypatch, tmp_path):
    from src.bot.bootstrap import CheckResult, Validators
    from src.bot.state import BotStateStore
    from src.envfile import read_env_value
    from src.storage.db import Database

    ok = lambda value: CheckResult(True, "ok")  # noqa: E731
    monkeypatch.setattr("src.cli.http_validators", lambda settings: Validators(ok, ok, ok))
    for name in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "NVIDIA_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    result = runner.invoke(app, ["init", "--no-start"], input="111:<BOT>\nnvapi-<NIM>\n\n")

    assert result.exit_code == 0, result.output
    env = tmp_path / ".env"
    assert read_env_value(env, "TELEGRAM_BOT_TOKEN") == "111:<BOT>"
    assert read_env_value(env, "NVIDIA_API_KEY") == "nvapi-<NIM>"
    assert (env.stat().st_mode & 0o777) == 0o600
    assert "<BOT>" not in result.output and "<NIM>" not in result.output
    assert "Send /start " in result.output
    from src.config import get_settings

    db = Database(get_settings().db_path)
    assert BotStateStore(db).get_pairing_code() is not None
