"""CLI: auto-import before every command, init, the no-profile exit and the profile sub-commands."""
import json
import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from src.cli import app
from src.profile.store import ProfileStore
from src.storage.db import Database

FIXTURES = Path(__file__).parent / "fixtures"
QUERIES_FIXTURE = FIXTURES / "legacy_queries.json"
PERSONA_FIXTURE = FIXTURES / "legacy_persona.md"

runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Own DB and working directory: no command can read the real queries.json / persona / .env."""
    db_path = tmp_path / "cli_profile.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    monkeypatch.chdir(tmp_path)
    from src.config import get_settings

    get_settings.cache_clear()
    yield db_path
    get_settings.cache_clear()


def _store(db_path: Path) -> ProfileStore:
    db = Database(db_path)
    db.init_db()
    return ProfileStore(db)


def _legacy_queries(tmp_path: Path) -> list:
    shutil.copy(QUERIES_FIXTURE, tmp_path / "queries.json")
    return json.loads(QUERIES_FIXTURE.read_text(encoding="utf-8"))


def _versions(db_path: Path) -> int:
    return len(_store(db_path).list_versions())


# --- run / auto-import --------------------------------------------------------------------------


def test_run_without_profile_or_legacy_exits_nonzero(monkeypatch):
    start = MagicMock()
    monkeypatch.setattr("src.scheduler.daemon.DaemonService.start", start)

    result = runner.invoke(app, ["run"])

    assert result.exit_code == 1
    assert "no profile, run `jobspy-coach init`" in result.output
    start.assert_not_called()


def test_run_with_profile_starts_the_daemon(tmp_path, monkeypatch):
    _legacy_queries(tmp_path)
    start = MagicMock()
    monkeypatch.setattr("src.scheduler.daemon.DaemonService.start", start)

    result = runner.invoke(app, ["run"])

    assert result.exit_code == 0, result.output
    start.assert_called_once()


def test_any_command_auto_imports_legacy_files_once(tmp_path, isolated):
    _legacy_queries(tmp_path)

    first = runner.invoke(app, ["stats"])
    second = runner.invoke(app, ["stats"])

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert "Imported legacy profile (version 1)" in first.output
    assert "Imported legacy profile" not in second.output
    snapshot = _store(isolated).read_snapshot()
    assert snapshot.version == 1
    assert _versions(isolated) == 1


def test_auto_import_failure_exits_nonzero_with_message(tmp_path, isolated):
    (tmp_path / "queries.json").write_text("[{", encoding="utf-8")

    result = runner.invoke(app, ["stats"])

    assert result.exit_code == 1
    assert "queries.json" in result.output
    assert _store(isolated).has_profile() is False


def test_unknown_board_in_queries_json_names_file_and_entry(tmp_path, isolated):
    entries = json.loads(QUERIES_FIXTURE.read_text(encoding="utf-8"))
    entries[1]["site_names"] = ["not_a_board"]
    (tmp_path / "queries.json").write_text(json.dumps(entries), encoding="utf-8")

    result = runner.invoke(app, ["stats"])

    assert result.exit_code == 1
    assert "queries.json" in result.output
    assert "Q-FIX-B" in result.output or "entries[1]" in result.output
    assert "not_a_board" in result.output
    assert "edit" in result.output.lower()
    assert _store(isolated).has_profile() is False


# --- init ---------------------------------------------------------------------------------------


def test_init_creates_paused_starter_profile(isolated):
    result = runner.invoke(app, ["init"])

    assert result.exit_code == 0, result.output
    snapshot = _store(isolated).read_snapshot()
    assert snapshot is not None and snapshot.fields.paused is True
    assert "paused" in result.output
    assert "profile export" in result.output and "profile import" in result.output


def test_init_after_legacy_import_keeps_imported_profile(tmp_path, isolated):
    _legacy_queries(tmp_path)

    result = runner.invoke(app, ["init"])

    assert result.exit_code == 0, result.output
    assert "already exists" in result.output
    snapshot = _store(isolated).read_snapshot()
    assert snapshot.fields.paused is False
    assert len(snapshot.pinned_queries) == len(json.loads(QUERIES_FIXTURE.read_text(encoding="utf-8")))
    assert snapshot.version == 1


def test_init_never_prompts():
    result = runner.invoke(app, ["init"], input="")

    assert result.exit_code == 0, result.output
    assert "token" not in result.output.lower()


# --- profile preview ----------------------------------------------------------------------------


def test_profile_preview_lists_queries_and_estimate(tmp_path):
    entries = _legacy_queries(tmp_path)

    result = runner.invoke(app, ["profile", "preview"])

    assert result.exit_code == 0, result.output
    for entry in entries:
        assert entry["query_id"] in result.output
    assert "Calls per cycle:" in result.output
    assert "LinkedIn:" in result.output
    assert "cap:" in result.output
    assert "Paused: no" in result.output


def test_profile_preview_shows_paused_starter(isolated):
    runner.invoke(app, ["init"])

    result = runner.invoke(app, ["profile", "preview"])

    assert result.exit_code == 0, result.output
    assert "Paused: yes" in result.output
    assert "Calls per cycle: 0" in result.output


def test_profile_preview_without_profile_exits_nonzero():
    result = runner.invoke(app, ["profile", "preview"])

    assert result.exit_code == 1
    assert "no profile" in result.output


# --- profile export / import --------------------------------------------------------------------


def test_profile_export_default_path(tmp_path):
    _legacy_queries(tmp_path)

    result = runner.invoke(app, ["profile", "export"])

    assert result.exit_code == 0, result.output
    assert (tmp_path / "local" / "profile-export.yaml").is_file()


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_profile_export_refuses_unignored_path_in_git_repo(tmp_path):
    _legacy_queries(tmp_path)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)

    refused = runner.invoke(app, ["profile", "export", "--path", "p.yaml"])
    forced = runner.invoke(app, ["profile", "export", "--path", "p.yaml", "--force"])

    assert refused.exit_code == 1
    assert "git-ignored" in refused.output
    assert forced.exit_code == 0, forced.output
    assert (tmp_path / "p.yaml").is_file()


def test_profile_export_without_profile_exits_nonzero():
    result = runner.invoke(app, ["profile", "export"])

    assert result.exit_code == 1
    assert "no profile" in result.output


_IMPORT_YAML = """\
snapshot_schema: 1
profile:
  paused: false
  wow_threshold: 8.5
  digest_threshold: 6.5
roles:
  - name: "<ROLE>"
    terms:
      - term: "<ROLE>"
markets:
  - name: "<MARKET>"
    location: "<MARKET>"
    countries: ["usa"]
    languages: ["en"]
"""


def test_profile_import_applies_and_prints_diff(tmp_path, isolated):
    runner.invoke(app, ["init"])
    (tmp_path / "in.yaml").write_text(_IMPORT_YAML, encoding="utf-8")

    result = runner.invoke(app, ["profile", "import", str(tmp_path / "in.yaml")])

    assert result.exit_code == 0, result.output
    assert "+ role <ROLE>" in result.output
    snapshot = _store(isolated).read_snapshot()
    assert snapshot.version == 2
    assert [r.name for r in snapshot.roles] == ["<ROLE>"]


def test_profile_import_invalid_exits_nonzero_and_writes_nothing(tmp_path, isolated):
    runner.invoke(app, ["init"])
    bad = _IMPORT_YAML.replace("digest_threshold: 6.5", "digest_threshold: 9.0")
    (tmp_path / "bad.yaml").write_text(bad, encoding="utf-8")

    result = runner.invoke(app, ["profile", "import", str(tmp_path / "bad.yaml")])

    assert result.exit_code == 1
    assert "digest" in result.output.lower()
    assert _store(isolated).read_snapshot().version == 1


def test_profile_import_missing_file_exits_nonzero(tmp_path):
    runner.invoke(app, ["init"])

    result = runner.invoke(app, ["profile", "import", str(tmp_path / "nope.yaml")])

    assert result.exit_code == 1
    assert "nope.yaml" in result.output


# --- final review fixes ------------------------------------------------------------------------


def test_init_with_invalid_env_thresholds_falls_back_with_a_note(isolated, monkeypatch):
    from src.config import get_settings

    monkeypatch.setenv("WOW_SCORE_THRESHOLD", "5.0")
    monkeypatch.setenv("DIGEST_SCORE_THRESHOLD", "9.0")
    get_settings.cache_clear()

    result = runner.invoke(app, ["init"])

    assert result.exit_code == 0, result.output
    snapshot = _store(isolated).read_snapshot()
    assert (snapshot.fields.wow_threshold, snapshot.fields.digest_threshold) == (8.5, 6.5)
    assert "8.5" in result.output and "6.5" in result.output


def test_persona_only_legacy_import_prints_paused_note(tmp_path):
    (tmp_path / ".candidate_profile.md").write_text("<PERSONA>\n", encoding="utf-8")

    result = runner.invoke(app, ["stats"])

    assert result.exit_code == 0, result.output
    assert "Imported legacy profile" in result.output
    assert "paused" in result.output and "no search queries" in result.output


def test_preview_escapes_markup_in_user_text(tmp_path, isolated):
    entries = [{"query_id": "Q-[/x]", "search_term": "<ROLE> [/y]", "location": "<M> [bold]",
                "site_names": ["linkedin"]}]
    (tmp_path / "queries.json").write_text(json.dumps(entries), encoding="utf-8")

    result = runner.invoke(app, ["profile", "preview"])

    assert result.exit_code == 0, result.output
    assert "Q-[/x]" in result.output and "[/y]" in result.output


def test_profile_export_oserror_is_a_one_line_message(tmp_path):
    _legacy_queries(tmp_path)
    (tmp_path / "adir").mkdir()

    result = runner.invoke(app, ["profile", "export", "--path", str(tmp_path / "adir")])

    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "cannot write" in result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)


@pytest.mark.parametrize("args", [["--help"], ["init", "--help"], ["profile", "--help"], ["profile", "preview", "--help"]])
def test_help_does_not_touch_the_database_or_import(tmp_path, isolated, args):
    _legacy_queries(tmp_path)

    result = runner.invoke(app, args)

    assert result.exit_code == 0, result.output
    assert not isolated.exists()


def test_concurrent_init_loser_reports_already_exists(isolated, monkeypatch):
    from src.profile.errors import VersionConflict

    real = ProfileStore.read_snapshot
    calls = {"n": 0}

    def racing(self):
        calls["n"] += 1
        if calls["n"] == 1:
            return None  # init's pre-check sees no profile ...
        return real(self)

    monkeypatch.setattr(ProfileStore, "read_snapshot", racing)
    runner.invoke(app, ["init"])  # ... but a concurrent init has created it by now
    calls["n"] = 0

    result = runner.invoke(app, ["init"])

    assert result.exit_code == 0, result.output
    assert "already exists" in result.output


# --- requeue and backtest ------------------------------------------------------------------------


def _profile_with_one_dropped_job(isolated):
    """An imported-legacy-free profile (init) plus one pre-filtered row that a (now empty) rule set passes."""
    runner.invoke(app, ["init"])
    db = Database(isolated)
    words = " ".join(f"word{n}" for n in range(60))
    from src.storage.models import JobPosting

    db.save_job(
        JobPosting(
            dedup_hash="h1", site="linkedin", job_url="https://example.test/1", title="<ROLE> one",
            company="Example Co", location="<MARKET>", description=words, pre_filtered=True,
            pre_filter_reason="USER_RULE: 'x'", pre_filter_rule="block:exact:x", eval_status="skipped",
            notification_status="dismissed",
        )
    )
    return db


def test_requeue_cli_dry_run_reports_counts(isolated):
    db = _profile_with_one_dropped_job(isolated)

    result = runner.invoke(app, ["requeue", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "requeued: 1" in result.output and "checked: 1" in result.output
    assert "<ROLE> one" in result.output
    assert db.get_unprocessed_jobs() == []

    live = runner.invoke(app, ["requeue"])
    assert live.exit_code == 0, live.output
    assert [j.title for j in db.get_unprocessed_jobs()] == ["<ROLE> one"]


def test_requeue_thresholds_cli(isolated):
    runner.invoke(app, ["init"])
    db = Database(isolated)
    from src.storage.models import JobPosting

    job_id, _ = db.save_job(
        JobPosting(
            dedup_hash="h2", site="linkedin", job_url="https://example.test/2", title="<ROLE> two",
            company="Example Co", location="<MARKET>", description="d", eval_status="evaluated",
            eval_score=3.0, notification_status="dismissed",
        )
    )
    with db._get_connection() as conn:  # save_job does not store the raw evaluation fields
        conn.execute(
            "UPDATE jobs SET eval_raw_score = 7.0, eval_is_core_role = 1, eval_bonus_flag = 0, "
            "eval_hard_reject = 0 WHERE id = ?",
            (job_id,),
        )
        conn.commit()

    dry = runner.invoke(app, ["requeue", "--thresholds", "--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert "changed: 1" in dry.output and "to digest: 1" in dry.output
    assert db.get_job_by_id(job_id).notification_status == "dismissed"

    live = runner.invoke(app, ["requeue", "--thresholds"])
    assert live.exit_code == 0, live.output
    row = db.get_job_by_id(job_id)
    assert (row.notification_status, row.eval_score) == ("pending", 7.0)


def test_requeue_without_profile_exits_nonzero():
    result = runner.invoke(app, ["requeue"])
    assert result.exit_code == 1
    assert "no profile" in result.output


def test_profile_import_prints_backtest_line(tmp_path, isolated):
    runner.invoke(app, ["init"])
    (tmp_path / "in.yaml").write_text(_IMPORT_YAML + "rules:\n  - {kind: block, term: \"<TERM>\", match_mode: exact, reason: USER_RULE, source: user}\n", encoding="utf-8")

    result = runner.invoke(app, ["profile", "import", str(tmp_path / "in.yaml")])

    assert result.exit_code == 0, result.output
    assert "backtest not meaningful: only 0 stored jobs (needs 200)" in result.output


def test_requeue_with_an_unreadable_profile_is_a_clean_error(tmp_path, isolated):
    runner.invoke(app, ["init"])
    (tmp_path / "in.yaml").write_text(_IMPORT_YAML, encoding="utf-8")
    assert runner.invoke(app, ["profile", "import", str(tmp_path / "in.yaml")]).exit_code == 0
    with Database(isolated)._get_connection() as conn:
        conn.execute("UPDATE markets SET countries = 'not json'")
        conn.commit()

    result = runner.invoke(app, ["requeue"])

    assert result.exit_code == 1
    assert "unreadable" in result.output
    assert isinstance(result.exception, SystemExit)  # a message, not a traceback


def test_profile_import_survives_a_failing_backtest(tmp_path, isolated, monkeypatch):
    import sqlite3

    runner.invoke(app, ["init"])
    (tmp_path / "in.yaml").write_text(_IMPORT_YAML, encoding="utf-8")

    def locked(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr("src.cli.backtest_snapshots", locked)

    result = runner.invoke(app, ["profile", "import", str(tmp_path / "in.yaml")])

    assert result.exit_code == 0, result.output
    assert "Profile is now at version 2." in result.output
    assert _store(isolated).read_snapshot().version == 2


def test_profile_import_backtests_against_the_applied_snapshots(tmp_path, isolated, monkeypatch):
    seen = {}

    def spy(db, old, new):
        seen["old"], seen["new"] = old, new

    monkeypatch.setattr("src.cli.backtest_snapshots", spy)
    runner.invoke(app, ["init"])
    (tmp_path / "in.yaml").write_text(_IMPORT_YAML, encoding="utf-8")

    result = runner.invoke(app, ["profile", "import", str(tmp_path / "in.yaml")])

    assert result.exit_code == 0, result.output
    assert (seen["old"].version, seen["new"].version) == (1, 2)


def test_requeue_dry_run_never_initialises_or_migrates(isolated, monkeypatch):
    import sqlite3

    runner.invoke(app, ["init"])
    calls = []
    real_init = Database.init_db
    monkeypatch.setattr(Database, "init_db", lambda self: calls.append(1) or real_init(self))

    for args in (["requeue", "--dry-run"], ["requeue", "--thresholds", "--dry-run"]):
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.output
    assert calls == []


def test_requeue_dry_run_reports_an_uninitialised_database_and_writes_nothing(isolated):
    import sqlite3

    raw = sqlite3.connect(isolated)
    raw.execute("CREATE TABLE jobs (id INTEGER)")
    raw.commit()
    raw.close()
    before = isolated.read_bytes()

    result = runner.invoke(app, ["requeue", "--dry-run"])

    assert result.exit_code == 1
    assert "schema_version" in result.output
    assert isolated.read_bytes() == before
    assert not Path(str(isolated) + "-wal").exists()


def test_requeue_dry_run_reports_an_outdated_schema(isolated):
    import sqlite3

    runner.invoke(app, ["init"])
    conn = sqlite3.connect(isolated)
    conn.execute("DELETE FROM schema_version WHERE version = 3")
    conn.commit()
    conn.close()

    result = runner.invoke(app, ["requeue", "--dry-run"])

    assert result.exit_code == 1
    assert "schema version" in result.output
