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
    countries: ["poland"]
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
