import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from src.profile.errors import ProfileCompileError, ProfileValidationError, VersionConflict
from src.profile.snapshot import (
    Market,
    PinnedQuery,
    ProfileFields,
    ProfileSnapshot,
    Role,
    RoleTerm,
    Rule,
    snapshot_from_dict,
    snapshot_to_dict,
)
from src.profile.store import ProfileStore
from src.storage.db import Database


@pytest.fixture
def db(tmp_path: Path) -> Database:
    database = Database(tmp_path / "profile.db")
    database.init_db()
    return database


@pytest.fixture
def store(db: Database) -> ProfileStore:
    return ProfileStore(db)


def _snapshot(persona: str = "<PERSONA>") -> ProfileSnapshot:
    return ProfileSnapshot(
        fields=ProfileFields(persona=persona, output_language="en", paused=True),
        roles=(
            Role(
                name="<ROLE>",
                terms=(RoleTerm("<TERM A>"), RoleTerm("<TERM B>", language="en")),
            ),
        ),
        markets=(
            Market(
                name="<MARKET>",
                location="<LOCATION>",
                countries=("germany", "poland"),
                remote=True,
                languages=("en", "de"),
            ),
        ),
        pinned_queries=(
            PinnedQuery("Q-1", "<TERM A>", "<LOCATION>", ("linkedin", "google")),
            PinnedQuery("Q-2", "<TERM B>", "<LOCATION>", ("indeed",), country="germany", is_remote=True,
                        google_search_term="x", enabled=False),
        ),
        rules=(
            Rule("block", "<NOISE>", "exact", "noise", "user"),
            Rule("soft", "<TRAP>", "prefix", "trap", "import"),
        ),
    )


def test_empty_db_has_no_profile(store: ProfileStore):
    assert store.has_profile() is False
    assert store.read_snapshot() is None
    assert store.list_versions() == []


def test_write_then_read_roundtrip(store: ProfileStore):
    snap = _snapshot()
    assert store.write_snapshot(snap, expected_version=0, label="create") == 1

    assert store.has_profile() is True
    assert store.read_snapshot() == replace(snap, version=1)


def test_stale_expected_version_raises_conflict_and_writes_nothing(store: ProfileStore):
    store.write_snapshot(_snapshot("<ONE>"), expected_version=0, label="a")

    with pytest.raises(VersionConflict):
        store.write_snapshot(_snapshot("<TWO>"), expected_version=0, label="b")
    with pytest.raises(VersionConflict):
        store.write_snapshot(_snapshot("<TWO>"), expected_version=5, label="b")

    current = store.read_snapshot()
    assert current is not None and current.version == 1 and current.fields.persona == "<ONE>"
    assert [v.version for v in store.list_versions()] == [1]


def test_create_when_profile_exists_conflicts(store: ProfileStore):
    store.write_snapshot(_snapshot(), expected_version=0, label="a")
    with pytest.raises(VersionConflict):
        store.write_snapshot(_snapshot(), expected_version=0, label="a")


def test_versions_pruned_to_last_10(store: ProfileStore):
    version = 0
    for i in range(12):
        version = store.write_snapshot(_snapshot(f"<P{i}>"), expected_version=version, label=f"w{i}")

    assert [v.version for v in store.list_versions()] == list(range(12, 2, -1))
    assert store.list_versions()[0].label == "w11"
    with pytest.raises(KeyError):
        store.load_version(2)


def test_load_version_returns_that_state(store: ProfileStore):
    v1 = store.write_snapshot(_snapshot("<ONE>"), expected_version=0, label="a")
    store.write_snapshot(_snapshot("<TWO>"), expected_version=v1, label="b")

    loaded = store.load_version(1)
    assert loaded.fields.persona == "<ONE>"
    assert loaded == replace(_snapshot("<ONE>"), version=1)
    assert store.load_version(2).fields.persona == "<TWO>"


def test_replace_removes_old_rows(store: ProfileStore):
    v1 = store.write_snapshot(_snapshot(), expected_version=0, label="a")
    smaller = ProfileSnapshot(fields=ProfileFields(persona="<X>"))
    store.write_snapshot(smaller, expected_version=v1, label="b")

    assert store.read_snapshot() == replace(smaller, version=2)


def test_read_snapshot_is_consistent_under_concurrent_write(
    db: Database, store: ProfileStore, monkeypatch
):
    store.write_snapshot(_snapshot("<OLD>"), expected_version=0, label="a")
    other = ProfileStore(db)
    original = ProfileStore._read_table
    fired = []

    def racing(self, conn, table):
        rows = original(self, conn, table)
        if table == "profile" and not fired:
            fired.append(True)
            other.write_snapshot(_snapshot("<NEW>"), expected_version=1, label="b")
        return rows

    monkeypatch.setattr(ProfileStore, "_read_table", racing)

    snap = store.read_snapshot()

    assert fired
    assert snap is not None
    assert snap.version == 1 and snap.fields.persona == "<OLD>"
    assert snap == replace(_snapshot("<OLD>"), version=1)
    monkeypatch.undo()
    latest = store.read_snapshot()
    assert latest is not None and latest.version == 2 and latest.fields.persona == "<NEW>"


def test_snapshot_dict_roundtrip():
    snap = _snapshot()
    data = snapshot_to_dict(snap)

    assert data["snapshot_schema"] == 2
    assert set(data) == {"snapshot_schema", "profile", "roles", "markets", "pinned_queries", "rules"}
    assert "version" not in data
    assert snapshot_from_dict(data, version=7) == replace(snap, version=7)


def test_snapshot_from_dict_takes_defaults_for_missing_keys():
    snap = snapshot_from_dict({"snapshot_schema": 1})
    assert snap == ProfileSnapshot(fields=ProfileFields())


def test_snapshot_from_dict_rejects_unknown_key():
    with pytest.raises(ProfileValidationError) as exc:
        snapshot_from_dict({"snapshot_schema": 1, "profile": {"persnoa": "x"}})
    assert any("persnoa" in m for m in exc.value.messages)


def test_snapshot_from_dict_rejects_unknown_top_level_key():
    with pytest.raises(ProfileValidationError):
        snapshot_from_dict({"snapshot_schema": 1, "extra": 1})


def test_snapshot_from_dict_rejects_wrong_types():
    with pytest.raises(ProfileValidationError):
        snapshot_from_dict({"snapshot_schema": 1, "profile": {"paused": "yes"}})
    with pytest.raises(ProfileValidationError):
        snapshot_from_dict({"snapshot_schema": 1, "profile": {"wow_threshold": "high"}})
    with pytest.raises(ProfileValidationError):
        snapshot_from_dict({"snapshot_schema": 1, "roles": [{"name": "r", "terms": "nope"}]})


def test_snapshot_from_dict_rejects_newer_or_missing_schema():
    with pytest.raises(ProfileValidationError):
        snapshot_from_dict({"snapshot_schema": 99, "profile": {}})
    with pytest.raises(ProfileValidationError):
        snapshot_from_dict({"profile": {}})


def test_failed_write_leaves_rows_version_and_history_untouched(store: ProfileStore):
    store.write_snapshot(_snapshot("<ONE>"), expected_version=0, label="a")
    before = store.read_snapshot()
    broken = ProfileSnapshot(
        fields=ProfileFields(persona="<TWO>"),
        roles=(Role("<ROLE>"), Role("<role>")),  # violates the NOCASE UNIQUE role name
    )

    with pytest.raises(sqlite3.IntegrityError):
        store.write_snapshot(broken, expected_version=1, label="b")

    assert store.read_snapshot() == before
    assert before is not None and before.version == 1 and before.fields.persona == "<ONE>"
    assert [v.version for v in store.list_versions()] == [1]


def test_read_snapshot_wraps_undecodable_rows_in_compile_error(db: Database, store: ProfileStore):
    store.write_snapshot(_snapshot(), expected_version=0, label="x")
    with db._get_connection() as conn:
        conn.execute("UPDATE markets SET countries = 'not json'")
        conn.commit()

    with pytest.raises(ProfileCompileError, match="stored profile is unreadable"):
        store.read_snapshot()


def test_read_snapshot_lets_sql_errors_propagate(db: Database, store: ProfileStore):
    store.write_snapshot(_snapshot(), expected_version=0, label="x")
    with db._get_connection() as conn:
        conn.execute("ALTER TABLE markets RENAME TO markets_gone")
        conn.commit()

    with pytest.raises(sqlite3.OperationalError):
        store.read_snapshot()
