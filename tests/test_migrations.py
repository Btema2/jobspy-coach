import sqlite3
import threading
from pathlib import Path

from src.profile import legacy_seed
from src.storage.db import Database
from src.storage.migrations import apply_migrations
from tests.test_storage import _OLD_SCHEMA

_JOBS_NEW = {
    "source_query", "remote_status", "eval_raw_score",
    "eval_is_core_role", "eval_bonus_flag", "eval_hard_reject",
}
_RUNS_NEW = {"rows_returned", "prefiltered", "outcome", "query_key"}


def _cols(db_file: Path, table: str) -> set:
    conn = sqlite3.connect(db_file)
    try:
        return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
    finally:
        conn.close()


def _scalar(db_file: Path, sql: str):
    conn = sqlite3.connect(db_file)
    try:
        return conn.execute(sql).fetchone()[0]
    finally:
        conn.close()


def test_fresh_db_has_foundations_schema(tmp_path: Path):
    db_file = tmp_path / "fresh.db"
    Database(db_file).init_db()

    assert _JOBS_NEW <= _cols(db_file, "jobs")
    assert _RUNS_NEW <= _cols(db_file, "scrape_runs")
    assert {"job_id", "query_key", "first_seen", "last_seen"} <= _cols(db_file, "job_hits")
    assert _scalar(db_file, "SELECT MAX(version) FROM schema_version") == 3
    assert _scalar(
        db_file,
        "SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND tbl_name='job_hits' "
        "AND sql LIKE '%query_key%'",
    ) >= 1


def test_old_schema_db_keeps_rows_and_gains_schema(tmp_path: Path):
    db_file = tmp_path / "old.db"
    conn = sqlite3.connect(db_file)
    conn.executescript(_OLD_SCHEMA)
    conn.execute(
        "INSERT INTO jobs (dedup_hash, site, job_url, title, company, location, description) "
        "VALUES ('h1', 'indeed', 'u', 't', 'c', 'l', 'd')"
    )
    conn.commit()
    conn.close()

    db = Database(db_file)
    db.init_db()
    db.init_db()

    job = db.get_job_by_hash("h1")
    assert job is not None and job.title == "t"
    assert _JOBS_NEW <= _cols(db_file, "jobs")
    assert _RUNS_NEW <= _cols(db_file, "scrape_runs")
    assert _scalar(db_file, "SELECT COUNT(*) FROM schema_version") == 3
    assert _scalar(db_file, "SELECT remote_status FROM jobs WHERE dedup_hash='h1'") == "unverified"


def test_init_db_is_safe_when_run_concurrently(tmp_path: Path):
    db_file = tmp_path / "race.db"
    barrier = threading.Barrier(2)
    errors = []

    def run():
        try:
            barrier.wait(timeout=5)
            Database(db_file).init_db()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert errors == []
    assert _scalar(db_file, "SELECT COUNT(*) FROM schema_version") == 3


def test_existing_ensure_column_import_still_works():
    from src.storage.db import _ensure_column

    assert callable(_ensure_column)


def test_profile_tables_created(tmp_path: Path):
    db_file = tmp_path / "profile.db"
    Database(db_file).init_db()

    conn = sqlite3.connect(db_file)
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        conn.close()
    assert {
        "profile", "roles", "role_terms", "markets", "pinned_queries", "rules", "profile_versions",
    } <= tables
    assert _scalar(db_file, "SELECT MAX(version) FROM schema_version") == 3


def test_bot_tables_not_created(tmp_path: Path):
    db_file = tmp_path / "nobot.db"
    Database(db_file).init_db()

    conn = sqlite3.connect(db_file)
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        conn.close()
    assert not ({"pending_change", "conversation", "bot_state", "llm_consent"} & tables)


def test_migration_3_adds_pre_filter_rule_column(tmp_path: Path):
    db_file = tmp_path / "m3.db"
    Database(db_file).init_db()

    assert "pre_filter_rule" in _cols(db_file, "jobs")
    assert _scalar(
        db_file,
        "SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND name='idx_jobs_prefiltered'",
    ) == 1
    assert _scalar(db_file, "SELECT name FROM schema_version WHERE version = 3") == "rules_and_rubric"


def _insert_profile(conn, core, adjacent, dealbreakers):
    conn.execute(
        "INSERT INTO profile (id, version, core_role_definition, adjacent_role_definition, dealbreakers, "
        "updated_at) VALUES (1, 4, ?, ?, ?, 'now')",
        (core, adjacent, dealbreakers),
    )


def test_migration_3_rewrites_only_unedited_seed_text(tmp_path: Path):
    db_file = tmp_path / "m3seed.db"
    Database(db_file).init_db()
    conn = sqlite3.connect(db_file)
    edited = legacy_seed.LEGACY_V1_ADJACENT_ROLE_DEFINITION + " My own edit."
    _insert_profile(conn, legacy_seed.LEGACY_V1_CORE_ROLE_DEFINITION, edited, legacy_seed.LEGACY_V1_DEALBREAKERS)
    conn.execute("DELETE FROM schema_version WHERE version = 3")
    conn.commit()

    assert apply_migrations(conn) == 3
    conn.commit()
    row = conn.execute(
        "SELECT version, core_role_definition, adjacent_role_definition, dealbreakers FROM profile"
    ).fetchone()
    assert row == (
        4, legacy_seed.LEGACY_CORE_ROLE_DEFINITION, edited, legacy_seed.LEGACY_DEALBREAKERS,
    )

    conn.execute("DELETE FROM schema_version WHERE version = 3")
    conn.commit()
    before = conn.total_changes
    apply_migrations(conn)
    # The second run rewrites nothing: the new bodies no longer equal the V1 texts.
    assert conn.total_changes - before == 1  # only the schema_version row
    assert conn.execute("SELECT core_role_definition FROM profile").fetchone()[0] == (
        legacy_seed.LEGACY_CORE_ROLE_DEFINITION
    )
    conn.close()
