import sqlite3
import threading
from pathlib import Path

from src.storage.db import Database
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
    assert _scalar(db_file, "SELECT MAX(version) FROM schema_version") == 2
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
    assert _scalar(db_file, "SELECT COUNT(*) FROM schema_version") == 2
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
    assert _scalar(db_file, "SELECT COUNT(*) FROM schema_version") == 2


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
    assert _scalar(db_file, "SELECT MAX(version) FROM schema_version") == 2


def test_bot_tables_not_created(tmp_path: Path):
    db_file = tmp_path / "nobot.db"
    Database(db_file).init_db()

    conn = sqlite3.connect(db_file)
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        conn.close()
    assert not ({"pending_change", "conversation", "bot_state", "llm_consent"} & tables)
