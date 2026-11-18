"""Add-only schema migrations.

Rules: migrations only add (tables, columns, indexes); they never rename or drop. Each migration is
idempotent, so a fresh database and an old one converge to the same schema and two processes running
``init_db`` at once are harmless (the version row is recorded with ``INSERT OR IGNORE``).
"""
import sqlite3
from datetime import datetime, timezone
from typing import Callable, List, Tuple


def ensure_column(conn: sqlite3.Connection, table: str, name: str, ddl: str) -> None:
    """Idempotently add a column; tolerant of a concurrent init_db winning the race."""
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if name in existing:
        return
    try:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")
    except sqlite3.OperationalError as exc:
        if "duplicate column" not in str(exc).lower():
            raise


def _migration_1_foundations(conn: sqlite3.Connection) -> None:
    ensure_column(conn, "jobs", "source_query", "TEXT")
    ensure_column(conn, "jobs", "remote_status", "TEXT NOT NULL DEFAULT 'unverified'")
    ensure_column(conn, "jobs", "eval_raw_score", "REAL")
    ensure_column(conn, "jobs", "eval_is_core_role", "INTEGER")
    ensure_column(conn, "jobs", "eval_bonus_flag", "INTEGER")
    ensure_column(conn, "jobs", "eval_hard_reject", "INTEGER")

    ensure_column(conn, "scrape_runs", "rows_returned", "INTEGER NOT NULL DEFAULT 0")
    ensure_column(conn, "scrape_runs", "prefiltered", "INTEGER NOT NULL DEFAULT 0")
    ensure_column(conn, "scrape_runs", "outcome", "TEXT")
    ensure_column(conn, "scrape_runs", "query_key", "TEXT")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS job_hits (
            job_id INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
            query_key TEXT NOT NULL,
            first_seen TEXT NOT NULL,
            last_seen TEXT NOT NULL,
            PRIMARY KEY (job_id, query_key)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_job_hits_query_key ON job_hits(query_key)")


MIGRATIONS: List[Tuple[int, str, Callable[[sqlite3.Connection], None]]] = [
    (1, "foundations", _migration_1_foundations),
]


def apply_migrations(conn: sqlite3.Connection) -> int:
    """Apply every migration not yet recorded; return the highest applied version."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            applied_at TEXT NOT NULL
        )
        """
    )
    applied = {row[0] for row in conn.execute("SELECT version FROM schema_version")}
    for version, name, func in MIGRATIONS:
        if version in applied:
            continue
        func(conn)
        conn.execute(
            "INSERT OR IGNORE INTO schema_version (version, name, applied_at) VALUES (?, ?, ?)",
            (version, name, datetime.now(timezone.utc).isoformat()),
        )
    return conn.execute("SELECT COALESCE(MAX(version), 0) FROM schema_version").fetchone()[0]
