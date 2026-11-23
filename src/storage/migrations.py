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


_PROFILE_TABLES = (
    """
    CREATE TABLE IF NOT EXISTS profile (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        version INTEGER NOT NULL,
        persona TEXT NOT NULL DEFAULT '',
        core_role_definition TEXT NOT NULL DEFAULT '',
        adjacent_role_definition TEXT NOT NULL DEFAULT '',
        dealbreakers TEXT NOT NULL DEFAULT '',
        bonus_criteria TEXT NOT NULL DEFAULT '',
        bonus_points REAL NOT NULL DEFAULT 1.5,
        output_language TEXT,
        target_level TEXT,
        wow_threshold REAL NOT NULL DEFAULT 8.5,
        digest_threshold REAL NOT NULL DEFAULT 6.5,
        max_calls_per_cycle INTEGER NOT NULL DEFAULT 100,
        paused INTEGER NOT NULL DEFAULT 0,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS roles (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL UNIQUE COLLATE NOCASE,
        enabled INTEGER NOT NULL DEFAULT 1
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS role_terms (
        id INTEGER PRIMARY KEY,
        role_id INTEGER NOT NULL REFERENCES roles(id) ON DELETE CASCADE,
        term TEXT NOT NULL,
        language TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS markets (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL UNIQUE COLLATE NOCASE,
        location TEXT NOT NULL,
        countries TEXT NOT NULL DEFAULT '[]',
        remote INTEGER NOT NULL DEFAULT 0,
        remote_only INTEGER NOT NULL DEFAULT 0,
        languages TEXT NOT NULL DEFAULT '[]',
        enabled INTEGER NOT NULL DEFAULT 1
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS pinned_queries (
        id INTEGER PRIMARY KEY,
        query_id TEXT NOT NULL UNIQUE,
        term TEXT NOT NULL,
        location TEXT NOT NULL,
        country TEXT,
        boards TEXT NOT NULL,
        is_remote INTEGER NOT NULL DEFAULT 0,
        google_search_term TEXT,
        enabled INTEGER NOT NULL DEFAULT 1
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS rules (
        id INTEGER PRIMARY KEY,
        kind TEXT NOT NULL,
        term TEXT NOT NULL,
        match_mode TEXT NOT NULL,
        reason TEXT NOT NULL,
        source TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS profile_versions (
        version INTEGER PRIMARY KEY,
        snapshot_schema INTEGER NOT NULL,
        snapshot TEXT NOT NULL,
        label TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
)


def _migration_2_profile_core(conn: sqlite3.Connection) -> None:
    for ddl in _PROFILE_TABLES:
        conn.execute(ddl)


def _migration_3_rules_and_rubric(conn: sqlite3.Connection) -> None:
    from src.profile.legacy_seed import LEGACY_FIELD_UPGRADES  # tracked seed text, no other imports

    ensure_column(conn, "jobs", "pre_filter_rule", "TEXT")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_prefiltered ON jobs(pre_filtered, id)")
    # Unedited seed definitions get the current body (exact match only; the profile version is unchanged
    # because the meaning is unchanged and an older process still reads valid text). The profile table may
    # be empty, in which case nothing matches.
    for field, (olds, new) in LEGACY_FIELD_UPGRADES.items():
        for old in olds:
            conn.execute(f"UPDATE profile SET {field} = ? WHERE {field} = ?", (new, old))


def _migration_4_bot_core(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS bot_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            update_offset INTEGER NOT NULL DEFAULT 0,
            paired_user_id INTEGER, paired_chat_id INTEGER,
            pairing_code_hash TEXT, pairing_expires_at REAL,
            pairing_attempts INTEGER NOT NULL DEFAULT 0,
            wizard_state TEXT,
            updated_at TEXT NOT NULL)
        """
    )
    conn.execute(
        "INSERT OR IGNORE INTO bot_state (id, updated_at) VALUES (1, ?)",
        (datetime.now(timezone.utc).isoformat(),),
    )


def _migration_5_llm_layer(conn: sqlite3.Connection) -> None:
    # No CHECK constraint on ``conversation.state``: add-only migrations must stay extensible.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS pending_change (
            chat_id INTEGER PRIMARY KEY,
            change_id TEXT NOT NULL,
            base_version INTEGER NOT NULL,
            mode TEXT NOT NULL,
            operations TEXT NOT NULL,
            message_id INTEGER,
            created_at REAL NOT NULL)
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS conversation (
            chat_id INTEGER PRIMARY KEY,
            state TEXT NOT NULL DEFAULT 'idle',
            context TEXT,
            updated_at REAL NOT NULL)
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS llm_consent (
            chat_id INTEGER PRIMARY KEY,
            notice_version INTEGER NOT NULL,
            acknowledged_at TEXT NOT NULL)
        """
    )


MIGRATIONS: List[Tuple[int, str, Callable[[sqlite3.Connection], None]]] = [
    (1, "foundations", _migration_1_foundations),
    (2, "profile_core", _migration_2_profile_core),
    (3, "rules_and_rubric", _migration_3_rules_and_rubric),
    (4, "bot_core", _migration_4_bot_core),
    (5, "llm_layer", _migration_5_llm_layer),
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
