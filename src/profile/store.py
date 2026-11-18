"""Persistence of the profile: the only code that touches the profile tables.

Reads run in one read transaction across all tables (a consistent snapshot even while another process
commits). Writes replace the whole state in one ``BEGIN IMMEDIATE`` transaction guarded by an optimistic
version check, and record the new state in ``profile_versions`` (the last ``KEEP_VERSIONS`` are kept).
"""
import json
import sqlite3
from datetime import datetime, timezone
from typing import List, Optional

from src.profile.errors import VersionConflict
from src.profile.snapshot import (
    SNAPSHOT_SCHEMA,
    Market,
    PinnedQuery,
    ProfileFields,
    ProfileSnapshot,
    Role,
    RoleTerm,
    Rule,
    VersionInfo,
    snapshot_from_dict,
    snapshot_to_dict,
)
from src.storage.db import Database

KEEP_VERSIONS = 10

_TABLES = frozenset({"profile", "roles", "role_terms", "markets", "pinned_queries", "rules"})
_FIELD_COLUMNS = (
    "persona", "core_role_definition", "adjacent_role_definition", "dealbreakers", "bonus_criteria",
    "bonus_points", "output_language", "target_level", "wow_threshold", "digest_threshold",
    "max_calls_per_cycle", "paused",
)


class ProfileStore:
    def __init__(self, db: Database):
        self._db = db

    # --- reads -----------------------------------------------------------------------------

    def has_profile(self) -> bool:
        with self._db._get_connection() as conn:
            return conn.execute("SELECT 1 FROM profile WHERE id = 1").fetchone() is not None

    def _read_table(self, conn: sqlite3.Connection, table: str) -> List[sqlite3.Row]:
        if table not in _TABLES:
            raise ValueError(f"unknown profile table {table!r}")
        return conn.execute(f"SELECT * FROM {table} ORDER BY id").fetchall()

    def read_snapshot(self) -> Optional[ProfileSnapshot]:
        with self._db._get_connection() as conn:
            conn.execute("BEGIN")
            try:
                # The first read pins the read snapshot, so `profile` must be read first.
                profile_rows = self._read_table(conn, "profile")
                if not profile_rows:
                    return None
                role_rows = self._read_table(conn, "roles")
                term_rows = self._read_table(conn, "role_terms")
                market_rows = self._read_table(conn, "markets")
                pinned_rows = self._read_table(conn, "pinned_queries")
                rule_rows = self._read_table(conn, "rules")
            finally:
                conn.execute("COMMIT")
        p = profile_rows[0]
        fields = ProfileFields(**{**{c: p[c] for c in _FIELD_COLUMNS}, "paused": bool(p["paused"])})
        terms_by_role: dict = {}
        for t in term_rows:
            terms_by_role.setdefault(t["role_id"], []).append(RoleTerm(t["term"], t["language"]))
        roles = tuple(
            Role(r["name"], bool(r["enabled"]), tuple(terms_by_role.get(r["id"], ()))) for r in role_rows
        )
        markets = tuple(
            Market(
                name=m["name"],
                location=m["location"],
                countries=tuple(json.loads(m["countries"])),
                remote=bool(m["remote"]),
                remote_only=bool(m["remote_only"]),
                languages=tuple(json.loads(m["languages"])),
                enabled=bool(m["enabled"]),
            )
            for m in market_rows
        )
        pinned = tuple(
            PinnedQuery(
                query_id=q["query_id"],
                term=q["term"],
                location=q["location"],
                boards=tuple(json.loads(q["boards"])),
                country=q["country"],
                is_remote=bool(q["is_remote"]),
                google_search_term=q["google_search_term"],
                enabled=bool(q["enabled"]),
            )
            for q in pinned_rows
        )
        rules = tuple(Rule(r["kind"], r["term"], r["match_mode"], r["reason"], r["source"]) for r in rule_rows)
        return ProfileSnapshot(fields, roles, markets, pinned, rules, p["version"])

    def list_versions(self) -> List[VersionInfo]:
        with self._db._get_connection() as conn:
            rows = conn.execute(
                "SELECT version, label, created_at FROM profile_versions ORDER BY version DESC"
            ).fetchall()
        return [VersionInfo(r["version"], r["label"], r["created_at"]) for r in rows]

    def load_version(self, version: int) -> ProfileSnapshot:
        with self._db._get_connection() as conn:
            row = conn.execute(
                "SELECT snapshot FROM profile_versions WHERE version = ?", (version,)
            ).fetchone()
        if row is None:
            raise KeyError(version)
        return snapshot_from_dict(json.loads(row["snapshot"]), version=version)

    # --- writes ----------------------------------------------------------------------------

    def write_snapshot(self, snapshot: ProfileSnapshot, *, expected_version: int, label: str) -> int:
        now = datetime.now(timezone.utc).isoformat()
        new_version = expected_version + 1
        with self._db._get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute("SELECT version FROM profile WHERE id = 1").fetchone()
                current = row["version"] if row else 0
                if current != expected_version:
                    raise VersionConflict(
                        f"profile is at version {current}, change was based on version {expected_version}"
                    )
                self._replace_rows(conn, snapshot, new_version, now, exists=row is not None)
                conn.execute(
                    "INSERT INTO profile_versions (version, snapshot_schema, snapshot, label, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (new_version, SNAPSHOT_SCHEMA, json.dumps(snapshot_to_dict(snapshot)), label, now),
                )
                conn.execute(
                    "DELETE FROM profile_versions WHERE version NOT IN "
                    "(SELECT version FROM profile_versions ORDER BY version DESC LIMIT ?)",
                    (KEEP_VERSIONS,),
                )
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        return new_version

    @staticmethod
    def _replace_rows(
        conn: sqlite3.Connection, snap: ProfileSnapshot, version: int, now: str, *, exists: bool
    ) -> None:
        f = snap.fields
        values = [getattr(f, c) for c in _FIELD_COLUMNS]
        values[_FIELD_COLUMNS.index("paused")] = int(f.paused)
        if exists:
            assignments = ", ".join(f"{c} = ?" for c in _FIELD_COLUMNS)
            conn.execute(
                f"UPDATE profile SET version = ?, {assignments}, updated_at = ? WHERE id = 1",
                [version, *values, now],
            )
        else:
            columns = ", ".join(_FIELD_COLUMNS)
            marks = ", ".join("?" for _ in _FIELD_COLUMNS)
            conn.execute(
                f"INSERT INTO profile (id, version, {columns}, updated_at) VALUES (1, ?, {marks}, ?)",
                [version, *values, now],
            )
        for table in ("role_terms", "roles", "markets", "pinned_queries", "rules"):
            conn.execute(f"DELETE FROM {table}")
        for role in snap.roles:
            cur = conn.execute("INSERT INTO roles (name, enabled) VALUES (?, ?)", (role.name, int(role.enabled)))
            for term in role.terms:
                conn.execute(
                    "INSERT INTO role_terms (role_id, term, language) VALUES (?, ?, ?)",
                    (cur.lastrowid, term.term, term.language),
                )
        for m in snap.markets:
            conn.execute(
                "INSERT INTO markets (name, location, countries, remote, remote_only, languages, enabled) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (m.name, m.location, json.dumps(list(m.countries)), int(m.remote), int(m.remote_only),
                 json.dumps(list(m.languages)), int(m.enabled)),
            )
        for q in snap.pinned_queries:
            conn.execute(
                "INSERT INTO pinned_queries (query_id, term, location, country, boards, is_remote, "
                "google_search_term, enabled) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (q.query_id, q.term, q.location, q.country, json.dumps(list(q.boards)), int(q.is_remote),
                 q.google_search_term, int(q.enabled)),
            )
        for r in snap.rules:
            conn.execute(
                "INSERT INTO rules (kind, term, match_mode, reason, source) VALUES (?, ?, ?, ?, ?)",
                (r.kind, r.term, r.match_mode, r.reason, r.source),
            )
