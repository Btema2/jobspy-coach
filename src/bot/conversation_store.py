"""Persistence of the free-text layer: the only code that touches ``pending_change``, ``conversation`` and
``llm_consent`` (all keyed by chat id, one row each).

Retention: ``pending_change`` holds the validated operation list only (never raw pasted text);
``conversation.context`` may hold raw earlier messages while a clarifying question is open and is purged
on apply, cancel, supersede and expiry. Every connection sets ``PRAGMA secure_delete = ON`` before it
writes, so deleted text is overwritten in the database file.
"""
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional

from src.storage.db import Database

STATES = ("idle", "awaiting_description", "awaiting_clarification", "awaiting_confirm")


@dataclass(frozen=True)
class PendingChange:
    chat_id: int
    change_id: str
    base_version: int
    mode: str  # "chat" or "onboarding"
    operations: str  # JSON of the validated operation list
    message_id: Optional[int]  # the proposal's last message (the one carrying the buttons)
    created_at: float


@dataclass(frozen=True)
class ConversationState:
    chat_id: int
    state: str
    context: Optional[Dict[str, Any]]
    updated_at: float


def _pending(row: Any) -> PendingChange:
    return PendingChange(
        chat_id=int(row["chat_id"]),
        change_id=row["change_id"],
        base_version=int(row["base_version"]),
        mode=row["mode"],
        operations=row["operations"],
        message_id=int(row["message_id"]) if row["message_id"] is not None else None,
        created_at=float(row["created_at"]),
    )


_PENDING_COLUMNS = "chat_id, change_id, base_version, mode, operations, message_id, created_at"


class ConversationStore:
    def __init__(self, db: Database):
        self._db = db

    @contextmanager
    def _conn(self) -> Generator[Any, None, None]:
        with self._db._get_connection() as conn:
            conn.execute("PRAGMA secure_delete = ON;")
            yield conn

    # --- pending proposal ------------------------------------------------------------------

    def put_pending(self, p: PendingChange) -> Optional[PendingChange]:
        """Store ``p`` as the chat's only proposal; the row it replaced (if any) is returned."""
        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            old = conn.execute(
                f"SELECT {_PENDING_COLUMNS} FROM pending_change WHERE chat_id = ?", (p.chat_id,)
            ).fetchone()
            conn.execute(
                f"INSERT OR REPLACE INTO pending_change ({_PENDING_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (p.chat_id, p.change_id, p.base_version, p.mode, p.operations, p.message_id, p.created_at),
            )
        return _pending(old) if old is not None else None

    def get_pending(self, chat_id: int) -> Optional[PendingChange]:
        with self._conn() as conn:
            row = conn.execute(
                f"SELECT {_PENDING_COLUMNS} FROM pending_change WHERE chat_id = ?", (chat_id,)
            ).fetchone()
        return _pending(row) if row is not None else None

    def take_pending(self, chat_id: int, change_id: str) -> Optional[PendingChange]:
        """Atomically remove and return the proposal when ``change_id`` matches, else None (row kept)."""
        with self._conn() as conn:
            row = conn.execute(
                f"DELETE FROM pending_change WHERE chat_id = ? AND change_id = ? RETURNING {_PENDING_COLUMNS}",
                (chat_id, change_id),
            ).fetchone()
        return _pending(row) if row is not None else None

    def delete_pending(self, chat_id: int) -> Optional[PendingChange]:
        with self._conn() as conn:
            row = conn.execute(
                f"DELETE FROM pending_change WHERE chat_id = ? RETURNING {_PENDING_COLUMNS}", (chat_id,)
            ).fetchone()
        return _pending(row) if row is not None else None

    def set_pending_message(self, chat_id: int, change_id: str, message_id: int) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE pending_change SET message_id = ? WHERE chat_id = ? AND change_id = ?",
                (message_id, chat_id, change_id),
            )

    def expired_pending(self, before: float) -> List[PendingChange]:
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT {_PENDING_COLUMNS} FROM pending_change WHERE created_at < ? ORDER BY created_at",
                (before,),
            ).fetchall()
        return [_pending(r) for r in rows]

    # --- conversation state ----------------------------------------------------------------

    def get_state(self, chat_id: int) -> ConversationState:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT state, context, updated_at FROM conversation WHERE chat_id = ?", (chat_id,)
            ).fetchone()
        if row is None:
            return ConversationState(chat_id, "idle", None, 0.0)
        context = json.loads(row["context"]) if row["context"] else None
        return ConversationState(chat_id, row["state"], context, float(row["updated_at"]))

    def set_state(
        self, chat_id: int, state: str, context: Optional[Dict[str, Any]] = None, now: Optional[float] = None
    ) -> None:
        if state not in STATES:
            raise ValueError(f"unknown conversation state '{state[:30]}'")
        stamp = time.time() if now is None else now
        payload = json.dumps(context) if context is not None else None
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO conversation (chat_id, state, context, updated_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(chat_id) DO UPDATE SET state = excluded.state, context = excluded.context, "
                "updated_at = excluded.updated_at",
                (chat_id, state, payload, stamp),
            )

    def purge_contexts(self, before: float) -> int:
        """Reset rows last touched before ``before`` to idle with no context; returns how many held one."""
        with self._conn() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM conversation WHERE updated_at < ? AND (context IS NOT NULL OR state != 'idle')",
                (before,),
            ).fetchone()[0]
            conn.execute(
                "UPDATE conversation SET state = 'idle', context = NULL WHERE updated_at < ? "
                "AND (context IS NOT NULL OR state != 'idle')",
                (before,),
            )
        return int(count)

    # --- consent ---------------------------------------------------------------------------

    def has_consent(self, chat_id: int, version: int) -> bool:
        with self._conn() as conn:
            row = conn.execute("SELECT notice_version FROM llm_consent WHERE chat_id = ?", (chat_id,)).fetchone()
        return row is not None and int(row["notice_version"]) >= version

    def record_consent(self, chat_id: int, version: int) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO llm_consent (chat_id, notice_version, acknowledged_at) VALUES (?, ?, ?)",
                (chat_id, version, datetime.now(timezone.utc).isoformat()),
            )

    # --- reset -----------------------------------------------------------------------------

    def clear_all(self) -> None:
        """Forget every proposal, conversation and consent (unpair and a new pairing)."""
        with self._conn() as conn:
            for table in ("pending_change", "conversation", "llm_consent"):
                conn.execute(f"DELETE FROM {table}")
