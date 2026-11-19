"""Persistence of the bot: the only code that touches the ``bot_state`` table (a single row, id 1)."""
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from src.storage.db import Database


@dataclass(frozen=True)
class Pairing:
    user_id: int
    chat_id: int


@dataclass(frozen=True)
class PairingCode:
    code_hash: str
    expires_at: float
    attempts: int


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class BotStateStore:
    def __init__(self, db: Database):
        self._db = db

    def _row(self):
        with self._db._get_connection() as conn:
            return conn.execute("SELECT * FROM bot_state WHERE id = 1").fetchone()

    def _update(self, assignments: str, params: tuple = ()) -> None:
        with self._db._get_connection() as conn:
            # The migration seeds the row; INSERT OR IGNORE keeps a manually emptied table working.
            conn.execute("INSERT OR IGNORE INTO bot_state (id, updated_at) VALUES (1, ?)", (_now(),))
            conn.execute(f"UPDATE bot_state SET {assignments}, updated_at = ? WHERE id = 1", (*params, _now()))

    # --- update offset ---------------------------------------------------------------------

    def get_offset(self) -> int:
        row = self._row()
        return int(row["update_offset"]) if row else 0

    def set_offset(self, offset: int) -> None:
        self._update("update_offset = ?", (int(offset),))  # may go down: Telegram can restart its ids

    # --- pairing ---------------------------------------------------------------------------

    def get_pairing(self) -> Optional[Pairing]:
        row = self._row()
        if row is None or row["paired_user_id"] is None or row["paired_chat_id"] is None:
            return None
        return Pairing(user_id=int(row["paired_user_id"]), chat_id=int(row["paired_chat_id"]))

    def set_pairing(self, user_id: int, chat_id: int) -> None:
        self._update(
            "paired_user_id = ?, paired_chat_id = ?, pairing_code_hash = NULL, "
            "pairing_expires_at = NULL, pairing_attempts = 0",
            (int(user_id), int(chat_id)),
        )

    def clear_pairing(self) -> None:
        self._update(
            "paired_user_id = NULL, paired_chat_id = NULL, pairing_code_hash = NULL, "
            "pairing_expires_at = NULL, pairing_attempts = 0, wizard_state = NULL"
        )

    def set_pairing_code(self, code_hash: str, expires_at: float) -> None:
        self._update(
            "pairing_code_hash = ?, pairing_expires_at = ?, pairing_attempts = 0",
            (code_hash, float(expires_at)),
        )

    def get_pairing_code(self) -> Optional[PairingCode]:
        row = self._row()
        if row is None or row["pairing_code_hash"] is None or row["pairing_expires_at"] is None:
            return None
        return PairingCode(
            code_hash=row["pairing_code_hash"],
            expires_at=float(row["pairing_expires_at"]),
            attempts=int(row["pairing_attempts"]),
        )

    def clear_pairing_code(self) -> None:
        self._update("pairing_code_hash = NULL, pairing_expires_at = NULL, pairing_attempts = 0")

    def record_failed_attempt(self) -> int:
        with self._db._get_connection() as conn:
            conn.execute("INSERT OR IGNORE INTO bot_state (id, updated_at) VALUES (1, ?)", (_now(),))
            conn.execute(
                "UPDATE bot_state SET pairing_attempts = pairing_attempts + 1, updated_at = ? WHERE id = 1",
                (_now(),),
            )
            return int(conn.execute("SELECT pairing_attempts FROM bot_state WHERE id = 1").fetchone()[0])

    def reset_attempts(self) -> None:
        self._update("pairing_attempts = 0")

    # --- wizard ----------------------------------------------------------------------------

    def get_wizard(self) -> Optional[Dict[str, Any]]:
        row = self._row()
        if row is None or not row["wizard_state"]:
            return None
        try:
            value = json.loads(row["wizard_state"])
        except ValueError:
            return None
        return value if isinstance(value, dict) else None

    def set_wizard(self, state: Optional[Dict[str, Any]]) -> None:
        self._update("wizard_state = ?", (json.dumps(state) if state is not None else None,))
