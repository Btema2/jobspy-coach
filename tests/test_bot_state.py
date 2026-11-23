import sqlite3

from src.bot.state import BotStateStore, Pairing, PairingCode
from src.storage.db import Database


def _store(tmp_path) -> BotStateStore:
    db = Database(tmp_path / "bot.db")
    db.init_db()
    return BotStateStore(db)


def test_initial_state_is_empty(tmp_path):
    store = _store(tmp_path)
    assert store.get_offset() == 0
    assert store.get_pairing() is None
    assert store.get_pairing_code() is None
    assert store.get_wizard() is None


def test_offset_is_stored_as_given_and_may_go_down(tmp_path):
    """Telegram can restart its update ids at a lower number; the offset must follow it."""
    store = _store(tmp_path)
    store.set_offset(1_000_000)
    store.set_offset(6)
    assert store.get_offset() == 6
    store.set_offset(11)
    assert store.get_offset() == 11


def test_pairing_roundtrip_clears_code(tmp_path):
    store = _store(tmp_path)
    store.set_pairing_code("hash", 123.5)
    assert store.get_pairing_code() == PairingCode("hash", 123.5, 0)
    store.record_failed_attempt()

    store.set_pairing(7, 8)

    assert store.get_pairing() == Pairing(user_id=7, chat_id=8)
    assert store.get_pairing_code() is None
    conn = sqlite3.connect(tmp_path / "bot.db")
    try:
        assert conn.execute("SELECT pairing_attempts FROM bot_state").fetchone()[0] == 0
    finally:
        conn.close()


def test_failed_attempts_count_and_reset(tmp_path):
    store = _store(tmp_path)
    store.set_pairing_code("hash", 99.0)
    assert store.record_failed_attempt() == 1
    assert store.record_failed_attempt() == 2
    assert store.get_pairing_code().attempts == 2
    store.reset_attempts()
    assert store.get_pairing_code().attempts == 0
    store.record_failed_attempt()
    store.set_pairing_code("hash2", 100.0)  # a new code starts at zero attempts
    assert store.get_pairing_code() == PairingCode("hash2", 100.0, 0)
    store.clear_pairing_code()
    assert store.get_pairing_code() is None


def test_clear_pairing_clears_wizard(tmp_path):
    store = _store(tmp_path)
    store.set_pairing(7, 7)
    store.set_pairing_code("h", 1.0)
    store.set_wizard({"step": "role"})

    store.clear_pairing()

    assert store.get_pairing() is None
    assert store.get_pairing_code() is None
    assert store.get_wizard() is None


def test_wizard_json_roundtrip(tmp_path):
    store = _store(tmp_path)
    state = {"step": "market", "nonce": "abc", "terms": ["<ROLE>", "ö"], "version": 3}
    store.set_wizard(state)
    assert store.get_wizard() == state
    store.set_wizard(None)
    assert store.get_wizard() is None
