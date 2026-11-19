"""ConversationStore: the only code touching pending_change, conversation and llm_consent."""
import sqlite3
from pathlib import Path

import pytest

from src.bot.conversation_store import STATES, ConversationState, ConversationStore, PendingChange
from src.storage.db import Database

CHAT = 4242


@pytest.fixture
def store(tmp_path: Path) -> ConversationStore:
    db = Database(tmp_path / "conv.db")
    db.init_db()
    return ConversationStore(db)


def pending(change_id: str = "aaaa1111", *, chat_id: int = CHAT, created_at: float = 100.0, version: int = 3):
    return PendingChange(chat_id, change_id, version, "chat", '[{"op": "set_paused", "paused": true}]', None, created_at)


def test_states_constant():
    assert STATES == ("idle", "awaiting_description", "awaiting_clarification", "awaiting_confirm")


def test_put_pending_returns_superseded(store):
    assert store.put_pending(pending("aaaa1111")) is None
    first = store.get_pending(CHAT)
    assert first == pending("aaaa1111")

    superseded = store.put_pending(pending("bbbb2222", created_at=200.0))

    assert superseded == first
    assert store.get_pending(CHAT) == pending("bbbb2222", created_at=200.0)
    assert store.put_pending(pending("cccc3333", chat_id=7)) is None  # another chat is independent
    assert store.get_pending(CHAT).change_id == "bbbb2222"


def test_take_pending_is_atomic_and_bound_to_change_id(store):
    store.put_pending(pending("aaaa1111"))

    assert store.take_pending(CHAT, "wrong000") is None
    assert store.get_pending(CHAT) is not None  # the row stays
    taken = store.take_pending(CHAT, "aaaa1111")
    assert taken == pending("aaaa1111")
    assert store.take_pending(CHAT, "aaaa1111") is None
    assert store.get_pending(CHAT) is None


def test_delete_pending_and_set_pending_message(store):
    assert store.delete_pending(CHAT) is None
    store.put_pending(pending("aaaa1111"))
    store.set_pending_message(CHAT, "other000", 5)  # wrong change id: no effect
    assert store.get_pending(CHAT).message_id is None
    store.set_pending_message(CHAT, "aaaa1111", 55)
    assert store.get_pending(CHAT).message_id == 55

    deleted = store.delete_pending(CHAT)

    assert deleted is not None and deleted.change_id == "aaaa1111" and deleted.message_id == 55
    assert store.get_pending(CHAT) is None


def test_expired_pending_and_purge_contexts(store):
    store.put_pending(pending("old00000", chat_id=1, created_at=10.0))
    store.put_pending(pending("new00000", chat_id=2, created_at=500.0))
    assert [p.change_id for p in store.expired_pending(100.0)] == ["old00000"]
    assert store.expired_pending(5.0) == []

    store.set_state(1, "awaiting_clarification", {"rounds": [["text", "q?"]]}, now=10.0)
    store.set_state(2, "awaiting_clarification", {"rounds": [["text", "q?"]]}, now=500.0)
    assert store.purge_contexts(100.0) == 1

    purged = store.get_state(1)
    assert (purged.state, purged.context) == ("idle", None)
    kept = store.get_state(2)
    assert kept.state == "awaiting_clarification" and kept.context == {"rounds": [["text", "q?"]]}


def test_state_roundtrip_and_unknown_state_rejected(store):
    missing = store.get_state(CHAT)
    assert missing == ConversationState(CHAT, "idle", None, missing.updated_at)

    for state in STATES:
        store.set_state(CHAT, state, {"k": [1, "x"]} if state != "idle" else None, now=123.5)
        got = store.get_state(CHAT)
        assert got.state == state and got.updated_at == 123.5
        assert got.context == ({"k": [1, "x"]} if state != "idle" else None)

    with pytest.raises(ValueError):
        store.set_state(CHAT, "typing", None, now=1.0)
    assert store.get_state(CHAT).state == "awaiting_confirm"  # the rejected write changed nothing


def test_consent_versioned(store):
    assert store.has_consent(CHAT, 1) is False
    store.record_consent(CHAT, 1)
    assert store.has_consent(CHAT, 1) is True
    assert store.has_consent(CHAT, 2) is False
    assert store.has_consent(CHAT + 1, 1) is False
    store.record_consent(CHAT, 2)
    assert store.has_consent(CHAT, 2) is True


def test_clear_all(store):
    store.put_pending(pending())
    store.set_state(CHAT, "awaiting_confirm", None, now=1.0)
    store.record_consent(CHAT, 1)

    store.clear_all()

    assert store.get_pending(CHAT) is None
    assert store.get_state(CHAT).state == "idle"
    assert store.has_consent(CHAT, 1) is False


def test_connections_enable_secure_delete(store, monkeypatch):
    seen = []
    real_connect = sqlite3.connect

    class Spy(sqlite3.Connection):
        def execute(self, sql, *args):
            seen.append(sql)
            return super().execute(sql, *args)

    monkeypatch.setattr(sqlite3, "connect", lambda *a, **k: real_connect(*a, factory=Spy, **k))
    store.put_pending(pending())
    store.take_pending(CHAT, "aaaa1111")

    assert any("secure_delete" in s.lower() for s in seen)
    first_write = next(i for i, s in enumerate(seen) if s.lstrip().upper().startswith(("INSERT", "DELETE")))
    assert any("secure_delete" in s.lower() for s in seen[:first_write])
