"""ConversationService: debounce, consent, clarification, pending proposals, confirm/cancel, uploads.

No network (``FakeTelegram`` plus a scripted fake translator), no real .env, placeholders only. The worker is
driven with ``process_next(now)`` and a fake clock; the real thread is exercised in two dedicated tests.
"""
import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, List, Optional

import pytest

import src.bot.conversation as conv
from src.bot.chat_llm import (
    ChatLLMLimited,
    ChatLLMNotConfigured,
    ChatLLMNotUnderstood,
    ChatLLMUnavailable,
)
from src.bot.client import TelegramBotClient
from src.bot.commands import TRIGGER_STARTED, BotContext
from src.bot.conversation import ConversationService
from src.bot.conversation_store import ConversationStore
from src.bot.privacy import CONSENT_NOTICE, CONSENT_VERSION
from src.bot.state import BotStateStore
from src.bot.translator import (
    Clarification,
    NoChange,
    TranslationProblem,
    operations_from_json,
    prepare_proposal,
)
from src.bot.updates import parse_update
from src.bot.wizard import SetupWizard
from src.config import Settings
from src.profile.errors import ProfileValidationError
from src.profile.ops import AddMarket, AddRole, SetField, apply_changes, create_starter_profile
from src.profile.snapshot import Market, RoleTerm
from src.profile.store import ProfileStore
from src.storage.db import Database
from tests.fake_telegram import FakeTelegram, callback_update, document_update, message_update

NOW = 1_700_000_000.0
TOKEN = "<TOKEN>"
MARKER = "<MARKER>"


class Clock:
    def __init__(self, value: float = NOW):
        self.value = value

    def __call__(self) -> float:
        return self.value


class Control:
    def __init__(self):
        self.trigger_calls = 0

    def trigger(self) -> str:
        self.trigger_calls += 1
        return TRIGGER_STARTED

    def is_running(self) -> bool:
        return False


def prop(*ops: dict):
    """A translator outcome: the dry-run proposal of ``ops`` on the snapshot the service passes in."""

    def make(snapshot, mode):
        return prepare_proposal(operations_from_json(json.dumps(list(ops))), snapshot, mode=mode)

    return make


def add_role(name: str) -> dict:
    return {"op": "add_role", "name": name, "synonyms": []}


class FakeTranslator:
    def __init__(self):
        self.script: List[Any] = []
        self.calls: List[dict] = []
        self.gate: Optional[threading.Event] = None
        self.entered = threading.Event()

    def add(self, *items: Any) -> "FakeTranslator":
        self.script.extend(items)
        return self

    def translate(self, parts, snapshot, *, mode, history=(), request_id):
        self.calls.append({"parts": list(parts), "mode": mode, "history": list(history), "snapshot": snapshot})
        self.entered.set()
        if self.gate is not None:
            assert self.gate.wait(10)
        item = self.script.pop(0) if self.script else NoChange()
        if isinstance(item, Exception):
            raise item
        return item(snapshot, mode) if callable(item) else item


@pytest.fixture
def db(tmp_path: Path) -> Database:
    database = Database(tmp_path / "conv.db")
    database.init_db()
    return database


@pytest.fixture
def store(db: Database) -> ProfileStore:
    profile_store = ProfileStore(db)
    create_starter_profile(profile_store, wow_threshold=8.5, digest_threshold=6.5, output_language="en")
    apply_changes(
        profile_store,
        [
            AddRole("<ROLE>", (RoleTerm("<SYN>", None),)),
            AddMarket(Market("<MARKET>", "<LOCATION>", ("poland",), languages=("pl", "en"))),
            SetField("paused", False),
        ],
        label="fixture",
    )
    return profile_store


@pytest.fixture
def conv_store(db: Database) -> ConversationStore:
    return ConversationStore(db)


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def control() -> Control:
    return Control()


@pytest.fixture
def telegram() -> FakeTelegram:
    return FakeTelegram()


@pytest.fixture
def translator() -> FakeTranslator:
    return FakeTranslator()


@pytest.fixture
def svc(tmp_path, db, store, conv_store, clock, control, telegram, translator) -> ConversationService:
    ctx = BotContext(
        db_path=db.db_path,
        env_path=tmp_path / ".env",
        settings=Settings(_env_file=None),
        trigger_cycle=control.trigger,
        is_cycle_running=control.is_running,
        scraper=None,
        clock=clock,
    )
    client = TelegramBotClient(TOKEN, transport=telegram.transport)
    wizard = SetupWizard(BotStateStore(db), store, clock=clock)
    conv_store.record_consent(111, CONSENT_VERSION)
    return ConversationService(ctx, client, wizard, translator=translator, clock=clock)


_ids = iter(range(1000, 10**6))


def text_update(text: str, **kw):
    return parse_update(message_update(next(_ids), text, **kw))


def callback(data: str, message_id: int = 5):
    return parse_update(callback_update(next(_ids), data, message_id=message_id))


def say(svc: ConversationService, text: str, **kw):
    return svc.on_text(text_update(text, **kw))


def texts(reply) -> str:
    return "\n".join(m.text for m in reply or [])


def sent(telegram: FakeTelegram) -> List[str]:
    return [c["text"] for c in telegram.calls("sendMessage")]


def last_markup(telegram: FakeTelegram) -> dict:
    return [c for c in telegram.calls("sendMessage") if c.get("reply_markup")][-1]["reply_markup"]


def button_data(markup: dict) -> List[str]:
    return [b["callback_data"] for row in markup["inline_keyboard"] for b in row]


def version(store: ProfileStore) -> int:
    return store.read_snapshot().version


def run_request(svc, clock, text="request", *, at: Optional[float] = None) -> None:
    """One text through the debounce and the worker."""
    clock.value = at if at is not None else clock.value
    say(svc, text)
    clock.value += conv.DEBOUNCE_SECONDS
    assert svc.process_next(clock.value) is True


def propose(svc, clock, translator, telegram, *ops: dict, text="request") -> str:
    """Run one request that ends in a proposal and return the confirm callback data."""
    translator.add(prop(*ops))
    run_request(svc, clock, text)
    return [d for d in button_data(last_markup(telegram)) if d.endswith(":a")][0]


def dump_db(path: Path) -> str:
    conn = sqlite3.connect(path)
    try:
        return "\n".join(conn.iterdump())
    finally:
        conn.close()


# --- debounce ------------------------------------------------------------------------------------


def test_parts_within_3s_merge_into_one_request(svc, clock, translator):
    clock.value = NOW
    assert say(svc, "A") == []
    clock.value = NOW + 1
    assert say(svc, "B") == []
    clock.value = NOW + 2
    assert say(svc, "C") == []
    assert svc.process_next(NOW + 4.9) is False  # three seconds after the last part, not before
    clock.value = NOW + 6  # 4 s after the last part: a new request, queued after the first
    say(svc, "D")
    assert svc.process_next(NOW + 6) is True
    assert [c["parts"] for c in translator.calls] == [["A", "B", "C"]]
    assert svc.process_next(NOW + 6) is False  # D is not due yet
    assert svc.process_next(NOW + 9) is True
    assert [c["parts"] for c in translator.calls] == [["A", "B", "C"], ["D"]]


def test_fourth_part_or_over_limit_drops_whole_paste(svc, clock, translator):
    for index, offset in enumerate((0, 0.5, 1.0)):
        clock.value = NOW + offset
        assert say(svc, f"part {index}") == []
    clock.value = NOW + 1.5
    assert texts(say(svc, "part 3")) == conv.TOO_LONG_REPLY
    clock.value = NOW + 2.0
    assert say(svc, "part 4") == []  # the tail of the same paste is swallowed silently
    assert svc.process_next(NOW + 100) is False
    assert translator.calls == []
    clock.value = NOW + 100  # a later message is an ordinary request again
    assert say(svc, "hello") == []
    assert svc.process_next(NOW + 103) is True and len(translator.calls) == 1

    clock.value = NOW + 200
    assert say(svc, "a" * 4000) == []
    assert say(svc, "b" * 4000) == []
    assert texts(say(svc, "c" * 4001)) == conv.TOO_LONG_REPLY  # 12,001 characters in total
    assert svc.process_next(NOW + 300) is False
    assert len(translator.calls) == 1
    clock.value = NOW + 400  # exactly 12,000 characters is accepted
    for ch in "abc":
        say(svc, ch * 4000)
    assert svc.process_next(NOW + 403) is True
    assert sum(len(p) for p in translator.calls[-1]["parts"]) == 12000


def test_request_queue_is_capped(svc, clock, translator):
    replies = []
    for index in range(conv.MAX_QUEUED_REQUESTS + 1):
        clock.value = NOW + index * 10  # each part is a separate request (more than 3 s apart)
        replies.append(say(svc, f"r{index}"))
    assert replies[:-1] == [[]] * conv.MAX_QUEUED_REQUESTS
    assert texts(replies[-1]) == conv.BUSY_REPLY


# --- uploads -------------------------------------------------------------------------------------


def test_upload_txt_md_up_to_20kb(svc, clock, translator, telegram, caplog):
    telegram.serve_file("f1", "documents/a.md", b"x" * 20480)
    caplog.set_level(logging.DEBUG)
    clock.value = NOW
    reply = svc.on_text(parse_update(document_update(1, "f1", "Notes.MD", caption="<CAPTION>")))
    assert reply == []
    assert svc.process_next(NOW) is True  # an upload is its own request: no debounce wait
    assert translator.calls[0]["parts"] == ["<CAPTION>", "x" * 20480]
    assert TOKEN not in caplog.text

    telegram.serve_file("f2", "documents/b.txt", b"y" * 20481)
    svc.on_text(parse_update(document_update(2, "f2", "b.txt")))
    assert svc.process_next(NOW) is True
    assert sent(telegram)[-1] == conv.FILE_TOO_LARGE_REPLY
    assert len(translator.calls) == 1

    reply = svc.on_text(parse_update(document_update(3, "f3", "c.txt", file_size=20481)))
    assert texts(reply) == conv.FILE_TOO_LARGE_REPLY  # known size: refused before any download
    assert len(telegram.calls("getFile")) == 2

    reply = svc.on_text(parse_update(document_update(4, "f4", "cv.pdf")))
    assert texts(reply) == conv.ONLY_TEXT_FILES_REPLY
    assert svc.process_next(NOW + 100) is False

    telegram.serve_file("f5", "documents/e.txt", b"\xff\xfe\xfa")
    svc.on_text(parse_update(document_update(5, "f5", "e.txt")))
    svc.process_next(NOW)
    assert sent(telegram)[-1] == conv.NOT_UTF8_REPLY

    telegram.serve_file("f6", "documents/f.txt", b"\xef\xbb\xbfhello")
    svc.on_text(parse_update(document_update(6, "f6", "f.txt")))
    svc.process_next(NOW)
    assert translator.calls[-1]["parts"] == ["hello"]  # BOM stripped, no caption part

    svc.on_text(parse_update(document_update(7, "unknown", "g.txt")))
    svc.process_next(NOW)
    assert sent(telegram)[-1] == conv.DOWNLOAD_FAILED_REPLY
    telegram.serve_file("f8", "documents/h.txt", b"  \n")
    svc.on_text(parse_update(document_update(8, "f8", "h.txt")))
    svc.process_next(NOW)
    assert sent(telegram)[-1] == conv.EMPTY_FILE_REPLY
    for payload in [c for _, c in telegram.requests]:
        assert TOKEN not in json.dumps(payload)


def test_upload_does_not_merge_with_text(svc, clock, translator, telegram):
    telegram.serve_file("f1", "documents/a.txt", b"file text")
    clock.value = NOW
    say(svc, "typed")
    svc.on_text(parse_update(document_update(1, "f1", "a.txt")))
    say(svc, "typed later")
    clock.value = NOW + 10
    while svc.process_next(NOW + 10):
        pass
    assert sorted(c["parts"] for c in translator.calls) == [["file text"], ["typed", "typed later"]]


# --- consent and masking -------------------------------------------------------------------------


def test_consent_required_once_and_text_held_in_memory(svc, clock, translator, telegram, conv_store, db):
    with sqlite3.connect(db.db_path) as conn:
        conn.execute("DELETE FROM llm_consent")
    run_request(svc, clock, f"{MARKER} wants a role")
    assert translator.calls == []
    notice = telegram.calls("sendMessage")[-1]
    assert notice["text"] == CONSENT_NOTICE
    yes, no = button_data(notice["reply_markup"])
    assert yes.startswith("cs:") and yes.endswith(":yes") and no.endswith(":no")
    assert MARKER not in dump_db(db.db_path)

    reply = svc.on_callback(callback(yes, message_id=77))
    assert reply == []
    assert conv_store.has_consent(111, CONSENT_VERSION)
    assert {"chat_id": 111, "message_id": 77} in telegram.calls("editMessageReplyMarkup")
    assert svc.process_next(clock.value) is True
    assert [c["parts"] for c in translator.calls] == [[f"{MARKER} wants a role"]]
    assert MARKER not in dump_db(db.db_path)

    shown = len(telegram.calls("sendMessage"))
    run_request(svc, clock, "second request")
    assert all(CONSENT_NOTICE != t for t in sent(telegram)[shown:])
    assert len(translator.calls) == 2

    # a stale or repeated button does nothing
    assert "no longer active" in texts(svc.on_callback(callback(yes)))

    with sqlite3.connect(db.db_path) as conn:
        conn.execute("DELETE FROM llm_consent")
    run_request(svc, clock, f"{MARKER} again")
    _, no = button_data(last_markup(telegram))
    reply = svc.on_callback(callback(no))
    assert texts(reply) == conv.CONSENT_DECLINED_REPLY and "/setup guided" in texts(reply)
    assert not conv_store.has_consent(111, CONSENT_VERSION)
    assert svc.process_next(clock.value + 100) is False
    assert len(translator.calls) == 2


def test_cancel_discards_held_text(svc, clock, translator, telegram, db):
    with sqlite3.connect(db.db_path) as conn:
        conn.execute("DELETE FROM llm_consent")
    run_request(svc, clock, "held")
    yes, _ = button_data(last_markup(telegram))
    svc.on_cancel(text_update("/cancel"))
    assert "no longer active" in texts(svc.on_callback(callback(yes)))
    assert svc.process_next(clock.value + 100) is False and translator.calls == []


def test_masking_applied_before_translator(svc, clock, translator):
    say(svc, "write to first.last@example.org or call +48 123 456 789")
    clock.value += 3
    svc.process_next(clock.value)
    assert translator.calls[0]["parts"] == ["write to [email] or call [phone]"]


# --- proposals -----------------------------------------------------------------------------------


def test_proposal_stored_and_rendered_with_buttons(svc, clock, translator, telegram, conv_store, store):
    data = propose(svc, clock, translator, telegram, add_role("<ROLE2>"))
    assert sent(telegram)[0] == conv.WORKING_REPLY
    state = conv_store.get_state(111)
    assert state.state == "awaiting_confirm" and state.context is None
    pending = conv_store.get_pending(111)
    assert pending.base_version == version(store) and pending.mode == "chat"
    assert pending.message_id == 100 + len(telegram.calls("sendMessage"))  # the proposal's last message
    assert data == f"pc:{pending.change_id}:{pending.base_version}:a"
    assert f"pc:{pending.change_id}:{pending.base_version}:c" in button_data(last_markup(telegram))
    assert "+ role &lt;ROLE2&gt;" in sent(telegram)[-1]
    assert [o["op"] for o in json.loads(pending.operations)] == ["add_role"]
    assert version(store) == pending.base_version  # nothing applied yet


def test_double_confirm_applies_once(svc, clock, translator, telegram, store):
    data = propose(svc, clock, translator, telegram, add_role("<ROLE2>"))
    before = version(store)
    first = svc.on_callback(callback(data, message_id=9))
    second = svc.on_callback(callback(data, message_id=9))
    assert version(store) == before + 1
    assert "<b>Profile v" in texts(first) and "+ role &lt;ROLE2&gt;" in texts(first)
    assert texts(second) == conv.NO_LONGER_ACTIVE_REPLY
    assert [r.name for r in store.read_snapshot().roles] == ["<ROLE>", "<ROLE2>"]
    assert store.list_versions()[0].label == "chat change (1 operations)"
    assert {"chat_id": 111, "message_id": 9} in telegram.calls("editMessageReplyMarkup")


def test_superseded_button_inactive(svc, clock, translator, telegram, conv_store, store):
    first = propose(svc, clock, translator, telegram, add_role("<ROLE2>"))
    first_message = conv_store.get_pending(111).message_id
    second = propose(svc, clock, translator, telegram, add_role("<ROLE3>"))
    assert conv.SUPERSEDED_REPLY in sent(telegram)
    assert {"chat_id": 111, "message_id": first_message} in telegram.calls("editMessageReplyMarkup")
    before = version(store)
    assert texts(svc.on_callback(callback(first))) == conv.NO_LONGER_ACTIVE_REPLY
    assert version(store) == before
    assert conv_store.get_pending(111) is not None  # the new proposal is untouched
    wrong_version = second.rsplit(":", 2)[0] + ":999:a"
    assert texts(svc.on_callback(callback(wrong_version))) == conv.NO_LONGER_ACTIVE_REPLY
    assert conv_store.get_pending(111) is not None
    svc.on_callback(callback(second))
    assert [r.name for r in store.read_snapshot().roles] == ["<ROLE>", "<ROLE3>"]


def test_version_moved_rejects_and_purges(svc, clock, translator, telegram, conv_store, store):
    data = propose(svc, clock, translator, telegram, add_role("<ROLE2>"))
    apply_changes(store, [AddRole("<OTHER>")], label="elsewhere")
    moved = version(store)
    reply = svc.on_callback(callback(data))
    assert texts(reply) == conv.CONFLICT_REPLY
    assert "profile changed" in texts(reply) and "send your request again" in texts(reply)
    assert conv_store.get_pending(111) is None
    assert version(store) == moved
    assert conv_store.get_state(111).state == "idle"


def test_expiry_notice_after_24h(svc, clock, translator, telegram, conv_store):
    data = propose(svc, clock, translator, telegram, add_role("<ROLE2>"))
    message_id = conv_store.get_pending(111).message_id
    svc.sweep(clock.value + 86399)
    assert conv_store.get_pending(111) is not None
    svc.sweep(clock.value + 86401)
    assert conv.EXPIRED_REPLY in sent(telegram)
    assert conv_store.get_pending(111) is None
    assert {"chat_id": 111, "message_id": message_id} in telegram.calls("editMessageReplyMarkup")
    assert conv_store.get_state(111).state == "idle"
    assert texts(svc.on_callback(callback(data))) == conv.NO_LONGER_ACTIVE_REPLY


def test_confirm_of_an_old_proposal_reports_expiry(svc, clock, translator, telegram, conv_store, store):
    data = propose(svc, clock, translator, telegram, add_role("<ROLE2>"))
    before = version(store)
    clock.value += 86401
    assert texts(svc.on_callback(callback(data))) == conv.EXPIRED_REPLY
    assert version(store) == before and conv_store.get_pending(111) is None


def test_cancel_button_discards_only_that_proposal(svc, clock, translator, telegram, conv_store, store):
    data = propose(svc, clock, translator, telegram, add_role("<ROLE2>"))
    before = version(store)
    reply = svc.on_callback(callback(data[:-1] + "c", message_id=12))
    assert texts(reply) == conv.CANCELLED_REPLY
    assert conv_store.get_pending(111) is None and version(store) == before
    assert conv_store.get_state(111).state == "idle"
    assert {"chat_id": 111, "message_id": 12} in telegram.calls("editMessageReplyMarkup")
    assert texts(svc.on_callback(callback(data))) == conv.NO_LONGER_ACTIVE_REPLY


def test_stored_operations_are_revalidated_at_apply(svc, clock, translator, telegram, conv_store, store):
    data = propose(svc, clock, translator, telegram, add_role("<ROLE2>"))
    pending = conv_store.get_pending(111)
    with sqlite3.connect(svc._db.db_path) as conn:
        conn.execute(
            "UPDATE pending_change SET operations = ? WHERE chat_id = 111",
            (json.dumps([{"op": "set_env", "name": "TELEGRAM_CHAT_ID"}]),),
        )
    before = version(store)
    assert texts(svc.on_callback(callback(data))) == conv.PROPOSAL_UNREADABLE_REPLY
    assert version(store) == before and conv_store.get_pending(111) is None
    assert pending.base_version == before


def test_apply_validation_error_is_reported(svc, clock, translator, telegram, conv_store, store, monkeypatch):
    data = propose(svc, clock, translator, telegram, add_role("<ROLE2>"))

    def refuse(*a, **kw):
        raise ProfileValidationError(["<one>", "two"])

    monkeypatch.setattr(conv, "apply_changes", refuse)
    reply = texts(svc.on_callback(callback(data)))
    assert reply.startswith("Not applied: &lt;one&gt;") and "two" in reply
    assert conv_store.get_pending(111) is None


def test_resume_via_chat_triggers_cycle(svc, clock, translator, telegram, control, store):
    apply_changes(store, [SetField("paused", True)], label="pause")
    data = propose(svc, clock, translator, telegram, {"op": "set_paused", "paused": False})
    assert control.trigger_calls == 0
    reply = texts(svc.on_callback(callback(data)))
    assert control.trigger_calls == 1
    assert "A cycle starts now." in reply
    assert store.read_snapshot().fields.paused is False


def test_awaiting_description_uses_onboarding_mode_and_paused_note(
    svc, clock, translator, telegram, conv_store, store
):
    apply_changes(store, [SetField("paused", True)], label="pause")
    conv_store.set_state(111, "awaiting_description", None, clock.value)
    translator.add(prop(add_role("<ROLE2>")))
    run_request(svc, clock, "describe")
    assert translator.calls[0]["mode"] == "onboarding"
    assert conv_store.get_pending(111).mode == "onboarding"
    data = [d for d in button_data(last_markup(telegram)) if d.endswith(":a")][0]
    reply = texts(svc.on_callback(callback(data)))
    assert conv.PAUSED_NOTE in reply
    assert store.list_versions()[0].label == "llm onboarding"
    assert conv_store.get_state(111).state == "idle"


# --- bare words, /ok, /cancel --------------------------------------------------------------------


def test_bare_words_routing(svc, clock, translator, telegram, conv_store, store):
    propose(svc, clock, translator, telegram, add_role("<ROLE2>"))
    before = version(store)
    # an "ok" that is 16 minutes old is ignored and the proposal is kept
    old = texts(say(svc, "ok", date=int(clock.value) - 16 * 60))
    assert old == conv.BARE_STALE_REPLY and '/ok' in old
    assert version(store) == before and conv_store.get_pending(111) is not None
    assert "<b>Profile v" in texts(say(svc, "OK!", date=int(clock.value)))
    assert version(store) == before + 1 and conv_store.get_pending(111) is None

    propose(svc, clock, translator, telegram, add_role("<ROLE3>"))
    assert texts(say(svc, "no")) == conv.CANCELLED_REPLY
    assert conv_store.get_pending(111) is None and conv_store.get_state(111).state == "idle"

    calls = len(translator.calls)
    assert texts(say(svc, "yes")) == conv.NO_PENDING_REPLY
    assert texts(say(svc, "No.")) == conv.NO_PENDING_REPLY
    clock.value += 10
    assert svc.process_next(clock.value) is False and len(translator.calls) == calls

    # as the answer to a clarifying question a bare word goes to the translator
    conv_store.set_state(
        111, "awaiting_clarification",
        {"mode": "chat", "rounds": [["earlier text", "Which one?"]], "created_at": clock.value}, clock.value,
    )
    assert say(svc, "yes") == []
    clock.value += 3
    assert svc.process_next(clock.value) is True
    assert translator.calls[-1]["parts"] == ["yes"]
    assert translator.calls[-1]["history"] == [("earlier text", "Which one?")]


@pytest.mark.parametrize("word", sorted(conv.BARE_CONFIRM))
def test_every_bare_confirm_word_applies(svc, clock, translator, telegram, store, word):
    propose(svc, clock, translator, telegram, add_role("<ROLE2>"))
    before = version(store)
    say(svc, word.upper() + "!!", date=int(clock.value))
    assert version(store) == before + 1


@pytest.mark.parametrize("word", sorted(conv.BARE_CANCEL))
def test_every_bare_cancel_word_cancels(svc, clock, translator, telegram, store, conv_store, word):
    propose(svc, clock, translator, telegram, add_role("<ROLE2>"))
    before = version(store)
    say(svc, word)
    assert version(store) == before and conv_store.get_pending(111) is None


def test_ok_and_cancel_commands(svc, clock, translator, telegram, conv_store, store):
    assert texts(svc.on_ok(text_update("/ok"))) == conv.NO_PENDING_REPLY
    assert texts(svc.on_cancel(text_update("/cancel"))) == conv.NOTHING_TO_CANCEL_REPLY

    propose(svc, clock, translator, telegram, add_role("<ROLE2>"))
    before = version(store)
    assert "<b>Profile v" in texts(svc.on_ok(text_update("/ok")))
    assert version(store) == before + 1 and conv_store.get_pending(111) is None
    assert texts(svc.on_ok(text_update("/ok"))) == conv.NO_PENDING_REPLY

    propose(svc, clock, translator, telegram, add_role("<ROLE3>"))
    message_id = conv_store.get_pending(111).message_id
    assert texts(svc.on_cancel(text_update("/cancel"))) == conv.CANCELLED_REPLY
    assert conv_store.get_pending(111) is None and version(store) == before + 1
    assert {"chat_id": 111, "message_id": message_id} in telegram.calls("editMessageReplyMarkup")

    # /cancel also clears the clarification context and the debounce buffer
    conv_store.set_state(
        111, "awaiting_clarification", {"mode": "chat", "rounds": [[MARKER, "Q?"]], "created_at": clock.value},
        clock.value,
    )
    say(svc, "buffered")
    assert texts(svc.on_cancel(text_update("/cancel"))) == conv.CANCELLED_REPLY
    state = conv_store.get_state(111)
    assert state.state == "idle" and state.context is None
    assert svc.process_next(clock.value + 100) is False


def test_clarification_rounds_capped_at_2(svc, clock, translator, telegram, conv_store):
    translator.add(Clarification("Q1?"), Clarification("Q2 <b>?"), Clarification("Q3?"))
    run_request(svc, clock, "first <request>")
    assert sent(telegram)[-1] == "Question: Q1?"
    state = conv_store.get_state(111)
    assert state.state == "awaiting_clarification"
    assert state.context["mode"] == "chat" and state.context["rounds"] == [["first <request>", "Q1?"]]
    assert "created_at" in state.context

    run_request(svc, clock, "second")
    assert translator.calls[1]["history"] == [("first <request>", "Q1?")]
    assert sent(telegram)[-1] == "Question: Q2 &lt;b&gt;?"
    assert len(conv_store.get_state(111).context["rounds"]) == 2

    run_request(svc, clock, "third")
    assert [len(c["history"]) for c in translator.calls] == [0, 1, 2]
    assert sent(telegram)[-1] == conv.NOT_UNDERSTOOD_REPLY
    state = conv_store.get_state(111)
    assert state.state == "idle" and state.context is None


def test_clarification_answer_then_proposal_purges_context(svc, clock, translator, telegram, conv_store):
    translator.add(Clarification("Which?"))
    run_request(svc, clock, f"{MARKER} text")
    translator.add(prop(add_role("<ROLE2>")))
    run_request(svc, clock, "the first one")
    state = conv_store.get_state(111)
    assert state.state == "awaiting_confirm" and state.context is None


def test_no_change_and_problems(svc, clock, translator, telegram, conv_store):
    translator.add(NoChange())
    run_request(svc, clock, "hello")
    assert sent(telegram)[-1] == conv.NO_CHANGE_REPLY
    assert conv_store.get_state(111).state == "idle"

    translator.add(TranslationProblem(["cannot <do> that", "or that"]))
    run_request(svc, clock, "x")
    assert sent(telegram)[-1].startswith("Not applied: cannot &lt;do&gt; that") and "or that" in sent(telegram)[-1]

    translator.add(ProfileValidationError(["digest above wow"]))
    run_request(svc, clock, "another one")
    assert sent(telegram)[-1] == "Not applied: digest above wow"


def test_problem_purges_clarification_context(svc, clock, translator, telegram, conv_store):
    translator.add(Clarification("Q?"), TranslationProblem("nope"))
    run_request(svc, clock, "a")
    run_request(svc, clock, "b")
    state = conv_store.get_state(111)
    assert state.state == "idle" and state.context is None


def test_cancel_during_call_discards_result(svc, clock, translator, telegram, conv_store):
    translator.gate = threading.Event()
    translator.add(prop(add_role("<ROLE2>")))
    say(svc, "request")
    clock.value += 3
    worker = threading.Thread(target=svc.process_next, args=(clock.value,), daemon=True)
    worker.start()
    assert translator.entered.wait(5)
    assert texts(svc.on_cancel(text_update("/cancel"))) == conv.CANCELLED_REPLY
    translator.gate.set()
    worker.join(5)
    assert not worker.is_alive()
    assert conv_store.get_pending(111) is None
    assert not any("Proposed change" in t for t in sent(telegram))
    assert conv_store.get_state(111).state == "idle"


# --- errors --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "error, expected",
    [
        (ChatLLMNotConfigured("x"), "Free-text requests need an LLM key (NVIDIA_API_KEY or OPENROUTER_API_KEY in .env); the commands keep working: /help"),
        (ChatLLMUnavailable("x"), "The language model is unavailable right now; try again later or use a command (/help)."),
        (ChatLLMNotUnderstood("x"), "Couldn't understand that, try a command (/help)."),
        (ChatLLMLimited(125.0), "Hourly limit of 30 LLM calls reached; try again in 3 min, or use the commands (/help)."),
        (ChatLLMLimited(1.0), "Hourly limit of 30 LLM calls reached; try again in 1 min, or use the commands (/help)."),
        (ValueError(MARKER), conv.ERROR_REPLY),
        (TypeError(MARKER), conv.ERROR_REPLY),
    ],
)
def test_llm_unavailable_and_not_configured_replies(svc, clock, translator, telegram, caplog, error, expected):
    caplog.set_level(logging.DEBUG)
    translator.add(error)
    run_request(svc, clock, "request")
    assert sent(telegram)[-1] == expected
    assert MARKER not in caplog.text  # only the exception type name is logged
    translator.add(NoChange())
    run_request(svc, clock, "again")  # the worker survived
    assert sent(telegram)[-1] == conv.NO_CHANGE_REPLY


def test_llm_error_keeps_clarification_context_unavailable_only(svc, clock, translator, conv_store):
    translator.add(Clarification("Q?"), ChatLLMUnavailable("x"))
    run_request(svc, clock, "a")
    run_request(svc, clock, "b")
    assert conv_store.get_state(111).state == "awaiting_clarification"


def test_not_configured_chat_llm_answers_at_once(tmp_path, db, store, clock, control, telegram):
    class NoKeys:
        def available(self) -> bool:
            return False

    ctx = BotContext(
        db_path=db.db_path, env_path=tmp_path / ".env", settings=Settings(_env_file=None),
        trigger_cycle=control.trigger, is_cycle_running=control.is_running, scraper=None, clock=clock,
        chat_llm=NoKeys(),
    )
    client = TelegramBotClient(TOKEN, transport=telegram.transport)
    svc = ConversationService(ctx, client, SetupWizard(BotStateStore(db), store, clock=clock), clock=clock)
    assert "need an LLM key" in texts(say(svc, "hello"))
    assert "need an LLM key" in texts(svc.on_text(parse_update(document_update(1, "f", "a.txt"))))
    assert svc.process_next(clock.value + 100) is False


def test_without_chat_llm_on_text_leaves_the_hint_to_the_router(tmp_path, db, store, clock, control, telegram):
    ctx = BotContext(
        db_path=db.db_path, env_path=tmp_path / ".env", settings=Settings(_env_file=None),
        trigger_cycle=control.trigger, is_cycle_running=control.is_running, scraper=None, clock=clock,
    )
    client = TelegramBotClient(TOKEN, transport=telegram.transport)
    svc = ConversationService(ctx, client, SetupWizard(BotStateStore(db), store, clock=clock), clock=clock)
    assert say(svc, "hello") is None
    assert svc.on_text(parse_update(document_update(1, "f", "a.txt"))) is None
    assert say(svc, "") is None


def test_wizard_text_goes_first(svc, clock, translator, store):
    svc._wizard.start(text_update("/setup"))
    reply = say(svc, "<ROLE3>")
    assert reply and "step 2" in texts(reply).lower()
    clock.value += 10
    assert svc.process_next(clock.value) is False and translator.calls == []


# --- privacy -------------------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["apply", "cancel", "supersede", "expiry", "bare_no"])
def test_raw_text_purged_after_apply_cancel_supersede_expiry(
    svc, clock, translator, telegram, conv_store, store, db, path
):
    translator.add(Clarification("Which?"))
    run_request(svc, clock, f"{MARKER} first")
    assert MARKER in json.dumps(conv_store.get_state(111).context)  # held only while the question is open
    data = propose(svc, clock, translator, telegram, add_role("<ROLE2>"), text=f"{MARKER} second")
    if path == "apply":
        svc.on_callback(callback(data))
    elif path == "cancel":
        svc.on_cancel(text_update("/cancel"))
    elif path == "bare_no":
        say(svc, "no")
    elif path == "supersede":
        propose(svc, clock, translator, telegram, add_role("<ROLE3>"), text=f"{MARKER} third")
        svc.on_cancel(text_update("/cancel"))
    else:
        svc.sweep(clock.value + 86401)
    dump = dump_db(db.db_path)
    assert MARKER not in dump
    state = conv_store.get_state(111)
    assert state.context is None and state.state == "idle"
    assert conv_store.get_pending(111) is None


def test_stale_clarification_context_is_purged_by_the_sweep(svc, clock, conv_store, db):
    conv_store.set_state(
        111, "awaiting_clarification", {"mode": "chat", "rounds": [[MARKER, "Q?"]], "created_at": clock.value},
        clock.value,
    )
    svc.sweep(clock.value + 86401)
    assert MARKER not in dump_db(db.db_path)
    assert conv_store.get_state(111).state == "idle"


def test_no_user_text_in_logs(svc, clock, translator, telegram, caplog, store):
    caplog.set_level(logging.DEBUG)
    translator.add(Clarification("<MARKER-OUT> which one?"))
    run_request(svc, clock, "<MARKER> request about first.last@example.org")
    data = propose(svc, clock, translator, telegram, add_role("<ROLE2>"), text="<MARKER> answer")
    svc.on_callback(callback(data))
    translator.add(ChatLLMUnavailable("<MARKER>"))
    run_request(svc, clock, "<MARKER> again")
    assert "<MARKER" not in caplog.text
    assert "example.org" not in caplog.text
    assert TOKEN not in caplog.text


def test_messages_are_chunked_and_escaped(svc, clock, translator, telegram):
    translator.add(Clarification("q" * 290))
    run_request(svc, clock, "x")
    from src.bot.render import utf16_len

    for payload in telegram.calls("sendMessage"):
        assert utf16_len(payload["text"]) <= 3800


# --- the real worker thread ----------------------------------------------------------------------


def test_worker_thread_flushes_after_debounce_and_stops_within_a_second(
    tmp_path, db, store, control, telegram, translator, monkeypatch, conv_store
):
    monkeypatch.setattr(conv, "DEBOUNCE_SECONDS", 0.05)
    stop = threading.Event()
    ctx = BotContext(
        db_path=db.db_path, env_path=tmp_path / ".env", settings=Settings(_env_file=None),
        trigger_cycle=control.trigger, is_cycle_running=control.is_running, scraper=None, stop_event=stop,
    )
    client = TelegramBotClient(TOKEN, transport=telegram.transport)
    service = ConversationService(
        ctx, client, SetupWizard(BotStateStore(db), store, clock=time.time), translator=translator
    )
    conv_store.record_consent(111, CONSENT_VERSION)
    thread = service.start()
    assert thread.name == "bot-conversation" and thread.daemon and thread.is_alive()
    service.on_text(text_update("hello", date=int(time.time())))
    deadline = time.monotonic() + 5
    while not translator.calls and time.monotonic() < deadline:
        time.sleep(0.02)
    assert translator.calls and translator.calls[0]["parts"] == ["hello"]
    stop.set()
    thread.join(1.0)
    assert not thread.is_alive()


def test_worker_loop_survives_an_unexpected_error(
    tmp_path, db, store, control, telegram, translator, monkeypatch, conv_store, caplog
):
    monkeypatch.setattr(conv, "DEBOUNCE_SECONDS", 0.05)
    stop = threading.Event()
    ctx = BotContext(
        db_path=db.db_path, env_path=tmp_path / ".env", settings=Settings(_env_file=None),
        trigger_cycle=control.trigger, is_cycle_running=control.is_running, scraper=None, stop_event=stop,
    )
    client = TelegramBotClient(TOKEN, transport=telegram.transport)
    service = ConversationService(
        ctx, client, SetupWizard(BotStateStore(db), store, clock=time.time), translator=translator
    )
    conv_store.record_consent(111, CONSENT_VERSION)
    boom = {"n": 0}
    original = service.process_next

    def flaky(now):
        boom["n"] += 1
        if boom["n"] == 1:
            raise RuntimeError(MARKER)
        return original(now)

    monkeypatch.setattr(service, "process_next", flaky)
    caplog.set_level(logging.DEBUG)
    thread = service.start()
    service.on_text(text_update("hello", date=int(time.time())))
    deadline = time.monotonic() + 5
    while not translator.calls and time.monotonic() < deadline:
        time.sleep(0.02)
    stop.set()
    thread.join(1.0)
    assert boom["n"] >= 2 and translator.calls and not thread.is_alive()
    assert MARKER not in caplog.text
