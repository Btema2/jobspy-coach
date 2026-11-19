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
    ChatFailure,
    ChatLLMError,
    ChatLLMLimited,
    ChatLLMNotConfigured,
    ChatLLMNotUnderstood,
    ChatLLMUnavailable,
    describe_failure,
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
    assert texts(say(svc, "part 4")) == conv.SWALLOWED_REPLY  # told once that the tail is being dropped
    clock.value = NOW + 2.5
    assert say(svc, "part 5") == []  # further parts of the same paste are dropped silently
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
    telegram.serve_file("f1", "documents/a.md", b"x" * 20000)
    caplog.set_level(logging.DEBUG)
    caplog.set_level(logging.WARNING, logger="httpx")  # as configure_logging does: httpx would print the URL
    caplog.set_level(logging.WARNING, logger="httpcore")
    clock.value = NOW
    reply = svc.on_text(parse_update(document_update(1, "f1", "Notes.MD", caption="<CAPTION>")))
    assert reply == []
    assert svc.process_next(NOW) is True  # an upload is its own request: no debounce wait
    assert translator.calls[0]["parts"] == ["<CAPTION>", "x" * 20000]
    assert "TOKEN" not in caplog.text

    telegram.serve_file("f0", "documents/z.md", b"x" * 20480)  # exactly the cap, no caption
    svc.on_text(parse_update(document_update(10, "f0", "z.md")))
    assert svc.process_next(NOW) is True
    assert translator.calls[1]["parts"] == ["x" * 20480]

    telegram.serve_file("f2", "documents/b.txt", b"y" * 20481)
    svc.on_text(parse_update(document_update(2, "f2", "b.txt")))
    assert svc.process_next(NOW) is True
    assert sent(telegram)[-1] == conv.FILE_TOO_LARGE_REPLY
    assert len(translator.calls) == 2

    reply = svc.on_text(parse_update(document_update(3, "f3", "c.txt", file_size=20481)))
    assert texts(reply) == conv.FILE_TOO_LARGE_REPLY  # known size: refused before any download
    assert len(telegram.calls("getFile")) == 3

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
        (ValueError(MARKER), conv.ERROR_REPLY),
        (TypeError(MARKER), conv.ERROR_REPLY),
    ],
)
def test_not_configured_and_unexpected_error_replies(svc, clock, translator, telegram, caplog, error, expected):
    caplog.set_level(logging.DEBUG)
    translator.add(error)
    run_request(svc, clock, "request")
    assert sent(telegram)[-1] == expected
    assert not telegram.calls("sendMessage")[-1].get("reply_markup")  # not configured: nothing to retry
    assert MARKER not in caplog.text  # only the exception type name is logged
    translator.add(NoChange())
    run_request(svc, clock, "again")  # the worker survived
    assert sent(telegram)[-1] == conv.NO_CHANGE_REPLY


def failure_error(failure: ChatFailure, retry_in=None) -> ChatLLMError:
    if failure is ChatFailure.CAPPED:
        return ChatLLMLimited(retry_in if retry_in is not None else 60.0)
    if failure in (ChatFailure.BAD_ANSWER, ChatFailure.TIMEOUT):
        return ChatLLMNotUnderstood("x", failure=failure, retry_in=retry_in)
    return ChatLLMUnavailable("x", failure=failure, retry_in=retry_in)


def retry_button(telegram: FakeTelegram) -> str:
    data = button_data(last_markup(telegram))
    assert len(data) == 1 and data[0].startswith("rt:") and data[0].endswith(":go") and len(data[0].encode()) <= 64
    return data[0]


@pytest.mark.parametrize("failure", list(ChatFailure))
@pytest.mark.parametrize("retry_in", [None, 125.0])
def test_each_failure_class_gets_its_reason_text_and_a_retry_button(svc, clock, translator, telegram, conv_store, failure, retry_in):
    translator.add(Clarification("Q?"), failure_error(failure, retry_in))
    run_request(svc, clock, "a")
    run_request(svc, clock, "b")
    expected_in = 60.0 if failure is ChatFailure.CAPPED and retry_in is None else retry_in  # ChatLLMLimited needs one
    assert sent(telegram)[-1] == describe_failure(failure, expected_in)
    retry_button(telegram)
    state = conv_store.get_state(111)  # the clarification stays open: nothing was purged
    assert state.state == "awaiting_clarification" and state.context["rounds"] == [["a", "Q?"]]


def test_daily_share_reply_says_today(svc, clock, translator, telegram):
    translator.add(ChatLLMLimited(3 * 3600.0, daily=True))
    run_request(svc, clock, "request")
    assert sent(telegram)[-1] == "Too many model requests today; try again in 3 h."
    assert "hour" not in sent(telegram)[-1]
    retry_button(telegram)


def test_hourly_cap_reply_keeps_the_fixed_head(svc, clock, translator, telegram):
    translator.add(ChatLLMLimited(125.0))
    run_request(svc, clock, "request")
    assert sent(telegram)[-1] == "Too many model requests this hour; try again in 3 min."


def test_translation_problem_has_no_retry_and_purges(svc, clock, translator, telegram, conv_store):
    translator.add(Clarification("Q?"), TranslationProblem(["Unknown country"]))
    run_request(svc, clock, "a")
    run_request(svc, clock, "b")
    assert not telegram.calls("sendMessage")[-1].get("reply_markup")
    assert conv_store.get_state(111).state == "idle" and conv_store.get_state(111).context is None
    assert svc._retry is None


def test_progress_note_text():
    assert conv.PROGRESS_REPLY == "Still working; this can take a few minutes."


def test_retry_button_reruns_the_same_masked_parts_without_resending(svc, clock, translator, telegram, conv_store):
    translator.add(Clarification("Q?"), ChatLLMUnavailable("x"))
    run_request(svc, clock, "first")
    say(svc, "write me at first.last@example.org")
    clock.value += 1
    say(svc, "second part")
    clock.value += conv.DEBOUNCE_SECONDS
    assert svc.process_next(clock.value) is True
    data = retry_button(telegram)
    assert translator.calls[1]["parts"] == ["write me at [email]", "second part"]
    translator.add(prop(add_role("<ROLE2>")))
    assert svc.on_callback(callback(data, message_id=77)) == []
    assert {"chat_id": 111, "message_id": 77} in telegram.calls("editMessageReplyMarkup")
    assert svc.process_next(clock.value) is True  # immediate: no debounce wait, nothing re-sent
    assert translator.calls[2]["parts"] == translator.calls[1]["parts"]
    assert translator.calls[2]["history"] == [("first", "Q?")]  # the clarification context was kept
    assert [d for d in button_data(last_markup(telegram)) if d.endswith(":a")]
    assert conv_store.get_state(111).state == "awaiting_confirm"
    assert svc._retry is None


def test_retry_that_fails_again_offers_a_new_button(svc, clock, translator, telegram):
    translator.add(ChatLLMUnavailable("x"), ChatLLMNotUnderstood("x", failure=ChatFailure.TIMEOUT))
    run_request(svc, clock, "request")
    first = retry_button(telegram)
    svc.on_callback(callback(first))
    assert svc.process_next(clock.value) is True
    assert sent(telegram)[-1] == describe_failure(ChatFailure.TIMEOUT)
    second = retry_button(telegram)
    assert second != first
    assert texts(svc.on_callback(callback(first))) == conv.CONSENT_STALE_REPLY


def test_stale_replaced_double_and_foreign_retry_buttons(svc, clock, translator, telegram):
    for bad in ("rt:deadbeef:go", "rt:deadbeef", "rt:deadbeef:stop", "rt::go"):
        assert texts(svc.on_callback(callback(bad))) == conv.CONSENT_STALE_REPLY
    translator.add(ChatLLMUnavailable("x"), ChatLLMUnavailable("x"))
    run_request(svc, clock, "one")
    first = retry_button(telegram)
    first_id = 100 + len(telegram.calls("sendMessage"))
    run_request(svc, clock, "two")
    second = retry_button(telegram)
    assert {"chat_id": 111, "message_id": first_id} in telegram.calls("editMessageReplyMarkup")  # old buttons go away
    assert texts(svc.on_callback(callback(first))) == conv.CONSENT_STALE_REPLY  # replaced
    assert svc.process_next(clock.value) is False
    translator.add(NoChange())
    assert svc.on_callback(callback(second)) == []
    assert texts(svc.on_callback(callback(second))) == conv.CONSENT_STALE_REPLY  # a double tap runs once
    assert svc.process_next(clock.value) is True
    assert svc.process_next(clock.value) is False
    assert [c["parts"] for c in translator.calls] == [["one"], ["two"], ["two"]]


def test_retry_from_another_chat_is_stale(svc, clock, translator, telegram):
    translator.add(ChatLLMUnavailable("x"))
    run_request(svc, clock, "request")
    data = retry_button(telegram)
    foreign = parse_update(callback_update(next(_ids), data, message_id=5, chat_id=222, user_id=222))
    assert texts(svc.on_callback(foreign)) == conv.CONSENT_STALE_REPLY
    assert svc._retry is not None


def test_cancel_drops_the_retry_holder_and_its_button(svc, clock, translator, telegram):
    translator.add(ChatLLMUnavailable("x"))
    run_request(svc, clock, "request")
    data = retry_button(telegram)
    message_id = 100 + len(telegram.calls("sendMessage"))
    assert texts(svc.on_cancel(text_update("/cancel"))) == conv.CANCELLED_REPLY  # something was in memory
    assert {"chat_id": 111, "message_id": message_id} in telegram.calls("editMessageReplyMarkup")
    assert svc._retry is None
    assert texts(svc.on_callback(callback(data))) == conv.CONSENT_STALE_REPLY


def test_pairing_change_drops_the_retry_holder(svc, clock, translator, telegram):
    translator.add(ChatLLMUnavailable("x"))
    run_request(svc, clock, "request")
    data = retry_button(telegram)
    svc.reset_memory()
    assert svc._retry is None
    assert texts(svc.on_callback(callback(data))) == conv.CONSENT_STALE_REPLY


def test_retry_holder_expires_after_24_hours(svc, clock, translator, telegram):
    translator.add(ChatLLMUnavailable("x"))
    run_request(svc, clock, "request")
    data = retry_button(telegram)
    svc.sweep(clock.value + 86399)
    assert svc._retry is not None
    svc.sweep(clock.value + 86401)
    assert svc._retry is None
    assert texts(svc.on_callback(callback(data))) == conv.CONSENT_STALE_REPLY


def test_expired_retry_button_is_stale_even_before_the_sweep(svc, clock, translator, telegram):
    translator.add(ChatLLMUnavailable("x"))
    run_request(svc, clock, "request")
    data = retry_button(telegram)
    clock.value += 86401
    assert texts(svc.on_callback(callback(data))) == conv.CONSENT_STALE_REPLY


def test_retry_button_after_a_restart_is_stale(svc, clock, translator, telegram, tmp_path, db, store, control):
    translator.add(ChatLLMUnavailable("x"))
    run_request(svc, clock, "request")
    data = retry_button(telegram)
    ctx = BotContext(
        db_path=db.db_path, env_path=tmp_path / ".env", settings=Settings(_env_file=None),
        trigger_cycle=control.trigger, is_cycle_running=control.is_running, scraper=None, clock=clock,
    )
    client = TelegramBotClient(TOKEN, transport=telegram.transport)
    fresh = ConversationService(ctx, client, SetupWizard(BotStateStore(db), store, clock=clock), translator=translator, clock=clock)
    assert texts(fresh.on_callback(callback(data))) == conv.CONSENT_STALE_REPLY


def test_cancel_during_the_failing_call_leaves_no_retry_button(svc, clock, translator, telegram):
    translator.gate = threading.Event()
    translator.add(ChatLLMUnavailable("x"))
    say(svc, "request")
    clock.value += conv.DEBOUNCE_SECONDS
    worker = threading.Thread(target=svc.process_next, args=(clock.value,))
    worker.start()
    assert translator.entered.wait(5)
    svc.on_cancel(text_update("/cancel"))
    translator.gate.set()
    worker.join(5)
    assert svc._retry is None
    assert describe_failure(ChatFailure.UNAVAILABLE) not in sent(telegram)


def test_no_prompt_text_in_logs_on_failure_and_retry(svc, clock, translator, telegram, caplog):
    caplog.set_level(logging.DEBUG)
    caplog.set_level(logging.WARNING, logger="httpx")
    caplog.set_level(logging.WARNING, logger="httpcore")
    translator.add(ChatLLMUnavailable(MARKER), ChatLLMNotUnderstood(MARKER))
    run_request(svc, clock, f"{MARKER} request first.last@example.org")
    svc.on_callback(callback(retry_button(telegram)))
    svc.process_next(clock.value)
    assert "<MARKER" not in caplog.text and "example.org" not in caplog.text


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
    caplog.set_level(logging.WARNING, logger="httpx")  # as configure_logging does: httpx would print the URL
    caplog.set_level(logging.WARNING, logger="httpcore")
    translator.add(Clarification("<MARKER-OUT> which one?"))
    run_request(svc, clock, "<MARKER> request about first.last@example.org")
    data = propose(svc, clock, translator, telegram, add_role("<ROLE2>"), text="<MARKER> answer")
    svc.on_callback(callback(data))
    translator.add(ChatLLMUnavailable("<MARKER>"))
    run_request(svc, clock, "<MARKER> again")
    assert "<MARKER" not in caplog.text
    assert "example.org" not in caplog.text
    assert "TOKEN" not in caplog.text


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


# --- review fixes ----------------------------------------------------------------------------------

HOUR = 3600.0


def test_context_expires_24h_after_it_was_first_stored(svc, clock, translator, telegram, conv_store, db):
    translator.add(Clarification("Q1?"), Clarification("Q2?"), NoChange())
    run_request(svc, clock, f"{MARKER} one")
    created = conv_store.get_state(111).context["created_at"]
    clock.value = created + 23 * HOUR
    run_request(svc, clock, "two")  # the second round refreshes updated_at but not created_at
    assert conv_store.get_state(111).context["created_at"] == created
    assert MARKER in dump_db(db.db_path)
    svc.sweep(created + 24 * HOUR + 1)
    assert MARKER not in dump_db(db.db_path)
    state = conv_store.get_state(111)
    assert state.state == "idle" and state.context is None


def test_expired_context_is_not_sent_again(svc, clock, translator, conv_store):
    translator.add(Clarification("Q1?"), Clarification("Q2?"), NoChange())
    run_request(svc, clock, f"{MARKER} one")
    created = conv_store.get_state(111).context["created_at"]
    clock.value = created + 23 * HOUR
    run_request(svc, clock, "two")
    clock.value = created + 24 * HOUR + 1
    run_request(svc, clock, "three")
    assert translator.calls[2]["history"] == []
    assert all(MARKER not in part for part in translator.calls[2]["parts"])


def test_cancel_between_store_and_send_sends_no_proposal(svc, clock, translator, telegram, conv_store, monkeypatch):
    original = svc._wconv.put_pending

    def put_then_cancel(pending):
        result = original(pending)
        svc.on_cancel(text_update("/cancel"))  # lands after the row is stored, before the message is sent
        return result

    monkeypatch.setattr(svc._wconv, "put_pending", put_then_cancel)
    translator.add(prop(add_role("<ROLE2>")))
    run_request(svc, clock, "request")
    assert not any("Proposed change" in t for t in sent(telegram))
    assert conv_store.get_pending(111) is None
    assert conv_store.get_state(111).state == "idle"


def test_confirm_keeps_state_when_a_newer_proposal_exists(svc, clock, translator, telegram, conv_store, monkeypatch):
    from src.bot.conversation_store import PendingChange

    data = propose(svc, clock, translator, telegram, add_role("<ROLE2>"))
    original = svc._conv.take_pending

    def take_then_new_proposal(chat_id, change_id):
        taken = original(chat_id, change_id)
        conv_store.put_pending(PendingChange(111, "newer000", taken.base_version, "chat", taken.operations, None, clock.value))
        conv_store.set_state(111, "awaiting_confirm", None, clock.value)
        return taken

    monkeypatch.setattr(svc._conv, "take_pending", take_then_new_proposal)
    svc.on_callback(callback(data))
    assert conv_store.get_pending(111).change_id == "newer000"
    assert conv_store.get_state(111).state == "awaiting_confirm"


def test_replaced_or_cancelled_consent_notice_loses_its_buttons(svc, clock, telegram, db):
    with sqlite3.connect(db.db_path) as conn:
        conn.execute("DELETE FROM llm_consent")
    run_request(svc, clock, "first")
    first_id = 100 + len(telegram.calls("sendMessage"))
    run_request(svc, clock, "second")
    assert {"chat_id": 111, "message_id": first_id} in telegram.calls("editMessageReplyMarkup")
    second_id = 100 + len(telegram.calls("sendMessage"))
    svc.on_cancel(text_update("/cancel"))
    assert {"chat_id": 111, "message_id": second_id} in telegram.calls("editMessageReplyMarkup")


def test_caption_plus_file_over_the_input_cap_is_refused(svc, clock, translator, telegram):
    telegram.serve_file("f1", "documents/a.txt", b"x" * 20480)
    svc.on_text(parse_update(document_update(1, "f1", "a.txt", caption="<CAPTION>")))
    svc.process_next(clock.value)
    assert sent(telegram)[-1] == conv.TOO_LONG_REPLY
    assert translator.calls == []


def test_upload_during_wizard_step_is_not_fed_to_the_wizard(svc, clock, translator, telegram):
    svc._wizard.start(text_update("/setup"))
    reply = svc.on_text(parse_update(document_update(1, "f1", "a.txt", caption="<CAPTION>")))
    assert texts(reply) == conv.FINISH_SETUP_REPLY
    assert svc._wizard._state.get_wizard()["role"] == ""  # the caption did not become the role
    assert svc.process_next(clock.value + 100) is False


def test_reset_memory_drops_buffers_held_text_and_running_result(svc, clock, translator, telegram, db):
    with sqlite3.connect(db.db_path) as conn:
        conn.execute("DELETE FROM llm_consent")
    run_request(svc, clock, "held text")
    say(svc, "buffered")
    svc.reset_memory()
    assert svc.process_next(clock.value + 100) is False
    yes, _ = button_data(last_markup(telegram))
    assert "no longer active" in texts(svc.on_callback(callback(yes)))

    translator.gate = threading.Event()
    translator.add(prop(add_role("<ROLE2>")))
    with sqlite3.connect(db.db_path) as conn:
        conn.execute(
            "INSERT INTO llm_consent (chat_id, notice_version, acknowledged_at) VALUES (111, ?, 'x')",
            (CONSENT_VERSION,),
        )
    say(svc, "request")
    clock.value += 3
    worker = threading.Thread(target=svc.process_next, args=(clock.value,), daemon=True)
    worker.start()
    assert translator.entered.wait(5)
    svc.reset_memory()
    translator.gate.set()
    worker.join(5)
    assert not any("Proposed change" in t for t in sent(telegram))


def _ok_reply(message_id: int) -> dict:
    return {"ok": True, "result": {"message_id": message_id}}


def test_undelivered_proposal_is_not_kept_for_a_bare_ok(svc, clock, translator, telegram, conv_store, store):
    telegram.respond("sendMessage", 200, _ok_reply(901))  # "Working on it..."
    telegram.respond("sendMessage", 500, {"ok": False, "error_code": 500, "description": "boom"})  # the proposal
    translator.add(prop(add_role("<ROLE2>")))
    before = version(store)
    run_request(svc, clock, "request")
    assert conv_store.get_pending(111) is None
    assert conv.DELIVERY_FAILED_REPLY in sent(telegram) and sent(telegram).count(conv.DELIVERY_FAILED_REPLY) == 1
    assert conv_store.get_state(111).state == "idle"
    assert texts(say(svc, "ok")) == conv.NO_PENDING_REPLY
    assert texts(svc.on_ok(text_update("/ok"))) == conv.NO_PENDING_REPLY
    assert version(store) == before


def test_proposal_with_a_lost_middle_chunk_is_discarded(svc, clock, translator, telegram, conv_store, store):
    persona = "\n".join(f"line {i} " + "x" * 60 for i in range(110))
    telegram.respond("sendMessage", 200, _ok_reply(901))  # "Working on it..."
    telegram.respond("sendMessage", 200, _ok_reply(902))  # first chunk
    telegram.respond("sendMessage", 500, {"ok": False, "error_code": 500, "description": "boom"})  # second chunk lost
    translator.add(prop({"op": "set_text", "field": "persona", "value": persona}))
    before = version(store)
    run_request(svc, clock, "request")
    assert conv_store.get_pending(111) is None
    assert conv.DELIVERY_FAILED_REPLY in sent(telegram)
    assert telegram.calls("editMessageReplyMarkup")  # the buttons of the last chunk were removed
    assert texts(say(svc, "yes")) == conv.NO_PENDING_REPLY
    assert version(store) == before


# --- the interview entry (update mode, routing, restart, purge) ----------------------------------------


def iv_nonce(telegram: FakeTelegram) -> str:
    for call in reversed(telegram.calls("sendMessage")):
        for row in (call.get("reply_markup") or {}).get("inline_keyboard", []):
            for button in row:
                if button["callback_data"].startswith("iv:"):
                    return button["callback_data"].split(":")[1]
    raise AssertionError("no interview buttons sent")


def start_interview(svc, clock, telegram) -> str:
    assert svc.start_setup(text_update("/setup")) == []
    assert svc.process_next(clock.value) is True
    return iv_nonce(telegram)


def test_interview_in_update_mode_when_profile_has_roles(svc, clock, telegram, conv_store):
    start_interview(svc, clock, telegram)
    state = conv_store.get_state(111)
    assert state.state == conv.STATE_INTERVIEW and state.context["mode"] == "update"


def test_interview_mode_follows_starter_rule(svc, clock, telegram, conv_store, store):
    # no enabled role and no market: onboarding; a disabled role alone still counts as starter
    snapshot = store.read_snapshot()
    from src.profile.ops import RemoveMarket, SetEnabled
    apply_changes(
        store,
        [SetEnabled("role", snapshot.roles[0].name, False), RemoveMarket(snapshot.markets[0].name)],
        label="empty",
    )
    start_interview(svc, clock, telegram)
    assert conv_store.get_state(111).context["mode"] == "onboarding"


def test_interview_text_is_tagged_and_never_reaches_translator(svc, clock, telegram, translator, conv_store):
    nonce = start_interview(svc, clock, telegram)
    assert say(svc, "<ANSWER>") == []
    buffer = svc._buffers[-1]
    assert (buffer.interview, buffer.nonce, buffer.slot, buffer.closed, buffer.immediate) == (
        True, nonce, "roles", True, True
    )
    assert svc.process_next(clock.value) is True
    assert translator.calls == []


def test_bare_words_in_interview_do_not_touch_a_proposal(svc, clock, telegram, translator, conv_store, store):
    start_interview(svc, clock, telegram)
    conv_store.put_pending(
        conv.PendingChange(111, "abcd0001", version(store), "chat", '[{"op": "set_paused", "paused": true}]', None,
                           clock.value)
    )
    before = version(store)
    for word in ("yes", "no", "OK!", "cancel"):
        assert say(svc, word, date=int(clock.value)) == []
    while svc.process_next(clock.value):
        pass
    assert version(store) == before and conv_store.get_pending(111) is not None
    assert translator.calls == []
    assert conv_store.get_state(111).state == conv.STATE_INTERVIEW


def test_after_cancel_free_text_reaches_translator_again(svc, clock, telegram, translator, conv_store):
    start_interview(svc, clock, telegram)
    assert texts(svc.on_cancel(text_update("/cancel"))) == conv.CANCELLED_REPLY
    assert conv_store.get_state(111).state == "idle" and 111 not in svc._interview
    run_request(svc, clock, "add a role")
    assert translator.calls and translator.calls[0]["mode"] == "chat"


def test_restart_keeps_row_and_rebuilds_mirror(svc, db, store, clock, telegram, translator, conv_store, tmp_path):
    nonce = start_interview(svc, clock, telegram)
    ctx = BotContext(
        db_path=db.db_path, env_path=tmp_path / ".env", settings=Settings(_env_file=None),
        trigger_cycle=lambda: TRIGGER_STARTED, is_cycle_running=lambda: False, scraper=None, clock=clock,
    )
    client = TelegramBotClient(TOKEN, transport=telegram.transport)
    fresh = ConversationService(ctx, client, SetupWizard(BotStateStore(db), store, clock=clock),
                                translator=translator, clock=clock)
    assert conv_store.get_state(111).state == conv.STATE_INTERVIEW
    assert fresh._interview == {}
    assert fresh.on_text(text_update("<ANSWER>")) == []
    buffer = fresh._buffers[-1]
    assert (buffer.interview, buffer.nonce, buffer.slot) == (True, nonce, None)
    assert fresh.process_next(clock.value) is True
    assert fresh._interview[111] == (nonce, "roles")
    assert translator.calls == []
    assert texts(fresh.on_callback(callback("iv:deadbeef:draft"))) == conv.INTERVIEW_STALE_REPLY


def _interview_row(conv_store, clock, *, created_at, updated_at):
    state = conv_store.get_state(111)
    context = dict(state.context)
    context["created_at"] = created_at
    conv_store.set_state(111, conv.STATE_INTERVIEW, context, updated_at)


def test_purge_counts_interview_from_created_at(svc, db, clock, telegram, conv_store):
    start_interview(svc, clock, telegram)
    t0 = clock.value
    clock.value = t0 + conv.CONTEXT_TTL_SECONDS + 1  # free text after the purge
    # the first answer was stored at t0; the row was rewritten an hour before the sweep
    _interview_row(conv_store, clock, created_at=t0, updated_at=t0 + conv.CONTEXT_TTL_SECONDS - 3600)
    svc.sweep(t0 + conv.CONTEXT_TTL_SECONDS - 60)
    assert conv_store.get_state(111).state == conv.STATE_INTERVIEW
    svc.sweep(t0 + conv.CONTEXT_TTL_SECONDS + 1)
    state = conv_store.get_state(111)
    assert state.state == "idle" and state.context is None
    assert 111 not in svc._interview
    _free_text_reaches_translator(svc, clock)


def _free_text_reaches_translator(svc, clock) -> None:
    """After the purge the chat is out of the interview: free text is a normal request again."""
    assert svc.on_text(text_update("add a role")) == []
    buffer = svc._buffers[-1]
    assert buffer.interview is False and buffer.closed is False
    clock.value += conv.DEBOUNCE_SECONDS
    calls = len(svc._translator.calls)
    assert svc.process_next(clock.value) is True
    assert len(svc._translator.calls) == calls + 1 and svc._translator.calls[-1]["mode"] == "chat"


def test_purge_counts_from_updated_at_while_created_at_is_null(svc, db, clock, telegram, conv_store):
    start_interview(svc, clock, telegram)
    t0 = clock.value
    _interview_row(conv_store, clock, created_at=None, updated_at=t0)
    svc.sweep(t0 + conv.CONTEXT_TTL_SECONDS - 60)
    assert conv_store.get_state(111).state == conv.STATE_INTERVIEW
    assert 111 in svc._interview
    clock.value = t0 + conv.CONTEXT_TTL_SECONDS + 1
    svc.sweep(clock.value)  # the SQL purge (updated_at rule) idles the row first
    assert conv_store.get_state(111).state == "idle"
    assert 111 not in svc._interview
    _free_text_reaches_translator(svc, clock)


def test_sweep_drops_mirror_entry_whose_row_has_another_nonce(svc, clock, telegram, conv_store):
    start_interview(svc, clock, telegram)
    state = conv_store.get_state(111)
    context = dict(state.context, nonce="0000ffff")
    conv_store.set_state(111, conv.STATE_INTERVIEW, context, clock.value)
    svc.sweep(clock.value)
    assert 111 not in svc._interview
    assert conv_store.get_state(111).state == conv.STATE_INTERVIEW  # the row itself is not touched


def test_onboarding_proposal_apply_labels_and_paused_note(svc, clock, conv_store, store, telegram):
    """A proposal stored with mode ``onboarding`` (the interview draft) is applied as ``llm onboarding``;
    a paused profile gets the note to /resume."""
    apply_changes(store, [SetField("paused", True)], label="pause")
    operations = json.dumps([add_role("<ROLE2>")])
    conv_store.put_pending(
        conv.PendingChange(111, "abcd0002", version(store), "onboarding", operations, None, clock.value)
    )
    conv_store.set_state(111, "awaiting_confirm", None, clock.value)
    before = version(store)
    reply = texts(svc.on_callback(callback(f"pc:abcd0002:{before}:a")))
    assert version(store) == before + 1
    assert store.list_versions()[0].label == "llm onboarding"
    assert conv.PAUSED_NOTE in reply
    assert [r.name for r in store.read_snapshot().roles if r.name == "<ROLE2>"] == ["<ROLE2>"]
    assert conv_store.get_pending(111) is None and conv_store.get_state(111).state == "idle"
