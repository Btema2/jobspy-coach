"""``/setup`` with a language model: consent first, the interview entry and its buttons, wizard fallbacks.

No network (``FakeTelegram``, scripted fake translator), no real .env, placeholders only.
"""
import sqlite3
import threading
import time
from pathlib import Path

import pytest

import src.bot.conversation as conv
from src.bot.client import TelegramBotClient
from src.bot.commands import TRIGGER_STARTED, BotContext, CommandRouter
from src.bot.conversation import ConversationService
from src.bot.conversation_store import ConversationStore
from src.bot.interview import QUESTIONS
from src.bot.privacy import CONSENT_NOTICE, CONSENT_VERSION
from src.bot.state import BotStateStore
from src.bot.updates import parse_update
from src.bot.wizard import SetupWizard
from src.config import Settings
from src.profile.ops import apply_changes, create_starter_profile, SetField
from src.profile.store import ProfileStore
from src.storage.db import Database
from tests.fake_telegram import FakeTelegram, callback_update, message_update
from tests.test_conversation import (
    Clock,
    FakeTranslator,
    add_role,
    button_data,
    callback,
    last_markup,
    prop,
    run_request,
    sent,
    text_update,
    texts,
    version,
)

TOKEN = "<TOKEN>"
NO_LLM_LINE = "No LLM is configured; using step-by-step questions."


class FakeChatLLM:
    def __init__(self, available: bool = True):
        self._available = available

    def available(self) -> bool:
        return self._available


@pytest.fixture
def db(tmp_path: Path) -> Database:
    database = Database(tmp_path / "onboarding.db")
    database.init_db()
    return database


@pytest.fixture
def store(db) -> ProfileStore:
    profile_store = ProfileStore(db)
    create_starter_profile(profile_store, wow_threshold=8.5, digest_threshold=6.5, output_language="en")
    return profile_store  # paused, no roles, no markets


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def telegram() -> FakeTelegram:
    return FakeTelegram()


@pytest.fixture
def translator() -> FakeTranslator:
    return FakeTranslator()


@pytest.fixture
def conv_store(db) -> ConversationStore:
    return ConversationStore(db)


def _ctx(tmp_path, db, clock, chat_llm):
    return BotContext(
        db_path=db.db_path, env_path=tmp_path / ".env", settings=Settings(_env_file=None),
        trigger_cycle=lambda: TRIGGER_STARTED, is_cycle_running=lambda: False, scraper=None, clock=clock,
        chat_llm=chat_llm,
    )


@pytest.fixture
def svc(tmp_path, db, store, clock, telegram, translator, conv_store) -> ConversationService:
    client = TelegramBotClient(TOKEN, transport=telegram.transport)
    wizard = SetupWizard(BotStateStore(db), store, clock=clock)
    conv_store.record_consent(111, CONSENT_VERSION)
    return ConversationService(
        _ctx(tmp_path, db, clock, FakeChatLLM()), client, wizard, translator=translator, clock=clock
    )


def setup(svc: ConversationService, args: str = ""):
    return svc.start_setup(parse_update(message_update(5000, ("/setup " + args).strip())))


def iv_nonce(telegram: FakeTelegram) -> str:
    """The nonce of the newest interview buttons."""
    for call in reversed(telegram.calls("sendMessage")):
        for row in (call.get("reply_markup") or {}).get("inline_keyboard", []):
            for button in row:
                if button["callback_data"].startswith("iv:"):
                    return button["callback_data"].split(":")[1]
    raise AssertionError("no interview buttons sent")


def wizard_open(db) -> bool:
    return BotStateStore(db).get_wizard() is not None


# --- /setup routing ---------------------------------------------------------------------------


def test_setup_without_llm_runs_wizard(tmp_path, db, store, clock):
    client = TelegramBotClient(TOKEN, transport=FakeTelegram().transport)
    router = CommandRouter(_ctx(tmp_path, db, clock, None), client=client)
    reply = router.handle(parse_update(message_update(1, "/setup")))
    text = "\n".join(m.text for m in reply)
    assert NO_LLM_LINE in text
    assert "Setup, step 1 of 4: send the name of the role you are looking for." in text
    assert wizard_open(db)
    assert ConversationStore(db).get_state(111).state == "idle"


def test_setup_with_unavailable_llm_runs_wizard_with_line(tmp_path, db, store, clock):
    client = TelegramBotClient(TOKEN, transport=FakeTelegram().transport)
    router = CommandRouter(_ctx(tmp_path, db, clock, FakeChatLLM(available=False)), client=client)
    text = "\n".join(m.text for m in router.handle(parse_update(message_update(1, "/setup"))))
    assert NO_LLM_LINE in text and "step 1 of 4" in text and wizard_open(db)


def test_setup_guided_runs_wizard(svc, db, conv_store):
    reply = setup(svc, "guided")
    assert NO_LLM_LINE not in texts(reply)  # the LLM is there; the user chose the wizard
    assert "step 1 of 4" in texts(reply)
    assert wizard_open(db)
    assert conv_store.get_state(111).state == "idle"


def test_setup_with_consent_starts_interview(svc, db, conv_store, clock, telegram, translator):
    BotStateStore(db).set_wizard({"nonce": "old", "step": "role", "updated_at": clock.value})
    assert setup(svc) == []  # the worker posts the intro
    assert not wizard_open(db)
    assert svc.process_next(clock.value) is True
    calls = telegram.calls("sendMessage")
    assert [c["text"] for c in calls] == [conv.INTERVIEW_INTRO, QUESTIONS["roles"]]
    nonce = iv_nonce(telegram)
    markup = calls[0]["reply_markup"]
    assert [b["text"] for row in markup["inline_keyboard"] for b in row] == [
        "Draft now", "Step-by-step questions instead", "Cancel"
    ]
    assert button_data(markup) == [f"iv:{nonce}:draft", f"iv:{nonce}:guided", f"iv:{nonce}:cancel"]
    state = conv_store.get_state(111)
    assert state.state == conv.STATE_INTERVIEW
    assert state.context["nonce"] == nonce and state.context["created_at"] is None
    assert state.context["mode"] == "onboarding"  # starter profile: no enabled role, no market
    assert state.context["current"] == "roles"
    assert svc._interview[111] == (nonce, "roles")
    assert translator.calls == []


def test_intro_names_buttons_and_retention():
    for words in ("Draft now", "Step-by-step questions instead", "Cancel", "24 hours", "confirm"):
        assert words in conv.INTERVIEW_INTRO


def test_setup_routes_through_the_router_command(tmp_path, db, store, clock):
    telegram = FakeTelegram()
    client = TelegramBotClient(TOKEN, transport=telegram.transport)
    ConversationStore(db).record_consent(111, CONSENT_VERSION)
    router = CommandRouter(_ctx(tmp_path, db, clock, FakeChatLLM()), client=client)
    prefixes = [prefix for prefix, _ in router._callbacks]
    assert "iv:" in prefixes and "rt:" in prefixes and "ob:" not in prefixes
    assert router.handle(parse_update(message_update(1, "/setup"))) == []
    assert router.conversation.process_next(clock.value) is True
    nonce = iv_nonce(telegram)
    reply = router.handle(parse_update(callback_update(2, f"iv:{nonce}:guided")))
    assert "step 1 of 4" in "\n".join(m.text for m in reply)
    reply = router.handle(parse_update(callback_update(3, "ob:guided")))
    assert "no longer active" in "\n".join(m.text for m in reply)


# --- consent first ----------------------------------------------------------------------------------


def _no_consent(db) -> None:
    with sqlite3.connect(db.db_path) as conn:
        conn.execute("DELETE FROM llm_consent")


def conversation_rows(db) -> int:
    with sqlite3.connect(db.db_path) as conn:
        return conn.execute("SELECT COUNT(*) FROM conversation").fetchone()[0]


def test_setup_without_consent_shows_notice_and_stores_nothing(svc, db, clock, telegram, translator):
    _no_consent(db)
    assert setup(svc) == []
    notice = telegram.calls("sendMessage")[-1]
    assert notice["text"] == CONSENT_NOTICE
    assert sorted(d.rsplit(":", 1)[1] for d in button_data(notice["reply_markup"])) == ["no", "yes"]
    assert conversation_rows(db) == 0
    assert svc.process_next(clock.value) is False  # nothing queued until the answer
    assert conversation_rows(db) == 0 and translator.calls == []
    assert svc._held is not None and svc._held.setup is True and svc._held.parts == []


def test_consent_v1_is_asked_again(svc, db, telegram, conv_store):
    _no_consent(db)
    conv_store.record_consent(111, 1)
    setup(svc)
    assert telegram.calls("sendMessage")[-1]["text"] == CONSENT_NOTICE
    assert conversation_rows(db) == 0


def test_setup_consent_decline_starts_wizard(svc, db, clock, translator, telegram, conv_store):
    _no_consent(db)
    setup(svc)
    notice = telegram.calls("sendMessage")[-1]
    no = [d for d in button_data(notice["reply_markup"]) if d.endswith(":no")][0]
    reply = svc.on_callback(callback(no, message_id=77))
    text = texts(reply)
    assert conv.CONSENT_DECLINED_SETUP_REPLY in text and "step 1 of 4" in text
    assert wizard_open(db)
    assert conversation_rows(db) == 0
    assert translator.calls == [] and not conv_store.has_consent(111, CONSENT_VERSION)


def test_setup_consent_yes_starts_interview(svc, db, clock, telegram, conv_store):
    _no_consent(db)
    setup(svc)
    notice = telegram.calls("sendMessage")[-1]
    yes = [d for d in button_data(notice["reply_markup"]) if d.endswith(":yes")][0]
    assert svc.on_callback(callback(yes, message_id=77)) == []
    assert conv_store.has_consent(111, CONSENT_VERSION)
    assert svc.process_next(clock.value) is True
    assert sent(telegram)[-2:] == [conv.INTERVIEW_INTRO, QUESTIONS["roles"]]
    state = conv_store.get_state(111)
    assert state.state == conv.STATE_INTERVIEW and state.context["created_at"] is None
    assert state.context["mode"] == "onboarding"


def test_chat_consent_decline_keeps_plain_reply(svc, db, clock, telegram):
    _no_consent(db)
    run_request(svc, clock, "add <ROLE>")
    no = [d for d in button_data(telegram.calls("sendMessage")[-1]["reply_markup"]) if d.endswith(":no")][0]
    assert texts(svc.on_callback(callback(no))) == conv.CONSENT_DECLINED_REPLY
    assert not wizard_open(db)


# --- interview buttons -------------------------------------------------------------------------------


def start_interview(svc, clock, telegram) -> str:
    setup(svc)
    assert svc.process_next(clock.value) is True
    return iv_nonce(telegram)


def test_guided_button_switches_to_wizard(svc, db, clock, conv_store, telegram):
    nonce = start_interview(svc, clock, telegram)
    reply = svc.on_callback(callback(f"iv:{nonce}:guided", message_id=42))
    assert "step 1 of 4" in texts(reply) and wizard_open(db)
    state = conv_store.get_state(111)
    assert state.state == "idle" and state.context is None
    assert 111 not in svc._interview
    assert {"chat_id": 111, "message_id": 42} in telegram.calls("editMessageReplyMarkup")
    # pressed again: the interview is over
    assert texts(svc.on_callback(callback(f"iv:{nonce}:guided"))) == conv.INTERVIEW_STALE_REPLY


def test_interview_button_stale_nonce(svc, db, clock, telegram, conv_store):
    assert texts(svc.on_callback(callback("iv:deadbeef:guided"))) == conv.INTERVIEW_STALE_REPLY
    nonce = start_interview(svc, clock, telegram)
    for action in ("draft", "cancel", "guided", "retry"):
        assert texts(svc.on_callback(callback(f"iv:deadbeef:{action}"))) == conv.INTERVIEW_STALE_REPLY
    assert texts(svc.on_callback(callback(f"iv:{nonce}:bogus"))) == conv.INTERVIEW_STALE_REPLY
    assert conv_store.get_state(111).state == conv.STATE_INTERVIEW and not wizard_open(db)


def test_interview_cancel_button_purges_row(svc, db, clock, telegram, conv_store, translator):
    nonce = start_interview(svc, clock, telegram)
    reply = svc.on_callback(callback(f"iv:{nonce}:cancel", message_id=42))
    assert texts(reply) == conv.CANCELLED_REPLY
    state = conv_store.get_state(111)
    assert state.state == "idle" and state.context is None
    assert 111 not in svc._interview
    assert texts(svc.on_callback(callback(f"iv:{nonce}:cancel"))) == conv.INTERVIEW_STALE_REPLY


@pytest.mark.parametrize("action", ["draft", "retry"])
def test_draft_and_retry_buttons_queue_a_worker_action(svc, clock, telegram, translator, conv_store, action):
    nonce = start_interview(svc, clock, telegram)
    assert svc.on_callback(callback(f"iv:{nonce}:{action}")) == []
    buffer = svc._buffers[-1]
    assert (buffer.action, buffer.interview, buffer.nonce, buffer.immediate) == (action, True, nonce, True)
    assert svc.process_next(clock.value) is True  # handled in Tasks 6b and 7; nothing else happens yet
    assert translator.calls == [] and conv_store.get_state(111).state == conv.STATE_INTERVIEW


def test_cancel_while_begin_in_flight_drops_the_result(svc, clock, telegram, conv_store, monkeypatch):
    """A /cancel that lands while the worker starts the interview: no row, no intro, no question."""
    entered, release = threading.Event(), threading.Event()
    real_new_state = conv.new_state

    def slow_new_state(nonce, mode):
        entered.set()
        assert release.wait(5)
        return real_new_state(nonce, mode)

    monkeypatch.setattr(conv, "new_state", slow_new_state)
    setup(svc)
    worker = threading.Thread(target=svc.process_next, args=(clock.value,))
    worker.start()
    assert entered.wait(5)
    svc.on_cancel(text_update("/cancel"))
    release.set()
    worker.join(5)
    assert conv.INTERVIEW_INTRO not in sent(telegram)
    state = conv_store.get_state(111)
    assert state.state == "idle" and state.context is None
    assert 111 not in svc._interview


def test_wizard_start_cancels_pending_proposal_with_notice(svc, clock, translator, telegram, conv_store, store):
    translator.add(prop(add_role("<ROLE2>")))
    run_request(svc, clock, "add a role")
    assert conv_store.get_pending(111) is not None
    before = version(store)
    reply = setup(svc, "guided")
    assert texts(reply).startswith(conv.PROPOSAL_DISCARDED_REPLY)
    assert "step 1 of 4" in texts(reply)
    assert conv_store.get_pending(111) is None and version(store) == before
    assert telegram.calls("editMessageReplyMarkup")  # the old buttons are gone


def test_interview_discards_pending_proposal_and_clarification(svc, clock, translator, conv_store):
    translator.add(prop(add_role("<ROLE2>")))
    run_request(svc, clock, "add a role")
    conv_store.set_state(111, "awaiting_clarification", {"mode": "chat", "rounds": [], "created_at": clock.value})
    reply = setup(svc)
    assert texts(reply).split("\n") == [conv.PROPOSAL_DISCARDED_REPLY, conv.CLARIFICATION_DISCARDED_REPLY]
    assert conv.CLARIFICATION_DISCARDED_REPLY == "Your open question was dropped."
    assert conv_store.get_pending(111) is None
    assert svc.process_next(clock.value) is True
    assert conv_store.get_state(111).state == conv.STATE_INTERVIEW


def test_setup_restarts_a_running_interview(svc, clock, telegram, conv_store):
    first = start_interview(svc, clock, telegram)
    assert setup(svc) == []  # no notice: an interview is neither a proposal nor a question
    assert texts(svc.on_callback(callback(f"iv:{first}:draft"))) == conv.INTERVIEW_STALE_REPLY
    assert svc.process_next(clock.value) is True
    second = iv_nonce(telegram)
    assert second != first and conv_store.get_state(111).context["nonce"] == second


def test_leftover_description_row_is_treated_as_idle(svc, clock, translator, conv_store):
    conv_store.set_state(111, "awaiting_description", None, clock.value)
    translator.add(prop(add_role("<ROLE2>")))
    run_request(svc, clock, "add a role")
    assert translator.calls[0]["mode"] == "chat"
    assert conv_store.get_pending(111).mode == "chat"
    conv_store.delete_pending(111)
    conv_store.set_state(111, "awaiting_description", None, clock.value)
    assert setup(svc) == []  # no clarification notice for the retired state


def test_setup_drops_a_running_request(svc, clock, translator, telegram):
    """A request still in flight when /setup starts loses its result."""
    translator.add(prop(add_role("<ROLE2>")))

    translator.gate = threading.Event()
    svc.on_text(text_update("add a role"))
    clock.value += conv.DEBOUNCE_SECONDS
    worker = threading.Thread(target=svc.process_next, args=(clock.value,))
    worker.start()
    assert translator.entered.wait(5)
    setup(svc, "guided")
    translator.gate.set()
    worker.join(5)
    assert not any("<ROLE2>" in t for t in sent(telegram))


def test_progress_note_firing_late_is_not_sent_after_the_proposal(svc, clock, translator, telegram, monkeypatch):
    import threading

    monkeypatch.setattr(conv, "PROGRESS_NOTE_SECONDS", 0.02)
    translator.add(prop(add_role("<ROLE2>")))
    translator.gate = threading.Event()
    in_note, release_note = threading.Event(), threading.Event()
    real_send = svc._send

    def slow_send(chat_id, messages):
        if messages and messages[0].text == conv.PROGRESS_REPLY:
            in_note.set()  # the note is firing ...
            assert release_note.wait(10)  # ... and stuck before it reaches Telegram
        return real_send(chat_id, messages)

    monkeypatch.setattr(svc, "_send", slow_send)
    svc.on_text(text_update("add a role"))
    clock.value += conv.DEBOUNCE_SECONDS
    worker = threading.Thread(target=svc.process_next, args=(clock.value,))
    worker.start()
    assert translator.entered.wait(5) and in_note.wait(5)
    translator.gate.set()  # the model answers while the note is still firing
    time.sleep(0.2)
    assert not any("pc:" in str(c.get("reply_markup")) for c in telegram.calls("sendMessage"))
    release_note.set()
    worker.join(10)
    texts_sent = sent(telegram)
    assert conv.PROGRESS_REPLY in texts_sent
    proposal = next(i for i, c in enumerate(telegram.calls("sendMessage")) if "pc:" in str(c.get("reply_markup")))
    assert texts_sent.index(conv.PROGRESS_REPLY) < proposal
