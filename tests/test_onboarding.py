"""``/setup`` with a language model: description prompt, onboarding translation, wizard fallbacks.

No network (``FakeTelegram``, scripted fake translator), no real .env, placeholders only.
"""
import sqlite3
from pathlib import Path

import pytest

import src.bot.conversation as conv
from src.bot.client import TelegramBotClient
from src.bot.commands import TRIGGER_STARTED, BotContext, CommandRouter
from src.bot.conversation import ConversationService
from src.bot.conversation_store import ConversationStore
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


def test_setup_with_llm_asks_for_description(svc, db, conv_store, clock):
    BotStateStore(db).set_wizard({"nonce": "old", "step": "role", "updated_at": clock.value})
    reply = setup(svc)
    assert texts(reply) == conv.ONBOARDING_PROMPT
    for word in ("roles", "synonyms", "countries", "level", "avoid"):
        assert word in conv.ONBOARDING_PROMPT.lower()
    assert button_data(reply[-1].reply_markup) == ["ob:guided"]
    assert reply[-1].reply_markup["inline_keyboard"][0][0]["text"] == "Step-by-step questions instead"
    assert conv_store.get_state(111).state == "awaiting_description"
    assert not wizard_open(db)


def test_setup_routes_through_the_router_command(tmp_path, db, store, clock):
    client = TelegramBotClient(TOKEN, transport=FakeTelegram().transport)
    router = CommandRouter(_ctx(tmp_path, db, clock, FakeChatLLM()), client=client)
    reply = router.handle(parse_update(message_update(1, "/setup")))
    assert reply[0].text == conv.ONBOARDING_PROMPT
    reply = router.handle(parse_update(callback_update(2, "ob:guided")))
    assert "step 1 of 4" in "\n".join(m.text for m in reply)


# --- the description ----------------------------------------------------------------------------


ONBOARDING_OPS = (
    {"op": "add_role", "name": "<ROLE>", "synonyms": [{"term": "<SYN>", "language": "en"}]},
    {"op": "add_market", "country": "germany", "remote": True, "remote_only": False},
    {"op": "set_text", "field": "persona", "value": "<PERSONA>"},
    {"op": "set_text", "field": "core_role_definition", "value": "<CORE>"},
    {"op": "set_text", "field": "dealbreakers", "value": "<DEALBREAKER>"},
    {"op": "set_target_level", "level": "senior"},
    {"op": "add_rule", "kind": "block", "term": "<BLOCKED>", "match_mode": "exact"},
)


def test_description_translated_in_onboarding_mode(svc, clock, translator, telegram, conv_store, store):
    setup(svc)
    translator.add(prop(*ONBOARDING_OPS))
    run_request(svc, clock, "I look for <ROLE> in <MARKET>")
    assert translator.calls[0]["mode"] == "onboarding"
    pending = conv_store.get_pending(111)
    assert pending.mode == "onboarding"
    shown = "\n".join(sent(telegram))
    for word in ("<ROLE>", "<PERSONA>", "<BLOCKED>", "senior"):
        assert word.replace("<", "&lt;").replace(">", "&gt;") in shown or word in shown
    data = [d for d in button_data(last_markup(telegram)) if d.endswith(":a")][0]
    before = version(store)
    reply = texts(svc.on_callback(callback(data)))
    assert version(store) == before + 1
    assert store.list_versions()[0].label == "llm onboarding"
    snapshot = store.read_snapshot()
    assert [r.name for r in snapshot.roles] == ["<ROLE>"]
    assert snapshot.fields.target_level == "senior" and snapshot.fields.persona == "<PERSONA>"
    assert [r.source for r in snapshot.rules if r.term == "<BLOCKED>"] == ["llm"]
    assert snapshot.fields.paused is True and conv.PAUSED_NOTE in reply
    assert conv_store.get_state(111).state == "idle"


def test_description_state_expires_after_24h(svc, clock, translator, conv_store):
    setup(svc)
    clock.value += conv.CONTEXT_TTL_SECONDS + 1
    translator.add(prop(add_role("<ROLE2>")))
    run_request(svc, clock, "later request")
    assert translator.calls[0]["mode"] == "chat"


def test_onboarding_consent_decline_starts_wizard(svc, db, clock, translator, telegram, conv_store):
    with sqlite3.connect(db.db_path) as conn:
        conn.execute("DELETE FROM llm_consent")
    setup(svc)
    run_request(svc, clock, "I look for <ROLE>")
    notice = telegram.calls("sendMessage")[-1]
    assert notice["text"] == CONSENT_NOTICE
    no = [d for d in button_data(notice["reply_markup"]) if d.endswith(":no")][0]
    reply = svc.on_callback(callback(no, message_id=77))
    text = texts(reply)
    assert "Nothing was sent" in text and "step 1 of 4" in text
    assert wizard_open(db)
    assert conv_store.get_state(111).state == "idle"
    assert translator.calls == []


def test_chat_consent_decline_keeps_plain_reply(svc, db, clock, telegram):
    with sqlite3.connect(db.db_path) as conn:
        conn.execute("DELETE FROM llm_consent")
    run_request(svc, clock, "add <ROLE>")
    no = [d for d in button_data(telegram.calls("sendMessage")[-1]["reply_markup"]) if d.endswith(":no")][0]
    assert texts(svc.on_callback(callback(no))) == conv.CONSENT_DECLINED_REPLY
    assert not wizard_open(db)


# --- switching between the two ways ----------------------------------------------------------------


def test_guided_button_switches_to_wizard(svc, db, conv_store, telegram):
    setup(svc)
    reply = svc.on_callback(callback("ob:guided", message_id=42))
    assert "step 1 of 4" in texts(reply) and wizard_open(db)
    assert conv_store.get_state(111).state == "idle"
    assert {"chat_id": 111, "message_id": 42} in telegram.calls("editMessageReplyMarkup")
    # pressed again (or after the state moved on): nothing happens
    assert "no longer active" in texts(svc.on_callback(callback("ob:guided")))


def test_guided_button_inactive_without_description_state(svc, db):
    assert "no longer active" in texts(svc.on_callback(callback("ob:guided")))
    assert not wizard_open(db)


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


def test_description_prompt_also_discards_pending_proposal(svc, clock, translator, conv_store):
    translator.add(prop(add_role("<ROLE2>")))
    run_request(svc, clock, "add a role")
    reply = setup(svc)
    assert conv.PROPOSAL_DISCARDED_REPLY in texts(reply) and conv.ONBOARDING_PROMPT in texts(reply)
    assert conv_store.get_pending(111) is None
    assert conv_store.get_state(111).state == "awaiting_description"


def test_setup_drops_a_running_request(svc, clock, translator, telegram):
    """A request still in flight when /setup starts loses its result."""
    translator.add(prop(add_role("<ROLE2>")))
    import threading

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
