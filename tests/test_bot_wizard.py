"""/setup wizard: stepwise role, synonyms, market, remote, confirm (router + FakeTelegram, no network)."""
import re
from pathlib import Path

import pytest

from src.bot.client import TelegramBotClient
from src.bot.commands import TRIGGER_STARTED, BotContext, CommandRouter
from src.bot.render import utf16_len
from src.bot.state import BotStateStore
from src.bot.updates import OutgoingMessage, parse_update
from src.bot.wizard import WIZARD_TTL_SECONDS, SetupWizard
from src.config import Settings
from src.profile.ops import AddRole, SetField, apply_changes, create_starter_profile
from src.profile.store import ProfileStore
from src.storage.db import Database
from tests.fake_telegram import FakeTelegram, callback_update, message_update

STALE = "This setup is no longer active. Send /setup to start again."


class Clock:
    def __init__(self):
        self.now = 1_000_000.0

    def __call__(self):
        return self.now


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def db(tmp_path: Path) -> Database:
    database = Database(tmp_path / "wizard.db")
    database.init_db()
    return database


@pytest.fixture
def store(db) -> ProfileStore:
    profile_store = ProfileStore(db)
    create_starter_profile(profile_store, wow_threshold=8.5, digest_threshold=6.5, output_language="en")
    return profile_store  # paused, no roles, no markets


@pytest.fixture
def router(tmp_path, db, store, clock) -> CommandRouter:
    ctx = BotContext(
        db_path=db.db_path, env_path=tmp_path / ".env", settings=Settings(_env_file=None),
        trigger_cycle=lambda **_: TRIGGER_STARTED, is_cycle_running=lambda: False, scraper=None, clock=clock,
    )
    return CommandRouter(ctx, client=TelegramBotClient("<TOKEN>", transport=FakeTelegram().transport))


class Chat:
    """Drives the router like the poller does and remembers the latest buttons."""

    def __init__(self, router: CommandRouter):
        self.router = router
        self.n = 0
        self.last: list = []

    def _handle(self, raw) -> str:
        self.last = self.router.handle(parse_update(raw))
        for m in self.last:
            assert utf16_len(m.text) <= 3800
        return "\n".join(m.text for m in self.last)

    def text(self, text: str) -> str:
        self.n += 1
        return self._handle(message_update(self.n, text))

    def press(self, data: str) -> str:
        self.n += 1
        return self._handle(callback_update(self.n, data))

    def buttons(self) -> list:
        out = []
        for m in self.last:
            for row in (m.reply_markup or {}).get("inline_keyboard", []):
                out.extend(row)
        return out

    def nonce(self) -> str:
        return self.buttons()[0]["callback_data"].split(":")[1]


@pytest.fixture
def chat(router) -> Chat:
    return Chat(router)


def _choice(chat: Chat, name: str) -> str:
    for b in chat.buttons():
        if b["callback_data"].endswith(":" + name):
            return b["callback_data"]
    raise AssertionError(f"no {name} button in {chat.buttons()}")


def _remote_data(chat: Chat) -> str:
    return _choice(chat, "remote")


def _to_confirm(chat: Chat, role="<ROLE>", market="germany") -> str:
    chat.text("/setup")
    chat.text(role)
    chat.press(_choice(chat, "skip"))
    chat.text(market)
    return chat.press(_choice(chat, "remote"))


def test_wizard_full_flow_applies_role_and_market(chat, store):
    before = store.read_snapshot().version
    first = chat.text("/setup")
    assert "role" in first.lower()
    assert any(b["callback_data"].endswith(":cancel") for b in chat.buttons())

    second = chat.text("<ROLE>")
    assert "synonym" in second.lower()
    assert {b["callback_data"].split(":")[2] for b in chat.buttons()} >= {"skip", "cancel"}

    third = chat.press(_choice(chat, "skip"))
    assert "market" in third.lower() or "country" in third.lower()

    fourth = chat.text("germany")
    assert {b["callback_data"].split(":")[2] for b in chat.buttons()} >= {"onsite", "remote", "remote_only", "cancel"}
    assert "remote" in fourth.lower()

    confirm = chat.press(_remote_data(chat))
    assert "&lt;ROLE&gt;" in confirm and "Germany" in confirm
    assert {b["callback_data"].split(":")[2] for b in chat.buttons()} == {"apply", "cancel"}
    assert store.read_snapshot().version == before  # nothing written before Apply

    done = chat.press(_choice(chat, "apply"))
    snap = store.read_snapshot()
    assert snap.version == before + 1
    assert [r.name for r in snap.roles] == ["<ROLE>"]
    assert snap.markets[0].countries == ("germany",) and snap.markets[0].remote is True
    assert snap.markets[0].remote_only is False
    assert "/resume" in done  # profile is paused
    assert store._db and BotStateStore(store._db).get_wizard() is None


def test_wizard_synonyms_step_with_text(chat, store):
    chat.text("/setup")
    chat.text("<ROLE>")
    chat.text("<SYN1>, <SYN2>, <syn1>")
    chat.text("poland")
    confirm = chat.press(_remote_data(chat))
    assert "&lt;SYN1&gt;" in confirm and "&lt;SYN2&gt;" in confirm
    chat.press(_choice(chat, "apply"))
    role = store.read_snapshot().roles[0]
    assert [t.term for t in role.terms] == ["<SYN1>", "<SYN2>"]


def test_wizard_onsite_and_remote_only(chat, store):
    chat.text("/setup")
    chat.text("<ROLE>")
    chat.press(_choice(chat, "skip"))
    chat.text("poland")
    chat.press(_choice(chat, "remote_only"))
    chat.press(_choice(chat, "apply"))
    market = store.read_snapshot().markets[0]
    assert market.remote_only is True and market.remote is True

    chat.text("/setup")
    chat.text("<ROLE2>")
    chat.press(_choice(chat, "skip"))
    chat.text("france")
    chat.press(_choice(chat, "onsite"))
    chat.press(_choice(chat, "apply"))
    market = [m for m in store.read_snapshot().markets if "France" in m.name][0]
    assert market.remote is False and market.remote_only is False


def test_wizard_ambiguous_country_offers_buttons(chat, store):
    chat.text("/setup")
    chat.text("<ROLE>")
    chat.press(_choice(chat, "skip"))
    reply = chat.text("united")
    nonce = chat.nonce()
    choices = [b["callback_data"] for b in chat.buttons() if not b["callback_data"].endswith(":cancel")]
    assert choices == [f"wz:{nonce}:c0", f"wz:{nonce}:c1", f"wz:{nonce}:c2"]
    labels = [b["text"] for b in chat.buttons() if b["callback_data"] in choices]
    assert len(labels) == 3 and "mean" in reply.lower()

    after = chat.press(choices[1])
    assert "remote" in after.lower()
    chat.press(_remote_data(chat))
    chat.press(_choice(chat, "apply"))
    market = store.read_snapshot().markets[0]
    assert market.name == labels[1]


def test_wizard_unknown_country_asks_again(chat):
    chat.text("/setup")
    chat.text("<ROLE>")
    chat.press(_choice(chat, "skip"))
    reply = chat.text("atlantis")
    assert "Unknown country" in reply and "atlantis" in reply
    assert any(b["callback_data"].endswith(":cancel") for b in chat.buttons())


def test_wizard_stale_nonce_and_expiry(chat, clock, store):
    chat.text("/setup")
    old_nonce = chat.nonce()
    chat.text("/setup")  # replaces the wizard
    new_nonce = chat.nonce()
    assert new_nonce != old_nonce
    assert chat.press(f"wz:{old_nonce}:cancel") == STALE

    chat.text("<ROLE>")
    clock.now += WIZARD_TTL_SECONDS + 60  # 31 minutes idle
    assert chat.press(f"wz:{new_nonce}:skip") == STALE
    assert BotStateStore(store._db).get_wizard() is None


def test_wizard_text_after_expiry_is_told_and_cleared(chat, clock, store):
    chat.text("/setup")
    clock.now += WIZARD_TTL_SECONDS + 1
    assert chat.text("<ROLE>") == STALE
    assert "Send /help" in chat.text("<ROLE>")  # no wizard any more: plain text falls through


def test_wizard_no_wizard_means_text_falls_through(chat):
    assert "Send /help" in chat.text("hello")
    assert chat.press("wz:deadbeef:apply") == STALE


def test_wizard_version_conflict(chat, store):
    _to_confirm(chat)
    apply_data = _choice(chat, "apply")
    before = store.read_snapshot().version
    chat.text("/add_role <OTHER>")
    assert store.read_snapshot().version == before + 1
    reply = chat.press(apply_data)
    assert "changed meanwhile" in reply and "/setup" in reply
    snap = store.read_snapshot()
    assert snap.version == before + 1 and [r.name for r in snap.roles] == ["<OTHER>"]
    assert BotStateStore(store._db).get_wizard() is None


def test_wizard_cancel_clears_state(chat, store):
    chat.text("/setup")
    chat.text("<ROLE>")
    before = store.read_snapshot().version
    reply = chat.press(_choice(chat, "cancel"))
    assert "cancel" in reply.lower()
    assert BotStateStore(store._db).get_wizard() is None
    assert store.read_snapshot().version == before
    assert "Send /help" in chat.text("<ROLE>")


def test_wizard_cancel_at_confirm(chat, store):
    _to_confirm(chat)
    chat.press(_choice(chat, "cancel"))
    assert BotStateStore(store._db).get_wizard() is None
    assert store.read_snapshot().roles == ()


def test_wizard_callback_data_under_64_bytes(chat):
    chat.text("/setup")
    seen = []

    def collect():
        seen.extend(b["callback_data"] for b in chat.buttons())

    collect()
    chat.text("<ROLE>" * 10)
    collect()
    chat.press(_choice(chat, "skip"))
    collect()
    chat.text("united")
    collect()
    chat.press(_choice(chat, "c0"))
    collect()
    chat.press(_remote_data(chat))
    collect()
    assert seen and all(len(d.encode("utf-8")) <= 64 for d in seen)
    assert all(re.fullmatch(r"wz:[0-9a-f]{8}:(skip|cancel|apply|onsite|remote|remote_only|c\d+)", d) for d in seen)


def test_wizard_rejects_invalid_role_name(chat, store):
    chat.text("/setup")
    reply = chat.text("a=b")
    assert "role" in reply.lower() and "=" in reply
    assert BotStateStore(store._db).get_wizard()["step"] == "role"
    assert any(b["callback_data"].endswith(":cancel") for b in chat.buttons())
    ok = chat.text("<ROLE>")
    assert "synonym" in ok.lower()


def test_wizard_rejects_existing_role_name(chat, store):
    apply_changes(store, [AddRole("<ROLE>")], label="x")
    chat.text("/setup")
    reply = chat.text("<role>")
    assert "duplicate" in reply.lower() or "already" in reply.lower()
    assert BotStateStore(store._db).get_wizard()["step"] == "role"


def test_wizard_rejects_invalid_synonym(chat, store):
    chat.text("/setup")
    chat.text("<ROLE>")
    reply = chat.text("fine, bad=term")
    assert "=" in reply
    assert BotStateStore(store._db).get_wizard()["step"] == "synonyms"


def test_wizard_text_on_button_step_reminds(chat):
    chat.text("/setup")
    chat.text("<ROLE>")
    chat.press(_choice(chat, "skip"))
    chat.text("germany")
    reply = chat.text("whatever")
    assert "button" in reply.lower()
    assert any(b["callback_data"].endswith(":remote_only") for b in chat.buttons())


def test_wizard_state_shape(chat, store):
    chat.text("/setup")
    chat.text("<ROLE>")
    state = BotStateStore(store._db).get_wizard()
    assert re.fullmatch(r"[0-9a-f]{8}", state["nonce"])
    assert state["step"] == "synonyms" and state["role"] == "<ROLE>"
    assert state["base_version"] == store.read_snapshot().version
    assert isinstance(state["updated_at"], float)


def test_wizard_running_profile_has_no_resume_note(chat, store):
    _to_confirm(chat)
    paused = chat.press(_choice(chat, "apply"))
    assert "/resume" in paused  # starter is paused

    chat.text("/setup")
    chat.text("<ROLE2>")
    chat.press(_choice(chat, "skip"))
    chat.text("poland")
    chat.press(_remote_data(chat))
    with store._db._get_connection() as conn:
        conn.execute("UPDATE profile SET paused = 0")
    done = chat.press(_choice(chat, "apply"))
    assert "/resume" not in done


def test_wizard_user_terms_escaped_in_confirm(chat):
    chat.text("/setup")
    chat.text("<b>& R&D")
    chat.press(_choice(chat, "skip"))
    chat.text("germany")
    confirm = chat.press(_remote_data(chat))
    assert "&lt;b&gt;&amp; R&amp;D" in confirm and "<b>&" not in confirm


def test_setup_wizard_class_direct(db, store, clock):
    wizard = SetupWizard(BotStateStore(db), store, clock=clock)
    assert wizard.on_text(parse_update(message_update(1, "hi"))) is None
    messages = wizard.start(parse_update(message_update(2, "/setup")))
    assert isinstance(messages[0], OutgoingMessage)


def test_replayed_apply_after_apply_is_rejected(chat, store):
    _to_confirm(chat)
    apply_data = _choice(chat, "apply")
    assert "Profile v" in chat.press(apply_data)
    version = store.read_snapshot().version
    assert chat.press(apply_data) == STALE  # a re-delivered or double-tapped button finds no wizard
    assert store.read_snapshot().version == version
