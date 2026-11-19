"""End to end: the real bot stack (client, state, pairing, router, poller thread) against a fake Telegram."""
import logging
import threading
import time
from pathlib import Path

import pytest

from src.bot.auth import Authorizer, format_code, issue_pairing_code
from src.bot.client import TelegramBotClient
from src.bot.commands import COMMANDS, TRIGGER_STARTED, BotContext, CommandRouter
from src.bot.poller import BotPoller
from src.bot.render import utf16_len
from src.bot.state import BotStateStore
from src.config import Settings
from src.profile.ops import AddMarket, AddRole, SetField, apply_changes, create_starter_profile
from src.profile.snapshot import Market, RoleTerm
from src.profile.store import ProfileStore
from src.storage.db import Database
from tests.fake_telegram import FakeTelegram, message_update

TOKEN = "123456:E2E-FAKE-TOKEN"
SYNONYM = "<SYN>"
GROUP_ID = -100500


class Harness:
    def __init__(self, tmp_path: Path):
        self.db = Database(tmp_path / "bot.db")
        self.db.init_db()
        self.store = ProfileStore(self.db)
        create_starter_profile(self.store, wow_threshold=8.5, digest_threshold=6.5, output_language="en")
        apply_changes(
            self.store,
            [
                AddRole("<BASE-ROLE>", (RoleTerm("<BASE-SYN>", None),)),
                AddMarket(Market("<MARKET>", "<LOCATION>", ("poland",), languages=("pl", "en"))),
                SetField("paused", False),
            ],
            label="fixture",
        )
        self.base_version = self.store.read_snapshot().version
        self.env_path = tmp_path / ".env"
        self.env_path.write_text("OTHER_KEY=keep\n", encoding="utf-8")
        self.state = BotStateStore(self.db)
        self.telegram = FakeTelegram()
        self.stop = threading.Event()
        self.triggers = 0
        ctx = BotContext(
            db_path=self.db.db_path,
            env_path=self.env_path,
            settings=Settings(_env_file=None),
            trigger_cycle=self._trigger,
            is_cycle_running=lambda: False,
            scraper=None,
        )
        client = TelegramBotClient(TOKEN, transport=self.telegram.transport, stop_event=self.stop)
        self.poller = BotPoller(
            client,
            self.state,
            Authorizer(self.state),
            CommandRouter(ctx, client=client),
            stop_event=self.stop,
            env_path=self.env_path,
            commands=COMMANDS,
        )

    def _trigger(self) -> str:
        self.triggers += 1
        return TRIGGER_STARTED

    def send_texts(self):
        return [c["text"] for c in self.telegram.calls("sendMessage")]


def _run_scenario(tmp_path: Path) -> Harness:
    h = Harness(tmp_path)
    code = format_code(issue_pairing_code(h.state, time.time()))
    h.telegram.queue_updates([message_update(1, "/start WRONG000", user_id=222, chat_id=GROUP_ID, chat_type="group")])
    h.telegram.queue_updates([message_update(2, f"/start {code}")])
    h.telegram.queue_updates([message_update(3, f"/add_role <ROLE-E2E> = {SYNONYM}")])
    h.telegram.queue_updates([message_update(4, "/undo")])
    h.telegram.queue_updates([message_update(4, "/undo")])  # Telegram re-delivers an already seen update
    h.telegram.queue_updates([message_update(5, "/stats")])

    thread = h.poller.start()
    deadline = time.monotonic() + 10
    while len(h.send_texts()) < 4 and time.monotonic() < deadline:  # pairing, add_role, undo, stats
        time.sleep(0.01)
    time.sleep(0.05)  # let a possible (wrong) extra reply surface before the stop
    started = time.monotonic()
    h.stop.set()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert time.monotonic() - started < 2
    return h


def test_bot_loop_against_fake_telegram(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    h = _run_scenario(tmp_path)

    # The stranger in the group got nothing; every reply went to the paired private chat.
    sends = h.telegram.calls("sendMessage")
    assert sends and all(call["chat_id"] == 111 for call in sends)
    assert not any(call["chat_id"] == GROUP_ID for call in sends)

    # Pairing wrote the chat id and kept the other .env lines.
    env_lines = h.env_path.read_text(encoding="utf-8").splitlines()
    assert "TELEGRAM_CHAT_ID=111" in env_lines
    assert "OTHER_KEY=keep" in env_lines

    # The menu was registered once, with every command.
    menus = h.telegram.calls("setMyCommands")
    assert len(menus) == 1
    assert [c["command"] for c in menus[0]["commands"]] == [name for name, _ in COMMANDS]

    texts = h.send_texts()
    assert any("+ role" in text for text in texts)  # the /add_role diff

    # add (+1) and undo (+1); the re-delivered /undo (same update id) did not run a second time.
    assert h.store.read_snapshot().version == h.base_version + 2

    # The offset stays one past the last update id.
    assert h.state.get_offset() == 6

    assert all(utf16_len(text) <= 3800 for text in texts)


def test_e2e_logs_carry_no_token_or_text(tmp_path, monkeypatch, caplog):
    monkeypatch.chdir(tmp_path)
    with caplog.at_level(logging.DEBUG):
        # What configure_logging() does in production: httpx logs the request URL (with the token) at INFO.
        caplog.set_level(logging.WARNING, logger="httpx")
        caplog.set_level(logging.WARNING, logger="httpcore")
        _run_scenario(tmp_path)

    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert "E2E-FAKE-TOKEN" not in logged
    assert SYNONYM not in logged
    assert "<ROLE-E2E>" not in logged
