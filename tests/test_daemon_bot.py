"""Daemon wiring of the bot: poller lifecycle, cycle lock, /run_now trigger, live chat id (no network)."""
import threading
import time
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import httpx
import pytest

from src.bot.commands import TRIGGER_BUSY, TRIGGER_NOT_READY, TRIGGER_STARTED
from src.bot.state import BotStateStore, Pairing
from src.bot.updates import parse_update
from src.config import Settings
from src.scheduler.daemon import DaemonService
from src.scraper.query_catalog import SearchQuery
from src.storage.db import Database
from tests.fake_telegram import message_update

REAL_START_BOT = DaemonService._start_bot  # conftest replaces it by a no-op for every other test


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(DaemonService, "_start_bot", REAL_START_BOT)


def _settings(**kw) -> Settings:
    return Settings(_env_file=None, **kw)


def _daemon(tmp_path: Path, scraper=None, notifier=None, **settings) -> DaemonService:
    daemon = DaemonService(
        db_path=tmp_path / "daemon.db", scraper=scraper or MagicMock(), notifier=notifier or MagicMock()
    )
    daemon.settings = _settings(DB_PATH=str(tmp_path / "daemon.db"), **settings)
    return daemon


class BlockingTelegram:
    """getUpdates answers after ``delay`` seconds with no updates; everything else answers at once."""

    def __init__(self, delay=0.3):
        self.delay = delay
        self.methods: List[str] = []
        self.transport = httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        method = request.url.path.rsplit("/", 1)[-1]
        self.methods.append(method)
        if method == "getUpdates":
            time.sleep(self.delay)
            return httpx.Response(200, json={"ok": True, "result": []})
        return httpx.Response(200, json={"ok": True, "result": True})


@pytest.fixture
def telegram(monkeypatch) -> BlockingTelegram:
    fake = BlockingTelegram()
    import src.scheduler.daemon as daemon_module

    real = daemon_module.TelegramBotClient

    def factory(token, **kw):
        return real(token, transport=fake.transport, **kw)

    monkeypatch.setattr(daemon_module, "TelegramBotClient", factory)
    return fake


def _stop(daemon: DaemonService) -> None:
    daemon.request_shutdown()
    thread = getattr(daemon, "_bot_thread", None)
    if thread is not None:
        thread.join(timeout=3)


# --- start order --------------------------------------------------------------------------


def test_poller_started_before_startup_cycle(monkeypatch, tmp_path):
    monkeypatch.setattr("src.scheduler.daemon.BlockingScheduler", MagicMock())
    monkeypatch.setattr("src.scheduler.daemon.check_jobspy_update", MagicMock(return_value=None))
    monkeypatch.setattr("signal.signal", MagicMock())
    daemon = _daemon(tmp_path)
    order: List[str] = []
    monkeypatch.setattr(daemon, "_start_bot", lambda: order.append("bot"))
    monkeypatch.setattr(daemon, "run_pipeline_cycle", lambda *a, **k: order.append("cycle") or (0, 0))
    daemon.start()
    assert order == ["bot", "cycle"]


def test_bot_start_failure_does_not_stop_daemon(monkeypatch, tmp_path, caplog):
    import src.scheduler.daemon as daemon_module

    def boom(token, **kw):
        raise RuntimeError("secret-detail")

    monkeypatch.setattr(daemon_module, "TelegramBotClient", boom)
    monkeypatch.setattr("src.scheduler.daemon.BlockingScheduler", MagicMock())
    monkeypatch.setattr("src.scheduler.daemon.check_jobspy_update", MagicMock(return_value=None))
    monkeypatch.setattr("signal.signal", MagicMock())
    daemon = _daemon(tmp_path, TELEGRAM_BOT_TOKEN="<TOKEN>")
    cycle = MagicMock(return_value=(0, 0))
    monkeypatch.setattr(daemon, "run_pipeline_cycle", cycle)
    with caplog.at_level("WARNING"):
        daemon.start()
    cycle.assert_called_once()
    assert daemon.scheduler.start.called
    assert any("RuntimeError" in r.getMessage() for r in caplog.records)
    assert not any("secret-detail" in r.getMessage() for r in caplog.records)
    assert daemon._start_bot() is None


def test_no_poller_without_token_or_when_disabled(monkeypatch, tmp_path):
    import src.scheduler.daemon as daemon_module

    created = []
    monkeypatch.setattr(daemon_module, "TelegramBotClient", lambda *a, **k: created.append(1))
    assert _daemon(tmp_path)._start_bot() is None
    assert _daemon(tmp_path, TELEGRAM_BOT_TOKEN="<TOKEN>", TELEGRAM_BOT_POLLING="false")._start_bot() is None
    assert created == []


# --- cycle lock and trigger ---------------------------------------------------------------


def test_trigger_cycle_refuses_while_running(tmp_path):
    daemon = _daemon(tmp_path)
    daemon.scheduler = MagicMock()
    assert daemon._cycle_lock.acquire(blocking=False)
    try:
        assert daemon.is_cycle_running() is True
        assert daemon.trigger_cycle() == TRIGGER_BUSY
        daemon.scheduler.add_job.assert_not_called()
    finally:
        daemon._cycle_lock.release()
    assert daemon.is_cycle_running() is False


def test_trigger_cycle_adds_immediate_job(tmp_path):
    daemon = _daemon(tmp_path)
    daemon.scheduler = MagicMock()
    assert daemon.trigger_cycle() == TRIGGER_STARTED
    daemon.scheduler.add_job.assert_called_once_with(
        daemon.run_pipeline_cycle, id="run_now", replace_existing=True, misfire_grace_time=None
    )


def test_trigger_cycle_not_ready(tmp_path):
    daemon = _daemon(tmp_path)
    assert daemon.scheduler is None
    assert daemon.trigger_cycle() == TRIGGER_NOT_READY
    daemon.scheduler = MagicMock()
    daemon.request_shutdown()
    assert daemon.trigger_cycle() == TRIGGER_NOT_READY
    daemon.scheduler.add_job.assert_not_called()


class SlowScraper:
    def __init__(self):
        self.calls = 0
        self.entered = threading.Event()
        self.release = threading.Event()

    def scrape_board(self, site, query):
        self.calls += 1
        self.entered.set()
        assert self.release.wait(10)
        return []


def test_scheduled_cycle_skipped_while_one_runs(tmp_path, caplog):
    scraper = SlowScraper()
    daemon = _daemon(tmp_path, scraper=scraper)
    daemon.evaluate_unprocessed = MagicMock(return_value=(0, 0))
    queries = [SearchQuery("Q1", "term", "loc", ["google"])]
    results = []
    first = threading.Thread(target=lambda: results.append(daemon.run_pipeline_cycle(queries=queries)))
    first.start()
    assert scraper.entered.wait(5)
    with caplog.at_level("WARNING"):
        assert daemon.run_pipeline_cycle(queries=queries) == (0, 0)
    assert "A pipeline cycle is already running; this one is skipped." in caplog.text
    assert scraper.calls == 1
    scraper.release.set()
    first.join(timeout=10)
    assert not first.is_alive() and scraper.calls == 1
    assert daemon.is_cycle_running() is False
    # The lock is free again: the next cycle runs.
    daemon.run_pipeline_cycle(queries=queries)
    assert scraper.calls == 2


def test_cycle_lock_released_when_the_cycle_raises(tmp_path):
    daemon = _daemon(tmp_path)
    daemon.current_profile = MagicMock(side_effect=RuntimeError("boom"))
    with pytest.raises(RuntimeError):
        daemon.run_pipeline_cycle()
    assert daemon.is_cycle_running() is False


# --- the poller ---------------------------------------------------------------------------


def test_daemon_shutdown_stops_poller(tmp_path, telegram):
    daemon = _daemon(tmp_path, TELEGRAM_BOT_TOKEN="<TOKEN>")
    poller = daemon._start_bot()
    assert poller is not None
    thread = daemon._bot_thread
    assert thread.is_alive() and thread.daemon and thread.name == "bot-poller"
    assert poller._stop is daemon._shutdown_requested

    daemon._stop_event.set()  # a batch aborting / re-arming must not touch the poller
    time.sleep(0.4)
    daemon._stop_event.clear()
    assert thread.is_alive()

    started = time.monotonic()
    daemon.request_shutdown()
    thread.join(timeout=1.0)
    assert not thread.is_alive()
    assert time.monotonic() - started < 1.0
    assert "getUpdates" in telegram.methods and "setMyCommands" in telegram.methods


def test_start_bot_registers_features_and_commands(tmp_path, telegram):
    daemon = _daemon(tmp_path, TELEGRAM_BOT_TOKEN="<TOKEN>", TELEGRAM_CHAT_ID="111")
    poller = daemon._start_bot()
    try:
        router = poller._router
        for command in ("/setup", "/stats", "/test", "/run_now"):
            reply = router.handle(parse_update(message_update(1, command)))
            assert reply and "Unknown command" not in reply[0].text
        assert router.handle(parse_update(message_update(2, "/stats")))[0].text
    finally:
        _stop(daemon)


def test_start_bot_adopts_positive_chat_id_and_sets_notifier(tmp_path, telegram):
    notifier = MagicMock()
    notifier.chat_id = ""
    daemon = _daemon(tmp_path, notifier=notifier, TELEGRAM_BOT_TOKEN="<TOKEN>", TELEGRAM_CHAT_ID="4242")
    try:
        assert daemon._start_bot() is not None
        assert BotStateStore(Database(daemon.db.db_path)).get_pairing() == Pairing(4242, 4242)
    finally:
        _stop(daemon)


def test_start_bot_does_not_adopt_group_chat_id(tmp_path, telegram):
    daemon = _daemon(tmp_path, TELEGRAM_BOT_TOKEN="<TOKEN>", TELEGRAM_CHAT_ID="-100123")
    try:
        daemon._start_bot()
        assert BotStateStore(Database(daemon.db.db_path)).get_pairing() is None
    finally:
        _stop(daemon)


def test_start_bot_sets_notifier_chat_id_from_existing_pairing(tmp_path, telegram):
    notifier = MagicMock()
    notifier.chat_id = ""
    daemon = _daemon(tmp_path, notifier=notifier, TELEGRAM_BOT_TOKEN="<TOKEN>")
    BotStateStore(daemon.db).set_pairing(77, 88)
    try:
        daemon._start_bot()
        assert notifier.chat_id == "88"
    finally:
        _stop(daemon)


def test_start_bot_keeps_existing_notifier_chat_id(tmp_path, telegram):
    notifier = MagicMock()
    notifier.chat_id = "999"
    daemon = _daemon(tmp_path, notifier=notifier, TELEGRAM_BOT_TOKEN="<TOKEN>")
    BotStateStore(daemon.db).set_pairing(77, 88)
    try:
        daemon._start_bot()
        assert notifier.chat_id == "999"
    finally:
        _stop(daemon)


def test_on_paired_updates_notifier_chat_id(tmp_path):
    notifier = MagicMock()
    notifier.chat_id = ""
    daemon = _daemon(tmp_path, notifier=notifier)
    daemon._on_paired(12345)
    assert notifier.chat_id == "12345"
