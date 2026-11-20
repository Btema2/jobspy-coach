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
from src.scheduler.daemon import DaemonService, EvalSummary
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
    for name in ("_bot_thread", "_conversation_thread"):
        thread = getattr(daemon, name, None)
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
    monkeypatch.setattr(daemon, "run_pipeline_cycle_report", lambda *a, **k: order.append("cycle"))
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
    cycle = MagicMock(return_value=None)
    monkeypatch.setattr(daemon, "run_pipeline_cycle_report", cycle)
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


def test_is_cycle_running_covers_the_digest_run(tmp_path):
    """/requeue rewrites job buckets; a digest sending and marking rows concurrently must block it too."""
    daemon = _daemon(tmp_path)
    daemon.db.init_db()
    seen = []
    real_pending = daemon.db.get_pending_digest_jobs

    def spy(*args, **kwargs):
        seen.append(daemon.is_cycle_running())
        return real_pending(*args, **kwargs)

    daemon.db.get_pending_digest_jobs = spy
    assert daemon.is_cycle_running() is False
    daemon.run_daily_digest_cycle(dry_run=True)
    assert seen == [True]
    assert daemon.is_cycle_running() is False
    # a digest in flight does not make /run_now report a running scrape cycle
    assert daemon._cycle_lock.locked() is False


def test_trigger_cycle_adds_immediate_job(tmp_path):
    daemon = _daemon(tmp_path)
    daemon.scheduler = MagicMock()
    assert daemon.trigger_cycle() == TRIGGER_STARTED
    daemon.scheduler.add_job.assert_called_once_with(
        daemon.run_pipeline_cycle_report,
        kwargs={"run_now": True},
        id="run_now",
        replace_existing=True,
        misfire_grace_time=None,
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
    daemon.evaluate_unprocessed = MagicMock(return_value=EvalSummary(0, 0, 0, 0, None))
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


def test_shutdown_returns_at_once_while_a_long_poll_is_in_flight(monkeypatch, tmp_path):
    """A getUpdates blocking for 2.5 s: the signal handler must not wait for it (the poller is a daemon thread)."""
    import signal

    import src.scheduler.daemon as daemon_module

    fake = BlockingTelegram(delay=2.5)
    real = daemon_module.TelegramBotClient
    monkeypatch.setattr(daemon_module, "TelegramBotClient", lambda token, **kw: real(token, transport=fake.transport, **kw))
    mock_scheduler = MagicMock()
    mock_scheduler.running = True
    monkeypatch.setattr("src.scheduler.daemon.BlockingScheduler", MagicMock(return_value=mock_scheduler))
    monkeypatch.setattr("src.scheduler.daemon.check_jobspy_update", MagicMock(return_value=None))
    handlers = {}
    monkeypatch.setattr("signal.signal", lambda sig, handler: handlers.update({sig: handler}))
    daemon = _daemon(tmp_path, TELEGRAM_BOT_TOKEN="<TOKEN>")
    monkeypatch.setattr(daemon, "run_pipeline_cycle_report", MagicMock(return_value=None))
    daemon.start()
    thread = daemon._bot_thread
    deadline = time.monotonic() + 2
    while "getUpdates" not in fake.methods and time.monotonic() < deadline:
        time.sleep(0.01)
    assert "getUpdates" in fake.methods and thread.is_alive()

    monkeypatch.setattr("sys.exit", MagicMock())
    started = time.monotonic()
    handlers[signal.SIGTERM](signal.SIGTERM, None)
    assert time.monotonic() - started < 1.0
    assert thread.is_alive() and thread.daemon  # still inside the poll: nothing waited for it
    thread.join(timeout=5)
    assert not thread.is_alive()  # and it ends by itself once the poll returns


def test_cycle_guard_is_atomic_and_covers_cycle_and_digest(tmp_path):
    daemon = _daemon(tmp_path)
    with daemon.cycle_guard() as acquired:
        assert acquired is True
        assert daemon.is_cycle_running()
        assert daemon.run_pipeline_cycle(queries=[]) == (0, 0)  # skipped: the guard holds the cycle lock
        with daemon.cycle_guard() as nested:
            assert nested is False
    assert not daemon.is_cycle_running()

    assert daemon._digest_lock.acquire(blocking=False)
    try:
        with daemon.cycle_guard() as acquired:
            assert acquired is False
        assert not daemon._cycle_lock.locked()  # a refused guard leaves nothing held
    finally:
        daemon._digest_lock.release()

    assert daemon._cycle_lock.acquire(blocking=False)
    try:
        with daemon.cycle_guard() as acquired:
            assert acquired is False
    finally:
        daemon._cycle_lock.release()
    assert not daemon._digest_lock.locked()


def test_requeue_holds_the_cycle_lock_and_is_refused_while_a_cycle_runs(tmp_path, telegram, monkeypatch):
    import src.bot.commands as commands
    from src.profile.ops import create_starter_profile
    from src.profile.store import ProfileStore

    daemon = _daemon(tmp_path, TELEGRAM_BOT_TOKEN="<TOKEN>")
    create_starter_profile(ProfileStore(daemon.db), wow_threshold=8.5, digest_threshold=6.5, output_language="en")
    poller = daemon._start_bot()
    seen = {}

    def fake_requeue(db, rules):
        seen["locked"] = daemon._cycle_lock.locked()
        seen["cycle"] = daemon.run_pipeline_cycle(queries=[])  # a scheduled cycle meanwhile is skipped
        from src.profile.requeue import RequeueResult

        return RequeueResult(0, 0, 0, 0, [])

    try:
        monkeypatch.setattr(commands, "requeue_prefiltered", fake_requeue)
        reply = poller._router.handle(parse_update(message_update(1, "/requeue")))
        assert seen == {"locked": True, "cycle": (0, 0)} and "checked: 0" in reply[0].text
        assert not daemon.is_cycle_running()

        assert daemon._cycle_lock.acquire(blocking=False)
        seen.clear()
        reply = poller._router.handle(parse_update(message_update(2, "/requeue")))
        assert "A cycle is running" in reply[0].text and seen == {}
        daemon._cycle_lock.release()
    finally:
        _stop(daemon)


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


# --- the language-model layer (conversation worker, ChatLLM) -------------------------------------


def _wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_conversation_worker_started_after_poller_and_stops_on_shutdown(tmp_path, telegram, monkeypatch):
    from src.bot.conversation import ConversationService
    from src.bot.poller import BotPoller

    order: List[str] = []
    real_poller_start, real_worker_start = BotPoller.start, ConversationService.start
    monkeypatch.setattr(BotPoller, "start", lambda self: order.append("poller") or real_poller_start(self))
    monkeypatch.setattr(ConversationService, "start", lambda self: order.append("worker") or real_worker_start(self))
    daemon = _daemon(tmp_path, TELEGRAM_BOT_TOKEN="<TOKEN>")
    assert daemon._start_bot() is not None
    thread = daemon._conversation_thread
    try:
        assert order == ["poller", "worker"]
        assert thread.is_alive() and thread.daemon and thread.name == "bot-conversation"
        daemon._stop_event.set()  # a batch aborting must not touch the worker
        time.sleep(0.6)
        daemon._stop_event.clear()
        assert thread.is_alive()
        started = time.monotonic()
        daemon.request_shutdown()
        thread.join(timeout=1.0)
        assert not thread.is_alive() and time.monotonic() - started < 1.0
    finally:
        _stop(daemon)


def test_chat_llm_uses_shutdown_event_and_chain_providers(tmp_path, telegram):
    from types import SimpleNamespace

    from tests.fake_llm import ScriptedLLM, make_providers

    providers = make_providers(ScriptedLLM().transport)
    daemon = _daemon(tmp_path, TELEGRAM_BOT_TOKEN="<TOKEN>")
    daemon.evaluator = SimpleNamespace(providers=providers)
    poller = daemon._start_bot()
    try:
        llm = daemon.chat_llm
        assert llm is not None and llm._stop_event is daemon._shutdown_requested
        assert llm.available() and llm._providers() == providers
        assert poller._router.conversation._ctx.chat_llm is llm
    finally:
        _stop(daemon)


def test_chat_llm_unconfigured_daemon_still_starts_bot(tmp_path, telegram):
    import src.bot.conversation as conv

    daemon = _daemon(tmp_path, TELEGRAM_BOT_TOKEN="<TOKEN>", TELEGRAM_CHAT_ID="111", NVIDIA_API_KEY="")
    poller = daemon._start_bot()
    try:
        assert poller is not None and daemon.chat_llm is not None and daemon.chat_llm.available() is False
        reply = poller._router.handle(parse_update(message_update(1, "please add a role")))
        assert reply[0].text == conv.NOT_CONFIGURED_REPLY
        assert "step-by-step" in poller._router.handle(parse_update(message_update(2, "/setup")))[0].text
    finally:
        _stop(daemon)


class PacedTelegram:
    """FakeTelegram that makes an empty getUpdates take a moment (a real long poll blocks)."""

    def __init__(self):
        from tests.fake_telegram import FakeTelegram

        self.fake = FakeTelegram()
        self.transport = httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        response = self.fake._handle(request)
        if request.url.path.endswith("/getUpdates") and response.json()["result"] == []:
            time.sleep(0.05)
        return response

    def texts(self) -> List[str]:
        return [c["text"] for c in self.fake.calls("sendMessage")]


def test_working_and_progress_note_and_poller_stays_responsive(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    import src.bot.conversation as conv
    import src.scheduler.daemon as daemon_module
    from src.bot.conversation_store import ConversationStore
    from src.bot.privacy import CONSENT_VERSION
    from src.profile.ops import create_starter_profile
    from src.profile.store import ProfileStore
    from tests.fake_llm import make_providers, nim_reply

    monkeypatch.setattr(conv, "DEBOUNCE_SECONDS", 0.05)
    monkeypatch.setattr(conv, "PROGRESS_NOTE_SECONDS", 0.05)
    release, entered = threading.Event(), threading.Event()
    answer = json.dumps({"kind": "proposal", "operations": [{"op": "add_role", "name": "<ROLE>", "synonyms": []}]})

    def slow_nim(request: httpx.Request) -> httpx.Response:
        entered.set()
        assert release.wait(15)
        return nim_reply(answer)

    telegram = PacedTelegram()
    real = daemon_module.TelegramBotClient
    monkeypatch.setattr(daemon_module, "TelegramBotClient", lambda token, **kw: real(token, transport=telegram.transport, **kw))
    daemon = _daemon(tmp_path, TELEGRAM_BOT_TOKEN="<TOKEN>", TELEGRAM_CHAT_ID="111")
    daemon.evaluator = SimpleNamespace(providers=make_providers(httpx.MockTransport(slow_nim)))
    create_starter_profile(ProfileStore(daemon.db), wow_threshold=8.5, digest_threshold=6.5, output_language="en")
    ConversationStore(daemon.db).record_consent(111, CONSENT_VERSION)
    poller = daemon._start_bot()
    try:
        assert poller is not None
        telegram.fake.queue_updates([message_update(1, "add the role <ROLE>")])
        assert _wait_for(lambda: conv.WORKING_REPLY in telegram.texts())
        assert entered.wait(5)
        assert _wait_for(lambda: conv.PROGRESS_REPLY in telegram.texts())
        assert not any("pc:" in json.dumps(c.get("reply_markup", {})) for c in telegram.fake.calls("sendMessage"))

        telegram.fake.queue_updates([message_update(2, "/help")])  # the poller answers while the model is stuck
        assert _wait_for(lambda: any(t.startswith("/help -") for t in telegram.texts()))
        assert not release.is_set()

        release.set()
        assert _wait_for(
            lambda: any("pc:" in json.dumps(c.get("reply_markup", {})) for c in telegram.fake.calls("sendMessage"))
        )
    finally:
        release.set()
        _stop(daemon)


def test_on_paired_clears_conversation_tables(tmp_path, telegram):
    from src.bot.conversation_store import ConversationStore, PendingChange

    notifier = MagicMock()
    notifier.chat_id = ""
    daemon = _daemon(tmp_path, notifier=notifier, TELEGRAM_BOT_TOKEN="<TOKEN>")
    daemon.db.init_db()
    store = ConversationStore(daemon.db)

    def fill():
        store.put_pending(PendingChange(5, "abcd1234", 1, "chat", "[]", None, time.time()))
        store.set_state(5, "awaiting_clarification", {"rounds": [["<MARKER>", "?"]]}, time.time())
        store.record_consent(5, 1)

    fill()
    daemon._on_paired(12345)  # no bot running: the tables are still cleared
    assert notifier.chat_id == "12345"
    assert store.get_pending(5) is None and store.get_state(5).state == "idle" and not store.has_consent(5, 1)

    daemon._start_bot()
    try:
        calls = []
        daemon._bot_router.conversation.reset_memory = lambda: calls.append(1)
        fill()
        daemon._on_paired(777)
        assert calls == [1]
        assert store.get_pending(5) is None and not store.has_consent(5, 1)
    finally:
        _stop(daemon)


def test_conversation_worker_start_failure_keeps_the_bot_running(tmp_path, telegram, monkeypatch, caplog):
    from src.bot.conversation import ConversationService

    def boom(self):
        raise RuntimeError("secret-detail")

    monkeypatch.setattr(ConversationService, "start", boom)
    daemon = _daemon(tmp_path, TELEGRAM_BOT_TOKEN="<TOKEN>")
    try:
        with caplog.at_level("WARNING"):
            poller = daemon._start_bot()
        assert poller is not None and daemon._bot_thread.is_alive()
        assert daemon._conversation_thread is None
        assert any("RuntimeError" in r.getMessage() for r in caplog.records)
        assert not any("secret-detail" in r.getMessage() or "Telegram bot could not start" in r.getMessage() for r in caplog.records)
    finally:
        _stop(daemon)


def test_unconfigured_chain_build_is_not_retried_on_every_message(tmp_path, telegram, monkeypatch):
    import src.scheduler.daemon as daemon_module
    from src.evaluator.errors import ConfigurationError

    builds = []

    def failing_build(*args, **kwargs):
        builds.append(1)
        raise ConfigurationError("no provider")

    monkeypatch.setattr(daemon_module, "build_chain_from_settings", failing_build)
    daemon = _daemon(tmp_path, TELEGRAM_BOT_TOKEN="<TOKEN>", NVIDIA_API_KEY="")
    try:
        daemon._start_bot()
        assert [daemon.chat_llm.available() for _ in range(4)] == [False] * 4
        assert len(builds) == 1
        daemon._chat_unconfigured_until = 0.0  # the short TTL has passed
        assert daemon.chat_llm.available() is False
        assert len(builds) == 2
    finally:
        _stop(daemon)
