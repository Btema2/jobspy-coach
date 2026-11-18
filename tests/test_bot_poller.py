"""BotPoller: offset-before-execute, stale commands, rejection logging, pairing, chunking, shutdown."""
import logging
import threading
import time

import httpx
import pytest

from src.bot.auth import Authorizer, format_code, issue_pairing_code
from src.bot.client import TelegramBotClient
from src.bot.poller import BotPoller
from src.bot.state import BotStateStore, Pairing
from src.bot.updates import OutgoingMessage, Update
from src.storage.db import Database
from tests.fake_telegram import FakeTelegram, callback_update, message_update

NOW = 1_700_000_000.0


class RecordingRouter:
    def __init__(self):
        self.handled = []
        self.replies = {}
        self.raises = False

    def is_mutating(self, update: Update) -> bool:
        return update.command in {"/undo"}

    def handle(self, update: Update):
        self.handled.append(update)
        if self.raises:
            raise RuntimeError("<SECRET-TEXT> boom")
        return self.replies.get(update.command, [OutgoingMessage(f"ok {update.command or update.text}")])


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "bot.db")
    database.init_db()
    return database


@pytest.fixture
def state(db):
    return BotStateStore(db)


@pytest.fixture
def telegram():
    return FakeTelegram()


def _make(telegram, state, tmp_path, *, router=None, stop=None, on_paired=None, commands=(), clock=None):
    client = TelegramBotClient("<TOKEN>", transport=telegram.transport)
    router = router or RecordingRouter()
    poller = BotPoller(
        client,
        state,
        Authorizer(state, clock=clock or (lambda: NOW)),
        router,
        stop_event=stop or threading.Event(),
        env_path=tmp_path / ".env",
        on_paired=on_paired or (lambda chat_id: None),
        commands=commands,
        clock=clock or (lambda: NOW),
    )
    return poller, router


def _paired(state):
    state.set_pairing(111, 111)


def _texts(telegram):
    return [c["text"] for c in telegram.calls("sendMessage")]


def test_offset_persisted_before_mutating_command(telegram, state, tmp_path):
    _paired(state)
    poller, router = _make(telegram, state, tmp_path)
    router.raises = True
    telegram.queue_updates([message_update(500, "/undo", date=int(NOW))])

    assert poller.poll_once() == 1

    assert state.get_offset() == 501
    assert len(router.handled) == 1

    # A restarted poller over the same DB asks for the saved offset: /undo is not handled again.
    telegram2 = FakeTelegram()
    poller2, router2 = _make(telegram2, BotStateStore(state._db), tmp_path)
    poller2.poll_once()
    assert telegram2.calls("getUpdates")[0]["offset"] == 501
    assert router2.handled == []


def test_offset_is_persisted_even_if_the_handler_would_kill_the_process(telegram, state, tmp_path):
    _paired(state)
    poller, router = _make(telegram, state, tmp_path)

    seen = {}

    def handle(update):
        seen["offset"] = state.get_offset()  # what a crash here would leave behind
        raise SystemExit

    router.handle = handle
    with pytest.raises(SystemExit):
        poller.process(message_update(700, "/undo", date=int(NOW)))
    assert seen["offset"] == 701
    assert state.get_offset() == 701


def test_stale_mutating_command_ignored_but_reads_run(telegram, state, tmp_path):
    _paired(state)
    poller, router = _make(telegram, state, tmp_path)
    old = int(NOW) - 16 * 60

    poller.process(message_update(1, "/undo", date=old))
    poller.process(message_update(2, "/help", date=old))

    assert [u.command for u in router.handled] == ["/help"]
    texts = _texts(telegram)
    assert "15 minutes" in texts[0] and "/undo" in texts[0]
    assert texts[1] == "ok /help"


def test_fresh_mutating_command_runs(telegram, state, tmp_path):
    _paired(state)
    poller, router = _make(telegram, state, tmp_path)

    poller.process(message_update(1, "/undo", date=int(NOW) - 14 * 60))

    assert [u.command for u in router.handled] == ["/undo"]


def test_rejected_update_logged_without_content(telegram, state, tmp_path, caplog):
    _paired(state)
    poller, router = _make(telegram, state, tmp_path)

    with caplog.at_level(logging.DEBUG):
        poller.process(message_update(42, "<SECRET-TEXT>", user_id=999, chat_id=999))
        poller.process(message_update(43, "/undo", chat_id=-5, chat_type="group\nfake log line"))

    assert telegram.calls("sendMessage") == []
    assert router.handled == []
    assert "rejected update 42 (message, private)" in caplog.text
    assert "rejected update 43 (message," in caplog.text
    assert "SECRET-TEXT" not in caplog.text and "fake log line" not in caplog.text
    assert state.get_offset() == 44


def test_callback_answered_and_routed(telegram, state, tmp_path):
    _paired(state)
    poller, router = _make(telegram, state, tmp_path)
    router.replies = {None: [OutgoingMessage("edited", reply_markup={"inline_keyboard": []}, edit_message_id=5)]}

    poller.process(callback_update(9, "w:1:go", message_id=5))
    poller.process(callback_update(10, "x", user_id=222))  # not the paired user

    assert [c["callback_query_id"] for c in telegram.calls("answerCallbackQuery")] == ["cb9"]
    edit = telegram.calls("editMessageText")[0]
    assert (edit["chat_id"], edit["message_id"], edit["text"]) == (111, 5, "edited")
    assert router.handled[0].kind == "callback" and router.handled[0].text == "w:1:go"
    assert telegram.calls("sendMessage") == []


def test_pairing_writes_chat_id_and_notifies(telegram, state, tmp_path):
    env = tmp_path / ".env"
    env.write_text("KEEP=1\nTELEGRAM_CHAT_ID=\n")
    paired = []
    poller, router = _make(telegram, state, tmp_path, on_paired=paired.append)
    code = issue_pairing_code(state, NOW)

    poller.process(message_update(1, f"/start {format_code(code)}", user_id=111, chat_id=111))

    assert state.get_pairing() == Pairing(111, 111)
    assert env.read_text() == "KEEP=1\nTELEGRAM_CHAT_ID=111\n"
    assert paired == [111]
    assert _texts(telegram) == ["Paired. Send /help to see the commands."]
    assert router.handled == []


def test_pairing_survives_env_write_failure(telegram, state, tmp_path, caplog, monkeypatch):
    paired = []
    poller, _ = _make(telegram, state, tmp_path, on_paired=paired.append)
    code = issue_pairing_code(state, NOW)

    def boom(path, updates):
        raise PermissionError("denied")

    monkeypatch.setattr("src.bot.poller.merge_env_file", boom)
    with caplog.at_level(logging.WARNING):
        poller.process(message_update(1, f"/start {code}"))

    assert state.get_pairing() == Pairing(111, 111)
    assert paired == [111]
    assert "PermissionError" in caplog.text


def test_wrong_code_reply_goes_to_sender(telegram, state, tmp_path):
    poller, _ = _make(telegram, state, tmp_path)
    issue_pairing_code(state, NOW)

    poller.process(message_update(1, "/start AAAA-AAAA", user_id=7, chat_id=7))

    sent = telegram.calls("sendMessage")
    assert [(c["chat_id"], c["text"]) for c in sent] == [(7, "Pairing code not accepted.")]
    assert state.get_pairing() is None


def test_router_error_replied_and_loop_continues(telegram, state, tmp_path, caplog):
    _paired(state)
    poller, router = _make(telegram, state, tmp_path)
    router.raises = True
    telegram.queue_updates([message_update(1, "/help"), message_update(2, "/help")])

    with caplog.at_level(logging.DEBUG):
        assert poller.poll_once() == 2

    assert _texts(telegram) == ["Something went wrong; nothing was changed unless a diff was shown."] * 2
    assert "RuntimeError" in caplog.text and "SECRET-TEXT" not in caplog.text
    assert state.get_offset() == 3


def test_long_reply_is_chunked(telegram, state, tmp_path):
    _paired(state)
    poller, router = _make(telegram, state, tmp_path)
    body = "\n".join(f"line {i} " + "x" * 80 for i in range(100))  # ~9,000 chars
    markup = {"inline_keyboard": [[{"text": "More", "callback_data": "p:2"}]]}
    router.replies = {"/roles": [OutgoingMessage(body, reply_markup=markup)]}

    poller.process(message_update(1, "/roles"))

    sent = telegram.calls("sendMessage")
    assert len(sent) == 3
    assert all(len(c["text"]) <= 3800 for c in sent)
    assert "reply_markup" not in sent[0] and "reply_markup" not in sent[1]
    assert sent[2]["reply_markup"] == markup
    assert "\n".join(c["text"] for c in sent) == body


def test_poller_backs_off_on_conflict_unauthorized_and_retry_after(telegram, state, tmp_path, caplog):
    class RecordingStop:
        def __init__(self, limit):
            self.waits = []
            self.limit = limit

        def is_set(self):
            return len(self.waits) >= self.limit

        def wait(self, timeout=None):
            self.waits.append(timeout)
            return self.is_set()

    # 409 twice: one WARNING, wait(30) each time
    stop = RecordingStop(2)
    poller, _ = _make(telegram, state, tmp_path, stop=stop)
    for _ in range(2):
        telegram.respond("getUpdates", 409, {"ok": False, "error_code": 409, "description": "Conflict"})
    with caplog.at_level(logging.WARNING):
        poller.run_forever()
    assert stop.waits == [30, 30]
    conflicts = [r for r in caplog.records if r.levelno == logging.WARNING and "another consumer" in r.getMessage()]
    assert len(conflicts) == 1

    # 429 retry_after=2 -> waits 2
    stop = RecordingStop(1)
    poller, _ = _make(telegram, state, tmp_path, stop=stop)
    telegram.respond("getUpdates", 429, {"ok": False, "error_code": 429, "description": "Too Many",
                                         "parameters": {"retry_after": 2}})
    poller.run_forever()
    assert stop.waits == [2]

    # a generic failure backs off exponentially, capped at 60 s
    stop = RecordingStop(8)
    poller, _ = _make(telegram, state, tmp_path, stop=stop)
    for _ in range(8):
        telegram.respond("getUpdates", 500, {"ok": False, "error_code": 500, "description": "boom"})
    poller.run_forever()
    assert stop.waits[0] < stop.waits[1] < stop.waits[2]
    assert max(stop.waits) <= 60

    # 401 -> one ERROR, run_forever returns without waiting
    stop = RecordingStop(10)
    poller, _ = _make(telegram, state, tmp_path, stop=stop)
    telegram.respond("getUpdates", 401, {"ok": False, "error_code": 401, "description": "Unauthorized"})
    caplog.clear()
    with caplog.at_level(logging.ERROR):
        poller.run_forever()
    assert stop.waits == []
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert [r.getMessage() for r in errors] == ["bot token rejected; bot polling stopped"]


def test_backoff_resets_after_a_success(telegram, state, tmp_path):
    class RecordingStop:
        def __init__(self):
            self.waits = []

        def is_set(self):
            return len(self.waits) >= 3

        def wait(self, timeout=None):
            self.waits.append(timeout)
            return self.is_set()

    stop = RecordingStop()
    poller, _ = _make(telegram, state, tmp_path, stop=stop)
    telegram.respond("getUpdates", 500, {"ok": False, "error_code": 500, "description": "x"})
    telegram.respond("getUpdates", 500, {"ok": False, "error_code": 500, "description": "x"})
    telegram.respond("getUpdates", 200, {"ok": True, "result": []})
    telegram.respond("getUpdates", 500, {"ok": False, "error_code": 500, "description": "x"})

    poller.run_forever()

    assert stop.waits[0] < stop.waits[1]
    assert stop.waits[2] == stop.waits[0]


def test_poller_stops_on_shutdown_event_while_blocked_in_a_poll(state, tmp_path):
    stop = threading.Event()

    def handler(request: httpx.Request) -> httpx.Response:
        stop.wait(30)  # a long poll that returns when the process shuts down
        return httpx.Response(200, json={"ok": True, "result": []})

    client = TelegramBotClient("<TOKEN>", transport=httpx.MockTransport(handler))
    poller = BotPoller(client, state, Authorizer(state), RecordingRouter(), stop_event=stop,
                       env_path=tmp_path / ".env")
    thread = poller.start()
    assert thread.daemon and thread.name == "bot-poller"
    time.sleep(0.1)
    started = time.monotonic()

    stop.set()
    thread.join(1)

    assert not thread.is_alive()
    assert time.monotonic() - started < 1


def test_updates_after_stop_are_not_processed(telegram, state, tmp_path):
    _paired(state)
    stop = threading.Event()
    poller, router = _make(telegram, state, tmp_path, stop=stop)
    telegram.queue_updates([message_update(1, "/help"), message_update(2, "/help")])
    original = router.handle

    def handle_and_stop(update):
        stop.set()
        return original(update)

    router.handle = handle_and_stop

    assert poller.poll_once() == 1
    assert state.get_offset() == 2  # update 2 stays unacknowledged and is re-delivered next start


def test_set_my_commands_sent_once_on_start(telegram, state, tmp_path):
    stop = threading.Event()
    poller, _ = _make(telegram, state, tmp_path, stop=stop, commands=[("help", "Show help"), ("undo", "Undo")])
    calls = {"n": 0}
    original = poller.poll_once

    def counted():
        calls["n"] += 1
        if calls["n"] >= 3:
            stop.set()
        return original()

    poller.poll_once = counted
    poller.run_forever()

    assert calls["n"] == 3
    sent = telegram.calls("setMyCommands")
    assert len(sent) == 1
    assert sent[0]["commands"] == [
        {"command": "help", "description": "Show help"},
        {"command": "undo", "description": "Undo"},
    ]


def test_set_my_commands_failure_does_not_stop_polling(telegram, state, tmp_path, caplog):
    stop = threading.Event()
    poller, _ = _make(telegram, state, tmp_path, stop=stop, commands=[("help", "Show help")])
    telegram.respond("setMyCommands", 500, {"ok": False, "error_code": 500, "description": "no"})
    original = poller.poll_once
    poller.poll_once = lambda: (stop.set(), original())[1]

    poller.run_forever()

    assert len(telegram.calls("getUpdates")) == 1
