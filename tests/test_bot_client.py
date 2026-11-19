import logging
import threading

import httpx
import pytest

from src.bot.client import (
    TelegramApiError,
    TelegramBotClient,
    TelegramConflict,
    TelegramNetworkError,
    TelegramRetryAfter,
    TelegramUnauthorized,
)
from tests.fake_telegram import FakeTelegram

# Token-shaped (so the redaction is really exercised) but assembled at runtime: no token literal in the source.
TOKEN = "".join(["123456", ":", "FAKE-" * 8])


def _client(fake: FakeTelegram) -> TelegramBotClient:
    return TelegramBotClient(TOKEN, transport=fake.transport)


def _error(status: int, description: str, **parameters):
    body = {"ok": False, "error_code": status, "description": description}
    if parameters:
        body["parameters"] = parameters
    return body


def test_send_message_posts_html_and_returns_message_id():
    fake = FakeTelegram()
    markup = {"inline_keyboard": [[{"text": "x", "callback_data": "y"}]]}

    message_id = _client(fake).send_message(5, "<b>hi</b>", reply_markup=markup)

    assert isinstance(message_id, int)
    method, body = fake.requests[0]
    assert method == "sendMessage"
    assert body["chat_id"] == 5 and body["text"] == "<b>hi</b>"
    assert body["parse_mode"] == "HTML"
    assert body["link_preview_options"] == {"is_disabled": True}
    assert body["reply_markup"] == markup


def test_get_updates_sends_offset_and_long_poll_timeout():
    fake = FakeTelegram()
    fake.queue_updates([{"update_id": 4}])

    result = _client(fake).get_updates(4)

    assert result == [{"update_id": 4}]
    body = fake.requests[0][1]
    assert body["offset"] == 4 and body["timeout"] == 25
    assert body["allowed_updates"] == ["message", "callback_query"]


def test_get_updates_http_timeout_exceeds_long_poll():
    seen = []

    def handler(request):
        seen.append(request.extensions["timeout"])
        return httpx.Response(200, json={"ok": True, "result": []})

    client = TelegramBotClient(TOKEN, transport=httpx.MockTransport(handler), timeout=15.0)
    client.get_updates(0, timeout=25)
    client.get_me()
    assert seen[0]["read"] == 35
    assert seen[1]["read"] == 15


def test_429_raises_retry_after_with_seconds():
    fake = FakeTelegram()
    fake.respond("getUpdates", 429, _error(429, "Too Many Requests: retry after 7", retry_after=7))

    with pytest.raises(TelegramRetryAfter) as info:
        _client(fake).get_updates(0)

    assert info.value.retry_after == 7
    assert info.value.status == 429


def test_429_without_parameters_defaults_to_five():
    fake = FakeTelegram()
    fake.respond("getMe", 429, _error(429, "slow down"))
    with pytest.raises(TelegramRetryAfter) as info:
        _client(fake).get_me()
    assert info.value.retry_after == 5


def test_send_message_retries_once_after_short_retry_after(monkeypatch):
    sleeps = []
    monkeypatch.setattr("src.bot.client.time.sleep", sleeps.append)
    fake = FakeTelegram()
    fake.respond("sendMessage", 429, _error(429, "Too Many Requests", retry_after=3))

    message_id = _client(fake).send_message(5, "hello")

    assert isinstance(message_id, int)
    assert sleeps == [3]
    assert len(fake.calls("sendMessage")) == 2


def test_send_message_retry_wait_is_interrupted_by_the_stop_event(monkeypatch):
    monkeypatch.setattr("src.bot.client.time.sleep", lambda s: pytest.fail("must wait on the stop event"))
    stop = threading.Event()
    stop.set()
    fake = FakeTelegram()
    fake.respond("sendMessage", 429, _error(429, "Too Many Requests", retry_after=30))
    client = TelegramBotClient(TOKEN, transport=fake.transport, stop_event=stop)

    assert client.send_message(5, "hello") is None
    assert len(fake.calls("sendMessage")) == 1  # no retry once stopping


def test_send_message_retries_when_the_stop_event_stays_clear():
    fake = FakeTelegram()
    fake.respond("sendMessage", 429, _error(429, "Too Many Requests", retry_after=0.01))
    client = TelegramBotClient(TOKEN, transport=fake.transport, stop_event=threading.Event())

    assert isinstance(client.send_message(5, "hello"), int)
    assert len(fake.calls("sendMessage")) == 2


def test_empty_token_does_not_garble_error_text():
    fake = FakeTelegram()
    fake.respond("getMe", 400, _error(400, "plain description"))
    with pytest.raises(TelegramApiError) as info:
        TelegramBotClient("", transport=fake.transport).get_me()
    assert info.value.description == "plain description"


@pytest.mark.parametrize("error", [httpx.InvalidURL("bad"), httpx.StreamError("closed")])
def test_other_httpx_errors_are_wrapped_like_transport_errors(error):
    def boom(request):
        raise error

    client = TelegramBotClient(TOKEN, transport=httpx.MockTransport(boom))
    with pytest.raises(TelegramNetworkError) as info:
        client.get_me()
    assert str(info.value) == type(error).__name__
    assert client.send_message(5, "x") is None


def test_send_message_does_not_wait_for_long_retry_after(monkeypatch):
    sleeps = []
    monkeypatch.setattr("src.bot.client.time.sleep", sleeps.append)
    fake = FakeTelegram()
    fake.respond("sendMessage", 429, _error(429, "Too Many Requests", retry_after=300))

    assert _client(fake).send_message(5, "hello") is None
    assert sleeps == []
    assert len(fake.calls("sendMessage")) == 1


def test_send_message_falls_back_to_plain_text_on_parse_error():
    fake = FakeTelegram()
    fake.respond("sendMessage", 400, _error(400, "Bad Request: can't parse entities: Unsupported start tag"))

    message_id = _client(fake).send_message(5, "<b>a &amp; b</b> &lt;x&gt;")

    assert isinstance(message_id, int)
    first, second = fake.calls("sendMessage")
    assert first["parse_mode"] == "HTML"
    assert "parse_mode" not in second
    assert second["text"] == "a & b <x>"


def test_send_message_never_raises(caplog):
    fake = FakeTelegram()
    fake.respond("sendMessage", 400, _error(400, "Bad Request: chat not found"))
    fake.respond("sendMessage", 500, _error(500, "boom"))
    client = _client(fake)
    with caplog.at_level(logging.WARNING):
        assert client.send_message(5, "x") is None
        assert client.send_message(5, "x") is None
    assert "sendMessage 400" in caplog.text and "sendMessage 500" in caplog.text


def test_send_message_returns_none_on_network_error():
    def boom(request):
        raise httpx.ConnectError("down")

    client = TelegramBotClient(TOKEN, transport=httpx.MockTransport(boom))
    assert client.send_message(5, "x") is None


def test_401_and_409_typed():
    fake = FakeTelegram()
    fake.respond("getMe", 401, _error(401, "Unauthorized"))
    fake.respond("getUpdates", 409, _error(409, "Conflict: terminated by other getUpdates request"))
    client = _client(fake)
    with pytest.raises(TelegramUnauthorized):
        client.get_me()
    with pytest.raises(TelegramConflict):
        client.get_updates(0)
    assert issubclass(TelegramUnauthorized, TelegramApiError)


def test_error_description_is_truncated():
    fake = FakeTelegram()
    fake.respond("getMe", 400, _error(400, "x" * 500))
    with pytest.raises(TelegramApiError) as info:
        _client(fake).get_me()
    assert len(info.value.description) <= 200
    assert info.value.method == "getMe"


def test_token_never_in_logs_or_errors(caplog):
    def boom(request):
        raise httpx.ConnectError(f"cannot reach {request.url}", request=request)

    client = TelegramBotClient(TOKEN, transport=httpx.MockTransport(boom))
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(TelegramNetworkError) as info:
            client.get_me()
        client.send_message(5, "x")
    assert str(info.value) == "ConnectError"
    for text in (caplog.text, str(info.value), repr(info.value)):
        assert TOKEN not in text
        assert "ABCDEFGHIJKLMNOP" not in text
        assert "api.telegram.org/bot" not in text


def test_error_text_never_contains_token():
    fake = FakeTelegram()
    fake.respond("getMe", 400, _error(400, f"bad https://api.telegram.org/bot{TOKEN}/getMe"))
    with pytest.raises(TelegramApiError) as info:
        _client(fake).get_me()
    assert TOKEN not in str(info.value) and TOKEN not in info.value.description


def test_helpers_post_expected_payloads():
    fake = FakeTelegram()
    client = _client(fake)
    assert client.get_me()["username"] == "fake_bot"
    assert client.edit_message_text(5, 9, "t", reply_markup={"inline_keyboard": []}) is True
    assert client.answer_callback_query("cb1", text="done") is True
    assert client.set_my_commands([("help", "Show help"), ("stats", "Stats")]) is True
    client.close()
    assert fake.calls("editMessageText")[0] == {
        "chat_id": 5, "message_id": 9, "text": "t", "parse_mode": "HTML",
        "link_preview_options": {"is_disabled": True}, "reply_markup": {"inline_keyboard": []},
    }
    assert fake.calls("answerCallbackQuery")[0] == {"callback_query_id": "cb1", "text": "done"}
    assert fake.calls("setMyCommands")[0] == {
        "commands": [
            {"command": "help", "description": "Show help"},
            {"command": "stats", "description": "Stats"},
        ]
    }


def test_helpers_return_false_on_failure():
    fake = FakeTelegram()
    fake.respond("editMessageText", 400, _error(400, "message is not modified"))
    fake.respond("answerCallbackQuery", 400, _error(400, "query is too old"))
    fake.respond("setMyCommands", 500, _error(500, "oops"))
    client = _client(fake)
    assert client.edit_message_text(5, 9, "t") is False
    assert client.answer_callback_query("cb1") is False
    assert client.set_my_commands([("a", "b")]) is False


# --- files and reply markup ----------------------------------------------------------------------


def test_get_file_path_and_download_roundtrip():
    fake = FakeTelegram()
    fake.serve_file("f1", "documents/file_1.txt", b"hello <PLACEHOLDER>")
    client = _client(fake)
    path = client.get_file_path("f1")
    assert path == "documents/file_1.txt"
    assert client.download_file(path, 1024) == b"hello <PLACEHOLDER>"
    assert fake.calls("getFile") == [{"file_id": "f1"}]
    assert fake.calls("download") == [{"file_path": "documents/file_1.txt"}]


def test_get_file_path_none_for_unknown_file():
    fake = FakeTelegram()
    assert _client(fake).get_file_path("nope") is None


def test_download_limit_is_exact():
    fake = FakeTelegram()
    fake.serve_file("f1", "d/a.txt", b"x" * 100)
    client = _client(fake)
    assert len(client.download_file("d/a.txt", 100)) == 100
    with pytest.raises(TelegramApiError) as info:
        client.download_file("d/a.txt", 99)
    assert (info.value.method, info.value.status, info.value.description) == ("download", 413, "file too large")


def test_download_stops_reading_at_the_limit_without_content_length():
    def handler(request):
        def chunks():
            for _ in range(1000):
                yield b"x" * 1024

        return httpx.Response(200, content=chunks())

    client = TelegramBotClient(TOKEN, transport=httpx.MockTransport(handler))
    with pytest.raises(TelegramApiError) as info:
        client.download_file("d/a.txt", 4096)
    assert info.value.status == 413


def test_download_http_error_and_network_error_never_leak_token(caplog):
    fake = FakeTelegram()
    client = _client(fake)
    with pytest.raises(TelegramApiError) as info:
        client.download_file("d/missing.txt", 100)
    assert info.value.status == 404

    def boom(request):
        raise httpx.ConnectError(f"cannot reach {request.url}", request=request)

    broken = TelegramBotClient(TOKEN, transport=httpx.MockTransport(boom))
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(TelegramNetworkError) as net:
            broken.download_file("d/a.txt", 100)
    assert str(net.value) == "ConnectError"
    for text in (caplog.text, str(net.value), repr(net.value), str(info.value)):
        assert TOKEN not in text and "api.telegram.org" not in text


def test_edit_message_reply_markup_payloads():
    fake = FakeTelegram()
    client = _client(fake)
    assert client.edit_message_reply_markup(5, 9) is True
    assert client.edit_message_reply_markup(5, 9, {"inline_keyboard": []}) is True
    assert fake.calls("editMessageReplyMarkup") == [
        {"chat_id": 5, "message_id": 9},
        {"chat_id": 5, "message_id": 9, "reply_markup": {"inline_keyboard": []}},
    ]
    fake.respond("editMessageReplyMarkup", 400, _error(400, "message is not modified"))
    assert client.edit_message_reply_markup(5, 9) is False
