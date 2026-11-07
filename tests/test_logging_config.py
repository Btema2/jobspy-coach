import logging

import httpx
import pytest

from src.logging_config import RedactingFilter, configure_logging, redact_secrets

# Obviously fake credentials (never real).
FAKE_TOKEN = "123456:FAKEFAKEFAKEFAKEFAKEFAKEFAKE_abc-XYZ"
FAKE_URL = f"https://api.telegram.org/bot{FAKE_TOKEN}/sendMessage"


@pytest.fixture
def restore_logging():
    root = logging.getLogger()
    saved_handlers, saved_level = list(root.handlers), root.level
    saved = {n: logging.getLogger(n).level for n in ("httpx", "httpcore")}
    yield
    root.handlers[:] = saved_handlers
    root.setLevel(saved_level)
    for n, lvl in saved.items():
        logging.getLogger(n).setLevel(lvl)


def _clean_root():
    # pytest installs capture handlers per phase; basicConfig is a no-op unless root is empty.
    logging.getLogger().handlers[:] = []


def _capture(logger_name="test.redact"):
    class ListHandler(logging.Handler):
        def __init__(self):
            super().__init__()
            self.lines = []

        def emit(self, record):
            self.lines.append(self.format(record))

    handler = ListHandler()
    handler.addFilter(RedactingFilter())
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers[:] = [handler]
    return logger, handler


def test_redact_secrets_telegram_token():
    out = redact_secrets(f"HTTP Request: POST {FAKE_URL} 200 OK")
    assert FAKE_TOKEN not in out
    assert "bot<redacted>/sendMessage" in out


def test_redact_secrets_bearer_and_authorization():
    out = redact_secrets("Authorization: Bearer sk-fake-1234567890abcdef")
    assert "sk-fake-1234567890abcdef" not in out
    out = redact_secrets("retry with Bearer abc.def-ghi now")
    assert "abc.def-ghi" not in out
    assert "Bearer <redacted> now" in out


def test_handler_redacts_httpx_style_info_message():
    logger, handler = _capture()
    logger.info('HTTP Request: %s %s "HTTP/1.1 200 OK"', "POST", FAKE_URL)
    assert len(handler.lines) == 1
    assert FAKE_TOKEN not in handler.lines[0]
    assert "bot<redacted>" in handler.lines[0]


def test_handler_redacts_preformatted_msg():
    logger, handler = _capture()
    logger.info(f"calling {FAKE_URL}")
    assert FAKE_TOKEN not in handler.lines[0]


def test_filter_handles_non_string_args_and_msg():
    logger, handler = _capture()
    logger.info("count=%d obj=%s none=%s", 5, {"url": FAKE_URL}, None)
    logger.info(12345)
    logger.info("dict %(a)s", {"a": FAKE_URL})
    assert all(FAKE_TOKEN not in line for line in handler.lines)
    assert any("count=5" in line for line in handler.lines)
    assert any("12345" in line for line in handler.lines)


def test_filter_never_raises_on_weird_record():
    record = logging.LogRecord("x", logging.INFO, __file__, 1, object(), (object(),), None)
    assert RedactingFilter().filter(record) is True


def test_configure_logging_sets_httpx_to_warning(restore_logging):
    logging.getLogger("httpx").setLevel(logging.DEBUG)
    logging.getLogger("httpcore").setLevel(logging.DEBUG)
    _clean_root()
    configure_logging()
    assert logging.getLogger("httpx").level == logging.WARNING
    assert logging.getLogger("httpcore").level == logging.WARNING


def test_configure_logging_attaches_filter_to_root_handlers(restore_logging, capsys):
    _clean_root()
    configure_logging()
    root = logging.getLogger()
    assert root.handlers
    assert all(any(isinstance(f, RedactingFilter) for f in h.filters) for h in root.handlers)
    # Idempotent: calling twice does not stack filters.
    _clean_root()
    configure_logging()
    for h in root.handlers:
        assert sum(isinstance(f, RedactingFilter) for f in h.filters) == 1


def test_configure_logging_redacts_end_to_end(restore_logging, capsys):
    _clean_root()
    configure_logging()
    logging.getLogger("some.module").info("url=%s", FAKE_URL)
    err = capsys.readouterr().err
    assert "url=" in err
    assert FAKE_TOKEN not in err


def test_httpx_client_logs_no_token_after_configure(restore_logging, capsys):
    _clean_root()
    configure_logging()
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"ok": True}))
    with httpx.Client(transport=transport) as client:
        client.post(FAKE_URL, json={})
    captured = capsys.readouterr()
    assert FAKE_TOKEN not in captured.err + captured.out


def test_handler_redacts_lazy_percent_s_token_shape():
    # The token alone (no literal "bot" prefix) arrives as an arg; only the formatted message shows the URL.
    logger, handler = _capture()
    logger.info("https://api.telegram.org/bot%s/send", FAKE_TOKEN)
    assert len(handler.lines) == 1
    assert FAKE_TOKEN not in handler.lines[0]
    assert "bot<redacted>/send" in handler.lines[0]


def test_clean_message_left_unchanged_and_args_kept():
    logger, handler = _capture()
    logger.info("count=%d name=%s", 5, "acme")
    assert handler.lines == ["count=5 name=acme"]
    record = logging.LogRecord("x", logging.INFO, __file__, 1, "n=%d", (7,), None)
    RedactingFilter().filter(record)
    assert record.msg == "n=%d" and record.args == (7,)


def test_filter_survives_unformattable_message():
    record = logging.LogRecord("x", logging.INFO, __file__, 1, "bad %d", ("nope",), None)
    assert RedactingFilter().filter(record) is True


@pytest.mark.parametrize(
    "header",
    [
        "Authorization: Basic dXNlcjpwYXNzd29yZA==",
        "Authorization: Bearer sk-fake-1234567890abcdef",
        "Authorization: Token abc123secretvalue",
        "authorization=rawsecretvalue99",
        "{'Authorization': 'Basic dXNlcjpwYXNzd29yZA=='}",
    ],
)
def test_redact_secrets_whole_authorization_value(header):
    out = redact_secrets(header)
    for leaked in ("dXNlcjpwYXNzd29yZA", "sk-fake-1234567890abcdef", "abc123secretvalue", "rawsecretvalue99"):
        assert leaked not in out
    assert "<redacted>" in out
