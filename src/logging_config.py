"""Central logging setup: quiet HTTP client loggers and redact secrets from log output."""

import logging
import re
from typing import Any

# Telegram bot API URLs embed the token: https://api.telegram.org/bot<id>:<secret>/method
_TELEGRAM_TOKEN_RE = re.compile(r"bot\d+:[A-Za-z0-9_-]{20,}")
# "Bearer <token>" style credentials (also covers "Authorization: Bearer <token>").
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")
# Whole "Authorization: <scheme> <credential>" value (any scheme) up to end of line or closing quote.
_AUTH_HEADER_RE = re.compile(r"(?i)(\bAuthorization\b['\"]?\s*[:=]\s*['\"]?)[^\r\n'\"]+")

_NOISY_LOGGERS = ("httpx", "httpcore")


def redact_secrets(text: str) -> str:
    """Return ``text`` with Telegram bot tokens and bearer/authorization secrets masked."""
    text = _TELEGRAM_TOKEN_RE.sub("bot<redacted>", text)
    text = _BEARER_RE.sub("Bearer <redacted>", text)
    text = _AUTH_HEADER_RE.sub(r"\1<redacted>", text)
    return text


def _redact_arg(arg: Any) -> Any:
    if isinstance(arg, str):
        return redact_secrets(arg)
    if isinstance(arg, (int, float, bool, type(None))):
        return arg
    try:
        rendered = str(arg)
    except Exception:
        return arg
    redacted = redact_secrets(rendered)
    return arg if redacted == rendered else redacted


class RedactingFilter(logging.Filter):
    """Masks secrets in a record's message and args. Never raises, never drops records."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            if isinstance(record.msg, str):
                record.msg = redact_secrets(record.msg)
            elif record.msg is not None:
                record.msg = redact_secrets(str(record.msg))
            if isinstance(record.args, dict):
                record.args = {k: _redact_arg(v) for k, v in record.args.items()}
            elif isinstance(record.args, tuple):
                record.args = tuple(_redact_arg(a) for a in record.args)
        except Exception:
            pass
        # A secret can span msg and args (e.g. "bot%s" % token), so redact the formatted result too.
        try:
            message = record.getMessage()
            redacted = redact_secrets(message)
            if redacted != message:
                record.msg = redacted
                record.args = None
        except Exception:
            pass
        return True


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging once for the CLI, the daemon and scripts."""
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    for handler in logging.getLogger().handlers:
        if not any(isinstance(f, RedactingFilter) for f in handler.filters):
            handler.addFilter(RedactingFilter())
