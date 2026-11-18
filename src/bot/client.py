"""Minimal Telegram Bot API client for the bot (long polling and replies).

Token hygiene (AGENTS.md 3.D.4): the bot token is part of every request URL, so nothing here ever logs or
raises the URL or ``str()`` of an httpx exception; error text goes through ``redact_secrets``.
"""
import html
import logging
import re
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx

from src.logging_config import redact_secrets

logger = logging.getLogger(__name__)

API_BASE = "https://api.telegram.org"
_MAX_DESCRIPTION = 200
_MAX_SHORT_RETRY_AFTER = 30.0
_DEFAULT_RETRY_AFTER = 5.0


class TelegramApiError(Exception):
    def __init__(self, method: str, status: int, description: str):
        self.method = method
        self.status = status
        self.description = redact_secrets(description)[:_MAX_DESCRIPTION]
        super().__init__(f"{method} {status}: {self.description}")


class TelegramUnauthorized(TelegramApiError):
    """401: the bot token is invalid or was revoked."""


class TelegramConflict(TelegramApiError):
    """409: another consumer is polling the same bot (or a webhook is set)."""


class TelegramRetryAfter(TelegramApiError):
    """429: flood control; ``retry_after`` seconds from ``parameters.retry_after`` (default 5)."""

    def __init__(self, method: str, status: int, description: str, retry_after: float = _DEFAULT_RETRY_AFTER):
        super().__init__(method, status, description)
        self.retry_after = retry_after


class TelegramNetworkError(TelegramApiError):
    """Transport error or timeout; the message is the exception type name only (httpx text embeds the URL)."""

    def __init__(self, method: str, error_name: str):
        super().__init__(method, 0, error_name)

    def __str__(self) -> str:
        return self.description


def _retry_after_seconds(body: Any) -> float:
    parameters = body.get("parameters") if isinstance(body, dict) else None
    value = parameters.get("retry_after") if isinstance(parameters, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        return _DEFAULT_RETRY_AFTER
    return value


class TelegramBotClient:
    def __init__(
        self,
        token: str,
        *,
        transport: Optional[httpx.BaseTransport] = None,
        timeout: float = 15.0,
        base_url: str = API_BASE,
        stop_event: Optional[threading.Event] = None,
    ):
        self._token = token
        self._stop = stop_event
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(transport=transport, timeout=timeout)

    def close(self) -> None:
        if not self._client.is_closed:
            self._client.close()

    def call(self, method: str, payload: Dict[str, Any], *, timeout: Optional[float] = None) -> Any:
        url = f"{self._base_url}/bot{self._token}/{method}"
        kwargs: Dict[str, Any] = {"json": payload}
        if timeout is not None:
            kwargs["timeout"] = timeout
        try:
            response = self._client.post(url, **kwargs)
        except (httpx.HTTPError, httpx.InvalidURL, httpx.StreamError) as exc:
            # No chaining: the httpx message (and a traceback showing it) embeds the URL with the token.
            raise TelegramNetworkError(method, type(exc).__name__) from None

        try:
            body = response.json()
        except ValueError:
            body = None
        if response.status_code < 400 and isinstance(body, dict) and body.get("ok"):
            return body.get("result")

        status = response.status_code
        if status < 400:  # HTTP 200 with ok false
            code = body.get("error_code") if isinstance(body, dict) else None
            status = code if isinstance(code, int) and code >= 400 else 502
        description = body.get("description") if isinstance(body, dict) else None
        text = str(description) if description else f"HTTP {response.status_code}"
        if self._token:
            text = text.replace(self._token, "<redacted>")
        if status == 401:
            raise TelegramUnauthorized(method, status, text)
        if status == 409:
            raise TelegramConflict(method, status, text)
        if status == 429:
            raise TelegramRetryAfter(method, status, text, _retry_after_seconds(body))
        raise TelegramApiError(method, status, text)

    def get_me(self) -> dict:
        return self.call("getMe", {})

    def get_updates(self, offset: int, timeout: int = 25) -> List[dict]:
        result = self.call(
            "getUpdates",
            {"offset": offset, "timeout": timeout, "allowed_updates": ["message", "callback_query"]},
            timeout=timeout + 10,
        )
        return result if isinstance(result, list) else []

    @staticmethod
    def _log_failure(error: TelegramApiError) -> None:
        if isinstance(error, TelegramNetworkError):
            logger.warning("Telegram %s failed: %s", error.method, error.description)
        else:
            logger.warning("Telegram %s %s", error.method, error.status)

    def send_message(self, chat_id: int, text: str, reply_markup: Optional[dict] = None) -> Optional[int]:
        """Send an HTML message; never raises. Returns the message id, or None when it was not delivered."""
        payload: Dict[str, Any] = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
            "link_preview_options": {"is_disabled": True},
        }
        if reply_markup:
            payload["reply_markup"] = reply_markup
        retried = False
        plain = False
        while True:
            try:
                result = self.call("sendMessage", payload)
            except TelegramRetryAfter as exc:
                if not retried and exc.retry_after <= _MAX_SHORT_RETRY_AFTER:
                    retried = True
                    if self._stop is not None:
                        if self._stop.wait(exc.retry_after):  # shutting down: do not delay it, do not retry
                            return None
                    else:
                        time.sleep(exc.retry_after)
                    continue
                self._log_failure(exc)
                return None
            except TelegramApiError as exc:
                if exc.status == 400 and "can't parse entities" in exc.description.lower() and not plain:
                    plain = True
                    payload = dict(payload)
                    payload.pop("parse_mode", None)
                    payload["text"] = html.unescape(re.sub("<[^>]+>", "", text))
                    continue
                self._log_failure(exc)
                return None
            message_id = result.get("message_id") if isinstance(result, dict) else None
            return message_id if isinstance(message_id, int) else None

    def _call_bool(self, method: str, payload: Dict[str, Any]) -> bool:
        try:
            self.call(method, payload)
        except TelegramApiError as exc:
            self._log_failure(exc)
            return False
        return True

    def edit_message_text(
        self, chat_id: int, message_id: int, text: str, reply_markup: Optional[dict] = None
    ) -> bool:
        payload: Dict[str, Any] = {
            "chat_id": chat_id,
            "message_id": message_id,
            "text": text,
            "parse_mode": "HTML",
            "link_preview_options": {"is_disabled": True},
        }
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup
        return self._call_bool("editMessageText", payload)

    def answer_callback_query(self, callback_query_id: str, text: Optional[str] = None) -> bool:
        payload: Dict[str, Any] = {"callback_query_id": callback_query_id}
        if text:
            payload["text"] = text
        return self._call_bool("answerCallbackQuery", payload)

    def set_my_commands(self, commands: Sequence[Tuple[str, str]]) -> bool:
        payload = {"commands": [{"command": c, "description": d} for c, d in commands]}
        return self._call_bool("setMyCommands", payload)
