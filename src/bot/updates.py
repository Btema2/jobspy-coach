"""Telegram updates as plain data: parsing never raises and never keeps more than the bot needs."""
import re
from dataclasses import dataclass
from typing import Any, Optional

_COMMAND_RE = re.compile(r"/([A-Za-z0-9_]{1,32})(?:@[A-Za-z0-9_]{1,64})?(?=\s|$)")
# Logged on rejected updates, so only values from this closed set are ever kept.
_CHAT_TYPES = {"private", "group", "supergroup", "channel", "sender"}


@dataclass(frozen=True)
class Update:
    update_id: int
    kind: str  # "message", "callback" or "other"
    user_id: Optional[int] = None
    chat_id: Optional[int] = None
    chat_type: Optional[str] = None
    date: Optional[int] = None  # message date; None for callbacks
    text: str = ""  # message text, or callback data
    command: Optional[str] = None  # lower-cased "/name" (an "@botname" suffix is dropped); None otherwise
    args: str = ""
    callback_id: Optional[str] = None
    message_id: Optional[int] = None  # the message a callback button sits on


@dataclass(frozen=True)
class OutgoingMessage:
    text: str
    reply_markup: Optional[dict] = None
    edit_message_id: Optional[int] = None


def _int(value: Any) -> Optional[int]:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _chat_type(chat: dict) -> Optional[str]:
    value = chat.get("type")
    if not isinstance(value, str):
        return None
    return value if value in _CHAT_TYPES else "unknown"


def _split_command(text: str):
    if not text.startswith("/"):
        return None, ""
    match = _COMMAND_RE.match(text)
    if match is None:
        return None, ""
    return "/" + match.group(1).lower(), text[match.end():].strip()


def parse_update(raw: dict) -> Update:
    try:
        return _parse(raw)
    except Exception:  # a hostile payload must never take the poller down
        return Update(update_id=-1, kind="other")


def _parse(raw: Any) -> Update:
    raw = _dict(raw)
    update_id = _int(raw.get("update_id"))
    update_id = update_id if update_id is not None and update_id >= 0 else -1

    message = raw.get("message")
    if isinstance(message, dict):
        chat = _dict(message.get("chat"))
        text = _text(message.get("text"))
        command, args = _split_command(text)
        return Update(
            update_id=update_id,
            kind="message",
            user_id=_int(_dict(message.get("from")).get("id")),
            chat_id=_int(chat.get("id")),
            chat_type=_chat_type(chat),
            date=_int(message.get("date")),
            text=text,
            command=command,
            args=args,
            message_id=_int(message.get("message_id")),
        )

    callback = raw.get("callback_query")
    if isinstance(callback, dict) and isinstance(callback.get("id"), str):
        button_message = _dict(callback.get("message"))
        chat = _dict(button_message.get("chat"))
        return Update(
            update_id=update_id,
            kind="callback",
            user_id=_int(_dict(callback.get("from")).get("id")),
            chat_id=_int(chat.get("id")),
            chat_type=_chat_type(chat),
            text=_text(callback.get("data")),
            callback_id=callback["id"],
            message_id=_int(button_message.get("message_id")),
        )

    return Update(update_id=update_id, kind="other")
