"""In-memory Telegram Bot API for tests: an ``httpx.MockTransport`` that records every request."""
import json
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx


class FakeTelegram:
    def __init__(self) -> None:
        self.requests: List[Tuple[str, dict]] = []
        self._queue: List[List[dict]] = []
        self._overrides: Dict[str, List[Tuple[int, Any]]] = {}
        self._next_message_id = 100
        self.transport = httpx.MockTransport(self._handle)

    def queue_updates(self, batch: List[dict]) -> None:
        """The next ``getUpdates`` call returns ``batch``."""
        self._queue.append(batch)

    def respond(self, method: str, status: int, body: Any) -> None:
        """The next call of ``method`` answers with this status and JSON body (FIFO per method)."""
        self._overrides.setdefault(method, []).append((status, body))

    def calls(self, method: str) -> List[dict]:
        return [body for name, body in self.requests if name == method]

    def _handle(self, request: httpx.Request) -> httpx.Response:
        method = request.url.path.rsplit("/", 1)[-1]
        body = json.loads(request.content.decode("utf-8")) if request.content else {}
        self.requests.append((method, body))
        pending = self._overrides.get(method)
        if pending:
            status, payload = pending.pop(0)
            return httpx.Response(status, json=payload)
        if method == "getUpdates":
            batch = self._queue.pop(0) if self._queue else []
            return httpx.Response(200, json={"ok": True, "result": batch})
        if method == "sendMessage":
            self._next_message_id += 1
            return httpx.Response(200, json={"ok": True, "result": {"message_id": self._next_message_id}})
        if method == "getMe":
            return httpx.Response(200, json={"ok": True, "result": {"id": 1, "is_bot": True, "username": "fake_bot"}})
        return httpx.Response(200, json={"ok": True, "result": True})


def message_update(
    update_id: int,
    text: str,
    *,
    user_id: int = 111,
    chat_id: int = 111,
    chat_type: str = "private",
    date: Optional[int] = None,
) -> dict:
    return {
        "update_id": update_id,
        "message": {
            "message_id": update_id,
            "date": int(time.time()) if date is None else date,
            "from": {"id": user_id, "is_bot": False, "first_name": "<NAME>"},
            "chat": {"id": chat_id, "type": chat_type},
            "text": text,
        },
    }


def callback_update(
    update_id: int,
    data: str,
    *,
    user_id: int = 111,
    chat_id: int = 111,
    chat_type: str = "private",
    message_id: int = 5,
) -> dict:
    return {
        "update_id": update_id,
        "callback_query": {
            "id": f"cb{update_id}",
            "from": {"id": user_id, "is_bot": False, "first_name": "<NAME>"},
            "data": data,
            "message": {
                "message_id": message_id,
                "date": int(time.time()),
                "chat": {"id": chat_id, "type": chat_type},
            },
        },
    }
