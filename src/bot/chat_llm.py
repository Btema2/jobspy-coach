"""Dedicated LLM path for the Telegram chat (free-text requests).

Drives the evaluation chain's own provider instances (so limiters and the OpenRouter daily
budget stay shared and per-process limits stay true) but never consults or updates their
circuit breakers, and its stop event is tied to process shutdown only (never the batch
evaluation stop event). Chat requests take rate-limit slots with priority.

Privacy: prompts and responses are never logged; only lengths, the request id and the
exception type name of a failure.
"""

import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, Optional, Sequence, TypeVar

import pydantic

from src.evaluator.errors import (
    ConfigurationError,
    EvaluatorError,
    InvalidResponse,
    ProviderTimeout,
    StopRequested,
)
from src.evaluator.providers import CompletionOptions

logger = logging.getLogger(__name__)

T = TypeVar("T")

CHAT_TIMEOUT_SECONDS = 120.0
CHAT_MAX_ATTEMPTS = 1
CHAT_HOURLY_CAP = 30
CHAT_LIMITER_MAX_WAIT_SECONDS = 30.0
CHAT_MAX_TOKENS = 8192


class ChatLLMError(Exception):
    """Base class for chat LLM failures a caller maps to a user message."""


class ChatLLMNotConfigured(ChatLLMError):
    """No LLM provider is configured (no key in ``.env``)."""


class ChatLLMUnavailable(ChatLLMError):
    """Every provider was skipped or failed provider-wide (quota, auth, 5xx, 429, limiter wait)."""


class ChatLLMNotUnderstood(ChatLLMError):
    """A provider timed out or answered invalid / schema-failing JSON and none succeeded."""


class ChatLLMLimited(ChatLLMError):
    """The hourly cap is spent; ``retry_in`` is the seconds until a slot frees up."""

    def __init__(self, retry_in: float):
        super().__init__("hourly chat llm limit reached")
        self.retry_in = retry_in


@dataclass
class ChatAnswer:
    value: Any
    provider: str
    model_used: str
    latency: float


class HourlyCap:
    """Sliding-window counter of provider attempts (per process)."""

    def __init__(self, limit: int = CHAT_HOURLY_CAP, period: float = 3600.0, time_fn: Callable[[], float] = time.monotonic):
        self.limit = limit
        self.period = period
        self._time = time_fn
        self._lock = threading.Lock()
        self._stamps: Deque[float] = deque()

    def try_take(self) -> Optional[float]:
        """None when a slot was taken, else the seconds until the oldest attempt leaves the window."""
        with self._lock:
            now = self._time()
            while self._stamps and self._stamps[0] + self.period <= now:
                self._stamps.popleft()
            if len(self._stamps) < self.limit:
                self._stamps.append(now)
                return None
            return self._stamps[0] + self.period - now


class ChatLLM:
    def __init__(
        self,
        providers_fn: Callable[[], Sequence[Any]],
        *,
        stop_event: threading.Event,
        cap: Optional[HourlyCap] = None,
        timeout: float = CHAT_TIMEOUT_SECONDS,
    ):
        self._providers_fn = providers_fn
        self._stop_event = stop_event
        self._cap = cap if cap is not None else HourlyCap()
        self._timeout = timeout

    def _providers(self) -> Sequence[Any]:
        try:
            return list(self._providers_fn())
        except ConfigurationError:
            return []

    def available(self) -> bool:
        return bool(self._providers())

    def complete(
        self,
        messages: Sequence[Dict[str, str]],
        *,
        validate: Callable[[Dict[str, Any]], T],
        request_id: str,
    ) -> ChatAnswer:
        providers = self._providers()
        if not providers:
            raise ChatLLMNotConfigured("no LLM provider is configured")
        options = CompletionOptions(
            timeout=self._timeout,
            max_attempts=CHAT_MAX_ATTEMPTS,
            stop_event=self._stop_event,
            priority=True,
            max_tokens=CHAT_MAX_TOKENS,
            limiter_max_wait=CHAT_LIMITER_MAX_WAIT_SECONDS,
        )
        chars_in = sum(len(m.get("content", "")) for m in messages)
        not_understood = False
        attempted = False
        for provider in providers:
            if self._stop_event.is_set():
                raise StopRequested("stop requested (chat llm)")
            if not provider.is_available():
                continue
            retry_in = self._cap.try_take()
            if retry_in is not None:
                if not attempted:
                    raise ChatLLMLimited(retry_in)
                break
            attempted = True
            started = time.monotonic()
            try:
                completion = provider.complete_json(messages, options)
            except StopRequested:
                raise
            except (ProviderTimeout, InvalidResponse) as exc:
                logger.warning("chat llm %s via %s failed: %s", request_id, provider.name, type(exc).__name__)
                not_understood = True
                continue
            except EvaluatorError as exc:
                logger.warning("chat llm %s via %s failed: %s", request_id, provider.name, type(exc).__name__)
                continue
            try:
                value = validate(completion.data)
            except (ValueError, pydantic.ValidationError) as exc:
                logger.warning("chat llm %s via %s rejected the answer: %s", request_id, provider.name, type(exc).__name__)
                not_understood = True
                continue
            logger.info(
                "chat llm %s via %s: %d chars in, %d chars out, %.1fs",
                request_id,
                provider.name,
                chars_in,
                len(json.dumps(completion.data)),
                time.monotonic() - started,
            )
            return ChatAnswer(value=value, provider=provider.name, model_used=completion.model_used, latency=completion.latency)
        if not_understood:
            raise ChatLLMNotUnderstood("no provider gave an answer that could be used")
        raise ChatLLMUnavailable("no provider is available")
