"""Dedicated LLM path for the Telegram chat (free-text requests).

Drives the evaluation chain's own provider instances (so limiters and the OpenRouter daily
budget stay shared and per-process limits stay true) but never consults or updates their
circuit breakers, and its stop event is tied to process shutdown only (never the batch
evaluation stop event). Chat requests take rate-limit slots with priority.

Privacy: prompts and responses are never logged; only lengths, the request id and the
exception type name of a failure.
"""

import datetime as dt
import json
import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Deque, Dict, Optional, Sequence, Tuple, TypeVar

import pydantic

from src.evaluator.errors import (
    ConfigurationError,
    EvaluatorError,
    InvalidResponse,
    ProviderTimeout,
    RateLimited,
    RateLimitWaitExceeded,
    ServerError,
    StopRequested,
    UpstreamErrorEnvelope,
)
from src.evaluator.providers import CompletionOptions

logger = logging.getLogger(__name__)

T = TypeVar("T")

CHAT_TIMEOUT_SECONDS = 120.0
CHAT_MAX_ATTEMPTS = 1
CHAT_HOURLY_CAP = 60
CHAT_OPENROUTER_DAILY_SHARE = 15
CHAT_LIMITER_MAX_WAIT_SECONDS = 30.0
CHAT_MAX_TOKENS = 8192
CHAT_DRAFT_TIMEOUT_SECONDS = 180.0
OPENROUTER_RESERVE_SECONDS = 60.0
MIN_RETRY_ATTEMPT_SECONDS = 30.0

# kind -> (per-attempt timeout, overall deadline); ``chat`` is the free-text translator (its attempt timeout
# is the ChatLLM ``timeout`` argument and its deadline twice that).
_PROFILES: Dict[str, Tuple[float, float]] = {
    "turn": (90.0, 150.0),
    "extract": (90.0, 150.0),
    "draft": (CHAT_DRAFT_TIMEOUT_SECONDS, 300.0),
}
CHAT_KINDS = ("chat", "turn", "extract", "draft")


class ChatFailure(str, Enum):
    """Why a chat LLM request failed; selects the code-written reply text (never LLM prose)."""

    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    UNAVAILABLE = "unavailable"
    BAD_ANSWER = "bad_answer"
    CAPPED = "capped"


# Highest wins when providers fail differently.
_PRECEDENCE = {
    ChatFailure.CAPPED: 5,
    ChatFailure.BAD_ANSWER: 4,
    ChatFailure.TIMEOUT: 3,
    ChatFailure.RATE_LIMITED: 2,
    ChatFailure.UNAVAILABLE: 1,
}

_FAILURE_HEADS = {
    ChatFailure.TIMEOUT: "The model timed out",
    ChatFailure.RATE_LIMITED: "The model is rate-limited",
    ChatFailure.UNAVAILABLE: "No model is reachable",
    ChatFailure.BAD_ANSWER: "The model's answer could not be used",
    ChatFailure.CAPPED: "Too many model requests this hour",
}


_DAILY_CAPPED_HEAD = "Too many model requests today"


def describe_failure(failure: ChatFailure, retry_in: Optional[float] = None, daily: bool = False) -> str:
    """Code-written reason text with a retry hint; callers add their own next-step sentence.

    ``daily`` (only meaningful for ``CAPPED``) names the OpenRouter daily share instead of the hourly cap;
    waits of more than two hours are given in hours (the daily share frees up at UTC midnight).
    """
    head = _DAILY_CAPPED_HEAD if daily and failure is ChatFailure.CAPPED else _FAILURE_HEADS[failure]
    if retry_in is None:
        hint = "try again in a minute"
    else:
        minutes = max(1, math.ceil(retry_in / 60))
        hint = f"try again in {math.ceil(retry_in / 3600)} h" if minutes > 120 else f"try again in {minutes} min"
    return f"{head}; {hint}."


class ChatLLMError(Exception):
    """Base class for chat LLM failures a caller maps to a user message."""

    failure: ChatFailure = ChatFailure.UNAVAILABLE

    def __init__(self, message: str = "", *, failure: Optional[ChatFailure] = None, retry_in: Optional[float] = None):
        super().__init__(message)
        if failure is not None:
            self.failure = failure
        self.retry_in = retry_in


class ChatLLMNotConfigured(ChatLLMError):
    """No LLM provider is configured (no key in ``.env``)."""


class ChatLLMUnavailable(ChatLLMError):
    """Every provider was skipped or failed provider-wide (quota, auth, 5xx, 429, limiter wait)."""


class ChatLLMNotUnderstood(ChatLLMError):
    """A provider timed out or answered invalid / schema-failing JSON and none succeeded."""

    failure = ChatFailure.BAD_ANSWER


class ChatLLMLimited(ChatLLMError):
    """The hourly cap (or the OpenRouter daily share) is spent; ``retry_in`` is the seconds until a slot frees up."""

    failure = ChatFailure.CAPPED

    def __init__(self, retry_in: float, daily: bool = False):
        super().__init__("chat llm limit reached", retry_in=retry_in)
        self.daily = daily  # True: the OpenRouter share of the UTC day, not the hourly cap


def _classify(exc: EvaluatorError) -> ChatFailure:
    if isinstance(exc, ProviderTimeout):
        return ChatFailure.TIMEOUT
    if isinstance(exc, (InvalidResponse, UpstreamErrorEnvelope)):
        return ChatFailure.BAD_ANSWER
    if isinstance(exc, RateLimited):  # includes RateLimitWaitExceeded
        return ChatFailure.RATE_LIMITED
    return ChatFailure.UNAVAILABLE  # quota, auth, ServerError, BadRequest, ModelNotFound, transport, ...


def _error_for(failure: ChatFailure, retry_in: Optional[float], daily: bool = False) -> ChatLLMError:
    if failure is ChatFailure.CAPPED:
        return ChatLLMLimited(retry_in if retry_in is not None else 60.0, daily=daily)
    if failure in (ChatFailure.BAD_ANSWER, ChatFailure.TIMEOUT):
        return ChatLLMNotUnderstood(f"no provider gave an answer that could be used ({failure.value})", failure=failure, retry_in=retry_in)
    return ChatLLMUnavailable(f"no provider is available ({failure.value})", failure=failure, retry_in=retry_in)


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
        clock: Callable[[], float] = time.monotonic,
        now_fn: Callable[[], float] = time.time,
    ):
        self._providers_fn = providers_fn
        self._stop_event = stop_event
        self._clock = clock
        self._now = now_fn
        self._cap = cap if cap is not None else HourlyCap(time_fn=clock)
        self._timeout = timeout
        self._share_lock = threading.Lock()
        self._share_day: Optional[dt.date] = None
        self._share_used = 0

    def _providers(self) -> Sequence[Any]:
        try:
            return list(self._providers_fn())
        except ConfigurationError:
            return []

    def available(self) -> bool:
        return bool(self._providers())

    def _profile(self, kind: str) -> Tuple[float, float]:
        if kind == "chat":
            return self._timeout, self._timeout * 2
        try:
            return _PROFILES[kind]
        except KeyError:
            raise ValueError(f"unknown chat llm kind: {kind!r}") from None

    # ----- OpenRouter's share of the daily budget -----

    def _utc_now(self) -> dt.datetime:
        return dt.datetime.fromtimestamp(self._now(), dt.timezone.utc)

    def _share_spent(self) -> Optional[float]:
        """None while the OpenRouter daily share has room, else the seconds until the next UTC midnight."""
        now = self._utc_now()
        with self._share_lock:
            if self._share_day != now.date():
                self._share_day = now.date()
                self._share_used = 0
            if self._share_used < CHAT_OPENROUTER_DAILY_SHARE:
                return None
        tomorrow = now.date() + dt.timedelta(days=1)
        midnight = dt.datetime(tomorrow.year, tomorrow.month, tomorrow.day, tzinfo=dt.timezone.utc)
        return max(1.0, (midnight - now).total_seconds())

    def _count_share(self) -> None:
        with self._share_lock:
            self._share_used += 1

    def complete(
        self,
        messages: Sequence[Dict[str, str]],
        *,
        validate: Callable[[Dict[str, Any]], T],
        request_id: str,
        kind: str = "chat",
    ) -> ChatAnswer:
        per_attempt, overall = self._profile(kind)
        providers = self._providers()
        if not providers:
            raise ChatLLMNotConfigured("no LLM provider is configured")
        deadline = self._clock() + overall
        chars_in = sum(len(m.get("content", "")) for m in messages)
        worst: Optional[ChatFailure] = None
        retry_in: Optional[float] = None
        worst_daily = False
        capped = False

        def note(failure: ChatFailure, hint: Optional[float] = None, daily: bool = False) -> None:
            nonlocal worst, retry_in, worst_daily
            if worst is None or _PRECEDENCE[failure] > _PRECEDENCE[worst]:
                worst, retry_in, worst_daily = failure, hint, daily
            elif failure is worst and hint is not None:
                if retry_in is None or hint > retry_in:
                    retry_in, worst_daily = hint, daily

        for provider in providers:
            if capped:
                break
            if self._stop_event.is_set():
                raise StopRequested("stop requested (chat llm)")
            if not provider.is_available():
                continue
            may_retry = provider.name == "nim"
            plan: Optional[Tuple[float, float]] = None  # (timeout, limiter wait) of the retry attempt
            while True:
                remaining = deadline - self._clock()
                if plan is not None:
                    attempt_timeout, limiter_wait = plan
                else:
                    attempt_timeout = min(per_attempt, remaining)
                    limiter_wait = min(CHAT_LIMITER_MAX_WAIT_SECONDS, remaining)
                if attempt_timeout <= 0:
                    note(ChatFailure.TIMEOUT)  # the overall deadline is spent
                    break
                if provider.name == "openrouter":
                    share_in = self._share_spent()
                    if share_in is not None:
                        logger.warning("chat llm %s via %s skipped: %s", request_id, provider.name, ChatFailure.CAPPED.value)
                        note(ChatFailure.CAPPED, share_in, daily=True)
                        break
                slot_in = self._cap.try_take()
                if slot_in is not None:
                    logger.warning("chat llm %s via %s skipped: %s", request_id, provider.name, ChatFailure.CAPPED.value)
                    note(ChatFailure.CAPPED, slot_in)
                    capped = True  # the hourly cap is shared by every provider
                    break
                if provider.name == "openrouter":
                    self._count_share()
                options = CompletionOptions(
                    timeout=attempt_timeout,
                    max_attempts=CHAT_MAX_ATTEMPTS,
                    stop_event=self._stop_event,
                    priority=True,
                    max_tokens=CHAT_MAX_TOKENS,
                    limiter_max_wait=limiter_wait,
                )
                started = self._clock()
                try:
                    completion = provider.complete_json(messages, options)
                except StopRequested:
                    raise
                except EvaluatorError as exc:
                    failure = _classify(exc)
                    status = getattr(exc, "status", None)
                    logger.warning(
                        "chat llm %s via %s failed: %s%s",
                        request_id,
                        provider.name,
                        failure.value,
                        f" (HTTP {status})" if status is not None else "",
                    )
                    hint = exc.retry_after if isinstance(exc, RateLimited) else None
                    note(failure, hint)
                    if may_retry and isinstance(exc, (ProviderTimeout, ServerError, RateLimited)) and not isinstance(exc, RateLimitWaitExceeded):
                        may_retry = False
                        plan = self._retry_plan(exc, per_attempt, deadline)
                        if plan is not None:
                            continue
                    break
                try:
                    value = validate(completion.data)
                except (ValueError, pydantic.ValidationError):
                    logger.warning("chat llm %s via %s rejected the answer: %s", request_id, provider.name, ChatFailure.BAD_ANSWER.value)
                    note(ChatFailure.BAD_ANSWER)
                    break
                logger.info(
                    "chat llm %s via %s: %d chars in, %d chars out, %.1fs",
                    request_id,
                    provider.name,
                    chars_in,
                    len(json.dumps(completion.data)),
                    self._clock() - started,
                )
                return ChatAnswer(value=value, provider=provider.name, model_used=completion.model_used, latency=completion.latency)
        raise _error_for(worst if worst is not None else ChatFailure.UNAVAILABLE, retry_in, worst_daily)

    def _retry_plan(self, exc: EvaluatorError, per_attempt: float, deadline: float) -> Optional[Tuple[float, float]]:
        """(attempt timeout, limiter wait) of the one extra NIM attempt, or None when it must not happen.

        The fallback keeps ``OPENROUTER_RESERVE_SECONDS`` of the deadline, so a retry needs a clipped timeout of at
        least ``MIN_RETRY_ATTEMPT_SECONDS``; a 429 whose ``Retry-After`` exceeds the wait bound is not retried.
        """
        spare = deadline - self._clock() - OPENROUTER_RESERVE_SECONDS
        timeout = min(per_attempt, spare)
        if timeout < MIN_RETRY_ATTEMPT_SECONDS:
            return None
        bound = min(CHAT_LIMITER_MAX_WAIT_SECONDS, spare)
        if isinstance(exc, RateLimited) and exc.retry_after is not None and exc.retry_after > bound:
            return None
        return timeout, bound
