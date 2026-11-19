"""OpenAI-compatible chat-completion providers (NVIDIA NIM, OpenRouter).

Responsibilities of this layer: one logical ``complete(system, user)`` call made of
at most ``MAX_HTTP_ATTEMPTS`` HTTP attempts, each gated by the provider's
rate limiter, with retry/backoff and error classification (plan section 3.4).

Circuit-breaker accounting is NOT done here. Providers only raise typed errors
(``AuthenticationFailed``, ``ModelNotFound``, ``BadRequest``, ``RateLimited`` /
``TransientProviderError``, ``QuotaExhausted``, ``InvalidResponse``); the chain
(``src/evaluator/chain.py``) is the single place that settles each provider's
``breaker`` (record_success / record_failure / open_immediately / release_probe).
Each provider merely owns its ``breaker`` instance so the chain can reach it.

Secrets: the API key is only ever placed in the Authorization header and is never
logged. Neither are prompts or response bodies.
"""

import datetime as dt
import logging
import random
import re
import threading
import time
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Dict, List, Optional, Sequence

import httpx

from src.evaluator.circuit_breaker import CircuitBreaker
from src.evaluator.errors import (
    AuthenticationFailed,
    BadRequest,
    InvalidResponse,
    ModelNotFound,
    ProviderTimeout,
    QuotaExhausted,
    RateLimited,
    ResponseTruncated,
    StopRequested,
    TransientProviderError,
    UnexpectedStatus,
)
from src.evaluator.parsing import extract_json_payload
from src.evaluator.rate_limiter import SlidingWindowRateLimiter

logger = logging.getLogger(__name__)

MAX_HTTP_ATTEMPTS = 3
# Only these statuses blame the request itself (job-specific); any other unexpected status is provider-wide.
BAD_REQUEST_STATUSES = frozenset({400, 413, 422})
MAX_CUMULATIVE_SLEEP_SECONDS = 180.0
MAX_RETRY_AFTER_SECONDS = 120.0
RATE_LIMIT_BACKOFF_BASE_SECONDS = 5.0  # 429 without Retry-After: 5 * 2**n
TRANSIENT_BACKOFF_BASE_SECONDS = 2.0  # 5xx / timeouts: 2 * 3**n -> 2, 6, 18
JITTER_FRACTION = 0.25
# A rate-limit reset further away than this is a quota window (daily), not a burst limit.
QUOTA_RESET_MIN_SECONDS = 300.0

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_THINK_CLOSE = re.compile(r"</think>", re.IGNORECASE)
_THINK_OPEN = re.compile(r"<think>", re.IGNORECASE)


def strip_think(text: str) -> str:
    """Remove ``<think>`` reasoning from model output and trim whitespace.

    Handles closed blocks, an orphan closing tag (template pre-opened the block:
    keep what follows the last one) and an unterminated block (drop the rest).
    """
    text = _THINK_BLOCK.sub("", text)
    closes = list(_THINK_CLOSE.finditer(text))
    if closes:
        text = text[closes[-1].end():]
    opened = _THINK_OPEN.search(text)
    if opened:
        text = text[: opened.start()]
    return text.strip()


@dataclass
class CompletionResult:
    content: str
    model_used: str
    latency: float
    finish_reason: Optional[str] = None


@dataclass(frozen=True)
class CompletionOptions:
    """Per-call settings of ``complete_json`` (the chat path): one bounded request, its own stop event."""

    timeout: float
    max_attempts: int = 1
    stop_event: Optional[threading.Event] = None
    priority: bool = False
    max_tokens: Optional[int] = None
    limiter_max_wait: Optional[float] = None


@dataclass
class JsonCompletion:
    data: Dict[str, Any]
    model_used: str
    latency: float


_CHAT_ROLES = frozenset({"system", "user", "assistant"})


def _parse_retry_after(value: Optional[str], now: Callable[[], float]) -> Optional[float]:
    if not value:
        return None
    value = value.strip()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
        if when.tzinfo is None:
            when = when.replace(tzinfo=dt.timezone.utc)
        return max(0.0, when.timestamp() - now())
    except (TypeError, ValueError):
        return None


def _epoch_from_header(value: Any) -> Optional[float]:
    """Parse an epoch timestamp in seconds or milliseconds."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number / 1000.0 if number > 1e11 else number


class OpenAICompatibleProvider:
    """Base class: shared client, headers, per-attempt limiter, retry and classification."""

    name = "provider"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str,
        limiter: Optional[SlidingWindowRateLimiter] = None,
        rpm_limit: int = 30,
        breaker: Optional[CircuitBreaker] = None,
        stop_event: Optional[threading.Event] = None,
        timeout: float = 240.0,
        max_tokens: int = 4096,
        temperature: float = 0.6,
        workers: int = 8,
        client: Optional[httpx.Client] = None,
        transport: Optional[httpx.BaseTransport] = None,
        wait_fn: Optional[Callable[[float], object]] = None,
        rand_fn: Callable[[], float] = random.random,
        time_fn: Callable[[], float] = time.monotonic,
        now_fn: Callable[[], float] = time.time,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.stop_event = stop_event if stop_event is not None else threading.Event()
        self.limiter = limiter if limiter is not None else SlidingWindowRateLimiter(
            rpm_limit, 60.0, stop_event=self.stop_event
        )
        self.breaker = breaker if breaker is not None else CircuitBreaker()
        self._wait_fn = wait_fn
        self._wait = wait_fn if wait_fn is not None else (lambda seconds: self.stop_event.wait(seconds))
        self._rand = rand_fn
        self._time = time_fn
        self._now = now_fn
        self._client = client if client is not None else httpx.Client(
            timeout=httpx.Timeout(connect=10.0, read=timeout, write=30.0, pool=30.0),
            limits=httpx.Limits(max_connections=workers + 4),
            transport=transport,
        )

    # ----- hooks for subclasses -----

    def _build_messages_body(self, messages: Sequence[Dict[str, str]], max_tokens: int) -> Dict[str, Any]:
        raise NotImplementedError

    def _build_body(self, system: str, user: str, max_tokens: int) -> Dict[str, Any]:
        return self._build_messages_body(
            [{"role": "system", "content": system}, {"role": "user", "content": user}], max_tokens
        )

    def _extra_headers(self) -> Dict[str, str]:
        return {}

    def is_available(self) -> bool:
        """False while the provider is disabled for non-breaker reasons (e.g. spent quota)."""
        return True

    def _reserve_request(self) -> None:
        """Called once per HTTP attempt, after the limiter; may raise QuotaExhausted."""

    def _check_429(self, response: httpx.Response) -> None:
        """Inspect a 429 before the generic retry logic; may raise QuotaExhausted."""

    # ----- internals -----

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        headers.update(self._extra_headers())
        return headers

    def _jittered(self, seconds: float) -> float:
        return seconds * (1.0 + JITTER_FRACTION * self._rand())

    def _check_stop(self, stop_event: Optional[threading.Event] = None) -> None:
        event = stop_event if stop_event is not None else self.stop_event
        if event.is_set():
            raise StopRequested(f"stop requested ({self.name})")

    def _sleep(self, seconds: float, stop_event: Optional[threading.Event] = None) -> None:
        if stop_event is not None and self._wait_fn is None:
            stop_event.wait(seconds)
        else:
            self._wait(seconds)
        self._check_stop(stop_event)

    def _check_budget(self) -> None:
        """Raise ``QuotaExhausted`` before any request when the provider is disabled (hook, see OpenRouter)."""
        if not self.is_available():
            raise QuotaExhausted(message=f"{self.name} unavailable (quota)")

    def complete(self, system: str, user: str, max_tokens: Optional[int] = None) -> CompletionResult:
        self._check_budget()
        body = self._build_body(system, user, max_tokens or self.max_tokens)
        started, response = self._post_with_retries(body)
        return self._parse(response, self._time() - started)

    def complete_json(self, messages: Sequence[Dict[str, str]], options: CompletionOptions) -> JsonCompletion:
        """One chat-style call that must answer a JSON object (the chat path; never touches ``breaker``)."""
        for message in messages:
            if message.get("role") not in _CHAT_ROLES or not isinstance(message.get("content"), str):
                raise ValueError("messages need a system, user or assistant role and text content")
        self._check_budget()
        body = self._build_messages_body(messages, options.max_tokens or self.max_tokens)
        started, response = self._post_with_retries(
            body,
            max_attempts=options.max_attempts,
            stop_event=options.stop_event,
            timeout=options.timeout,
            priority=options.priority,
            limiter_max_wait=options.limiter_max_wait,
        )
        latency = self._time() - started
        result = self._parse(response, latency)
        try:
            data = extract_json_payload(result.content)
        except ValueError:
            raise InvalidResponse(f"{self.name}: reply is not JSON ({len(result.content)} chars)") from None
        if not isinstance(data, dict):
            raise InvalidResponse(f"{self.name}: reply is not a JSON object ({len(result.content)} chars)")
        return JsonCompletion(data=data, model_used=result.model_used, latency=latency)

    def _post_with_retries(
        self,
        body: Dict[str, Any],
        *,
        max_attempts: int = MAX_HTTP_ATTEMPTS,
        stop_event: Optional[threading.Event] = None,
        timeout: Optional[float] = None,
        priority: bool = False,
        limiter_max_wait: Optional[float] = None,
    ):
        url = f"{self.base_url}/chat/completions"
        chat_path = priority or stop_event is not None or limiter_max_wait is not None
        request_timeout = (
            httpx.Timeout(connect=10.0, read=timeout, write=30.0, pool=30.0) if timeout is not None else httpx.USE_CLIENT_DEFAULT
        )
        slept = 0.0
        attempt = 0
        while True:
            attempt += 1
            self._check_stop(stop_event)
            if chat_path:
                self.limiter.acquire(priority=priority, stop_event=stop_event, max_wait=limiter_max_wait)
            else:
                self.limiter.acquire()  # wraps EVERY HTTP attempt, retries included
            self._reserve_request()
            started = self._time()
            error: TransientProviderError
            delay: float
            try:
                response = self._client.post(url, json=body, headers=self._headers(), timeout=request_timeout)
            except httpx.TransportError as exc:
                logger.warning("%s transport error (%s) on attempt %d/%d", self.name, type(exc).__name__, attempt, max_attempts)
                error_type = ProviderTimeout if isinstance(exc, httpx.TimeoutException) else TransientProviderError
                error = error_type(f"{self.name}: {type(exc).__name__}")
                delay = self._jittered(TRANSIENT_BACKOFF_BASE_SECONDS * 3 ** (attempt - 1))
            else:
                status = response.status_code
                if status == 200:
                    return started, response
                if status == 429:
                    self._check_429(response)
                    retry_after = _parse_retry_after(response.headers.get("Retry-After"), self._now)
                    if retry_after is not None:
                        delay = min(retry_after, MAX_RETRY_AFTER_SECONDS)
                    else:
                        delay = self._jittered(RATE_LIMIT_BACKOFF_BASE_SECONDS * 2 ** (attempt - 1))
                    # All workers must back off, whether or not this call retries.
                    self.limiter.pause_for(delay)
                    logger.warning("%s HTTP 429 on attempt %d/%d (retry-after=%s)", self.name, attempt, max_attempts, retry_after)
                    error = RateLimited(retry_after, f"{self.name}: HTTP 429")
                elif status >= 500 or status == 408:
                    logger.warning("%s HTTP %d on attempt %d/%d", self.name, status, attempt, max_attempts)
                    error = TransientProviderError(f"{self.name}: HTTP {status}")
                    delay = self._jittered(TRANSIENT_BACKOFF_BASE_SECONDS * 3 ** (attempt - 1))
                elif status in (401, 403):
                    logger.error("%s rejected the API key (HTTP %d); check the configured key", self.name, status)
                    raise AuthenticationFailed(f"{self.name}: HTTP {status}")
                elif status == 404:
                    logger.error("%s model or endpoint not found (HTTP 404), model=%s", self.name, self.model)
                    raise ModelNotFound(f"{self.name}: HTTP 404 for model {self.model}")
                elif status in BAD_REQUEST_STATUSES:
                    logger.warning("%s rejected the request (HTTP %d)", self.name, status)
                    raise BadRequest(f"{self.name}: HTTP {status}")
                else:
                    # 402, 405, 409, 3xx, other 4xx: not the job's fault; the chain counts it against the provider.
                    logger.warning("%s answered with unexpected HTTP %d", self.name, status)
                    raise UnexpectedStatus(f"{self.name}: unexpected HTTP {status}")

            if attempt >= max_attempts or slept + delay > MAX_CUMULATIVE_SLEEP_SECONDS:
                raise error
            self._sleep(delay, stop_event)
            slept += delay

    def _parse(self, response: httpx.Response, latency: float) -> CompletionResult:
        try:
            data = response.json()
        except ValueError:
            raise InvalidResponse(f"{self.name}: response body is not JSON") from None
        if not isinstance(data, dict):
            raise InvalidResponse(f"{self.name}: unexpected response shape")
        choices = data.get("choices")
        if not choices:
            if data.get("error"):
                # HTTP 200 carrying an upstream error envelope (seen on OpenRouter).
                raise TransientProviderError(f"{self.name}: upstream error in HTTP 200 body")
            raise InvalidResponse(f"{self.name}: response has no choices")
        if not isinstance(choices, list):
            raise InvalidResponse(f"{self.name}: unexpected shape of choices")
        choice = choices[0] if isinstance(choices[0], dict) else {}
        message = choice.get("message") or {}
        finish_reason = choice.get("finish_reason")
        model_used = data.get("model") or self.model
        if finish_reason == "length":
            raise ResponseTruncated(f"{self.name}: output truncated by max_tokens")
        raw = message.get("content") if isinstance(message, dict) else None
        content = strip_think(raw) if isinstance(raw, str) else ""
        if not content:
            raise InvalidResponse(f"{self.name}: empty content")
        logger.debug("%s answered via %s in %.1fs", self.name, model_used, latency)
        return CompletionResult(content=content, model_used=model_used, latency=latency, finish_reason=finish_reason)


class NvidiaNimProvider(OpenAICompatibleProvider):
    """NVIDIA NIM hosted endpoint. Reasoning is controlled with ``reasoning_effort``.

    Deliberately sends no ``response_format`` (unsupported) and no
    ``thinking_token_budget`` (runs to max_tokens with empty content).
    """

    name = "nim"

    def __init__(self, *, reasoning_effort: str = "low", **kwargs: Any):
        super().__init__(**kwargs)
        self.reasoning_effort = reasoning_effort

    def _build_messages_body(self, messages: Sequence[Dict[str, str]], max_tokens: int) -> Dict[str, Any]:
        return {
            "model": self.model,
            "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            "reasoning_effort": self.reasoning_effort,
        }


class OpenRouterProvider(OpenAICompatibleProvider):
    """OpenRouter with pinned free models (``models`` array) and a local daily budget.

    The daily counter is in-memory, rolls over at 00:00 UTC and counts every HTTP
    attempt; a process restart resets it (accepted and documented).
    """

    name = "openrouter"

    def __init__(
        self,
        *,
        models: Sequence[str],
        daily_budget: int = 45,
        **kwargs: Any,
    ):
        models = [m for m in models if m]
        if not models:
            raise ValueError("OpenRouterProvider needs at least one pinned model")
        kwargs.setdefault("model", models[0])
        kwargs.setdefault("temperature", 0.3)
        kwargs.setdefault("timeout", 45.0)
        kwargs.setdefault("rpm_limit", 20)
        super().__init__(**kwargs)
        self.models: List[str] = list(models)
        self.daily_budget = daily_budget
        self._quota_lock = threading.Lock()
        self._day: Optional[dt.date] = None
        self._used = 0
        self._disabled_until = 0.0

    # ----- daily budget -----

    def _today(self) -> dt.date:
        return dt.datetime.fromtimestamp(self._now(), dt.timezone.utc).date()

    def _next_reset(self) -> float:
        tomorrow = self._today() + dt.timedelta(days=1)
        return dt.datetime(tomorrow.year, tomorrow.month, tomorrow.day, tzinfo=dt.timezone.utc).timestamp()

    def _roll_day(self) -> None:  # caller holds the lock
        today = self._today()
        if self._day != today:
            self._day = today
            self._used = 0

    def is_available(self) -> bool:
        with self._quota_lock:
            self._roll_day()
            return self._now() >= self._disabled_until and self._used < self.daily_budget

    def _reset_at(self) -> float:  # caller holds the lock
        if self._now() < self._disabled_until:
            return self._disabled_until
        return self._next_reset()

    def _reserve_request(self) -> None:
        with self._quota_lock:
            self._roll_day()
            if self._now() < self._disabled_until or self._used >= self.daily_budget:
                raise QuotaExhausted(self._reset_at(), "openrouter daily budget spent")
            self._used += 1

    def _check_budget(self) -> None:
        with self._quota_lock:
            self._roll_day()
            if self._now() < self._disabled_until or self._used >= self.daily_budget:
                raise QuotaExhausted(self._reset_at(), "openrouter daily budget spent")

    def _check_429(self, response: httpx.Response) -> None:
        try:
            text = response.text or ""
        except Exception:  # pragma: no cover - undecodable body
            text = ""
        reset_at = _epoch_from_header(response.headers.get("X-RateLimit-Reset"))
        far_reset = reset_at is not None and reset_at - self._now() > QUOTA_RESET_MIN_SECONDS
        if "free-models-per-day" in text or far_reset:
            with self._quota_lock:
                reset = reset_at if far_reset else self._next_reset()
                self._disabled_until = max(self._disabled_until, reset)
            logger.warning("openrouter daily free quota spent; disabled until reset")
            raise QuotaExhausted(reset, "openrouter free daily quota spent")

    # ----- request -----

    def _extra_headers(self) -> Dict[str, str]:
        return {"HTTP-Referer": "https://github.com/jobspy-coach", "X-Title": "JobSpy Coach Evaluator"}

    def _build_messages_body(self, messages: Sequence[Dict[str, str]], max_tokens: int) -> Dict[str, Any]:
        return {
            "models": list(self.models),
            "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "provider": {"require_parameters": True},
        }
