"""Scripted LLM HTTP transport for tests: no network, no sleeping, placeholder keys only."""

import json
import threading
from typing import Any, Dict, List, Optional

import httpx

from src.config import Settings
from src.evaluator.providers import NvidiaNimProvider, OpenRouterProvider
from src.evaluator.rate_limiter import SlidingWindowRateLimiter

TIMEOUT = "timeout"  # script item: the transport raises httpx.ReadTimeout


def nim_reply(content: str, status: int = 200) -> httpx.Response:
    """A chat-completions answer carrying ``content`` (any non-200 status has an empty body)."""
    if status != 200:
        return httpx.Response(status)
    message = {"role": "assistant", "content": content, "reasoning_content": "thinking..."}
    return httpx.Response(200, json={"model": "actual/model", "choices": [{"message": message, "finish_reason": "stop"}]})


class ScriptedLLM:
    """An ``httpx.MockTransport`` handler serving queued responses and recording request bodies.

    Script items: an ``httpx.Response``, an exception instance (raised), or ``TIMEOUT``.
    """

    def __init__(self, *script: Any):
        self.script: List[Any] = list(script)
        self.requests: List[httpx.Request] = []
        self.bodies: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def queue(self, *items: Any) -> "ScriptedLLM":
        with self._lock:
            self.script.extend(items)
        return self

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        with self._lock:
            self.requests.append(request)
            self.bodies.append(json.loads(request.content))
            if not self.script:
                raise AssertionError("ScriptedLLM: no queued response left")
            item = self.script.pop(0)
        if item == TIMEOUT:
            raise httpx.ReadTimeout("scripted timeout", request=request)
        if isinstance(item, Exception):
            raise item
        return item


def make_providers(
    nim_transport: Optional[httpx.BaseTransport],
    openrouter_transport: Optional[httpx.BaseTransport] = None,
    *,
    stop_event: Optional[threading.Event] = None,
    rpm_limit: Optional[int] = None,
) -> List[Any]:
    """NIM (and OpenRouter when a transport is given) built like ``build_chain_from_settings``."""
    stop_event = stop_event if stop_event is not None else threading.Event()
    settings = Settings(
        _env_file=None,
        NVIDIA_API_KEY="<NIM-KEY>" if nim_transport is not None else "",
        OPENROUTER_API_KEY="<OR-KEY>" if openrouter_transport is not None else "",
        OPENROUTER_MODELS="vendor/model-a:free,vendor/model-b:free" if openrouter_transport is not None else "",
    )
    providers: List[Any] = []
    if settings.nim_enabled:
        providers.append(
            NvidiaNimProvider(
                api_key=settings.nvidia_api_key.strip(),
                model=settings.nvidia_model,
                base_url=settings.nvidia_base_url,
                limiter=SlidingWindowRateLimiter(rpm_limit or settings.nvidia_rpm_limit, 60.0, stop_event=stop_event),
                stop_event=stop_event,
                timeout=settings.nvidia_timeout_seconds,
                max_tokens=settings.nvidia_max_tokens,
                temperature=settings.nvidia_temperature,
                reasoning_effort=settings.nvidia_reasoning_effort,
                workers=settings.eval_max_workers,
                transport=nim_transport,
            )
        )
    if settings.openrouter_enabled:
        providers.append(
            OpenRouterProvider(
                api_key=settings.openrouter_api_key.strip(),
                models=settings.openrouter_models,
                base_url=settings.openrouter_base_url,
                limiter=SlidingWindowRateLimiter(rpm_limit or settings.openrouter_rpm_limit, 60.0, stop_event=stop_event),
                stop_event=stop_event,
                timeout=settings.openrouter_timeout_seconds,
                daily_budget=settings.openrouter_daily_budget,
                workers=settings.eval_max_workers,
                transport=openrouter_transport,
            )
        )
    return providers
