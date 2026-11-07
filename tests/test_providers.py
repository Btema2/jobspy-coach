import datetime as dt
import json
import logging
import threading

import httpx
import pytest

from src.evaluator.errors import (
    AuthenticationFailed,
    BadRequest,
    InvalidResponse,
    ModelNotFound,
    PermanentProviderError,
    QuotaExhausted,
    RateLimited,
    ResponseTruncated,
    StopRequested,
    TransientProviderError,
    UnexpectedStatus,
)
from src.evaluator.providers import (
    CompletionResult,
    NvidiaNimProvider,
    OpenRouterProvider,
    strip_think,
)

SECRET = "sk-test-secret-value"


class FakeLimiter:
    def __init__(self):
        self.acquires = 0
        self.pauses = []

    def acquire(self):
        self.acquires += 1
        return 0.0

    def pause_for(self, seconds):
        self.pauses.append(seconds)


class Harness:
    """Scripted HTTP responses + recorded requests/waits; nothing touches the network or sleeps."""

    def __init__(self, *script):
        self.script = list(script)
        self.requests = []
        self.waits = []
        self.limiter = FakeLimiter()
        self.stop_event = threading.Event()

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def common(self):
        return dict(
            transport=httpx.MockTransport(self.handler),
            limiter=self.limiter,
            stop_event=self.stop_event,
            wait_fn=self.waits.append,
            rand_fn=lambda: 1.0,  # maximum jitter -> deterministic upper bound
        )

    def nim(self, **kw):
        args = dict(api_key=SECRET, model="nim/model", base_url="https://nim.test/v1", **self.common())
        args.update(kw)
        return NvidiaNimProvider(**args)

    def openrouter(self, **kw):
        args = dict(
            api_key=SECRET,
            models=["or/a:free", "or/b:free"],
            base_url="https://or.test/api/v1",
            **self.common(),
        )
        args.update(kw)
        return OpenRouterProvider(**args)

    def body(self, index=-1):
        return json.loads(self.requests[index].content)


def ok(content='{"score": 7}', model="actual/model", finish="stop", reasoning="thinking...", **extra):
    message = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    data = {"model": model, "choices": [{"message": message, "finish_reason": finish}]}
    data.update(extra)
    return httpx.Response(200, json=data)


# ---------- strip_think ----------

def test_strip_think_removes_closed_block():
    assert strip_think("<think>hmm {not json}</think>\n{\"a\": 1}") == '{"a": 1}'


def test_strip_think_handles_multiline_and_case():
    assert strip_think("<THINK>a\nb</THINK> x") == "x"


def test_strip_think_orphan_closing_tag_keeps_text_after_it():
    assert strip_think("reasoning here</think>{\"a\": 1}") == '{"a": 1}'


def test_strip_think_unterminated_block_drops_rest():
    assert strip_think('{"a": 1} <think>never closed') == '{"a": 1}'


def test_strip_think_leaves_plain_text_alone():
    assert strip_think("  plain  ") == "plain"


# ---------- request shape ----------

def test_nim_request_shape():
    h = Harness(ok())
    provider = h.nim(temperature=0.6, max_tokens=4096, reasoning_effort="low")
    provider.complete("SYS", "USER")
    request = h.requests[0]
    assert str(request.url) == "https://nim.test/v1/chat/completions"
    assert request.headers["Authorization"] == f"Bearer {SECRET}"
    body = h.body()
    assert body["model"] == "nim/model"
    assert body["temperature"] == 0.6
    assert body["max_tokens"] == 4096
    assert body["reasoning_effort"] == "low"
    assert body["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USER"},
    ]
    assert "response_format" not in body
    assert "thinking_token_budget" not in body
    assert "stream" not in body or body["stream"] is False


def test_openrouter_request_shape():
    h = Harness(ok())
    provider = h.openrouter(temperature=0.3, max_tokens=2048)
    provider.complete("SYS", "USER")
    body = h.body()
    assert str(h.requests[0].url) == "https://or.test/api/v1/chat/completions"
    assert body["models"] == ["or/a:free", "or/b:free"]
    assert "model" not in body
    assert body["provider"] == {"require_parameters": True}
    assert body["response_format"] == {"type": "json_object"}
    assert body["temperature"] == 0.3
    assert body["max_tokens"] == 2048
    assert "reasoning_effort" not in body
    assert h.requests[0].headers["Authorization"] == f"Bearer {SECRET}"


def test_max_tokens_override_per_call():
    h = Harness(ok())
    h.nim(max_tokens=4096).complete("s", "u", max_tokens=6144)
    assert h.body()["max_tokens"] == 6144


def test_provider_names():
    h = Harness()
    assert h.nim().name == "nim"
    assert h.openrouter().name == "openrouter"


# ---------- response handling ----------

def test_complete_returns_completion_result():
    h = Harness(ok(content='{"x": 1}', model="actual/model", finish="stop"))
    result = h.nim().complete("s", "u")
    assert isinstance(result, CompletionResult)
    assert result.content == '{"x": 1}'
    assert result.model_used == "actual/model"
    assert result.finish_reason == "stop"
    assert result.latency >= 0.0


def test_model_used_falls_back_to_configured_model():
    response = httpx.Response(200, json={"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]})
    h = Harness(response)
    assert h.nim().complete("s", "u").model_used == "nim/model"


def test_reasoning_content_is_ignored():
    h = Harness(ok(content='{"x": 1}', reasoning='{"secret": "reasoning"}'))
    assert h.nim().complete("s", "u").content == '{"x": 1}'


def test_think_tags_stripped_from_content():
    h = Harness(ok(content='<think>draft {"bad": 1}</think>{"x": 1}'))
    assert h.nim().complete("s", "u").content == '{"x": 1}'


@pytest.mark.parametrize("content", ["", "   ", None, "<think>only thoughts</think>"])
def test_empty_content_is_invalid_response(content):
    h = Harness(ok(content=content))
    with pytest.raises(InvalidResponse):
        h.nim().complete("s", "u")


def test_finish_reason_length_is_truncated_invalid_response():
    h = Harness(ok(content='{"partial": ', finish="length"))
    with pytest.raises(ResponseTruncated) as exc_info:
        h.nim().complete("s", "u")
    assert isinstance(exc_info.value, InvalidResponse)


def test_non_json_200_body_is_invalid_response():
    h = Harness(httpx.Response(200, content=b"<html>proxy page</html>"))
    with pytest.raises(InvalidResponse):
        h.nim().complete("s", "u")


def test_missing_choices_is_invalid_response():
    h = Harness(httpx.Response(200, json={"model": "m", "choices": []}))
    with pytest.raises(InvalidResponse):
        h.nim().complete("s", "u")


@pytest.mark.parametrize(
    "choices",
    [{"message": {"content": "{}"}}, 5, "text", [], [None], [5], [{}], [{"message": 7}], [{"message": None}]],
)
def test_unexpected_choices_shape_is_invalid_response(choices):
    h = Harness(httpx.Response(200, json={"model": "m", "choices": choices}))
    with pytest.raises(InvalidResponse):
        h.nim().complete("s", "u")


def test_http_200_error_envelope_is_transient():
    envelope = httpx.Response(200, json={"error": {"code": 502, "message": "Provider returned error"}})
    h = Harness(envelope)
    with pytest.raises(TransientProviderError):
        h.openrouter().complete("s", "u")
    assert len(h.requests) == 1


# ---------- 429 ----------

def test_429_with_retry_after_waits_and_pauses_limiter():
    h = Harness(httpx.Response(429, headers={"Retry-After": "7"}), ok())
    result = h.nim().complete("s", "u")
    assert result.content
    assert h.waits == [7.0]
    assert h.limiter.pauses == [7.0]
    assert len(h.requests) == 2
    assert h.limiter.acquires == 2  # limiter wraps every HTTP attempt


def test_retry_after_capped_at_120():
    h = Harness(httpx.Response(429, headers={"Retry-After": "9999"}), ok())
    h.nim().complete("s", "u")
    assert h.waits == [120.0]
    assert h.limiter.pauses == [120.0]


def test_429_without_header_uses_exponential_backoff_with_jitter():
    h = Harness(httpx.Response(429), httpx.Response(429), ok())
    h.nim().complete("s", "u")
    # rand_fn == 1.0 -> +25% jitter exactly: 5*1.25 and 10*1.25
    assert h.waits == [pytest.approx(6.25), pytest.approx(12.5)]
    assert h.limiter.pauses == [pytest.approx(6.25), pytest.approx(12.5)]


def test_429_backoff_without_jitter_lower_bound():
    h = Harness(httpx.Response(429), ok())
    h.nim(rand_fn=lambda: 0.0).complete("s", "u")
    assert h.waits == [pytest.approx(5.0)]


def test_429_nim_empty_body_title_shape_is_just_a_429():
    body = {"status": 429, "title": "Too Many Requests"}
    h = Harness(httpx.Response(429, json=body), ok())
    assert h.nim().complete("s", "u").content


def test_429_attempts_are_bounded_at_three():
    h = Harness(httpx.Response(429), httpx.Response(429), httpx.Response(429), ok())
    with pytest.raises(RateLimited):
        h.nim().complete("s", "u")
    assert len(h.requests) == 3
    assert len(h.waits) == 2  # no sleep after the final attempt
    assert h.limiter.acquires == 3
    # the unused fourth scripted response was never requested
    assert len(h.script) == 1


def test_cumulative_sleep_cap_gives_up_early():
    h = Harness(
        httpx.Response(429, headers={"Retry-After": "120"}),
        httpx.Response(429, headers={"Retry-After": "120"}),
        ok(),
    )
    with pytest.raises(RateLimited):
        h.nim().complete("s", "u")
    # 120 + 120 would exceed 180 -> the second wait is not taken
    assert len(h.requests) == 2
    assert h.waits == [120.0]
    # but the limiter is still paused so other workers back off
    assert h.limiter.pauses == [120.0, 120.0]


def test_rate_limited_after_exhaustion_carries_retry_after():
    h = Harness(*[httpx.Response(429, headers={"Retry-After": "3"})] * 3)
    with pytest.raises(RateLimited) as exc_info:
        h.nim().complete("s", "u")
    assert exc_info.value.retry_after == 3.0


# ---------- 5xx / timeouts / network ----------

@pytest.mark.parametrize("status", [500, 502, 503, 408])
def test_transient_status_backoff_schedule(status):
    h = Harness(httpx.Response(status), httpx.Response(status), ok())
    h.nim().complete("s", "u")
    assert h.waits == [pytest.approx(2.0 * 1.25), pytest.approx(6.0 * 1.25)]
    assert h.limiter.pauses == []  # only 429 pauses the shared limiter


def test_5xx_exhausts_after_three_attempts():
    h = Harness(httpx.Response(503), httpx.Response(503), httpx.Response(503), ok())
    with pytest.raises(TransientProviderError) as exc_info:
        h.nim().complete("s", "u")
    assert not isinstance(exc_info.value, RateLimited)
    assert len(h.requests) == 3
    assert len(h.waits) == 2


def test_timeout_retried_then_succeeds():
    h = Harness(httpx.ReadTimeout("slow"), ok())
    assert h.nim().complete("s", "u").content
    assert h.waits == [pytest.approx(2.5)]


def test_timeouts_exhaust_into_transient_error():
    h = Harness(httpx.ReadTimeout("t"), httpx.ConnectTimeout("t"), httpx.ReadTimeout("t"))
    with pytest.raises(TransientProviderError):
        h.nim().complete("s", "u")
    assert len(h.requests) == 3


def test_network_error_retried():
    h = Harness(httpx.ConnectError("refused"), ok())
    assert h.nim().complete("s", "u").content
    assert len(h.requests) == 2


# ---------- permanent errors ----------

@pytest.mark.parametrize("status", [401, 403])
def test_auth_failure_is_permanent_and_not_retried(status):
    h = Harness(httpx.Response(status, json={"error": "bad key"}), ok())
    with pytest.raises(AuthenticationFailed) as exc_info:
        h.nim().complete("s", "u")
    assert isinstance(exc_info.value, PermanentProviderError)
    assert len(h.requests) == 1
    assert h.waits == []


def test_404_is_model_not_found():
    h = Harness(httpx.Response(404), ok())
    with pytest.raises(ModelNotFound) as exc_info:
        h.nim().complete("s", "u")
    assert isinstance(exc_info.value, PermanentProviderError)
    assert len(h.requests) == 1


@pytest.mark.parametrize("status", [400, 413, 422])
def test_bad_request_is_permanent_call_only(status):
    h = Harness(httpx.Response(status, text="nope"), ok())
    with pytest.raises(BadRequest) as exc_info:
        h.nim().complete("s", "u")
    assert isinstance(exc_info.value, PermanentProviderError)
    assert not isinstance(exc_info.value, (AuthenticationFailed, ModelNotFound))
    assert len(h.requests) == 1


@pytest.mark.parametrize("status", [307, 402, 405, 409, 410, 418])
def test_unexpected_status_is_provider_wide_not_bad_request(status):
    h = Harness(httpx.Response(status, text="x"), ok())
    with pytest.raises(UnexpectedStatus) as exc_info:
        h.nim().complete("s", "u")
    assert isinstance(exc_info.value, PermanentProviderError)
    assert not isinstance(exc_info.value, BadRequest)
    assert len(h.requests) == 1


# ---------- stop handling ----------

def test_stop_during_backoff_raises_stop_requested():
    h = Harness(httpx.Response(503), ok())

    def stopping_wait(seconds):
        h.waits.append(seconds)
        h.stop_event.set()

    provider = h.nim(wait_fn=stopping_wait)
    with pytest.raises(StopRequested):
        provider.complete("s", "u")
    assert len(h.requests) == 1


def test_stop_set_before_call_raises_without_request():
    h = Harness(ok())
    h.stop_event.set()
    with pytest.raises(StopRequested):
        h.nim().complete("s", "u")
    assert h.requests == []


def test_default_wait_uses_stop_event(monkeypatch):
    h = Harness(httpx.Response(503), ok())
    event = h.stop_event
    seen = []
    monkeypatch.setattr(event, "wait", lambda s: seen.append(s) or False)
    kwargs = h.common()
    del kwargs["wait_fn"]
    provider = NvidiaNimProvider(api_key=SECRET, model="m", base_url="https://nim.test/v1", **kwargs)
    provider.complete("s", "u")
    assert seen == [pytest.approx(2.5)]


# ---------- secrets ----------

def test_api_key_never_logged(caplog):
    h = Harness(httpx.Response(429), httpx.Response(503), httpx.Response(401, text="bad"))
    caplog.set_level(logging.DEBUG)
    with pytest.raises(AuthenticationFailed):
        h.nim().complete("system text", "job text")
    assert SECRET not in caplog.text
    assert "job text" not in caplog.text
    assert "system text" not in caplog.text


# ---------- OpenRouter quota ----------

class Clock:
    def __init__(self, when):
        self.when = when

    def __call__(self):
        return self.when.timestamp()


def utc(y, m, d, hh=12, mm=0):
    return dt.datetime(y, m, d, hh, mm, tzinfo=dt.timezone.utc)


def test_openrouter_daily_budget_exhausts_without_http_request():
    clock = Clock(utc(2026, 11, 5))
    h = Harness(ok(), ok(), ok())
    provider = h.openrouter(daily_budget=2, now_fn=clock)
    provider.complete("s", "u")
    provider.complete("s", "u")
    assert provider.is_available() is False
    with pytest.raises(QuotaExhausted) as exc_info:
        provider.complete("s", "u")
    assert len(h.requests) == 2
    assert exc_info.value.reset_at == utc(2026, 11, 6, 0, 0).timestamp()


def test_openrouter_budget_resets_at_utc_midnight():
    clock = Clock(utc(2026, 11, 5, 23, 59))
    h = Harness(ok(), ok())
    provider = h.openrouter(daily_budget=1, now_fn=clock)
    provider.complete("s", "u")
    assert provider.is_available() is False
    clock.when = utc(2026, 11, 6, 0, 1)
    assert provider.is_available() is True
    provider.complete("s", "u")
    assert len(h.requests) == 2


def test_openrouter_retries_count_against_budget():
    clock = Clock(utc(2026, 11, 5))
    h = Harness(httpx.Response(503), ok())
    provider = h.openrouter(daily_budget=2, now_fn=clock)
    provider.complete("s", "u")
    assert provider.is_available() is False


def test_openrouter_daily_429_disables_until_reset():
    clock = Clock(utc(2026, 11, 5, 10))
    body = {"error": {"message": "Rate limit exceeded: free-models-per-day. Add credits", "code": 429}}
    h = Harness(httpx.Response(429, json=body), ok())
    provider = h.openrouter(now_fn=clock)
    with pytest.raises(QuotaExhausted) as exc_info:
        provider.complete("s", "u")
    assert len(h.requests) == 1  # no retries
    assert h.waits == []
    assert exc_info.value.reset_at == utc(2026, 11, 6, 0, 0).timestamp()
    assert provider.is_available() is False
    with pytest.raises(QuotaExhausted):
        provider.complete("s", "u")
    assert len(h.requests) == 1  # skipped without a request
    clock.when = utc(2026, 11, 6, 0, 5)
    assert provider.is_available() is True


def test_openrouter_daily_429_reset_header_in_far_future():
    clock = Clock(utc(2026, 11, 5, 10))
    reset_ms = int(utc(2026, 11, 5, 20).timestamp() * 1000)
    h = Harness(httpx.Response(429, json={"error": {"message": "limit"}}, headers={"X-RateLimit-Reset": str(reset_ms)}))
    provider = h.openrouter(now_fn=clock)
    with pytest.raises(QuotaExhausted) as exc_info:
        provider.complete("s", "u")
    assert exc_info.value.reset_at == pytest.approx(reset_ms / 1000)


def test_openrouter_ordinary_429_is_retried_not_quota():
    clock = Clock(utc(2026, 11, 5, 10))
    soon_ms = int((utc(2026, 11, 5, 10).timestamp() + 30) * 1000)
    h = Harness(
        httpx.Response(429, json={"error": {"message": "upstream busy"}}, headers={"X-RateLimit-Reset": str(soon_ms), "Retry-After": "4"}),
        ok(),
    )
    provider = h.openrouter(now_fn=clock)
    assert provider.complete("s", "u").content
    assert h.waits == [4.0]
    assert provider.is_available() is True


def test_nim_is_always_available():
    assert Harness().nim().is_available() is True


def test_providers_expose_a_breaker():
    h = Harness()
    assert h.nim().breaker.allow() is True
    assert h.openrouter().breaker.allow() is True


def test_openrouter_daily_counter_is_thread_safe():
    clock = Clock(utc(2026, 11, 5))
    budget, threads_n = 7, 24
    seen = []
    seen_lock = threading.Lock()

    def handler(request):
        with seen_lock:
            seen.append(request)
        return ok()

    h = Harness()
    provider = h.openrouter(daily_budget=budget, now_fn=clock, transport=httpx.MockTransport(handler))
    outcomes = []
    barrier = threading.Barrier(threads_n)

    def worker():
        barrier.wait()
        try:
            provider.complete("s", "u")
            result = "ok"
        except QuotaExhausted:
            result = "quota"
        with seen_lock:
            outcomes.append(result)

    threads = [threading.Thread(target=worker) for _ in range(threads_n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)
    assert outcomes.count("ok") == budget
    assert outcomes.count("quota") == threads_n - budget
    assert len(seen) == budget
