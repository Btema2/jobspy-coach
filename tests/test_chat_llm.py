import json
import logging
import threading

import httpx
import pytest

from src.bot.chat_llm import (
    CHAT_HOURLY_CAP,
    CHAT_LIMITER_MAX_WAIT_SECONDS,
    CHAT_MAX_ATTEMPTS,
    CHAT_MAX_TOKENS,
    CHAT_TIMEOUT_SECONDS,
    ChatAnswer,
    ChatLLM,
    ChatLLMLimited,
    ChatLLMNotConfigured,
    ChatLLMNotUnderstood,
    ChatLLMUnavailable,
    HourlyCap,
)
from src.evaluator.errors import ConfigurationError, StopRequested
from tests.fake_llm import TIMEOUT, ScriptedLLM, make_providers, nim_reply

MESSAGES = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "USER"}]
GOOD = json.dumps({"kind": "ok", "n": 1})


def validate_ok(data):
    if data.get("kind") != "ok":
        raise ValueError("bad kind")
    return data["n"]


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def make_chat(nim, openrouter=None, *, stop=None, batch_stop=None, cap=None, rpm_limit=None):
    providers = make_providers(
        nim.transport if nim else None,
        openrouter.transport if openrouter else None,
        stop_event=batch_stop,
        rpm_limit=rpm_limit,
    )
    chat = ChatLLM(lambda: providers, stop_event=stop or threading.Event(), cap=cap)
    return chat, providers


def run(chat, validate=validate_ok, request_id="r1"):
    return chat.complete(MESSAGES, validate=validate, request_id=request_id)


def test_constants():
    assert (CHAT_TIMEOUT_SECONDS, CHAT_MAX_ATTEMPTS, CHAT_HOURLY_CAP) == (120.0, 1, 30)
    assert (CHAT_LIMITER_MAX_WAIT_SECONDS, CHAT_MAX_TOKENS) == (30.0, 8192)


def test_success_on_nim_returns_validated_value():
    nim = ScriptedLLM(nim_reply(GOOD))
    chat, _ = make_chat(nim)
    answer = run(chat)
    assert isinstance(answer, ChatAnswer)
    assert (answer.value, answer.provider, answer.model_used) == (1, "nim", "actual/model")
    assert answer.latency >= 0
    body = nim.bodies[0]
    assert body["max_tokens"] == CHAT_MAX_TOKENS
    assert nim.requests[0].extensions["timeout"]["read"] == CHAT_TIMEOUT_SECONDS


def test_nim_then_openrouter_on_timeout():
    nim = ScriptedLLM(TIMEOUT)
    openrouter = ScriptedLLM(nim_reply(GOOD))
    chat, _ = make_chat(nim, openrouter)
    answer = run(chat)
    assert answer.provider == "openrouter"
    assert len(nim.requests) == 1 and len(openrouter.requests) == 1


def test_schema_failure_falls_through_then_not_understood():
    nim = ScriptedLLM(nim_reply(json.dumps({"kind": "nope"})))
    openrouter = ScriptedLLM(nim_reply("not json at all"))
    chat, _ = make_chat(nim, openrouter)
    with pytest.raises(ChatLLMNotUnderstood):
        run(chat)
    assert len(nim.requests) == 1 and len(openrouter.requests) == 1


def test_schema_failure_on_nim_is_rescued_by_openrouter():
    nim = ScriptedLLM(nim_reply(json.dumps({"kind": "nope"})))
    openrouter = ScriptedLLM(nim_reply(GOOD))
    chat, _ = make_chat(nim, openrouter)
    assert run(chat).provider == "openrouter"


def test_all_unavailable_is_unavailable():
    nim = ScriptedLLM(nim_reply("", status=503))
    openrouter = ScriptedLLM(httpx.Response(429, text="free-models-per-day exhausted"))
    chat, _ = make_chat(nim, openrouter)
    with pytest.raises(ChatLLMUnavailable):
        run(chat)


def test_unavailable_then_not_understood_flag_wins():
    nim = ScriptedLLM(TIMEOUT)
    openrouter = ScriptedLLM(nim_reply("", status=503))
    chat, _ = make_chat(nim, openrouter)
    with pytest.raises(ChatLLMNotUnderstood):
        run(chat)


def test_not_configured():
    def raising():
        raise ConfigurationError("no provider")

    chat = ChatLLM(raising, stop_event=threading.Event())
    assert chat.available() is False
    with pytest.raises(ChatLLMNotConfigured):
        run(chat)
    empty = ChatLLM(lambda: [], stop_event=threading.Event())
    assert empty.available() is False
    with pytest.raises(ChatLLMNotConfigured):
        run(empty)


def test_available_when_a_provider_exists():
    chat, _ = make_chat(ScriptedLLM())
    assert chat.available() is True


def test_hourly_cap_30():
    clock = Clock()
    cap = HourlyCap(time_fn=clock)
    nim = ScriptedLLM(*[nim_reply(GOOD) for _ in range(31)])
    chat, _ = make_chat(nim, cap=cap, rpm_limit=1000)
    for _ in range(30):
        run(chat)
    clock.now += 10.0
    with pytest.raises(ChatLLMLimited) as exc_info:
        run(chat)
    assert exc_info.value.retry_in > 0
    assert exc_info.value.retry_in == pytest.approx(3590.0)
    assert len(nim.requests) == 30
    clock.now += 3600.0
    assert run(chat).value == 1


def test_hourly_cap_counts_every_provider_attempt_and_stops_later():
    clock = Clock()
    cap = HourlyCap(limit=1, time_fn=clock)
    nim = ScriptedLLM(TIMEOUT)
    openrouter = ScriptedLLM(nim_reply(GOOD))
    chat, _ = make_chat(nim, openrouter, cap=cap)
    with pytest.raises(ChatLLMNotUnderstood):  # the second attempt was refused: stop trying, report what happened
        run(chat)
    assert openrouter.requests == []


def test_hourly_cap_try_take_semantics():
    clock = Clock()
    cap = HourlyCap(limit=2, period=100.0, time_fn=clock)
    assert cap.try_take() is None
    clock.now += 10
    assert cap.try_take() is None
    assert cap.try_take() == pytest.approx(90.0)
    clock.now += 90
    assert cap.try_take() is None


def test_chat_ignores_batch_stop_event():
    batch_stop = threading.Event()
    batch_stop.set()
    nim = ScriptedLLM(nim_reply(GOOD))
    chat, providers = make_chat(nim, batch_stop=batch_stop)
    assert run(chat).value == 1
    with pytest.raises(StopRequested):
        providers[0].complete("s", "u")  # the evaluation path still honours the batch event


def test_shutdown_event_aborts():
    stop = threading.Event()
    stop.set()
    nim = ScriptedLLM(nim_reply(GOOD))
    chat, _ = make_chat(nim, stop=stop)
    with pytest.raises(StopRequested):
        run(chat)
    assert nim.requests == []


def test_breakers_untouched():
    nim = ScriptedLLM(TIMEOUT)
    openrouter = ScriptedLLM(nim_reply("", status=503))
    chat, providers = make_chat(nim, openrouter)
    before = [(p.breaker.state, p.breaker._failures) for p in providers]
    with pytest.raises(ChatLLMNotUnderstood):
        run(chat)
    nim.queue(nim_reply(GOOD))
    run(chat)
    assert [(p.breaker.state, p.breaker._failures) for p in providers] == before


def test_no_prompt_or_response_text_in_logs(caplog):
    caplog.set_level(logging.DEBUG)
    nim = ScriptedLLM(nim_reply(json.dumps({"kind": "ok", "n": 1, "text": "<MARKER-OUT>"})), nim_reply("<MARKER-OUT> not json"))
    chat, _ = make_chat(nim)
    messages = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "<MARKER-IN>"}]
    chat.complete(messages, validate=validate_ok, request_id="req-42")
    with pytest.raises(ChatLLMNotUnderstood):
        chat.complete(messages, validate=validate_ok, request_id="req-43")
    text = "\n".join(record.getMessage() for record in caplog.records)
    assert "<MARKER-IN>" not in text and "<MARKER-OUT>" not in text
    assert "req-42" in text
    assert "req-43" in text
