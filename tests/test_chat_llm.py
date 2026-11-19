import json
import logging
import threading

import httpx
import pytest

from src.bot.chat_llm import (
    CHAT_HOURLY_CAP,
    CHAT_OPENROUTER_DAILY_SHARE,
    MIN_RETRY_ATTEMPT_SECONDS,
    OPENROUTER_RESERVE_SECONDS,
    ChatFailure,
    describe_failure,
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
from src.bot.chat_llm import ChatLLMError
from src.evaluator.errors import ConfigurationError, StopRequested
from tests.fake_llm import TIMEOUT, FakeClock, ScriptedLLM, make_providers, nim_reply

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


def make_chat(nim, openrouter=None, *, stop=None, batch_stop=None, cap=None, rpm_limit=None, clock=None):
    providers = make_providers(
        nim.transport if nim else None,
        openrouter.transport if openrouter else None,
        stop_event=batch_stop,
        rpm_limit=rpm_limit,
        clock=clock,
    )
    extra = {"clock": clock.now, "now_fn": clock.now} if clock is not None else {}
    chat = ChatLLM(lambda: providers, stop_event=stop or threading.Event(), cap=cap, **extra)
    return chat, providers


def run(chat, validate=validate_ok, request_id="r1", kind="chat"):
    return chat.complete(MESSAGES, validate=validate, request_id=request_id, kind=kind)


def timeout_after(clock, seconds=None):
    """A scripted item: the request runs for ``seconds`` (default: its full timeout) and times out."""

    def fn(request):
        clock.advance(seconds if seconds is not None else request.extensions["timeout"]["read"])
        raise httpx.ReadTimeout("scripted", request=request)

    return fn


def read_timeouts(llm):
    return [request.extensions["timeout"]["read"] for request in llm.requests]


def spy_acquire(provider):
    calls = []
    original = provider.limiter.acquire

    def acquire(*args, **kwargs):
        calls.append(kwargs.get("max_wait"))
        return original(*args, **kwargs)

    provider.limiter.acquire = acquire
    return calls


def test_constants():
    assert (CHAT_TIMEOUT_SECONDS, CHAT_MAX_ATTEMPTS, CHAT_HOURLY_CAP) == (120.0, 1, 60)
    assert (CHAT_OPENROUTER_DAILY_SHARE, OPENROUTER_RESERVE_SECONDS, MIN_RETRY_ATTEMPT_SECONDS) == (15, 60.0, 30.0)
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
    nim = ScriptedLLM(TIMEOUT, TIMEOUT)  # a quick timeout is retried once on NIM
    openrouter = ScriptedLLM(nim_reply(GOOD))
    chat, _ = make_chat(nim, openrouter)
    answer = run(chat)
    assert answer.provider == "openrouter"
    assert len(nim.requests) == 2 and len(openrouter.requests) == 1


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
    nim = ScriptedLLM(nim_reply("", status=503), nim_reply("", status=503))
    openrouter = ScriptedLLM(httpx.Response(429, text="free-models-per-day exhausted"))
    chat, _ = make_chat(nim, openrouter)
    with pytest.raises(ChatLLMUnavailable):
        run(chat)


def test_unavailable_then_not_understood_flag_wins():
    nim = ScriptedLLM(TIMEOUT, TIMEOUT)
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


def test_hourly_cap_is_60():
    assert CHAT_HOURLY_CAP == 60
    clock = Clock()
    cap = HourlyCap(time_fn=clock)
    assert cap.limit == 60
    nim = ScriptedLLM(*[nim_reply(GOOD) for _ in range(61)])
    chat, _ = make_chat(nim, cap=cap, rpm_limit=1000)
    for _ in range(60):
        run(chat)
    clock.now += 10.0
    with pytest.raises(ChatLLMLimited) as exc_info:
        run(chat)
    assert exc_info.value.retry_in > 0
    assert exc_info.value.retry_in == pytest.approx(3590.0)
    assert exc_info.value.daily is False
    assert len(nim.requests) == 60
    clock.now += 3600.0
    assert run(chat).value == 1


def test_hourly_cap_counts_every_provider_attempt_and_stops_later():
    clock = Clock()
    cap = HourlyCap(limit=1, time_fn=clock)
    nim = ScriptedLLM(TIMEOUT, TIMEOUT)
    openrouter = ScriptedLLM(nim_reply(GOOD))
    chat, _ = make_chat(nim, openrouter, cap=cap)
    with pytest.raises(ChatLLMLimited):  # the retry was refused: capped outranks the timeout
        run(chat)
    assert openrouter.requests == [] and len(nim.requests) == 1


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
    nim = ScriptedLLM(TIMEOUT, TIMEOUT)
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


# ---------- typed failures, retry loop, deadlines, caps ----------

PROFILES = {"chat": (120.0, 240.0), "turn": (90.0, 150.0), "draft": (180.0, 300.0)}


@pytest.mark.parametrize("kind", ["chat", "turn", "draft"])
def test_nim_retries_once_after_fast_429_and_spends_two_cap_slots(kind):
    clock = FakeClock()
    cap = HourlyCap(time_fn=clock.now)
    nim = ScriptedLLM(nim_reply("", status=429), nim_reply(GOOD))
    chat, _ = make_chat(nim, clock=clock, cap=cap)
    answer = run(chat, kind=kind)
    assert (answer.value, answer.provider) == (1, "nim")
    assert len(nim.requests) == 2
    assert len(cap._stamps) == 2
    assert read_timeouts(nim) == [PROFILES[kind][0]] * 2  # fast failure leaves the whole attempt timeout


def test_nim_retries_once_after_5xx():
    clock = FakeClock()
    nim = ScriptedLLM(nim_reply("", status=503), nim_reply(GOOD))
    chat, _ = make_chat(nim, clock=clock)
    assert run(chat).provider == "nim"
    assert len(nim.requests) == 2
    nim = ScriptedLLM(nim_reply("", status=500), nim_reply("", status=500))
    openrouter = ScriptedLLM(nim_reply(GOOD))
    chat, _ = make_chat(nim, openrouter, clock=clock)
    assert run(chat).provider == "openrouter"
    assert len(nim.requests) == 2  # at most one extra attempt


@pytest.mark.parametrize(
    "item",
    [
        nim_reply("not json at all"),
        httpx.Response(200, json={"error": {"code": 502, "message": "upstream"}}),
        httpx.ConnectError("down"),
        nim_reply(json.dumps({"kind": "nope"})),
        httpx.Response(400),
    ],
    ids=["invalid", "envelope", "transport", "schema", "bad-request"],
)
def test_no_retry_after_invalid_response_or_envelope_or_transport_error(item):
    clock = FakeClock()
    nim = ScriptedLLM(item, nim_reply(GOOD))
    openrouter = ScriptedLLM(nim_reply(GOOD))
    chat, _ = make_chat(nim, openrouter, clock=clock)
    assert run(chat).provider == "openrouter"
    assert len(nim.requests) == 1 and len(openrouter.requests) == 1


def test_full_length_nim_timeout_goes_straight_to_openrouter_reserve_rule():
    clock = FakeClock()
    nim = ScriptedLLM(timeout_after(clock), nim_reply(GOOD))
    openrouter = ScriptedLLM(nim_reply(GOOD))
    chat, _ = make_chat(nim, openrouter, clock=clock)
    answer = run(chat, kind="turn")  # 90 s attempt of a 150 s deadline: 60 s left, nothing after the reserve
    assert answer.provider == "openrouter"
    assert len(nim.requests) == 1 and len(openrouter.requests) == 1
    assert read_timeouts(openrouter) == [pytest.approx(60.0)]


def test_retry_timeout_is_clipped_to_remaining_minus_reserve():
    clock = FakeClock()
    nim = ScriptedLLM(timeout_after(clock), nim_reply(GOOD))
    chat, _ = make_chat(nim, clock=clock)
    assert run(chat, kind="draft").provider == "nim"  # 300 - 180 = 120 left; 120 - 60 = 60 >= 30
    assert read_timeouts(nim) == [180.0, pytest.approx(60.0)]


@pytest.mark.parametrize("retry_after_header", [True, False])
def test_429_pause_over_bound_moves_to_next_provider(retry_after_header):
    clock = FakeClock()
    holder = {}

    def long_pause(request):
        holder["providers"][0].limiter.pause_for(100.0)
        return httpx.Response(429)

    first = httpx.Response(429, headers={"Retry-After": "100"}) if retry_after_header else long_pause
    nim = ScriptedLLM(first, nim_reply(GOOD))
    openrouter = ScriptedLLM(nim_reply(GOOD))
    chat, providers = make_chat(nim, openrouter, clock=clock)
    holder["providers"] = providers
    waits = spy_acquire(providers[0])
    answer = run(chat, kind="turn")
    assert answer.provider == "openrouter"
    assert len(nim.requests) == 1
    if retry_after_header:
        assert waits == [30.0]  # a known retry-after above the bound skips the retry entirely
    else:
        assert waits == [30.0, 30.0]  # the retry is bounded; the limiter refuses the wait


def test_retry_after_clips_first_wait_to_the_remaining_deadline():
    clock = FakeClock()
    nim = ScriptedLLM(timeout_after(clock, 140.0))
    openrouter = ScriptedLLM(nim_reply(GOOD))
    chat, providers = make_chat(nim, openrouter, clock=clock)
    waits = spy_acquire(providers[1])
    assert run(chat, kind="turn").provider == "openrouter"
    assert len(nim.requests) == 1  # 10 s left: no retry
    assert waits == [pytest.approx(10.0)]
    assert read_timeouts(openrouter) == [pytest.approx(10.0)]


def test_overall_deadline_ends_request_with_timeout_class():
    clock = FakeClock()
    nim = ScriptedLLM(timeout_after(clock, 200.0))
    openrouter = ScriptedLLM(nim_reply(GOOD))
    chat, _ = make_chat(nim, openrouter, clock=clock)
    with pytest.raises(ChatLLMNotUnderstood) as exc_info:
        run(chat, kind="turn")
    assert exc_info.value.failure is ChatFailure.TIMEOUT
    assert openrouter.requests == [] and len(nim.requests) == 1


def test_openrouter_never_retried():
    clock = FakeClock()
    for first in (nim_reply("", status=503), nim_reply("", status=429), TIMEOUT):
        openrouter = ScriptedLLM(first, nim_reply(GOOD))
        chat, _ = make_chat(None, openrouter, clock=clock)
        with pytest.raises(ChatLLMError):
            run(chat)
        assert len(openrouter.requests) == 1


def test_openrouter_daily_share_stops_at_15():
    clock = FakeClock()
    openrouter = ScriptedLLM(*[nim_reply(GOOD) for _ in range(17)])
    chat, _ = make_chat(None, openrouter, clock=clock)
    for _ in range(CHAT_OPENROUTER_DAILY_SHARE):
        assert run(chat).provider == "openrouter"
    with pytest.raises(ChatLLMLimited) as exc_info:
        run(chat)
    assert exc_info.value.failure is ChatFailure.CAPPED
    assert exc_info.value.retry_in > 0
    assert exc_info.value.daily is True  # the share is per UTC day, the hourly cap is not
    assert len(openrouter.requests) == 15
    clock.advance(86400.0)  # a new UTC day
    assert run(chat).provider == "openrouter"
    assert len(openrouter.requests) == 16


@pytest.mark.parametrize(
    "case, expected, exc_type",
    [
        ("timeout+quota", ChatFailure.TIMEOUT, ChatLLMNotUnderstood),
        ("429+invalid", ChatFailure.BAD_ANSWER, ChatLLMNotUnderstood),
        ("5xx+5xx", ChatFailure.UNAVAILABLE, ChatLLMUnavailable),
        ("429+429", ChatFailure.RATE_LIMITED, ChatLLMUnavailable),
        ("capped", ChatFailure.CAPPED, ChatLLMLimited),
        ("no-key", ChatFailure.UNAVAILABLE, ChatLLMNotConfigured),
    ],
)
def test_precedence_table(case, expected, exc_type):
    clock = FakeClock()
    cap = None
    if case == "timeout+quota":
        nim = ScriptedLLM(TIMEOUT, TIMEOUT)
        openrouter = ScriptedLLM(httpx.Response(429, text="free-models-per-day exhausted"))
    elif case == "429+invalid":
        nim = ScriptedLLM(nim_reply("", status=429), nim_reply("", status=429))
        openrouter = ScriptedLLM(nim_reply("not json"))
    elif case == "5xx+5xx":
        nim = ScriptedLLM(nim_reply("", status=503), nim_reply("", status=503))
        openrouter = ScriptedLLM(nim_reply("", status=502))
    elif case == "429+429":
        nim = ScriptedLLM(nim_reply("", status=429), nim_reply("", status=429))
        openrouter = ScriptedLLM(nim_reply("", status=429))
    elif case == "capped":
        cap = HourlyCap(limit=1, time_fn=clock.now)
        assert cap.try_take() is None
        nim, openrouter = ScriptedLLM(nim_reply(GOOD)), ScriptedLLM(nim_reply(GOOD))
    else:
        chat = ChatLLM(lambda: [], stop_event=threading.Event(), clock=clock.now, now_fn=clock.now)
        with pytest.raises(exc_type) as exc_info:
            run(chat)
        assert exc_info.value.failure is expected
        return
    chat, _ = make_chat(nim, openrouter, clock=clock, cap=cap)
    with pytest.raises(exc_type) as exc_info:
        run(chat)
    assert exc_info.value.failure is expected
    if expected is ChatFailure.CAPPED:
        assert exc_info.value.retry_in == pytest.approx(3600.0)
        assert nim.requests == [] and openrouter.requests == []


def test_every_error_class_carries_failure_and_retry_in():
    assert ChatLLMLimited(30.0).failure is ChatFailure.CAPPED and ChatLLMLimited(30.0).retry_in == 30.0
    for cls in (ChatLLMNotConfigured, ChatLLMUnavailable, ChatLLMNotUnderstood):
        err = cls("x")
        assert isinstance(err.failure, ChatFailure)
        assert err.retry_in is None
    assert ChatLLMNotConfigured("x").failure is ChatFailure.UNAVAILABLE


def test_unknown_kind_is_rejected():
    chat, _ = make_chat(ScriptedLLM(nim_reply(GOOD)))
    with pytest.raises(ValueError):
        run(chat, kind="bogus")


def test_describe_failure_texts():
    heads = {
        ChatFailure.TIMEOUT: "The model timed out",
        ChatFailure.RATE_LIMITED: "The model is rate-limited",
        ChatFailure.UNAVAILABLE: "No model is reachable",
        ChatFailure.BAD_ANSWER: "The model's answer could not be used",
        ChatFailure.CAPPED: "Too many model requests this hour",
    }
    assert set(heads) == set(ChatFailure)
    for failure, head in heads.items():
        assert describe_failure(failure) == f"{head}; try again in a minute."
        assert describe_failure(failure, 125.0) == f"{head}; try again in 3 min."
        assert describe_failure(failure, 5.0) == f"{head}; try again in 1 min."


def test_logs_contain_only_class_and_status(caplog):
    caplog.set_level(logging.DEBUG)
    clock = FakeClock()
    nim = ScriptedLLM(nim_reply("", status=503), nim_reply("", status=503))
    openrouter = ScriptedLLM(httpx.Response(200, json={"error": {"message": "<MARKER-ERR> upstream"}}))
    chat, _ = make_chat(nim, openrouter, clock=clock)
    messages = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "<MARKER-IN>"}]
    with pytest.raises(ChatLLMNotUnderstood):
        chat.complete(messages, validate=validate_ok, request_id="req-77")
    # httpx's own INFO line carries the URL; production keeps httpx at WARNING (configure_logging)
    text = "\n".join(record.getMessage() for record in caplog.records if record.name.startswith("src."))
    assert "req-77" in text
    assert "503" in text and ChatFailure.UNAVAILABLE.value in text and ChatFailure.BAD_ANSWER.value in text
    for forbidden in ("<MARKER-IN>", "<MARKER-ERR>", "https://", "http://", "<NIM-KEY>", "<OR-KEY>", "nim.test", "upstream"):
        assert forbidden not in text


def test_describe_failure_daily_wording_and_long_waits():
    assert describe_failure(ChatFailure.CAPPED, 125.0) == "Too many model requests this hour; try again in 3 min."
    assert describe_failure(ChatFailure.CAPPED, 125.0, daily=True) == "Too many model requests today; try again in 3 min."
    assert describe_failure(ChatFailure.CAPPED, None, daily=True) == "Too many model requests today; try again in a minute."
    assert describe_failure(ChatFailure.CAPPED, 3 * 3600.0, daily=True) == "Too many model requests today; try again in 3 h."
    assert describe_failure(ChatFailure.CAPPED, 2.5 * 3600.0, daily=True) == "Too many model requests today; try again in 3 h."
    assert describe_failure(ChatFailure.TIMEOUT, 3 * 3600.0, daily=True) == "The model timed out; try again in 3 h."  # daily only changes the capped head
    assert ChatLLMLimited(30.0).daily is False and ChatLLMLimited(30.0, daily=True).daily is True
