import json
import logging
import threading
import time

import pytest

from src.evaluator import chain as chain_module
from src.evaluator.chain import DIGEST_MAX_SCORE, ProviderChainEvaluator, build_chain_from_settings
from src.evaluator.circuit_breaker import CircuitBreaker
from src.evaluator.errors import (
    AuthenticationFailed,
    BadRequest,
    EvaluationRejected,
    InvalidResponse,
    ModelNotFound,
    PermanentProviderError,
    ProvidersUnavailable,
    QuotaExhausted,
    RateLimited,
    ResponseTruncated,
    StopRequested,
    TransientProviderError,
    UnexpectedStatus,
)
from src.evaluator import prompts as prompts_module
from src.evaluator.providers import CompletionResult
from src.evaluator.scoring import ScoringPolicy
from src.evaluator.schemas import JobEvaluationResult
from src.storage.models import JobPosting


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class FakeProvider:
    """Scripted stand-in: each script item is an Exception to raise, or content (str) to return."""

    def __init__(self, name, script=(), *, available=True, breaker=None, max_tokens=4096, model=None):
        self.name = name
        self.script = list(script)
        self.breaker = breaker or CircuitBreaker(failure_threshold=3, cooldown_seconds=600)
        self.max_tokens = max_tokens
        self.model_name = model or f"{name}/model"
        self.available = available
        self.calls = []
        self._lock = threading.Lock()

    def is_available(self):
        return self.available

    def complete(self, system, user, max_tokens=None):
        with self._lock:
            self.calls.append({"system": system, "user": user, "max_tokens": max_tokens})
            item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(item, Exception):
            raise item
        return CompletionResult(content=item, model_used=self.model_name, latency=0.1, finish_reason="stop")


def payload(**overrides):
    data = {
        "score": 7.0,
        "summary": "Коротке резюме вакансії для кандидата.",
        "reasoning": "Reasonable fit.",
        "language_requirements": ["English"],
        "pros": ["remote"],
        "cons": [],
        "hard_reject": False,
        "is_core_role": False,
        "matches_bonus_criteria": False,
    }
    data.update(overrides)
    return json.dumps(data)


def make_job(title="Career Coach", **kw):
    args = dict(
        dedup_hash="h",
        site="indeed",
        job_url="https://example.test/job/1",
        title=title,
        company="Acme",
        location="Remote",
        description="Coach people through career transitions. " * 10,
        is_remote=True,
    )
    args.update(kw)
    return JobPosting(**args)


def make_chain(*providers, **kw):
    kw.setdefault("system_prompt", "SYSTEM PROMPT")
    return ProviderChainEvaluator(list(providers), **kw)


# ---------- ordering and basics ----------

def test_returns_validated_result_from_first_provider():
    nim = FakeProvider("nim", [payload(score=7.4)])
    router = FakeProvider("openrouter", [payload()])
    result = make_chain(nim, router).evaluate_job(make_job())
    assert isinstance(result, JobEvaluationResult)
    assert result.score == 7.4
    assert len(nim.calls) == 1
    assert router.calls == []


def test_eval_model_is_model_that_answered():
    nim = FakeProvider("nim", [payload()], model="actual/nim-model")
    assert make_chain(nim).evaluate_job(make_job()).eval_model == "actual/nim-model"


def test_is_wow_is_never_set_by_the_chain():
    nim = FakeProvider("nim", [payload(score=10, is_wow=True)])
    assert make_chain(nim).evaluate_job(make_job()).is_wow is False


def test_prompts_built_from_job_and_system_prompt_used_verbatim():
    nim = FakeProvider("nim", [payload()])
    make_chain(nim, system_prompt="MY SYSTEM").evaluate_job(make_job(title="Wellbeing Coach", company="Globex"))
    call = nim.calls[0]
    assert call["system"] == "MY SYSTEM"
    assert "Wellbeing Coach" in call["user"] and "Globex" in call["user"]


def test_chain_construction_never_reads_profile_file(monkeypatch):
    def boom():
        raise AssertionError("candidate profile must not be read at construction")

    monkeypatch.setattr(prompts_module, "load_candidate_persona", boom)
    nim = FakeProvider("nim", [payload()])
    chain = ProviderChainEvaluator([nim])  # must not raise
    assert nim.calls == []

    built = []
    monkeypatch.setattr(
        chain_module, "build_system_prompt", lambda persona=None: built.append(persona) or "BUILT"
    )
    chain.evaluate_job(make_job())
    chain.evaluate_job(make_job())
    assert len(built) == 1
    assert [c["system"] for c in nim.calls] == ["BUILT", "BUILT"]


def test_evaluate_job_uses_per_call_system_prompt():
    nim = FakeProvider("nim", [payload()])
    chain = make_chain(nim, system_prompt="FROZEN")
    chain.evaluate_job(make_job(), system_prompt="X")
    chain.evaluate_job(make_job())
    assert nim.calls[0]["system"] == "X"
    assert nim.calls[1]["system"] == "FROZEN"


def test_per_call_prompt_does_not_change_chain_default():
    nim = FakeProvider("nim", [payload()])
    chain = make_chain(nim, system_prompt="FROZEN")
    chain.evaluate_job(make_job(), system_prompt="OVERRIDE")
    chain.evaluate_job(make_job())
    assert nim.calls[-1]["system"] == "FROZEN"
    assert chain._system_prompt == "FROZEN"


def test_per_call_prompt_reaches_retry_and_fallback_provider():
    nim = FakeProvider("nim", ["not json", "still not json"])
    router = FakeProvider("openrouter", [payload()])
    make_chain(nim, router, system_prompt="FROZEN").evaluate_job(make_job(), system_prompt="X")
    assert [c["system"] for c in nim.calls] == ["X", "X"]
    assert [c["system"] for c in router.calls] == ["X"]


def test_requires_at_least_one_provider():
    with pytest.raises(ValueError):
        ProviderChainEvaluator([])


@pytest.mark.parametrize(
    "content",
    [
        "```json\n" + payload(score=6.5) + "\n```",
        "<think>draft {\"score\": 1}</think>" + payload(score=6.5),
        "Here you go:\n" + payload(score=6.5) + "\nUser Safety: safe.",
    ],
)
def test_tolerates_fences_think_and_trailing_text(content):
    nim = FakeProvider("nim", [content])
    assert make_chain(nim).evaluate_job(make_job()).score == 6.5


# ---------- fallback ----------

@pytest.mark.parametrize("error", [RateLimited(None), TransientProviderError("boom")])
def test_falls_back_to_next_provider_on_transient_error(error):
    nim = FakeProvider("nim", [error])
    router = FakeProvider("openrouter", [payload(score=6.8)], model="or/model")
    result = make_chain(nim, router).evaluate_job(make_job())
    assert result.score == 6.8
    assert result.eval_model == "or/model"


def test_invalid_json_retries_same_provider_then_falls_back():
    nim = FakeProvider("nim", ["not json at all"])
    router = FakeProvider("openrouter", [payload()])
    make_chain(nim, router).evaluate_job(make_job())
    assert len(nim.calls) == 2
    assert len(router.calls) == 1


def test_invalid_json_retry_on_same_provider_can_recover():
    nim = FakeProvider("nim", ["garbage", payload(score=7.7)])
    router = FakeProvider("openrouter", [payload()])
    result = make_chain(nim, router).evaluate_job(make_job())
    assert result.score == 7.7
    assert len(nim.calls) == 2
    assert router.calls == []


def test_schema_invalid_json_counts_as_invalid():
    bad = payload(summary="short")
    nim = FakeProvider("nim", [bad, payload(score=6.6)])
    assert make_chain(nim).evaluate_job(make_job()).score == 6.6
    assert len(nim.calls) == 2


def test_json_that_is_not_an_object_is_invalid():
    nim = FakeProvider("nim", ["[1, 2, 3]", payload()])
    make_chain(nim).evaluate_job(make_job())
    assert len(nim.calls) == 2


def test_truncated_response_retries_with_one_and_a_half_times_max_tokens():
    nim = FakeProvider("nim", [ResponseTruncated("len"), payload()], max_tokens=4096)
    make_chain(nim).evaluate_job(make_job())
    assert nim.calls[0]["max_tokens"] is None or nim.calls[0]["max_tokens"] == 4096
    assert nim.calls[1]["max_tokens"] == 6144


def test_plain_invalid_response_retry_keeps_default_max_tokens():
    nim = FakeProvider("nim", [InvalidResponse("empty"), payload()])
    make_chain(nim).evaluate_job(make_job())
    assert nim.calls[1]["max_tokens"] in (None, 4096)


# ---------- unavailable vs rejected ----------

def test_all_providers_transient_raises_providers_unavailable():
    nim = FakeProvider("nim", [TransientProviderError("x")])
    router = FakeProvider("openrouter", [RateLimited(2.0)])
    with pytest.raises(ProvidersUnavailable):
        make_chain(nim, router).evaluate_job(make_job())


def test_all_providers_skipped_raises_providers_unavailable():
    nim = FakeProvider("nim", [payload()])
    nim.breaker.open_immediately()
    router = FakeProvider("openrouter", [payload()], available=False)
    with pytest.raises(ProvidersUnavailable):
        make_chain(nim, router).evaluate_job(make_job())
    assert nim.calls == [] and router.calls == []


def test_invalid_everywhere_raises_evaluation_rejected():
    nim = FakeProvider("nim", ["garbage"])
    router = FakeProvider("openrouter", [InvalidResponse("empty")])
    with pytest.raises(EvaluationRejected):
        make_chain(nim, router).evaluate_job(make_job())
    assert len(nim.calls) == 2 and len(router.calls) == 2


def test_single_provider_invalid_is_rejected():
    nim = FakeProvider("nim", ["garbage"])
    with pytest.raises(EvaluationRejected):
        make_chain(nim).evaluate_job(make_job())


def test_provider_wide_failure_plus_invalid_is_rejected():
    # No provider answered and one reachable provider judged the job unusable:
    # count an attempt, otherwise one bad job would stall every batch forever.
    nim = FakeProvider("nim", [TransientProviderError("down")])
    router = FakeProvider("openrouter", ["garbage"])
    with pytest.raises(EvaluationRejected):
        make_chain(nim, router).evaluate_job(make_job())


def test_invalid_plus_quota_exhausted_fallback_is_rejected():
    nim = FakeProvider("nim", ["garbage"])
    router = FakeProvider("openrouter", [payload()], available=False)
    with pytest.raises(EvaluationRejected):
        make_chain(nim, router).evaluate_job(make_job())
    assert router.calls == []


def test_invalid_plus_quota_exhausted_during_call_is_rejected():
    nim = FakeProvider("nim", ["garbage"])
    router = FakeProvider("openrouter", [QuotaExhausted(1.0)])
    with pytest.raises(EvaluationRejected):
        make_chain(nim, router).evaluate_job(make_job())


def test_invalid_plus_breaker_open_fallback_is_rejected():
    nim = FakeProvider("nim", ["garbage"])
    router = FakeProvider("openrouter", [payload()])
    router.breaker.open_immediately()
    with pytest.raises(EvaluationRejected):
        make_chain(nim, router).evaluate_job(make_job())
    assert router.calls == []


def test_only_provider_wide_failures_is_unavailable():
    nim = FakeProvider("nim", [AuthenticationFailed("401")])
    router = FakeProvider("openrouter", [ModelNotFound("404")])
    with pytest.raises(ProvidersUnavailable):
        make_chain(nim, router).evaluate_job(make_job())


@pytest.mark.parametrize("error", [PermanentProviderError("HTTP 402"), UnexpectedStatus("HTTP 409")])
def test_other_permanent_errors_are_provider_wide_breaker_failures(error):
    nim = FakeProvider("nim", [error], breaker=CircuitBreaker(failure_threshold=2))
    chain = make_chain(nim)
    with pytest.raises(ProvidersUnavailable):
        chain.evaluate_job(make_job())
    assert nim.breaker.state == "closed"
    with pytest.raises(ProvidersUnavailable):
        chain.evaluate_job(make_job())
    assert nim.breaker.state == "open"


def test_bad_request_everywhere_is_rejected_and_not_a_breaker_failure():
    nim = FakeProvider("nim", [BadRequest("400")], breaker=CircuitBreaker(failure_threshold=1))
    with pytest.raises(EvaluationRejected):
        make_chain(nim).evaluate_job(make_job())
    assert nim.breaker.state == "closed"


def test_bad_request_falls_through_to_next_provider():
    nim = FakeProvider("nim", [BadRequest("422")])
    router = FakeProvider("openrouter", [payload()])
    make_chain(nim, router).evaluate_job(make_job())
    assert len(router.calls) == 1


# ---------- breaker accounting ----------

def test_open_breaker_skips_provider_without_a_request():
    nim = FakeProvider("nim", [payload()])
    nim.breaker.open_immediately()
    router = FakeProvider("openrouter", [payload()])
    make_chain(nim, router).evaluate_job(make_job())
    assert nim.calls == []
    assert len(router.calls) == 1


def test_unavailable_provider_is_skipped_without_a_request():
    nim = FakeProvider("nim", [payload()], available=False)
    router = FakeProvider("openrouter", [payload()])
    make_chain(nim, router).evaluate_job(make_job())
    assert nim.calls == []


def test_transient_failures_trip_breaker_after_threshold():
    nim = FakeProvider("nim", [TransientProviderError("x")], breaker=CircuitBreaker(failure_threshold=3))
    router = FakeProvider("openrouter", [payload()])
    chain = make_chain(nim, router)
    for _ in range(3):
        chain.evaluate_job(make_job())
    assert nim.breaker.state == "open"
    chain.evaluate_job(make_job())
    assert len(nim.calls) == 3  # fourth job skipped NIM


def test_success_resets_breaker_failure_count():
    nim = FakeProvider(
        "nim",
        [TransientProviderError("x"), TransientProviderError("x"), payload(), TransientProviderError("x")],
        breaker=CircuitBreaker(failure_threshold=3),
    )
    router = FakeProvider("openrouter", [payload()])
    chain = make_chain(nim, router)
    for _ in range(4):
        chain.evaluate_job(make_job())
    assert nim.breaker.state == "closed"


def test_auth_failure_opens_breaker_immediately():
    nim = FakeProvider("nim", [AuthenticationFailed("401")])
    router = FakeProvider("openrouter", [payload()])
    make_chain(nim, router).evaluate_job(make_job())
    assert nim.breaker.state == "open"


def test_model_not_found_is_a_breaker_failure():
    nim = FakeProvider("nim", [ModelNotFound("404")], breaker=CircuitBreaker(failure_threshold=2))
    router = FakeProvider("openrouter", [payload()])
    chain = make_chain(nim, router)
    chain.evaluate_job(make_job())
    assert nim.breaker.state == "closed"
    chain.evaluate_job(make_job())
    assert nim.breaker.state == "open"


def test_quota_exhausted_skips_provider_without_breaker_failure():
    nim = FakeProvider("nim", [payload()])
    router = FakeProvider(
        "openrouter", [QuotaExhausted(1.0)], breaker=CircuitBreaker(failure_threshold=1)
    )
    nim.breaker.open_immediately()
    with pytest.raises(ProvidersUnavailable):
        make_chain(nim, router).evaluate_job(make_job())
    assert router.breaker.state == "closed"


def test_invalid_responses_trip_breaker_only_after_three_consecutive_jobs():
    nim = FakeProvider("nim", ["garbage"], breaker=CircuitBreaker(failure_threshold=1))
    router = FakeProvider("openrouter", [payload()])
    chain = make_chain(nim, router)
    chain.evaluate_job(make_job())
    chain.evaluate_job(make_job())
    assert nim.breaker.state == "closed"
    chain.evaluate_job(make_job())
    assert nim.breaker.state == "open"


def test_streak_is_not_reset_at_threshold_each_further_invalid_job_is_a_failure():
    clock = Clock()
    breaker = CircuitBreaker(failure_threshold=2, cooldown_seconds=100, time_fn=clock)
    nim = FakeProvider("nim", ["garbage"], breaker=breaker)
    router = FakeProvider("openrouter", [payload()])
    chain = make_chain(nim, router)
    for _ in range(3):  # streak 1, 2, 3 -> one failure recorded (threshold is 2, so closed)
        chain.evaluate_job(make_job())
    assert breaker.state == "closed"
    chain.evaluate_job(make_job())  # streak 4 -> second failure -> open
    assert breaker.state == "open"


def test_half_open_probe_with_long_invalid_streak_reopens_breaker():
    clock = Clock()
    breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=100, time_fn=clock)
    nim = FakeProvider("nim", ["garbage"], breaker=breaker)
    router = FakeProvider("openrouter", [payload()])
    chain = make_chain(nim, router)
    for _ in range(3):
        chain.evaluate_job(make_job())
    assert breaker.state == "open"
    clock.now += 101
    chain.evaluate_job(make_job())  # probe, still invalid, streak >= 3 -> failure -> reopened
    assert breaker.state == "open"
    clock.now += 50
    assert breaker.allow() is False  # cooldown restarted at the probe failure


def test_valid_response_resets_consecutive_invalid_count():
    nim = FakeProvider(
        "nim",
        ["garbage", "garbage", payload(), "garbage", "garbage", "garbage", "garbage"],
        breaker=CircuitBreaker(failure_threshold=1),
    )
    router = FakeProvider("openrouter", [payload()])
    chain = make_chain(nim, router)
    # job1: 2 garbage -> invalid; job2: valid ... sequence per call, two calls per invalid job
    chain.evaluate_job(make_job())  # garbage, garbage -> invalid (1)
    chain.evaluate_job(make_job())  # valid -> reset
    chain.evaluate_job(make_job())  # garbage, garbage -> invalid (1)
    chain.evaluate_job(make_job())  # garbage, garbage -> invalid (2)
    assert nim.breaker.state == "closed"


def test_half_open_probe_released_after_invalid_response():
    clock = Clock()
    breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=100, time_fn=clock)
    breaker.open_immediately()
    clock.now += 101
    nim = FakeProvider("nim", ["garbage"], breaker=breaker)
    with pytest.raises(EvaluationRejected):
        make_chain(nim).evaluate_job(make_job())
    # the probe was handed back: the next caller can claim it again
    assert breaker.state == "half_open"
    assert breaker.allow() is True


def test_half_open_probe_success_closes_breaker():
    clock = Clock()
    breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=100, time_fn=clock)
    breaker.open_immediately()
    clock.now += 101
    nim = FakeProvider("nim", [payload()], breaker=breaker)
    make_chain(nim).evaluate_job(make_job())
    assert breaker.state == "closed"


def test_half_open_probe_failure_reopens_breaker():
    clock = Clock()
    breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=100, time_fn=clock)
    breaker.open_immediately()
    clock.now += 101
    nim = FakeProvider("nim", [TransientProviderError("still down")], breaker=breaker)
    router = FakeProvider("openrouter", [payload()])
    make_chain(nim, router).evaluate_job(make_job())
    assert breaker.state == "open"


def test_only_one_job_claims_the_half_open_probe():
    clock = Clock()
    breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=100, time_fn=clock)
    breaker.open_immediately()
    clock.now += 101
    claimed = breaker.allow()  # another caller holds the probe
    assert claimed
    nim = FakeProvider("nim", [payload()], breaker=breaker)
    router = FakeProvider("openrouter", [payload()])
    make_chain(nim, router).evaluate_job(make_job())
    assert nim.calls == []
    # our job did not touch the other caller's probe
    assert breaker.allow() is False


def test_stop_requested_propagates_and_releases_probe_without_failure():
    clock = Clock()
    breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=100, time_fn=clock)
    breaker.open_immediately()
    clock.now += 101
    nim = FakeProvider("nim", [StopRequested("bye")], breaker=breaker)
    router = FakeProvider("openrouter", [payload()])
    with pytest.raises(StopRequested):
        make_chain(nim, router).evaluate_job(make_job())
    assert router.calls == []
    assert breaker.allow() is True  # probe was released, not failed


def test_unexpected_exception_still_releases_probe():
    clock = Clock()
    breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=100, time_fn=clock)
    breaker.open_immediately()
    clock.now += 101
    nim = FakeProvider("nim", [RuntimeError("bug")], breaker=breaker)
    with pytest.raises(RuntimeError):
        make_chain(nim).evaluate_job(make_job())
    assert breaker.allow() is True


# ---------- stop ----------

def test_stop_sets_event_and_aborts_evaluation():
    nim = FakeProvider("nim", [payload()])
    chain = make_chain(nim)
    chain.stop()
    assert chain.stop_event.is_set()
    with pytest.raises(StopRequested):
        chain.evaluate_job(make_job())
    assert nim.calls == []


def test_shared_stop_event_is_used():
    event = threading.Event()
    chain = make_chain(FakeProvider("nim", [payload()]), stop_event=event)
    chain.stop()
    assert event.is_set()


# ---------- post-processing ----------

def evaluate(job=None, bonus_enabled=True, **overrides):
    nim = FakeProvider("nim", [payload(**overrides)])
    return make_chain(nim, bonus_enabled=bonus_enabled).evaluate_job(job or make_job())


def test_score_clamped_into_range():
    assert evaluate(score=14, is_core_role=True).score == 10.0
    assert evaluate(score=-3, hard_reject=False).score <= 2.0


def test_bonus_applies_when_base_at_least_six():
    assert evaluate(score=6.0, matches_bonus_criteria=True).score == 7.5


def test_per_call_policy_bonus_and_cap():
    nim = FakeProvider("nim", [payload(score=7.0, matches_bonus_criteria=True, is_core_role=False)])
    chain = make_chain(nim)
    bonus_on = ScoringPolicy(system_prompt="P1", bonus_enabled=True, core_cap=8.4)
    bonus_off = ScoringPolicy(system_prompt="P2", bonus_enabled=False, core_cap=None)
    assert chain.evaluate_job(make_job(), policy=bonus_on).score == 8.4  # 7.0 + 1.5 capped, non core
    assert chain.evaluate_job(make_job(), policy=bonus_off).score == 7.0
    uncapped = ScoringPolicy(system_prompt="P3", bonus_enabled=True, bonus_points=2.5, core_cap=None)
    assert chain.evaluate_job(make_job(), policy=uncapped).score == 9.5
    assert [c["system"] for c in nim.calls] == ["P1", "P2", "P3"]  # policy prompt used per call


def test_system_prompt_arg_wins_over_policy_prompt():
    nim = FakeProvider("nim", [payload()])
    chain = make_chain(nim, system_prompt="FROZEN")
    chain.evaluate_job(make_job(), system_prompt="ARG", policy=ScoringPolicy(system_prompt="POLICY"))
    chain.evaluate_job(make_job(), policy=ScoringPolicy(system_prompt="POLICY"))
    assert [c["system"] for c in nim.calls] == ["ARG", "POLICY"]


def test_title_safety_net_uses_policy_rules():
    nim = FakeProvider("nim", [payload(score=7.0)])
    chain = make_chain(nim)
    job = make_job(title="Scrum Master")
    assert chain.evaluate_job(job, policy=ScoringPolicy(system_prompt="P", rules=())).score == 7.0
    assert chain.evaluate_job(job).score == 2.0  # default policy: legacy rules


def test_job_prompt_remote_line():
    nim = FakeProvider("nim", [payload()])
    chain = make_chain(nim)
    policy = ScoringPolicy(system_prompt="P", remote_filter_query_ids=frozenset({"Q-REMOTE"}))
    chain.evaluate_job(make_job(remote_status="verified", source_query="Q-REMOTE|linkedin"), policy=policy)
    chain.evaluate_job(make_job(remote_status="unverified", source_query="Q-REMOTE|linkedin"), policy=policy)
    chain.evaluate_job(make_job(remote_status="unverified", source_query="Q-OTHER|linkedin"), policy=policy)
    chain.evaluate_job(make_job(remote_status="unverified", source_query=None), policy=policy)
    remote_lines = [
        next(line for line in c["user"].splitlines() if line.startswith("Remote:")) for c in nim.calls
    ]
    assert remote_lines == [
        "Remote: yes",
        "Remote: unverified (found with the remote filter)",
        "Remote: unverified",
        "Remote: unverified",
    ]


def test_explicit_system_prompt_without_kwarg_disables_bonus():
    nim = FakeProvider("nim", [payload(score=7.0, matches_bonus_criteria=True)])
    assert make_chain(nim).evaluate_job(make_job()).score == 7.0


def test_bonus_capped_at_ten():
    assert evaluate(score=9.0, matches_bonus_criteria=True, is_core_role=True).score == 10.0


def test_bonus_gate_blocks_mediocre_base():
    assert evaluate(score=5.0, matches_bonus_criteria=True).score == 5.0
    assert evaluate(score=5.9, matches_bonus_criteria=True).score == 5.9


def test_bonus_applied_once_only():
    assert evaluate(score=7.0, matches_bonus_criteria=True, is_core_role=True).score == 8.5


def test_no_bonus_flag_no_change():
    assert evaluate(score=6.0).score == 6.0


def test_hard_reject_caps_at_two_and_ignores_bonus():
    result = evaluate(score=8.0, hard_reject=True, matches_bonus_criteria=True)
    assert result.score == 2.0


def test_low_score_gets_no_bonus():
    assert evaluate(score=2.0, matches_bonus_criteria=True).score == 2.0
    assert evaluate(score=1.5, matches_bonus_criteria=True).score == 1.5


def test_score_rounded_to_one_decimal():
    assert evaluate(score=7.26).score == 7.3
    assert evaluate(score=6.0 + 0.04, matches_bonus_criteria=True).score == 7.5


def test_title_safety_net_caps_score_and_notes_reasoning():
    result = evaluate(job=make_job(title="Director of Sales"), score=9.0)
    assert result.score == 2.0
    assert "title" in result.reasoning.lower()
    assert "Reasonable fit." in result.reasoning


def test_title_safety_net_ignores_bonus():
    result = evaluate(job=make_job(title="Director of Sales"), score=9.0, matches_bonus_criteria=True)
    assert result.score == 2.0


def test_title_safety_net_leaves_clean_titles_alone():
    assert evaluate(job=make_job(title="Career Coach"), score=9.0, is_core_role=True).score == 9.0


def test_title_safety_net_does_not_raise_low_scores():
    assert evaluate(job=make_job(title="Director of Sales"), score=1.4).score == 1.4


# ---------- logging ----------

def test_info_logs_do_not_contain_job_text(caplog):
    caplog.set_level(logging.INFO)
    nim = FakeProvider("nim", [payload()])
    job = make_job(title="Zebra Unique Title Coach", company="Quokka Corp", job_url="https://secret.test/xyz")
    make_chain(nim).evaluate_job(job)
    assert "Zebra" not in caplog.text
    assert "Quokka" not in caplog.text
    assert "secret.test" not in caplog.text


# ---------- concurrency ----------

class SlowProvider(FakeProvider):
    def complete(self, system, user, max_tokens=None):
        time.sleep(0.005)
        return super().complete(system, user, max_tokens)


def test_concurrent_evaluate_job_on_shared_instance():
    nim = SlowProvider("nim", [payload(score=7.0)])
    chain = make_chain(nim)
    results, errors = [], []
    lock = threading.Lock()

    def worker(n):
        try:
            for i in range(5):
                res = chain.evaluate_job(make_job(title=f"Career Coach {n}-{i}"))
                with lock:
                    results.append(res)
        except Exception as exc:  # pragma: no cover - failure path
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)
    assert errors == []
    assert len(results) == 40
    assert all(r.score == 7.0 and r.eval_model == "nim/model" for r in results)
    assert len(nim.calls) == 40


def test_concurrent_fallback_and_breaker_with_threads():
    nim = SlowProvider("nim", [TransientProviderError("x")], breaker=CircuitBreaker(failure_threshold=3))
    router = SlowProvider("openrouter", [payload(score=6.5)])
    chain = make_chain(nim, router)
    results = []
    lock = threading.Lock()

    def worker():
        for _ in range(5):
            res = chain.evaluate_job(make_job())
            with lock:
                results.append(res)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)
    assert len(results) == 30
    assert all(r.score == 6.5 for r in results)
    assert nim.breaker.state == "open"


# ---------- factory ----------

from src.config import Settings  # noqa: E402
from src.evaluator.chain import build_chain_from_settings  # noqa: E402
from src.evaluator.errors import ConfigurationError  # noqa: E402
from src.evaluator.providers import NvidiaNimProvider, OpenRouterProvider  # noqa: E402

_PROVIDER_ENV = [
    "NVIDIA_API_KEY", "OPENROUTER_API_KEY", "OPENROUTER_MODELS", "NVIDIA_MODEL",
    "NVIDIA_RPM_LIMIT", "OPENROUTER_RPM_LIMIT", "OPENROUTER_DAILY_BUDGET",
    "LLM_BREAKER_FAILURES", "LLM_BREAKER_COOLDOWN_SECONDS", "EVAL_MAX_WORKERS",
]


@pytest.fixture
def clean_env(monkeypatch):
    for name in _PROVIDER_ENV:
        monkeypatch.delenv(name, raising=False)


def settings_with(**env):
    return Settings(_env_file=None, **env)


def test_factory_raises_clear_error_when_no_provider_enabled(clean_env):
    with pytest.raises(ConfigurationError) as exc_info:
        build_chain_from_settings(settings_with(), system_prompt="p")
    assert isinstance(exc_info.value, ValueError)
    message = str(exc_info.value)
    assert "NVIDIA_API_KEY" in message and "OPENROUTER" in message


def test_factory_openrouter_key_without_models_is_not_enabled(clean_env, caplog):
    caplog.set_level(logging.WARNING)
    with pytest.raises(ConfigurationError):
        build_chain_from_settings(settings_with(OPENROUTER_API_KEY="or-key"), system_prompt="p")
    assert "OPENROUTER_MODELS" in caplog.text
    assert "or-key" not in caplog.text


def test_factory_nim_only(clean_env):
    settings = settings_with(NVIDIA_API_KEY="nv-key", NVIDIA_MODEL="vendor/model", NVIDIA_RPM_LIMIT=12,
                             LLM_BREAKER_FAILURES=4, LLM_BREAKER_COOLDOWN_SECONDS=90)
    chain = build_chain_from_settings(settings, system_prompt="p")
    (nim,) = chain.providers
    assert isinstance(nim, NvidiaNimProvider)
    assert nim.model == "vendor/model"
    assert nim.reasoning_effort == settings.nvidia_reasoning_effort
    assert nim.temperature == settings.nvidia_temperature
    assert nim.max_tokens == settings.nvidia_max_tokens
    assert nim.limiter.max_calls == 12
    assert nim.breaker.failure_threshold == 4
    assert nim.breaker.cooldown_seconds == 90


def test_factory_both_providers_in_order(clean_env):
    settings = settings_with(NVIDIA_API_KEY="nv-key", OPENROUTER_API_KEY="or-key",
                             OPENROUTER_MODELS="a/one:free, b/two:free", OPENROUTER_DAILY_BUDGET=7,
                             OPENROUTER_RPM_LIMIT=9)
    chain = build_chain_from_settings(settings, system_prompt="p")
    nim, router = chain.providers
    assert isinstance(nim, NvidiaNimProvider) and isinstance(router, OpenRouterProvider)
    assert router.models == ["a/one:free", "b/two:free"]
    assert router.daily_budget == 7
    assert router.limiter.max_calls == 9
    assert router.limiter is not nim.limiter
    assert router.breaker is not nim.breaker


def test_factory_openrouter_only(clean_env):
    settings = settings_with(OPENROUTER_API_KEY="or-key", OPENROUTER_MODELS="a/one:free")
    chain = build_chain_from_settings(settings, system_prompt="p")
    assert [type(p) for p in chain.providers] == [OpenRouterProvider]


def test_factory_shares_one_stop_event_with_limiters(clean_env):
    event = threading.Event()
    settings = settings_with(NVIDIA_API_KEY="nv-key", OPENROUTER_API_KEY="or-key", OPENROUTER_MODELS="a/one:free")
    chain = build_chain_from_settings(settings, stop_event=event, system_prompt="p")
    assert chain.stop_event is event
    for provider in chain.providers:
        assert provider.stop_event is event
        assert provider.limiter.stop_event is event
    chain.stop()
    assert event.is_set()


def test_factory_defaults_to_cached_settings(clean_env, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # hermetic: no developer .env in the working directory
    monkeypatch.setenv("NVIDIA_API_KEY", "nv-key")
    chain = build_chain_from_settings(system_prompt="p")
    assert chain.providers[0].name == "nim"


# ---------- error text hygiene ----------

def test_parse_errors_do_not_embed_model_output():
    marker = "SENSITIVE-MODEL-OUTPUT-xyz"
    with pytest.raises(InvalidResponse) as exc_info:
        ProviderChainEvaluator._parse(f"{marker} no json here")
    assert marker not in str(exc_info.value)
    with pytest.raises(InvalidResponse) as exc_info:
        ProviderChainEvaluator._parse(json.dumps({"score": "x", "summary": marker}))
    assert marker not in str(exc_info.value)
    with pytest.raises(InvalidResponse) as exc_info:
        ProviderChainEvaluator._parse(f'["{marker}"]')
    assert marker not in str(exc_info.value)


# ---------- WOW only for direct coaching ----------


def test_non_direct_coaching_raw_nine_is_capped_at_digest_max():
    result = evaluate(score=9.0, is_core_role=False)
    assert result.score == DIGEST_MAX_SCORE == 8.4


def test_non_direct_coaching_bonus_cannot_lift_past_digest_max():
    result = evaluate(score=8.0, matches_bonus_criteria=True, is_core_role=False)
    assert result.score == 8.4


def test_non_direct_coaching_below_cap_is_unchanged():
    assert evaluate(score=7.0, is_core_role=False).score == 7.0
    assert evaluate(score=6.0, matches_bonus_criteria=True, is_core_role=False).score == 7.5


def test_direct_coaching_with_bonus_reaches_wow_band():
    result = evaluate(score=8.0, matches_bonus_criteria=True, is_core_role=True)
    assert result.score == 9.5


def test_direct_coaching_is_not_capped():
    assert evaluate(score=9.0, is_core_role=True).score == 9.0


def test_hard_reject_still_wins_over_direct_coaching():
    result = evaluate(score=9.0, hard_reject=True, is_core_role=True, matches_bonus_criteria=True)
    assert result.score == 2.0


def test_missing_direct_coaching_flag_is_treated_as_false_and_capped():
    data = json.loads(payload(score=9.2))
    del data["is_core_role"]
    result = make_chain(FakeProvider("nim", [json.dumps(data)])).evaluate_job(make_job())
    assert result.is_core_role is False
    assert result.score == 8.4


def test_cap_is_logged_with_title_raw_and_final_score(caplog):
    with caplog.at_level(logging.INFO, logger="src.evaluator.chain"):
        evaluate(job=make_job(title="Training Lead"), score=9.0, is_core_role=False)
    messages = [r.getMessage() for r in caplog.records if r.levelno >= logging.INFO]
    assert any("Training Lead" in m and "9.0" in m and "8.4" in m for m in messages)


def test_cap_is_silent_when_it_does_not_bite(caplog):
    with caplog.at_level(logging.INFO, logger="src.evaluator.chain"):
        evaluate(score=7.0, is_core_role=False)
    assert not any("capped" in r.getMessage().lower() for r in caplog.records)


# ---------- raw score recording ----------

def test_postprocess_records_raw_score_before_adjustments():
    boosted = evaluate(score=7.0, matches_bonus_criteria=True, is_core_role=True)
    assert boosted.score == 8.5  # 7.0 + BONUS_POINTS (1.5), direct coaching so no cap
    assert boosted.raw_score == 7.0

    rejected = evaluate(score=9.0, hard_reject=True)
    assert rejected.score == 2.0
    assert rejected.raw_score == 9.0


def test_postprocess_raw_score_ignores_non_core_cap_and_is_clamped():
    capped = evaluate(score=9.5, is_core_role=False)
    assert capped.score == DIGEST_MAX_SCORE
    assert capped.raw_score == 9.5

    assert evaluate(score=14, is_core_role=True).raw_score == 10.0


def test_old_model_keys_round_trip_to_the_stored_columns(tmp_path):
    """A model that still answers with the old keys ends with eval_is_core_role / summary stored (not NULL)."""
    from src.storage.db import Database

    old = json.dumps({
        "score": 9.0, "summary_uk": "Старе резюме вакансії для кандидата.", "reasoning": "r",
        "role_is_direct_coaching": True,
    })
    result = make_chain(FakeProvider("nim", [old])).evaluate_job(make_job())
    assert (result.summary, result.is_core_role, result.score) == ("Старе резюме вакансії для кандидата.", True, 9.0)

    db = Database(tmp_path / "old_keys.db")
    db.init_db()
    job_id, _ = db.save_job(make_job(dedup_hash="old-keys"))
    db.update_evaluation(job_id, result)
    stored = db.get_job_by_id(job_id)
    assert stored.eval_is_core_role is True
    assert stored.eval_summary_uk == "Старе резюме вакансії для кандидата."


def test_cap_log_uses_a_compact_number(caplog):
    nim = FakeProvider("nim", [payload(score=9.0, is_core_role=False)])
    policy = ScoringPolicy(system_prompt="P", wow_threshold=8.25, digest_threshold=8.15, core_cap=8.15)
    with caplog.at_level(logging.INFO, logger="src.evaluator.chain"):
        make_chain(nim).evaluate_job(make_job(), policy=policy)
    text = " ".join(r.getMessage() for r in caplog.records if "Core-role cap" in r.getMessage())
    assert "capped at 8.15" in text and "final score 8.15" in text


def test_default_policy_is_built_once_without_taking_the_lock_again():
    chain = make_chain(FakeProvider("nim", [payload()]))
    first = chain._get_default_policy()
    chain._default_lock.acquire()
    try:  # a held lock must not block the fast path
        assert chain._get_default_policy() is first
    finally:
        chain._default_lock.release()


def test_default_policy_comes_from_the_shared_no_profile_helper():
    from src.evaluator.scoring import no_profile_policy

    chain = make_chain(FakeProvider("nim", [payload()]), bonus_enabled=True, wow_threshold=9.0, digest_threshold=7.0)
    assert chain._get_default_policy() == no_profile_policy("SYSTEM PROMPT", 9.0, 7.0, bonus_enabled=True)
    plain = make_chain(FakeProvider("nim", [payload()]))
    assert plain._get_default_policy() == no_profile_policy("SYSTEM PROMPT")


def test_build_chain_from_settings_forwards_the_env_thresholds():
    from src.config import Settings

    settings = Settings(_env_file=None, NVIDIA_API_KEY="k", WOW_SCORE_THRESHOLD=9.0, DIGEST_SCORE_THRESHOLD=7.0)
    chain = build_chain_from_settings(settings, system_prompt="p")
    policy = chain._get_default_policy()
    assert (policy.wow_threshold, policy.digest_threshold, policy.core_cap) == (9.0, 7.0, 8.9)
