import json
import logging
import threading
import time

import pytest

from src.evaluator import chain as chain_module
from src.evaluator.chain import DIGEST_MAX_SCORE, ProviderChainEvaluator
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
from src.evaluator.providers import CompletionResult
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
        "summary_uk": "Коротке резюме вакансії для кандидата.",
        "reasoning": "Reasonable fit.",
        "language_requirements": ["English"],
        "pros": ["remote"],
        "cons": [],
        "hard_reject": False,
        "role_is_direct_coaching": False,
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


def test_system_prompt_built_once_at_construction(monkeypatch):
    built = []
    monkeypatch.setattr(chain_module, "build_system_prompt", lambda persona=None: built.append(persona) or "BUILT")
    monkeypatch.setattr(chain_module, "load_candidate_persona", lambda: None)
    nim = FakeProvider("nim", [payload()])
    chain = ProviderChainEvaluator([nim])
    chain.evaluate_job(make_job())
    chain.evaluate_job(make_job())
    assert len(built) == 1
    assert nim.calls[0]["system"] == "BUILT"


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
    bad = payload(summary_uk="short")
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
    assert evaluate(score=14, role_is_direct_coaching=True).score == 10.0
    assert evaluate(score=-3, hard_reject=False).score <= 2.0


def test_bonus_applies_when_base_at_least_six():
    assert evaluate(score=6.0, matches_bonus_criteria=True).score == 7.5


def _chain_from_persona(monkeypatch, persona, payload_json):
    monkeypatch.setattr(chain_module, "load_candidate_persona", lambda: persona)
    nim = FakeProvider("nim", [payload_json])
    return ProviderChainEvaluator([nim])


def test_generic_persona_ignores_model_bonus_flag(monkeypatch, caplog):
    chain = _chain_from_persona(monkeypatch, None, payload(score=7.0, matches_bonus_criteria=True))
    with caplog.at_level(logging.DEBUG, logger="src.evaluator.chain"):
        assert chain.evaluate_job(make_job()).score == 7.0
    assert any("bonus" in r.getMessage().lower() and r.levelno == logging.DEBUG for r in caplog.records)


def test_persona_without_bonus_section_ignores_flag(monkeypatch):
    chain = _chain_from_persona(monkeypatch, "- Domain: Coaching", payload(score=7.0, matches_bonus_criteria=True))
    assert chain.evaluate_job(make_job()).score == 7.0


def test_persona_with_bonus_section_applies_bonus(monkeypatch):
    persona = "- Domain: Coaching\n\n## Bonus criteria\n- Something valued"
    chain = _chain_from_persona(
        monkeypatch, persona, payload(score=7.0, matches_bonus_criteria=True, role_is_direct_coaching=True)
    )
    assert chain.evaluate_job(make_job()).score == 8.5


def test_persona_with_bonus_section_still_gated_by_base_score(monkeypatch):
    persona = "## Bonus criteria\n- Something valued"
    chain = _chain_from_persona(monkeypatch, persona, payload(score=5.0, matches_bonus_criteria=True))
    assert chain.evaluate_job(make_job()).score == 5.0


def test_explicit_kwarg_wins_over_persona(monkeypatch):
    monkeypatch.setattr(chain_module, "load_candidate_persona", lambda: "## Bonus criteria\n- x")
    off = ProviderChainEvaluator([FakeProvider("nim", [payload(score=7.0, matches_bonus_criteria=True)])], bonus_enabled=False)
    assert off.evaluate_job(make_job()).score == 7.0
    monkeypatch.setattr(chain_module, "load_candidate_persona", lambda: None)
    on = ProviderChainEvaluator(
        [FakeProvider("nim", [payload(score=7.0, matches_bonus_criteria=True, role_is_direct_coaching=True)])],
        bonus_enabled=True,
    )
    assert on.evaluate_job(make_job()).score == 8.5


def test_explicit_system_prompt_without_kwarg_disables_bonus():
    nim = FakeProvider("nim", [payload(score=7.0, matches_bonus_criteria=True)])
    assert make_chain(nim).evaluate_job(make_job()).score == 7.0


def test_bonus_capped_at_ten():
    assert evaluate(score=9.0, matches_bonus_criteria=True, role_is_direct_coaching=True).score == 10.0


def test_bonus_gate_blocks_mediocre_base():
    assert evaluate(score=5.0, matches_bonus_criteria=True).score == 5.0
    assert evaluate(score=5.9, matches_bonus_criteria=True).score == 5.9


def test_bonus_applied_once_only():
    assert evaluate(score=7.0, matches_bonus_criteria=True, role_is_direct_coaching=True).score == 8.5


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
    assert evaluate(job=make_job(title="Career Coach"), score=9.0, role_is_direct_coaching=True).score == 9.0


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
        ProviderChainEvaluator._parse(json.dumps({"score": "x", "summary_uk": marker}))
    assert marker not in str(exc_info.value)
    with pytest.raises(InvalidResponse) as exc_info:
        ProviderChainEvaluator._parse(f'["{marker}"]')
    assert marker not in str(exc_info.value)


# ---------- WOW only for direct coaching ----------


def test_non_direct_coaching_raw_nine_is_capped_at_digest_max():
    result = evaluate(score=9.0, role_is_direct_coaching=False)
    assert result.score == DIGEST_MAX_SCORE == 8.4


def test_non_direct_coaching_bonus_cannot_lift_past_digest_max():
    result = evaluate(score=8.0, matches_bonus_criteria=True, role_is_direct_coaching=False)
    assert result.score == 8.4


def test_non_direct_coaching_below_cap_is_unchanged():
    assert evaluate(score=7.0, role_is_direct_coaching=False).score == 7.0
    assert evaluate(score=6.0, matches_bonus_criteria=True, role_is_direct_coaching=False).score == 7.5


def test_direct_coaching_with_bonus_reaches_wow_band():
    result = evaluate(score=8.0, matches_bonus_criteria=True, role_is_direct_coaching=True)
    assert result.score == 9.5


def test_direct_coaching_is_not_capped():
    assert evaluate(score=9.0, role_is_direct_coaching=True).score == 9.0


def test_hard_reject_still_wins_over_direct_coaching():
    result = evaluate(score=9.0, hard_reject=True, role_is_direct_coaching=True, matches_bonus_criteria=True)
    assert result.score == 2.0


def test_missing_direct_coaching_flag_is_treated_as_false_and_capped():
    data = json.loads(payload(score=9.2))
    del data["role_is_direct_coaching"]
    result = make_chain(FakeProvider("nim", [json.dumps(data)])).evaluate_job(make_job())
    assert result.role_is_direct_coaching is False
    assert result.score == 8.4


def test_cap_is_logged_with_title_raw_and_final_score(caplog):
    with caplog.at_level(logging.INFO, logger="src.evaluator.chain"):
        evaluate(job=make_job(title="Training Lead"), score=9.0, role_is_direct_coaching=False)
    messages = [r.getMessage() for r in caplog.records if r.levelno >= logging.INFO]
    assert any("Training Lead" in m and "9.0" in m and "8.4" in m for m in messages)


def test_cap_is_silent_when_it_does_not_bite(caplog):
    with caplog.at_level(logging.INFO, logger="src.evaluator.chain"):
        evaluate(score=7.0, role_is_direct_coaching=False)
    assert not any("capped" in r.getMessage().lower() for r in caplog.records)
