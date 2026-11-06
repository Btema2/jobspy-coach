"""Provider chain: NIM first, OpenRouter as fallback, one validated result per job.

``ProviderChainEvaluator.evaluate_job(job)`` is synchronous and thread-safe; one
shared instance serves every worker thread. It takes no alert thresholds and
never sets ``is_wow`` (the daemon derives that from the final score).

Breaker accounting lives HERE and only here. For every provider the chain calls
``breaker.allow()`` once; only when that returned True does it settle the breaker,
exactly once, in a ``finally`` (record_success / record_failure /
open_immediately, else release_probe). A caller whose ``allow()`` was refused
never touches that breaker, so it cannot free another thread's half-open probe.
Providers merely raise typed errors (see ``providers.py``).

Outcome rules per job:
  * valid answer from a provider           -> post-process and return it
  * provider-wide trouble (transient, 429, quota, bad key, missing model,
    open breaker)                          -> try the next provider
  * invalid answer (unparseable, schema-invalid, empty, truncated) -> retry once
    on the same provider (x1.5 max_tokens after truncation), then next provider
  * nobody answered validly and at least one reachable provider judged the job
    unusable (invalid / truncated after the retry, or HTTP 400/413/422)
                                           -> ``EvaluationRejected`` (job-specific, counts an
    attempt; otherwise one bad job would stall every batch whenever a fallback is down)
  * every provider was skipped or failed provider-wide (open breaker, spent quota,
    bad key, missing model, unexpected status, transient exhaustion)
                                           -> ``ProvidersUnavailable``

Invalid-answer breaker policy: a per-provider streak of consecutive invalid JOBS
(thread-safe, reset by any valid answer) is kept. It is NOT reset when it reaches
the threshold (3): from then on every further invalid job records one breaker failure.
"""

import logging
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import ValidationError

from src.evaluator.circuit_breaker import CircuitBreaker
from src.evaluator.errors import (
    AuthenticationFailed,
    BadRequest,
    ConfigurationError,
    EvaluationRejected,
    InvalidResponse,
    ModelNotFound,
    PermanentProviderError,
    ProvidersUnavailable,
    QuotaExhausted,
    ResponseTruncated,
    StopRequested,
    TransientProviderError,
)
from src.evaluator.parsing import extract_json_payload
from src.evaluator.prompts import build_job_user_prompt, build_system_prompt, load_candidate_persona
from src.evaluator.providers import NvidiaNimProvider, OpenRouterProvider, strip_think
from src.evaluator.rate_limiter import SlidingWindowRateLimiter
from src.evaluator.schemas import JobEvaluationResult
from src.filters.pre_filter import classify_title
from src.storage.models import JobPosting

logger = logging.getLogger(__name__)

INVALID_RETRIES_PER_PROVIDER = 1
INVALID_STREAK_BREAKER_THRESHOLD = 3
TRUNCATION_RETRY_FACTOR = 1.5
BONUS_POINTS = 1.5
BONUS_MIN_BASE_SCORE = 6.0
REJECT_CAP = 2.0

_OK = "ok"
_JOB_FAULT = "job"  # this provider judged the job unusable; not an outage
_UNAVAILABLE = "unavailable"  # provider-wide: skipped or failed


class ProviderChainEvaluator:
    def __init__(
        self,
        providers: Sequence[Any],
        stop_event: Optional[threading.Event] = None,
        system_prompt: Optional[str] = None,
        invalid_streak_threshold: int = INVALID_STREAK_BREAKER_THRESHOLD,
        bonus_enabled: Optional[bool] = None,
    ):
        if not providers:
            raise ValueError("ProviderChainEvaluator needs at least one provider")
        self.providers: List[Any] = list(providers)
        self.stop_event = stop_event if stop_event is not None else threading.Event()
        self._invalid_streak_threshold = invalid_streak_threshold
        self._streak_lock = threading.Lock()
        self._invalid_streaks: Dict[str, int] = {}
        # The +BONUS_POINTS only applies when the local profile defines bonus criteria
        # (it mentions "bonus"); an explicit ``bonus_enabled`` always wins. With an explicit
        # ``system_prompt`` and no flag the bonus stays off.
        persona: Optional[str] = None
        if system_prompt is None:
            persona = load_candidate_persona()
            logger.info("Local candidate profile %s", "found" if persona else "not found; using generic persona")
            system_prompt = build_system_prompt(persona)
        self._system_prompt = system_prompt
        if bonus_enabled is None:
            bonus_enabled = persona is not None and "bonus" in persona.lower()
        self._bonus_enabled = bool(bonus_enabled)

    def stop(self) -> None:
        """Wake every blocked worker and make further evaluations raise StopRequested."""
        self.stop_event.set()

    # ----- public -----

    def evaluate_job(self, job: JobPosting) -> JobEvaluationResult:
        self._check_stop()
        user_prompt = build_job_user_prompt(
            title=job.title,
            company=job.company,
            location=job.location,
            is_remote=job.is_remote,
            description=job.description,
        )
        job_fault = False
        for provider in self.providers:
            self._check_stop()
            if not provider.is_available():
                logger.debug("Skipping %s: unavailable (quota)", provider.name)
                continue
            status, result, model_used, latency = self._try_provider(provider, user_prompt)
            if status == _OK:
                result.eval_model = model_used
                final = self._postprocess(result, job)
                logger.info(
                    "Evaluated via %s (model %s) in %.1fs -> score %.1f",
                    provider.name, model_used, latency, final.score,
                )
                return final
            if status == _JOB_FAULT:
                job_fault = True
        if job_fault:
            raise EvaluationRejected("no provider produced a valid answer for this job")
        raise ProvidersUnavailable("no provider could evaluate the job")

    # ----- internals -----

    def _check_stop(self) -> None:
        if self.stop_event.is_set():
            raise StopRequested("stop requested")

    def _try_provider(
        self, provider: Any, user_prompt: str
    ) -> Tuple[str, Optional[JobEvaluationResult], Optional[str], float]:
        breaker: CircuitBreaker = provider.breaker
        if not breaker.allow():
            logger.debug("Skipping %s: circuit breaker open", provider.name)
            return _UNAVAILABLE, None, None, 0.0

        settled = False
        try:
            max_tokens: Optional[int] = None
            result: Optional[JobEvaluationResult] = None
            completion = None
            for _ in range(1 + INVALID_RETRIES_PER_PROVIDER):
                try:
                    completion = provider.complete(self._system_prompt, user_prompt, max_tokens)
                    result = self._parse(completion.content)
                except InvalidResponse as exc:
                    result = None
                    logger.warning("%s returned an invalid response (%s)", provider.name, type(exc).__name__)
                    if isinstance(exc, ResponseTruncated):
                        max_tokens = int(provider.max_tokens * TRUNCATION_RETRY_FACTOR)
                    continue
                break

            if result is not None:
                breaker.record_success()
                settled = True
                self._reset_streak(provider)
                return _OK, result, completion.model_used, completion.latency

            if self._bump_streak(provider):
                breaker.record_failure()
                settled = True
            return _JOB_FAULT, None, None, 0.0
        except AuthenticationFailed:
            breaker.open_immediately()
            settled = True
            return _UNAVAILABLE, None, None, 0.0
        except BadRequest:
            return _JOB_FAULT, None, None, 0.0  # request-specific: not a breaker failure
        except (ModelNotFound, TransientProviderError, PermanentProviderError):
            breaker.record_failure()
            settled = True
            return _UNAVAILABLE, None, None, 0.0
        except QuotaExhausted:
            return _UNAVAILABLE, None, None, 0.0
        finally:
            if not settled:
                breaker.release_probe()

    def _bump_streak(self, provider: Any) -> bool:
        """Count one invalid job; True once the streak is at/after the threshold (never auto-reset)."""
        with self._streak_lock:
            count = self._invalid_streaks.get(provider.name, 0) + 1
            self._invalid_streaks[provider.name] = count
            return count >= self._invalid_streak_threshold

    def _reset_streak(self, provider: Any) -> None:
        with self._streak_lock:
            self._invalid_streaks[provider.name] = 0

    @staticmethod
    def _parse(content: str) -> JobEvaluationResult:
        try:
            data = extract_json_payload(strip_think(content or ""))
        except ValueError:
            raise InvalidResponse(f"response is not valid JSON ({len(content or '')} chars)") from None
        if not isinstance(data, dict):
            raise InvalidResponse("JSON payload is not an object")
        try:
            return JobEvaluationResult.model_validate(data)
        except ValidationError as exc:
            raise InvalidResponse(f"schema-invalid payload ({exc.error_count()} errors)") from None

    def _postprocess(self, result: JobEvaluationResult, job: JobPosting) -> JobEvaluationResult:
        score = min(10.0, max(1.0, float(result.score)))
        if result.hard_reject or score <= REJECT_CAP:
            score = min(score, REJECT_CAP)
        elif result.matches_bonus_criteria and score >= BONUS_MIN_BASE_SCORE:
            if self._bonus_enabled:
                score = min(10.0, score + BONUS_POINTS)
            else:
                logger.debug("Model set matches_bonus_criteria but no bonus criteria are configured; bonus ignored")

        title_reason = classify_title(job.title)
        if title_reason and score > REJECT_CAP:
            score = REJECT_CAP
            result.reasoning = f"{result.reasoning} [Title safety net: {title_reason}; score capped at {REJECT_CAP}]"

        result.score = round(score, 1)
        result.is_wow = False  # decided by the daemon from the final score, never by the model
        return result


def build_chain_from_settings(
    settings: Optional[Any] = None,
    stop_event: Optional[threading.Event] = None,
    system_prompt: Optional[str] = None,
    bonus_enabled: Optional[bool] = None,
) -> ProviderChainEvaluator:
    """Build the chain from ``Settings``: NIM iff a key is set, OpenRouter iff key + pinned models.

    Raises ``ConfigurationError`` when no provider is enabled. Called lazily by the
    daemon, never at import or construction time.
    """
    if settings is None:
        from src.config import get_settings

        settings = get_settings()
    stop_event = stop_event if stop_event is not None else threading.Event()

    def make_breaker() -> CircuitBreaker:
        return CircuitBreaker(settings.llm_breaker_failures, settings.llm_breaker_cooldown_seconds)

    providers: List[Any] = []
    if settings.nim_enabled:
        providers.append(
            NvidiaNimProvider(
                api_key=settings.nvidia_api_key.strip(),
                model=settings.nvidia_model,
                base_url=settings.nvidia_base_url,
                limiter=SlidingWindowRateLimiter(settings.nvidia_rpm_limit, 60.0, stop_event=stop_event),
                breaker=make_breaker(),
                stop_event=stop_event,
                timeout=settings.nvidia_timeout_seconds,
                max_tokens=settings.nvidia_max_tokens,
                temperature=settings.nvidia_temperature,
                reasoning_effort=settings.nvidia_reasoning_effort,
                workers=settings.eval_max_workers,
            )
        )
    if settings.openrouter_enabled:
        providers.append(
            OpenRouterProvider(
                api_key=settings.openrouter_api_key.strip(),
                models=settings.openrouter_models,
                base_url=settings.openrouter_base_url,
                limiter=SlidingWindowRateLimiter(settings.openrouter_rpm_limit, 60.0, stop_event=stop_event),
                breaker=make_breaker(),
                stop_event=stop_event,
                timeout=settings.openrouter_timeout_seconds,
                daily_budget=settings.openrouter_daily_budget,
                workers=settings.eval_max_workers,
            )
        )
    elif settings.openrouter_api_key and settings.openrouter_api_key.strip():
        logger.warning(
            "OPENROUTER_API_KEY is set but OPENROUTER_MODELS is empty: the OpenRouter fallback is disabled "
            "(the legacy OPENROUTER_MODEL router is no longer used)."
        )

    if not providers:
        raise ConfigurationError(
            "No LLM provider is configured. Set NVIDIA_API_KEY (primary), and/or OPENROUTER_API_KEY "
            "together with OPENROUTER_MODELS (fallback), in .env."
        )
    return ProviderChainEvaluator(
        providers, stop_event=stop_event, system_prompt=system_prompt, bonus_enabled=bonus_enabled
    )
