import json
import logging
import re
import time
from typing import Any, Dict, Optional

import httpx

from src.evaluator.prompts import SYSTEM_PROMPT, build_job_user_prompt
from src.evaluator.schemas import JobEvaluationResult
from src.storage.models import JobPosting

logger = logging.getLogger(__name__)


def extract_json_payload(raw_text: str) -> Dict[str, Any]:
    if not raw_text or not isinstance(raw_text, str):
        raise ValueError("Empty or non-string response payload")

    text = raw_text.strip()

    # 1. Direct JSON parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Extract from markdown code fences ```json ... ``` or ``` ... ```
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 3. Regex search for outermost JSON object { ... }
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract valid JSON from response: {raw_text[:200]}")


class OpenRouterEvaluator:
    def __init__(
        self,
        api_key: str,
        model: str = "openrouter/free",
        base_url: str = "https://openrouter.ai/api/v1",
        rpm_limit: int = 20,
        timeout: float = 45.0,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.rpm_limit = rpm_limit
        self.timeout = timeout
        self.min_interval = 60.0 / max(rpm_limit, 1)
        self.last_call_time = 0.0

    @classmethod
    def from_settings(cls, settings: Optional[Any] = None) -> "OpenRouterEvaluator":
        if settings is None:
            from src.config import get_settings

            settings = get_settings()
        return cls(
            api_key=settings.openrouter_api_key,
            model=settings.openrouter_model,
            base_url=settings.openrouter_base_url,
            rpm_limit=settings.openrouter_rpm_limit,
            timeout=settings.openrouter_timeout_seconds,
        )

    def _rate_limit_wait(self) -> None:
        elapsed = time.time() - self.last_call_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_call_time = time.time()

    def evaluate_job(self, job: JobPosting) -> JobEvaluationResult:
        self._rate_limit_wait()

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/jobspy-coach",
            "X-Title": "JobSpy Coach Evaluator",
            "Content-Type": "application/json",
        }

        user_content = build_job_user_prompt(
            title=job.title,
            company=job.company,
            location=job.location,
            is_remote=job.is_remote,
            description=job.description,
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.2,
        }

        for attempt in range(1, 4):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                    )
                    if resp.status_code == 429:
                        if attempt == 3:
                            raise RuntimeError("OpenRouter rate limit (429) exceeded after 3 attempts")
                        wait_sec = attempt * 5.0
                        logger.warning(
                            "OpenRouter 429 Rate Limited. Sleeping %ss (attempt %d/3)...",
                            wait_sec,
                            attempt,
                        )
                        time.sleep(wait_sec)
                        continue

                    resp.raise_for_status()
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"]
                    parsed = extract_json_payload(content)
                    result = JobEvaluationResult.model_validate(parsed)
                    actual_model = data.get("model", self.model) or self.model
                    result.eval_model = actual_model
                    logger.info(
                        "OpenRouter evaluated '%s' via model '%s' -> Score: %.1f, is_wow: %s",
                        job.title,
                        actual_model,
                        result.score,
                        result.is_wow,
                    )
                    return result
            except Exception as e:
                if attempt == 3:
                    logger.error(
                        "Failed evaluation for job '%s' after 3 attempts: %s",
                        job.title,
                        e,
                    )
                    raise
                wait_sec = attempt * 2.0
                logger.warning(
                    "Error evaluating job '%s' (attempt %d/3): %s. Sleeping %ss...",
                    job.title,
                    attempt,
                    e,
                    wait_sec,
                )
                time.sleep(wait_sec)

        raise RuntimeError("Evaluation unexpectedly exited retry loop")
