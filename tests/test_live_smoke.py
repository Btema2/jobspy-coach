"""Live smoke test: two real calls through the real provider chain.

Skipped by default. Run explicitly with ``pytest -m live`` and at least one provider
configured (``NVIDIA_API_KEY`` and/or ``OPENROUTER_API_KEY`` + ``OPENROUTER_MODELS``).
It uses persona-neutral fixture jobs and the generic built-in persona, and never prints
keys or job/response content.
"""
import json
from pathlib import Path

import pytest

from src.config import Settings
from src.evaluator.chain import build_chain_from_settings
from src.evaluator.prompts import DEFAULT_CANDIDATE_PERSONA, build_system_prompt
from src.storage.models import JobPosting

FIXTURE = Path(__file__).parent / "fixtures" / "eval_cases.json"
TRAP_ID = "trap-it-pm-coaching-side-duty"
RELEVANT_ID = "relevant-career-coach-edtech"

pytestmark = pytest.mark.live


def _job(case: dict) -> JobPosting:
    return JobPosting(
        dedup_hash=f"live-{case['id']}",
        site="fixture",
        job_url=f"https://example.invalid/{case['id']}",
        title=case["title"],
        company=case["company"],
        location=case["location"],
        description=case["description"],
        is_remote=bool(case.get("is_remote")),
    )


def test_live_chain_scores_a_trap_low_and_a_relevant_job_high():
    settings = Settings()  # reads the real .env / environment on purpose
    if not (settings.nim_enabled or settings.openrouter_enabled):
        pytest.skip("no LLM provider configured (set NVIDIA_API_KEY and/or OPENROUTER_API_KEY + OPENROUTER_MODELS)")

    cases = {c["id"]: c for c in json.loads(FIXTURE.read_text(encoding="utf-8"))}
    chain = build_chain_from_settings(
        settings, system_prompt=build_system_prompt(DEFAULT_CANDIDATE_PERSONA)
    )

    trap = chain.evaluate_job(_job(cases[TRAP_ID]))
    relevant = chain.evaluate_job(_job(cases[RELEVANT_ID]))

    trap_score, trap_model = trap.score, trap.eval_model
    rel_score, rel_model = relevant.score, relevant.eval_model
    assert 1.0 <= trap_score <= 10.0 and 1.0 <= rel_score <= 10.0, "score out of range"
    assert isinstance(trap_model, str) and trap_model, "missing eval_model on trap result"
    assert isinstance(rel_model, str) and rel_model, "missing eval_model on relevant result"
    trap_max = cases[TRAP_ID]["expected"]["max"] + 1.0  # bake-off backup bound (3.0)
    rel_min = cases[RELEVANT_ID]["expected"]["min"]
    assert trap_score <= trap_max, f"trap scored {trap_score} (max {trap_max})"
    assert rel_score >= rel_min, f"relevant scored {rel_score} (min {rel_min})"
