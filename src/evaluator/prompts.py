import logging
from pathlib import Path
from typing import Optional

from src.evaluator.rubric import RubricInputs, render_rubric
from src.profile.legacy_seed import (
    LEGACY_ADJACENT_ROLE_DEFINITION,
    LEGACY_CORE_ROLE_DEFINITION,
    LEGACY_DEALBREAKERS,
)

logger = logging.getLogger(__name__)

DEFAULT_CANDIDATE_PERSONA = """- Domain Expertise: Professional Coaching, Career Advisory, Corporate Training & L&D.
- Excluded Roles: Sports coaching, fitness/gym training, musical/vocal coaching, purely technical IT/software delivery without people development.
- Candidate Location: Remote / Global or Local (configurable).
- Language Capabilities: Professional working proficiency.
"""


def load_candidate_persona() -> Optional[str]:
    """Loads private candidate persona from local git-ignored profile file if present."""
    profile_paths = [
        Path(".candidate_profile.md"),
        Path("config/candidate_profile.md"),
    ]
    for p in profile_paths:
        if p.exists():
            try:
                content = p.read_text(encoding="utf-8").strip()
                if content:
                    return content
            except Exception as e:
                logger.warning("Could not read local candidate profile from %s: %s", p, e)
    return None


def legacy_rubric_inputs(persona: str) -> RubricInputs:
    """Rubric slots of an imported legacy profile: the seed definitions, 8.5 / 6.5, bonus from the persona."""
    from src.profile.importer import extract_bonus_criteria  # lazy: the importer imports the compiler

    return RubricInputs(
        persona=persona,
        core_role_definition=LEGACY_CORE_ROLE_DEFINITION,
        adjacent_role_definition=LEGACY_ADJACENT_ROLE_DEFINITION,
        dealbreakers=LEGACY_DEALBREAKERS,
        bonus_criteria=extract_bonus_criteria(persona),
        output_language=None,
        wow_threshold=8.5,
        digest_threshold=6.5,
        remote_only=False,
    )


def build_system_prompt(custom_persona: str | None = None) -> str:
    """No-profile fallback prompt: the legacy rubric around the persona from the local profile file.

    Used when no stored profile exists (bake-off, a chain built without a policy). Persona text comes only
    from ``.candidate_profile.md`` or the argument; the default persona is generic.
    """
    persona = (custom_persona or load_candidate_persona() or DEFAULT_CANDIDATE_PERSONA).strip()
    return render_rubric(legacy_rubric_inputs(persona))


def build_job_user_prompt(
    title: str,
    company: str,
    location: str,
    remote: str,
    description: str,
) -> str:
    desc = (description or "").strip()
    return f"""JOB TO EVALUATE:
Title: {title}
Company: {company}
Location: {location}
Remote: {remote}

Job Description:
{desc[:4000]}
"""
