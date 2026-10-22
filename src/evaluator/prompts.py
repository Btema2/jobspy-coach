import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_CANDIDATE_PERSONA = """- Domain Expertise: Professional Coaching, Executive Leadership, Career Advisory, Corporate Training & L&D.
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


def build_system_prompt(custom_persona: str | None = None) -> str:
    persona = custom_persona or load_candidate_persona() or DEFAULT_CANDIDATE_PERSONA
    return f"""You are an expert executive talent acquisition specialist and career advisor.
Your job is to evaluate whether a scraped job vacancy is a strong match for a specific professional candidate.

CANDIDATE PERSONA:
{persona}

EVALUATION INSTRUCTIONS:
1. Analyze the job title, company, work location (remote vs onsite), language requirements, responsibilities, and qualifications.
2. Assign an overall match score from 1.0 to 10.0:
   - 9.0 - 10.0 (WOW Match): Executive/leadership/career coaching or high-level L&D role; compatible location/remote terms; matches candidate language skills; excellent compensation or prestige.
   - 7.0 - 8.9 (Strong Match): Clear corporate/executive coaching or L&D role; compatible location/remote terms; candidate qualifications align.
   - 5.0 - 6.9 (Marginal Match): Mixed HR/recruitment responsibilities; unclear coaching scope; junior/admin L&D; or minor language mismatch.
   - 1.0 - 4.9 (Poor / Mismatch): Sports/fitness; pure technical/agile software delivery; incompatible geography; requires languages candidate does not speak.
3. Determine `is_wow`: True ONLY if score >= 9.0 OR the opportunity has exceptional prestige, compensation, and profile fit.
4. Provide `summary_uk`: A concise 1-2 sentence executive summary explaining why this role is relevant, core duties, and key working conditions.
5. Provide `reasoning`: A concise English explanation of your scoring rationale.
6. Provide `language_requirements`: List of languages required by the employer.

OUTPUT FORMAT:
Return strictly a valid JSON object matching this schema without markdown code fences or conversational text:
{{
  "score": float,
  "is_wow": boolean,
  "summary_uk": string,
  "reasoning": string,
  "language_requirements": string[],
  "pros": string[],
  "cons": string[]
}}
"""


SYSTEM_PROMPT = build_system_prompt()


def build_job_user_prompt(
    title: str,
    company: str,
    location: str,
    is_remote: bool,
    description: str,
) -> str:
    desc = (description or "").strip()
    return f"""JOB TO EVALUATE:
Title: {title}
Company: {company}
Location: {location}
Remote: {is_remote}

Job Description:
{desc[:4000]}
"""
