import re
from pathlib import Path

import pytest

import src.evaluator.prompts as prompts
from src.evaluator.prompts import (
    DEFAULT_CANDIDATE_PERSONA,
    build_job_user_prompt,
    build_system_prompt,
    load_candidate_persona,
)


@pytest.fixture
def no_profile(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_generic_fallback_without_profile_file(no_profile):
    assert load_candidate_persona() is None
    prompt = build_system_prompt()
    assert DEFAULT_CANDIDATE_PERSONA in prompt


def test_persona_injected_from_tmp_profile(no_profile):
    (no_profile / ".candidate_profile.md").write_text("- Test persona marker XYZ-123", encoding="utf-8")
    prompt = build_system_prompt()
    assert "Test persona marker XYZ-123" in prompt
    assert DEFAULT_CANDIDATE_PERSONA not in prompt


def test_explicit_persona_overrides_file(no_profile):
    (no_profile / ".candidate_profile.md").write_text("- From file", encoding="utf-8")
    prompt = build_system_prompt(custom_persona="- Explicit persona")
    assert "- Explicit persona" in prompt
    assert "From file" not in prompt


def test_prompt_is_built_lazily_not_at_import():
    assert not hasattr(prompts, "SYSTEM_PROMPT")


def test_prompt_encodes_generic_rubric_tiers(no_profile):
    prompt = build_system_prompt()
    for expected in ("HARD REJECT", "1.0", "2.0", "6.5", "8.4", "8.5", "10.0", "DIGEST", "WOW"):
        assert expected in prompt
    # The output language of summary_uk comes from the local profile (English if none states one).
    assert "Ukrainian" not in prompt
    assert "output language" in prompt.lower()
    assert "English if none" in prompt


def test_prompt_declares_json_contract_with_generic_flags(no_profile):
    prompt = build_system_prompt()
    for field in (
        "score",
        "summary_uk",
        "reasoning",
        "language_requirements",
        "pros",
        "cons",
        "hard_reject",
        "role_is_direct_coaching",
        "matches_bonus_criteria",
    ):
        assert f'"{field}"' in prompt
    # The LLM no longer decides is_wow; the pipeline derives it from the final score.
    assert '"is_wow"' not in prompt


def test_prompt_defers_bonus_criteria_to_profile(no_profile):
    prompt = build_system_prompt()
    assert "BONUS CRITERIA" in prompt
    assert "candidate profile" in prompt.lower()


def test_prompts_module_source_has_no_persona_identifiers():
    source = Path(prompts.__file__).read_text(encoding="utf-8")
    # Persona-specific flag names (values_*) must stay out of tracked prompt code.
    assert not re.search(r"values_\w+", source)


def test_prompt_without_profile_file_has_no_profile_content(no_profile):
    # Nothing from a local candidate profile may leak when no profile file exists.
    (no_profile / "unrelated.md").write_text("- Local marker QWE-987", encoding="utf-8")
    assert "QWE-987" not in build_system_prompt()


def test_user_prompt_truncates_description():
    text = build_job_user_prompt("T", "C", "L", True, "x" * 5000)
    assert text.count("x") == 4000
