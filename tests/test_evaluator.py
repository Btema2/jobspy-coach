import logging
from pathlib import Path
import pytest
from pydantic import ValidationError

from src.evaluator.prompts import (
    DEFAULT_CANDIDATE_PERSONA,
    build_system_prompt,
    load_candidate_persona,
)
from src.evaluator.schemas import JobEvaluationResult


def test_job_evaluation_result_schema_validation():
    valid_data = {
        "score": 8.5,
        "is_wow": False,
        "summary_uk": "Відмінна вакансія бізнес-тренера для міжнародної компанії.",
        "reasoning": "Strong corporate fit for executive coaching.",
        "language_requirements": ["EN", "PL"],
        "pros": ["Remote", "Competitive salary"],
        "cons": ["High travel requirement"],
    }
    result = JobEvaluationResult.model_validate(valid_data)
    assert result.score == 8.5
    assert result.is_wow is False
    assert result.summary_uk == "Відмінна вакансія бізнес-тренера для міжнародної компанії."
    assert result.reasoning == "Strong corporate fit for executive coaching."
    assert result.language_requirements == ["EN", "PL"]
    assert result.pros == ["Remote", "Competitive salary"]
    assert result.cons == ["High travel requirement"]

    # Test default empty lists
    minimal_data = {
        "score": 9.5,
        "is_wow": True,
        "summary_uk": "Чудова позиція Executive коуча.",
        "reasoning": "Fits perfectly.",
    }
    result_min = JobEvaluationResult.model_validate(minimal_data)
    assert result_min.language_requirements == []
    assert result_min.pros == []
    assert result_min.cons == []

    # Out-of-range scores are clamped rather than rejected
    assert JobEvaluationResult.model_validate({**valid_data, "score": 0.9}).score == 1.0
    assert JobEvaluationResult.model_validate({**valid_data, "score": 10.1}).score == 10.0

    # Non-numeric score is still invalid
    with pytest.raises(ValidationError):
        JobEvaluationResult.model_validate({**valid_data, "score": "excellent"})

    # summary_uk too short (< 10 chars)
    with pytest.raises(ValidationError):
        JobEvaluationResult.model_validate({**valid_data, "summary_uk": "Коротко"})

    # Missing required reasoning
    invalid_data = {k: v for k, v in valid_data.items() if k != "reasoning"}
    with pytest.raises(ValidationError):
        JobEvaluationResult.model_validate(invalid_data)


def test_load_candidate_persona_file_exists(tmp_path, monkeypatch):
    # Test primary path: .candidate_profile.md
    monkeypatch.chdir(tmp_path)
    primary_profile = tmp_path / ".candidate_profile.md"
    primary_profile.write_text("  - Domain: Senior Executive Coach\n- Languages: English, Ukrainian  \n", encoding="utf-8")

    loaded = load_candidate_persona()
    assert loaded == "- Domain: Senior Executive Coach\n- Languages: English, Ukrainian"

    # Test secondary fallback path: config/candidate_profile.md
    primary_profile.unlink()
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    secondary_profile = config_dir / "candidate_profile.md"
    secondary_profile.write_text("- Domain: Secondary Config Coach", encoding="utf-8")

    loaded_secondary = load_candidate_persona()
    assert loaded_secondary == "- Domain: Secondary Config Coach"


def test_load_candidate_persona_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # Ensure neither .candidate_profile.md nor config/candidate_profile.md exists
    assert load_candidate_persona() is None


def test_load_candidate_persona_exception(tmp_path, monkeypatch, caplog):
    monkeypatch.chdir(tmp_path)
    bad_file = tmp_path / ".candidate_profile.md"
    bad_file.write_text("content", encoding="utf-8")

    def mock_read_text(*args, **kwargs):
        raise OSError("Disk I/O failure")

    monkeypatch.setattr(Path, "read_text", mock_read_text)

    with caplog.at_level(logging.WARNING):
        result = load_candidate_persona()

    assert result is None
    assert "Could not read local candidate profile" in caplog.text
    assert "Disk I/O failure" in caplog.text


def test_build_system_prompt_custom_vs_default(monkeypatch):
    # 1. Custom persona explicitly passed to build_system_prompt
    custom_persona = "- Custom Executive Profile with 15+ years experience"
    prompt_custom = build_system_prompt(custom_persona=custom_persona)
    assert custom_persona in prompt_custom
    assert "CANDIDATE PERSONA:\n- Custom Executive Profile with 15+ years experience" in prompt_custom

    # 2. No custom persona passed and load_candidate_persona returns None -> falls back to DEFAULT_CANDIDATE_PERSONA
    monkeypatch.setattr("src.evaluator.prompts.load_candidate_persona", lambda: None)
    prompt_default = build_system_prompt()
    assert DEFAULT_CANDIDATE_PERSONA in prompt_default

    # 3. No custom persona passed and load_candidate_persona returns loaded profile from disk
    monkeypatch.setattr("src.evaluator.prompts.load_candidate_persona", lambda: "- Loaded Disk Persona")
    prompt_loaded = build_system_prompt()
    assert "- Loaded Disk Persona" in prompt_loaded
    assert DEFAULT_CANDIDATE_PERSONA not in prompt_loaded




