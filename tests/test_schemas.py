import pytest
from pydantic import ValidationError

from src.evaluator.schemas import JobEvaluationResult

BASE = {
    "score": 7.0,
    "summary_uk": "Коротке резюме вакансії для кандидата.",
    "reasoning": "Reasonable fit.",
}


def test_is_wow_is_optional_and_defaults_false():
    result = JobEvaluationResult.model_validate(BASE)
    assert result.is_wow is False


def test_is_wow_can_still_be_provided():
    result = JobEvaluationResult.model_validate({**BASE, "is_wow": True})
    assert result.is_wow is True


@pytest.mark.parametrize(
    "raw,expected",
    [(0, 1.0), (0.0, 1.0), (-3, 1.0), (11, 10.0), (10.1, 10.0), (100, 10.0), (5.5, 5.5), (1, 1.0), (10, 10.0)],
)
def test_score_is_clamped_not_rejected(raw, expected):
    assert JobEvaluationResult.model_validate({**BASE, "score": raw}).score == expected


def test_score_numeric_string_is_accepted_and_clamped():
    assert JobEvaluationResult.model_validate({**BASE, "score": "12"}).score == 10.0


@pytest.mark.parametrize("bad", ["high", None, [], {}, "nan"])
def test_non_numeric_score_is_still_invalid(bad):
    with pytest.raises(ValidationError):
        JobEvaluationResult.model_validate({**BASE, "score": bad})


def test_missing_score_is_invalid():
    data = {k: v for k, v in BASE.items() if k != "score"}
    with pytest.raises(ValidationError):
        JobEvaluationResult.model_validate(data)


def test_generic_flags_default_false():
    result = JobEvaluationResult.model_validate(BASE)
    assert result.hard_reject is False
    assert result.matches_bonus_criteria is False
    assert result.role_is_direct_coaching is False


def test_generic_flags_can_be_set():
    result = JobEvaluationResult.model_validate(
        {**BASE, "hard_reject": True, "matches_bonus_criteria": True, "role_is_direct_coaching": True}
    )
    assert result.hard_reject and result.matches_bonus_criteria and result.role_is_direct_coaching


def test_eval_model_assignable_after_validation():
    result = JobEvaluationResult.model_validate(BASE)
    assert result.eval_model is None
    result.eval_model = "some/model"
    assert result.eval_model == "some/model"
