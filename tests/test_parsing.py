import pytest

from src.evaluator import extract_json_payload  # noqa: F401  (public re-export)
from src.evaluator.parsing import extract_json_payload as _direct


def test_extract_json_payload_is_reexported_from_package():
    assert extract_json_payload is _direct


def test_extract_json_payload_clean():
    raw = '{"score": 9.5, "is_wow": true, "summary_uk": "Чудова позиція.", "reasoning": "Fits well", "language_requirements": ["EN"], "pros": [], "cons": []}'
    parsed = extract_json_payload(raw)
    assert parsed["score"] == 9.5
    assert parsed["is_wow"] is True
    assert parsed["summary_uk"] == "Чудова позиція."


def test_extract_json_payload_markdown_and_noise():
    raw = """Here is your evaluation:
    ```json
    {
      "score": 8.5,
      "is_wow": false,
      "summary_uk": "Відмінна вакансія бізнес-тренера.",
      "reasoning": "Strong corporate fit",
      "language_requirements": ["PL", "EN"],
      "pros": ["Good salary"],
      "cons": ["Hybrid"]
    }
    ```
    Best regards!"""
    parsed = extract_json_payload(raw)
    assert parsed["score"] == 8.5
    assert parsed["is_wow"] is False
    assert parsed["summary_uk"] == "Відмінна вакансія бізнес-тренера."


def test_extract_json_payload_malformed_raises_value_error():
    # Completely non-JSON text
    with pytest.raises(ValueError, match="Could not extract valid JSON"):
        extract_json_payload("Here is an answer with no JSON object at all.")

    # Broken/unparseable JSON syntax
    with pytest.raises(ValueError):
        extract_json_payload("{score: 9.0, missing_closing_bracket")


def test_extract_json_payload_none_raises_value_error():
    with pytest.raises(ValueError, match="Empty or non-string response payload"):
        extract_json_payload(None)

    with pytest.raises(ValueError, match="Empty or non-string response payload"):
        extract_json_payload("")

    with pytest.raises(ValueError, match="Empty or non-string response payload"):
        extract_json_payload(123)  # type: ignore[arg-type]


def test_extract_json_payload_outermost_json_without_fences():
    raw = 'Sure, here is the result: {"score": 7.5, "is_wow": false, "summary_uk": "Звичайна вакансія.", "reasoning": "Ok", "language_requirements": [], "pros": [], "cons": []}. Hope this helps!'
    parsed = extract_json_payload(raw)
    assert parsed["score"] == 7.5
    assert parsed["summary_uk"] == "Звичайна вакансія."


def test_extract_json_payload_broken_fence_fallback_to_json_regex():
    # If code fence contains broken/invalid text, but outer JSON regex can locate a valid JSON block
    raw = '```markdown\nBroken fence here\n```\nHere is valid JSON: {"score": 8.0, "is_wow": false, "summary_uk": "Тестова вакансія.", "reasoning": "Valid", "language_requirements": [], "pros": [], "cons": []}'
    parsed = extract_json_payload(raw)
    assert parsed["score"] == 8.0
    assert parsed["summary_uk"] == "Тестова вакансія."
