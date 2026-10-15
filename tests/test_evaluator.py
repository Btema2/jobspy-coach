import json
from unittest.mock import MagicMock
import httpx
import pytest
from pydantic import ValidationError

from src.evaluator.schemas import JobEvaluationResult
from src.evaluator.llm_evaluator import OpenRouterEvaluator, extract_json_payload
from src.storage.models import JobPosting


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

    # Score below 1.0
    with pytest.raises(ValidationError):
        JobEvaluationResult.model_validate({**valid_data, "score": 0.9})

    # Score above 10.0
    with pytest.raises(ValidationError):
        JobEvaluationResult.model_validate({**valid_data, "score": 10.1})

    # summary_uk too short (< 10 chars)
    with pytest.raises(ValidationError):
        JobEvaluationResult.model_validate({**valid_data, "summary_uk": "Коротко"})

    # Missing required reasoning
    invalid_data = {k: v for k, v in valid_data.items() if k != "reasoning"}
    with pytest.raises(ValidationError):
        JobEvaluationResult.model_validate(invalid_data)


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


def test_openrouter_evaluator_successful_evaluation(monkeypatch):
    mock_post = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "score": 9.2,
                            "is_wow": True,
                            "summary_uk": "Престижна позиція Executive коуча для міжнародного банку.",
                            "reasoning": "Direct match for senior coach persona with Ukrainian and English.",
                            "language_requirements": ["English", "Ukrainian"],
                            "pros": ["Top tier bank", "Full remote"],
                            "cons": [],
                        }
                    )
                }
            }
        ]
    }
    mock_post.return_value = mock_response
    monkeypatch.setattr("httpx.Client.post", mock_post)

    evaluator = OpenRouterEvaluator(
        api_key="fake-key",
        model="openrouter/free",
        base_url="https://openrouter.ai/api/v1",
        rpm_limit=100,
    )

    job = JobPosting(
        dedup_hash="abc",
        site="linkedin",
        job_url="https://example.com/job",
        title="Executive Coach",
        company="Global Bank",
        location="Remote Poland",
        description="We need an Executive Coach to work with leadership in Warsaw.",
        is_remote=True,
    )

    res = evaluator.evaluate_job(job)
    assert isinstance(res, JobEvaluationResult)
    assert res.score == 9.2
    assert res.is_wow is True
    assert "Executive коуча" in res.summary_uk

    # Verify request payload and headers
    assert mock_post.call_count == 1
    call_args, call_kwargs = mock_post.call_args
    assert call_args[0] == "https://openrouter.ai/api/v1/chat/completions"
    assert call_kwargs["headers"]["Authorization"] == "Bearer fake-key"
    assert call_kwargs["headers"]["HTTP-Referer"] == "https://github.com/jobspy-coach"
    assert call_kwargs["json"]["model"] == "openrouter/free"
    assert call_kwargs["json"]["temperature"] == 0.2
    assert len(call_kwargs["json"]["messages"]) == 2
    assert call_kwargs["json"]["messages"][0]["role"] == "system"
    assert call_kwargs["json"]["messages"][1]["role"] == "user"


# Alias for compatibility if referenced by different naming conventions
test_openrouter_evaluator_success = test_openrouter_evaluator_successful_evaluation


def test_openrouter_evaluator_rate_limit_backoff(monkeypatch):
    mock_post = MagicMock()
    resp_429 = MagicMock()
    resp_429.status_code = 429

    resp_200 = MagicMock()
    resp_200.status_code = 200
    resp_200.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "score": 8.0,
                            "is_wow": False,
                            "summary_uk": "Відмінна вакансія коуча.",
                            "reasoning": "Good fit",
                            "language_requirements": ["English"],
                            "pros": [],
                            "cons": [],
                        }
                    )
                }
            }
        ]
    }
    mock_post.side_effect = [resp_429, resp_200]
    monkeypatch.setattr("httpx.Client.post", mock_post)
    mock_sleep = MagicMock()
    monkeypatch.setattr("time.sleep", mock_sleep)

    evaluator = OpenRouterEvaluator(
        api_key="fake-key",
        rpm_limit=100,
    )
    job = JobPosting(
        dedup_hash="abc",
        site="linkedin",
        job_url="https://example.com/job",
        title="Career Coach",
        company="Company",
        location="Remote",
        description="Career coaching role.",
    )
    res = evaluator.evaluate_job(job)
    assert res.score == 8.0
    assert mock_post.call_count == 2
    # Verify sleep was called for 429 backoff (attempt 1 * 5.0s = 5.0s)
    mock_sleep.assert_any_call(5.0)


def test_openrouter_evaluator_rate_limiter_pacing(monkeypatch):
    sleep_calls = []
    fake_time = [1000.0]

    def mock_time():
        return fake_time[0]

    def mock_sleep(seconds):
        sleep_calls.append(seconds)
        fake_time[0] += seconds

    monkeypatch.setattr("time.time", mock_time)
    monkeypatch.setattr("time.sleep", mock_sleep)

    # 20 RPM -> min_interval is 60 / 20 = 3.0 seconds
    evaluator = OpenRouterEvaluator(api_key="fake-key", rpm_limit=20)
    assert evaluator.min_interval == 3.0

    # First call: last_call_time is 0.0 -> elapsed is 1000.0 >= 3.0 -> no sleep
    evaluator._rate_limit_wait()
    assert sleep_calls == []
    assert evaluator.last_call_time == 1000.0

    # Simulate 1.0 second elapsed since last call
    fake_time[0] = 1001.0
    # Second call: elapsed is 1.0 < 3.0 -> should sleep for 2.0s
    evaluator._rate_limit_wait()
    assert len(sleep_calls) == 1
    assert pytest.approx(sleep_calls[0]) == 2.0
    # After sleep(2.0), fake_time is 1003.0, last_call_time is 1003.0
    assert evaluator.last_call_time == 1003.0

    # Third call immediately after: elapsed is 0.0 < 3.0 -> should sleep 3.0s
    evaluator._rate_limit_wait()
    assert len(sleep_calls) == 2
    assert pytest.approx(sleep_calls[1]) == 3.0


def test_openrouter_evaluator_all_attempts_429_raises_runtime_error(monkeypatch):
    mock_post = MagicMock()
    resp_429 = MagicMock()
    resp_429.status_code = 429
    mock_post.return_value = resp_429
    monkeypatch.setattr("httpx.Client.post", mock_post)
    monkeypatch.setattr("time.sleep", MagicMock())

    evaluator = OpenRouterEvaluator(api_key="fake-key", rpm_limit=100)
    job = JobPosting(
        dedup_hash="abc",
        site="linkedin",
        job_url="https://example.com/job",
        title="Career Coach",
        company="Company",
        location="Remote",
        description="Career coaching role.",
    )
    with pytest.raises(RuntimeError):
        evaluator.evaluate_job(job)
    assert mock_post.call_count == 3


def test_openrouter_evaluator_http_error_retries_and_raises(monkeypatch):
    mock_post = MagicMock()
    resp_500 = MagicMock()
    resp_500.status_code = 500
    resp_500.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Internal Server Error",
        request=MagicMock(),
        response=resp_500,
    )
    mock_post.return_value = resp_500
    monkeypatch.setattr("httpx.Client.post", mock_post)
    mock_sleep = MagicMock()
    monkeypatch.setattr("time.sleep", mock_sleep)

    evaluator = OpenRouterEvaluator(api_key="fake-key", rpm_limit=100)
    job = JobPosting(
        dedup_hash="abc",
        site="linkedin",
        job_url="https://example.com/job",
        title="Career Coach",
        company="Company",
        location="Remote",
        description="Career coaching role.",
    )
    with pytest.raises(httpx.HTTPStatusError):
        evaluator.evaluate_job(job)
    assert mock_post.call_count == 3
    # Check that retry sleeps were invoked: attempt 1 (2.0s), attempt 2 (4.0s)
    mock_sleep.assert_any_call(2.0)
    mock_sleep.assert_any_call(4.0)


def test_openrouter_evaluator_from_settings():
    from src.config import Settings

    settings = Settings(
        OPENROUTER_API_KEY="test-settings-key",
        OPENROUTER_MODEL="custom/model",
        OPENROUTER_BASE_URL="https://custom.openrouter.ai/api/v1/",
        OPENROUTER_RPM_LIMIT=30,
        OPENROUTER_TIMEOUT_SECONDS=60.0,
    )
    evaluator = OpenRouterEvaluator.from_settings(settings)
    assert evaluator.api_key == "test-settings-key"
    assert evaluator.model == "custom/model"
    assert evaluator.base_url == "https://custom.openrouter.ai/api/v1"
    assert evaluator.rpm_limit == 30
    assert evaluator.timeout == 60.0


def test_openrouter_evaluator_from_settings_default(monkeypatch):
    from src.config import Settings
    import src.config

    custom_settings = Settings(
        OPENROUTER_API_KEY="default-env-key",
        OPENROUTER_MODEL="default-model",
    )
    monkeypatch.setattr(src.config, "get_settings", lambda: custom_settings)
    evaluator = OpenRouterEvaluator.from_settings()
    assert evaluator.api_key == "default-env-key"
    assert evaluator.model == "default-model"


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


