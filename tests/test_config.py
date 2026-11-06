import os
import pytest
from src.config import Settings, get_settings


def test_settings_default_values(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:test-bot-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "987654321")

    settings = Settings(_env_file=None)
    assert settings.app_env == "production"
    assert settings.log_level == "INFO"
    assert settings.timezone == "UTC"
    assert str(settings.db_path) == "data/jobs.db"
    assert not hasattr(settings, "openrouter_model")  # legacy single-model setting was removed
    assert settings.openrouter_base_url == "https://openrouter.ai/api/v1"
    assert settings.openrouter_rpm_limit == 20
    assert settings.openrouter_timeout_seconds == 45.0
    assert settings.wow_score_threshold == 8.5
    assert settings.digest_score_threshold == 6.5
    assert settings.digest_max_items == 10
    assert settings.scrape_interval_hours == 6
    assert settings.hours_old_window == 48
    assert settings.scraper_jitter_min_seconds == 4.0
    assert settings.scraper_jitter_max_seconds == 9.0
    assert settings.scraper_results_per_query == 10


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "custom-key")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "custom-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "112233")
    monkeypatch.setenv("WOW_SCORE_THRESHOLD", "9.2")
    monkeypatch.setenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
    monkeypatch.setenv("DB_PATH", "/tmp/custom_jobs.db")
    monkeypatch.setenv("SCRAPER_RESULTS_PER_QUERY", "50")

    settings = Settings(_env_file=None)
    assert settings.wow_score_threshold == 9.2
    assert not hasattr(settings, "openrouter_model")  # legacy var is accepted but ignored
    assert str(settings.db_path) == "/tmp/custom_jobs.db"
    assert settings.scraper_results_per_query == 50


def test_get_settings_caching(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "cached-key")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "cached-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123")

    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


def test_scraper_enabled_boards_and_proxies_parsing(monkeypatch):
    monkeypatch.delenv("SCRAPER_ENABLED_BOARDS", raising=False)
    monkeypatch.delenv("SCRAPER_PROXIES", raising=False)
    monkeypatch.delenv("SCRAPER_RESULTS_PER_QUERY", raising=False)

    # Defaults
    settings = Settings(_env_file=None)
    assert settings.scraper_results_per_query == 10
    assert settings.scraper_enabled_boards is None
    assert settings.scraper_proxies is None

    # Comma-delimited parsing with stripping
    monkeypatch.setenv("SCRAPER_ENABLED_BOARDS", "linkedin, indeed, google")
    monkeypatch.setenv("SCRAPER_PROXIES", " http://p1:8080 , http://p2:8080 ")
    settings = Settings(_env_file=None)
    assert settings.scraper_enabled_boards == ["linkedin", "indeed", "google"]
    assert settings.scraper_proxies == ["http://p1:8080", "http://p2:8080"]

    # Empty or whitespace string parses into None
    monkeypatch.setenv("SCRAPER_ENABLED_BOARDS", "")
    monkeypatch.setenv("SCRAPER_PROXIES", "   ")
    settings = Settings(_env_file=None)
    assert settings.scraper_enabled_boards is None
    assert settings.scraper_proxies is None


NEW_ENV_VARS = [
    "NVIDIA_API_KEY", "NVIDIA_MODEL", "NVIDIA_BASE_URL", "NVIDIA_RPM_LIMIT",
    "NVIDIA_REASONING_EFFORT", "NVIDIA_MAX_TOKENS", "NVIDIA_TEMPERATURE",
    "NVIDIA_TIMEOUT_SECONDS", "OPENROUTER_API_KEY", "OPENROUTER_MODELS",
    "OPENROUTER_DAILY_BUDGET", "EVAL_MAX_WORKERS", "EVAL_MAX_ATTEMPTS",
    "EVAL_BATCH_LIMIT", "LLM_BREAKER_FAILURES", "LLM_BREAKER_COOLDOWN_SECONDS",
]


def _clean_env(monkeypatch):
    for name in NEW_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_new_engine_setting_defaults(monkeypatch):
    _clean_env(monkeypatch)
    settings = Settings(_env_file=None)
    assert settings.nvidia_api_key is None
    assert settings.nvidia_model == "nvidia/nemotron-3-ultra-550b-a55b"
    assert settings.nvidia_base_url == "https://integrate.api.nvidia.com/v1"
    assert settings.nvidia_rpm_limit == 30
    assert settings.nvidia_reasoning_effort == "low"
    assert settings.nvidia_max_tokens == 4096
    assert settings.nvidia_temperature == 0.6
    assert settings.nvidia_timeout_seconds == 240
    assert settings.openrouter_models is None
    assert settings.openrouter_daily_budget == 45
    assert settings.eval_max_workers == 8
    assert settings.eval_max_attempts == 5
    assert settings.eval_batch_limit == 300
    assert settings.llm_breaker_failures == 3
    assert settings.llm_breaker_cooldown_seconds == 600


def test_new_engine_settings_env_override(monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setenv("NVIDIA_API_KEY", "nv-test")
    monkeypatch.setenv("NVIDIA_RPM_LIMIT", "10")
    monkeypatch.setenv("NVIDIA_TEMPERATURE", "0.2")
    monkeypatch.setenv("EVAL_MAX_WORKERS", "2")
    monkeypatch.setenv("OPENROUTER_DAILY_BUDGET", "5")
    settings = Settings(_env_file=None)
    assert settings.nvidia_api_key == "nv-test"
    assert settings.nvidia_rpm_limit == 10
    assert settings.nvidia_temperature == 0.2
    assert settings.eval_max_workers == 2
    assert settings.openrouter_daily_budget == 5


def test_openrouter_models_comma_parsing(monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setenv("OPENROUTER_MODELS", "a,b , c")
    assert Settings(_env_file=None).openrouter_models == ["a", "b", "c"]

    monkeypatch.setenv("OPENROUTER_MODELS", "  ")
    assert Settings(_env_file=None).openrouter_models is None


def test_nim_enabled_rule(monkeypatch):
    _clean_env(monkeypatch)
    assert Settings(_env_file=None).nim_enabled is False
    monkeypatch.setenv("NVIDIA_API_KEY", "")
    assert Settings(_env_file=None).nim_enabled is False
    monkeypatch.setenv("NVIDIA_API_KEY", "nv-test")
    assert Settings(_env_file=None).nim_enabled is True


def test_openrouter_enabled_requires_key_and_models(monkeypatch):
    _clean_env(monkeypatch)
    assert Settings(_env_file=None).openrouter_enabled is False

    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")
    assert Settings(_env_file=None).openrouter_enabled is False  # no models

    monkeypatch.setenv("OPENROUTER_MODELS", "a,b")
    assert Settings(_env_file=None).openrouter_enabled is True

    monkeypatch.delenv("OPENROUTER_API_KEY")
    assert Settings(_env_file=None).openrouter_enabled is False  # no key


def test_env_example_enables_no_provider_and_no_proxy(monkeypatch):
    """`cp .env.example .env` must not enable providers or route scraping through a fake proxy."""
    from pathlib import Path

    for name in (
        "NVIDIA_API_KEY", "OPENROUTER_API_KEY", "OPENROUTER_MODELS", "OPENROUTER_MODEL",
        "SCRAPER_PROXIES", "SCRAPER_ENABLED_BOARDS",
    ):
        monkeypatch.delenv(name, raising=False)
    example = Path(__file__).resolve().parent.parent / ".env.example"
    s = Settings(_env_file=str(example))
    assert s.nim_enabled is False
    assert s.openrouter_enabled is False
    assert s.scraper_proxies is None


def test_legacy_openrouter_model_env_var_still_loads(monkeypatch, tmp_path):
    for name in ("OPENROUTER_MODELS", "OPENROUTER_API_KEY", "NVIDIA_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    env = tmp_path / ".env"
    env.write_text("OPENROUTER_MODEL=openrouter/free\nWOW_SCORE_THRESHOLD=9.1\n", encoding="utf-8")
    s = Settings(_env_file=str(env))
    assert s.wow_score_threshold == 9.1
    assert not hasattr(s, "openrouter_model")
