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
    assert settings.timezone == "Europe/Warsaw"
    assert str(settings.db_path) == "data/jobs.db"
    assert settings.openrouter_model == "openrouter/free"
    assert settings.openrouter_base_url == "https://openrouter.ai/api/v1"
    assert settings.openrouter_rpm_limit == 20
    assert settings.openrouter_timeout_seconds == 45.0
    assert settings.wow_score_threshold == 9.0
    assert settings.digest_score_threshold == 6.0
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
    monkeypatch.setenv("WOW_SCORE_THRESHOLD", "8.5")
    monkeypatch.setenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
    monkeypatch.setenv("DB_PATH", "/tmp/custom_jobs.db")
    monkeypatch.setenv("SCRAPER_RESULTS_PER_QUERY", "50")

    settings = Settings(_env_file=None)
    assert settings.wow_score_threshold == 8.5
    assert settings.openrouter_model == "meta-llama/llama-3.3-70b-instruct:free"
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
