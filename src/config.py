from functools import lru_cache
from pathlib import Path
from typing import Any, List, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # General
    app_env: str = Field(default="production", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    timezone: str = Field(default="UTC", alias="TIMEZONE")
    db_path: Path = Field(default=Path("data/jobs.db"), alias="DB_PATH")

    # OpenRouter LLM
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1", alias="OPENROUTER_BASE_URL")
    openrouter_rpm_limit: int = Field(default=20, alias="OPENROUTER_RPM_LIMIT")
    openrouter_timeout_seconds: float = Field(default=45.0, alias="OPENROUTER_TIMEOUT_SECONDS")

    openrouter_models: Optional[List[str]] = Field(default=None, alias="OPENROUTER_MODELS")
    openrouter_daily_budget: int = Field(default=45, alias="OPENROUTER_DAILY_BUDGET")

    # NVIDIA NIM LLM (primary provider)
    nvidia_api_key: Optional[str] = Field(default=None, alias="NVIDIA_API_KEY")
    nvidia_base_url: str = Field(default="https://integrate.api.nvidia.com/v1", alias="NVIDIA_BASE_URL")
    nvidia_model: str = Field(default="nvidia/nemotron-3-ultra-550b-a55b", alias="NVIDIA_MODEL")
    nvidia_rpm_limit: int = Field(default=30, alias="NVIDIA_RPM_LIMIT")
    nvidia_reasoning_effort: str = Field(default="low", alias="NVIDIA_REASONING_EFFORT")
    nvidia_max_tokens: int = Field(default=4096, alias="NVIDIA_MAX_TOKENS")
    nvidia_temperature: float = Field(default=0.6, alias="NVIDIA_TEMPERATURE")
    nvidia_timeout_seconds: float = Field(default=240.0, alias="NVIDIA_TIMEOUT_SECONDS")

    # Evaluation engine
    eval_max_workers: int = Field(default=8, alias="EVAL_MAX_WORKERS")
    eval_max_attempts: int = Field(default=5, alias="EVAL_MAX_ATTEMPTS")
    eval_batch_limit: int = Field(default=300, alias="EVAL_BATCH_LIMIT")
    llm_breaker_failures: int = Field(default=3, alias="LLM_BREAKER_FAILURES")
    llm_breaker_cooldown_seconds: float = Field(default=600.0, alias="LLM_BREAKER_COOLDOWN_SECONDS")

    # Telegram
    telegram_bot_token: str = Field(default="", alias="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: str = Field(default="", alias="TELEGRAM_CHAT_ID")

    # Scheduler & Scraper
    scrape_interval_hours: int = Field(default=6, alias="SCRAPE_INTERVAL_HOURS")
    digest_time_cet: str = Field(default="09:00", alias="DIGEST_TIME_CET")
    hours_old_window: int = Field(default=48, alias="HOURS_OLD_WINDOW")
    scraper_jitter_min_seconds: float = Field(default=4.0, alias="SCRAPER_JITTER_MIN_SECONDS")
    scraper_jitter_max_seconds: float = Field(default=9.0, alias="SCRAPER_JITTER_MAX_SECONDS")
    scraper_results_per_query: int = Field(default=10, alias="SCRAPER_RESULTS_PER_QUERY")
    scraper_enabled_boards: Optional[List[str]] = Field(default=None, alias="SCRAPER_ENABLED_BOARDS")
    scraper_proxies: Optional[List[str]] = Field(default=None, alias="SCRAPER_PROXIES")

    # Evaluation Thresholds
    wow_score_threshold: float = Field(default=8.5, alias="WOW_SCORE_THRESHOLD")
    digest_score_threshold: float = Field(default=6.5, alias="DIGEST_SCORE_THRESHOLD")
    digest_max_items: int = Field(default=10, alias="DIGEST_MAX_ITEMS")

    @field_validator("scraper_enabled_boards", "scraper_proxies", "openrouter_models", mode="before")
    @classmethod
    def parse_comma_delimited_list(cls, v: Any) -> Optional[List[str]]:
        if v is None:
            return None
        if isinstance(v, list):
            items = [str(item).strip() for item in v if str(item).strip()]
            return items if items else None
        if isinstance(v, str):
            items = [item.strip() for item in v.split(",") if item.strip()]
            return items if items else None
        return None

    @property
    def nim_enabled(self) -> bool:
        """NVIDIA NIM is enabled iff an API key is configured."""
        return bool(self.nvidia_api_key and self.nvidia_api_key.strip())

    @property
    def openrouter_enabled(self) -> bool:
        """OpenRouter fallback is enabled iff an API key and pinned models are configured."""
        return bool(self.openrouter_api_key and self.openrouter_api_key.strip() and self.openrouter_models)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
