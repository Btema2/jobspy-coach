"""Pytest configuration and shared fixtures."""
import pytest
from src.config import get_settings


@pytest.fixture(autouse=True)
def reset_settings_cache():
    """Ensure cached settings singleton does not leak across unit tests."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
