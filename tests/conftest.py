"""Pytest configuration and shared fixtures."""
import pytest
from src.config import get_settings


@pytest.fixture(autouse=True)
def reset_settings_cache():
    """Ensure cached settings singleton does not leak across unit tests."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def pytest_collection_modifyitems(config, items):
    """Skip ``live`` tests (real network calls) unless the run was selected with ``-m live``."""
    if "live" in (config.getoption("markexpr") or ""):
        return
    skip_live = pytest.mark.skip(reason="live test: run explicitly with -m live")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
