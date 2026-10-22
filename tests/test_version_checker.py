from unittest.mock import MagicMock, patch
import httpx
import pytest
from importlib.metadata import PackageNotFoundError

from src.scraper.version_checker import VersionCheckResult, check_jobspy_update
import src.scraper.version_checker as vc


@pytest.fixture(autouse=True)
def reset_cache():
    """Reset the module-level in-memory cache before and after each test."""
    vc._cached_result = None
    vc._cached_timestamp = 0.0
    yield
    vc._cached_result = None
    vc._cached_timestamp = 0.0


def test_version_check_update_available():
    """Test update available when PyPI has a newer version than installed."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"info": {"version": "1.4.0"}}
    mock_resp.raise_for_status = MagicMock()

    with patch("importlib.metadata.version", return_value="1.3.0"):
        with patch("httpx.Client.get", return_value=mock_resp) as mock_get:
            result = check_jobspy_update(package_name="python-jobspy")

            assert result is not None
            assert result.has_update is True
            assert result.installed_version == "1.3.0"
            assert result.latest_version == "1.4.0"
            mock_get.assert_called_once_with("https://pypi.org/pypi/python-jobspy/json")


def test_version_check_already_latest():
    """Test has_update is False when installed matches PyPI version."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"info": {"version": "1.3.0"}}
    mock_resp.raise_for_status = MagicMock()

    with patch("importlib.metadata.version", return_value="1.3.0"):
        with patch("httpx.Client.get", return_value=mock_resp):
            result = check_jobspy_update(package_name="python-jobspy")

            assert result is not None
            assert result.has_update is False
            assert result.installed_version == "1.3.0"
            assert result.latest_version == "1.3.0"


def test_version_check_installed_newer():
    """Test has_update is False when installed version is newer than PyPI (e.g. dev/git version)."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"info": {"version": "1.2.0"}}
    mock_resp.raise_for_status = MagicMock()

    with patch("importlib.metadata.version", return_value="1.3.0"):
        with patch("httpx.Client.get", return_value=mock_resp):
            result = check_jobspy_update(package_name="python-jobspy")

            assert result is not None
            assert result.has_update is False
            assert result.installed_version == "1.3.0"
            assert result.latest_version == "1.2.0"


def test_version_check_pypi_error_returns_none():
    """Test that PyPI HTTP errors or timeouts gracefully return None without raising."""
    with patch("importlib.metadata.version", return_value="1.3.0"):
        # HTTPStatusError
        with patch("httpx.Client.get", side_effect=httpx.ConnectTimeout("Connection timed out")):
            result = check_jobspy_update(package_name="python-jobspy")
            assert result is None

        # HTTP error status
        err_resp = MagicMock(status_code=500)
        with patch(
            "httpx.Client.get",
            side_effect=httpx.HTTPStatusError("500 Server Error", request=MagicMock(), response=err_resp),
        ):
            result = check_jobspy_update(package_name="python-jobspy")
            assert result is None

        # Generic unexpected exception
        with patch("httpx.Client.get", side_effect=RuntimeError("Unexpected network failure")):
            result = check_jobspy_update(package_name="python-jobspy")
            assert result is None


def test_version_check_package_not_found_returns_none():
    """Test that PackageNotFoundError returns None without querying PyPI."""
    with patch("importlib.metadata.version", side_effect=PackageNotFoundError("python-jobspy")):
        with patch("httpx.Client.get") as mock_get:
            result = check_jobspy_update(package_name="python-jobspy")
            assert result is None
            mock_get.assert_not_called()


def test_version_check_caching():
    """Test that consecutive calls within cache_ttl_seconds use the cached result."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"info": {"version": "1.4.0"}}
    mock_resp.raise_for_status = MagicMock()

    with patch("importlib.metadata.version", return_value="1.3.0"):
        with patch("httpx.Client.get", return_value=mock_resp) as mock_get:
            res1 = check_jobspy_update(package_name="python-jobspy", cache_ttl_seconds=3600.0)
            assert res1 is not None
            assert res1.has_update is True
            assert mock_get.call_count == 1

            # Second call immediately should hit cache
            res2 = check_jobspy_update(package_name="python-jobspy", cache_ttl_seconds=3600.0)
            assert res2 == res1
            assert mock_get.call_count == 1

            # Call with force_refresh=True should bypass cache and query PyPI again
            res3 = check_jobspy_update(
                package_name="python-jobspy",
                cache_ttl_seconds=3600.0,
                force_refresh=True,
            )
            assert res3 == res1
            assert mock_get.call_count == 2


def test_version_check_cache_expiry():
    """Test that when cache expires after cache_ttl_seconds, PyPI is queried again."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"info": {"version": "1.4.0"}}
    mock_resp.raise_for_status = MagicMock()

    with patch("importlib.metadata.version", return_value="1.3.0"):
        with patch("httpx.Client.get", return_value=mock_resp) as mock_get:
            check_jobspy_update(package_name="python-jobspy", cache_ttl_seconds=10.0)
            assert mock_get.call_count == 1

            # Simulate passage of time by setting _cached_timestamp in past
            vc._cached_timestamp -= 15.0

            check_jobspy_update(package_name="python-jobspy", cache_ttl_seconds=10.0)
            assert mock_get.call_count == 2


def test_version_check_malformed_pypi_response_returns_none():
    """Test that invalid json structure from PyPI returns None."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"unexpected": "payload"}
    mock_resp.raise_for_status = MagicMock()

    with patch("importlib.metadata.version", return_value="1.3.0"):
        with patch("httpx.Client.get", return_value=mock_resp):
            result = check_jobspy_update(package_name="python-jobspy")
            assert result is None
