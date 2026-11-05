"""Lightweight, non-blocking PyPI version checker for python-jobspy."""

from dataclasses import dataclass
import importlib.metadata
import logging
import time
from typing import Optional

import httpx
from packaging.version import parse

logger = logging.getLogger(__name__)

# Module-level in-memory cache
_cached_result: Optional["VersionCheckResult"] = None
_cached_timestamp: float = 0.0


@dataclass
class VersionCheckResult:
    has_update: bool
    installed_version: str
    latest_version: str


def check_jobspy_update(
    package_name: str = "python-jobspy",
    timeout: float = 3.0,
    cache_ttl_seconds: float = 43200.0,
    force_refresh: bool = False,
) -> Optional[VersionCheckResult]:
    """Check PyPI for a newer version of the specified package.

    Caches results in-memory for `cache_ttl_seconds` (default 12 hours) to
    prevent redundant outbound network requests. Fails gracefully returning
    `None` without raising if network, timeout, or parsing issues occur.
    """
    global _cached_result, _cached_timestamp

    now = time.time()
    if not force_refresh and (now - _cached_timestamp < cache_ttl_seconds) and (_cached_result is not None):
        return _cached_result

    # 1. Determine currently installed version
    try:
        installed_version = importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        logger.warning("Package '%s' not found in installed metadata.", package_name)
        return None
    except Exception as e:
        logger.warning("Failed to determine installed version for '%s': %s", package_name, e)
        return None

    # 2. Query PyPI JSON API
    url = f"https://pypi.org/pypi/{package_name}/json"
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(url)
            resp.raise_for_status()
            data = resp.json()

        latest_version = data.get("info", {}).get("version")
        if not latest_version:
            logger.debug("PyPI response for '%s' missing 'info.version'.", package_name)
            return None

        has_update = parse(latest_version) > parse(installed_version)
        result = VersionCheckResult(
            has_update=has_update,
            installed_version=installed_version,
            latest_version=latest_version,
        )

        _cached_result = result
        _cached_timestamp = now
        return result
    except (httpx.HTTPError, httpx.TimeoutException) as exc:
        logger.debug("Network error checking PyPI for '%s': %s", package_name, exc)
        return None
    except Exception as exc:
        logger.debug("Unexpected error checking PyPI for '%s': %s", package_name, exc)
        return None
