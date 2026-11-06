"""The ``live`` marker is registered and live tests are skipped unless ``-m live`` is given."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _run_pytest(*args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    # Blank any provider keys so this check can never reach the network.
    env.update({"NVIDIA_API_KEY": "", "OPENROUTER_API_KEY": "", "OPENROUTER_MODELS": ""})
    return subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_live_smoke.py", "-rs", "-q", "-p", "no:cacheprovider", *args],
        cwd=ROOT, capture_output=True, text=True, env=env, timeout=120,
    )


def test_live_tests_are_skipped_by_default():
    proc = _run_pytest()
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out
    assert "1 skipped" in out, out
    assert "-m live" in out, out  # skipped by the marker hook, not by the missing-key guard


def test_live_marker_is_registered():
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--markers", "-p", "no:cacheprovider"],
        cwd=ROOT, capture_output=True, text=True, timeout=120,
    )
    assert "@pytest.mark.live" in proc.stdout
