"""Regression guards: tracked files must not contain machine-specific or personal identifiers."""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# Machine-specific absolute paths / service users that leak the developer's environment.
LEAK_PATTERNS = [
    re.compile(r"/home/[A-Za-z0-9._-]+/"),
    re.compile(r"file:///"),
    re.compile(r"^\s*(User|Group)=(?!<)\S+", re.MULTILINE),
]


def _tracked_text_files():
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("git metadata unavailable")
    out = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    for rel in out:
        if rel == "tests/test_privacy.py":
            continue
        path = ROOT / rel
        if path.suffix in {".py", ".md", ".json", ".toml", ".example", ".txt", ".yml", ".yaml"} or path.name.startswith(".env"):
            if path.exists():
                yield rel, path.read_text(encoding="utf-8", errors="ignore")


def test_tracked_files_have_no_machine_specific_paths_or_users():
    offenders = []
    for rel, text in _tracked_text_files():
        for pat in LEAK_PATTERNS:
            m = pat.search(text)
            if m:
                offenders.append(f"{rel}: {m.group(0).strip()!r}")
    assert not offenders, "Machine-specific identifiers in tracked files:\n" + "\n".join(offenders)


def test_env_example_has_neutral_timezone():
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "TIMEZONE=UTC" in text
    assert "Warsaw" not in text
