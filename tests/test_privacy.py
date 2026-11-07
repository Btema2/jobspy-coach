"""Regression guards: tracked files must not contain machine-specific or personal identifiers."""
import hashlib
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


# Lowercase sha256 digests of real organisation/borough tokens, so the names are not spelled out in tracked files.
FORBIDDEN_TOKEN_DIGESTS = {
    "cbe2275d90e3604c3cf933c9c5601b012644c621843609c8bb39cae40b0ea756",
    "335cd93d3ef7754196551357b1a47d670e9b6d7499d063d545a90c7f99e6086b",
    "e39c6386825bef98a3df9cac493f674d78aa775d442d5f59bbaf735e708c51d4",
    "9fbaf5bf9f7b720a51130d616c21bb12447b6e3e69bd023638700b05ff2aed5d",
}


def test_readme_and_dedup_fixtures_carry_no_real_org_or_home_city_examples():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "Warsaw" not in readme, "README.md contains 'Warsaw'"
    for rel in ("README.md", "tests/test_dedup.py"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        tokens = re.findall(r"[A-Za-z]+", text)
        digests = {hashlib.sha256(t.lower().encode("utf-8")).hexdigest() for t in tokens}
        hits = digests & FORBIDDEN_TOKEN_DIGESTS
        assert not hits, f"{rel} contains forbidden tokens (sha256 digests): {sorted(hits)}"
