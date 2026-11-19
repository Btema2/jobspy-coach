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


# --- masking of personal data before text goes to an LLM (src/bot/privacy.py) -------------------------

import pytest  # noqa: E402

from src.bot.privacy import CONSENT_NOTICE, CONSENT_VERSION, mask_personal_data  # noqa: E402


def test_masks_email_and_phones():
    text = "Mail a.b@example.org or call +48 123 456 789, (030) 1234-5678 or 0612345678 today."

    masked, count = mask_personal_data(text)

    assert masked == "Mail [email] or call [phone], [phone] or [phone] today."
    assert count == 4


def test_masks_phone_variants_and_counts_each():
    masked, count = mask_personal_data("x.y+tag@sub.example.co.uk / +1.555.123.4567 / 555-123-4567 / 0049 30 123 456 78")
    assert masked == "[email] / [phone] / [phone] / [phone]"
    assert count == 4


def test_leaves_dates_years_salaries_and_ids():
    text = "2019-2024, 2024-11-18, 120000-150000, EUR 4,500, ISO 9001, <ROLE>2024, v1.2.3, 12:30-14:00"

    masked, count = mask_personal_data(text)

    assert masked == text
    assert count == 0


def test_never_masks_a_run_inside_a_longer_token():
    for text in ("ID0612345678", "0612345678X", "ref_0612345678", "a1234567890123456789"):
        assert mask_personal_data(text) == (text, 0)
    assert mask_personal_data("12345678") == ("12345678", 0)  # 8 digits: too short
    assert mask_personal_data("1234567890123456") == ("1234567890123456", 0)  # 16 digits: too long


@pytest.mark.parametrize("number", [
    "+49 15112345678", "+48 600123456", "030 12345678", "+44 7700 900123", "030/1234567", "0612345678",
    "+48 123 456 789", "(030) 1234-5678", "+1 (555) 123-4567", "555.123.4567", "123456789", "123456789012345",
    "+49 (0) 30 1234567",
])
def test_masks_common_phone_formats(number):
    assert mask_personal_data(f"call {number} now") == ("call [phone] now", 1)


@pytest.mark.parametrize("text", [
    "2024-11-18", "2024-11-18 10:30", "2024-11-18 10:30:00", "10:30", "2019-2024", "120000-150000", "EUR 4,500",
    "18.11.2024 10.30", "18/11/2024", "2019 2020 2021", "12:30-14:00", "ISO 9001", "v1.2.3", "12345678",
    "1234567890123456", "12 34 56",
])
def test_phone_rule_keeps_dates_years_and_salary_ranges(text):
    assert mask_personal_data(f"see {text} ok") == (f"see {text} ok", 0)


def test_phone_over_masking_is_documented_best_effort():
    # Recall over precision: bare reference numbers and IP addresses look like phone numbers.
    assert mask_personal_data("ref 123456789") == ("ref [phone]", 1)
    assert mask_personal_data("host 192.168.100.200") == ("host [phone]", 1)


def test_phone_separators_are_single_characters():
    assert mask_personal_data("+48  600 123 456") == ("+48  [phone]", 1)  # a double space ends the number
    assert mask_personal_data("600,123,456") == ("600,123,456", 0)


def test_empty_and_plain_text_unchanged():
    assert mask_personal_data("") == ("", 0)
    assert mask_personal_data("I want <ROLE> work in <MARKET>.") == ("I want <ROLE> work in <MARKET>.", 0)


def test_consent_notice_names_providers():
    assert CONSENT_VERSION == 1
    notice = CONSENT_NOTICE
    assert "NVIDIA NIM" in notice and "OpenRouter" in notice
    assert "free" in notice.lower() and "logging" in notice.lower() and "training" in notice.lower()
    assert "names" in notice and "addresses" in notice and "phone numbers" in notice
    assert "e-mail addresses" in notice and "masked" in notice


def test_masking_is_linear_on_hostile_input():
    import time

    hostile = [
        "1-" * 10000 + "a", "a." * 10000, "a@" + "b." * 5000, "(1) " * 5000 + "z", "1/" * 10000 + "a",
        "12 " * 7000 + "x", "1" * 20480, "+" * 20480, "2024-11-18 " * 1800, "(" * 20480, "1 (" * 6000,
        "12.34." * 3400 + "z", "a@b." * 5000,
    ]
    for text in hostile:
        assert len(text) <= 21100
        started = time.monotonic()
        mask_personal_data(text)
        assert time.monotonic() - started < 0.5, text[:12]
