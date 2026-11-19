"""Consent notice and best-effort masking of personal data before text reaches an LLM.

Masking is local and deliberately conservative: it catches obvious e-mail addresses and phone numbers
and leaves dates, years, salary ranges and identifiers alone. Names and addresses cannot be recognised
reliably, which is why the notice asks the user to leave them out.
"""
import re
from typing import Tuple

CONSENT_VERSION = 1

CONSENT_NOTICE = (
    "Before I use a language model for free text: what you write is sent to NVIDIA NIM and, when that "
    "is unavailable, to free models on OpenRouter. Their logging and training policies may differ, and I "
    "cannot control what they keep. Please leave out names, addresses and phone numbers. Obvious e-mail addresses "
    "and phone numbers are masked here before sending when I recognise them, but that is best effort. "
    "Choose below to continue; the commands (/help) work without any language model."
)

# Every repetition is bounded, so hostile input cannot make the matching quadratic.
_EMAIL_RE = re.compile(
    r"(?<![A-Za-z0-9._%+\-])[A-Za-z0-9._%+\-]{1,64}@[A-Za-z0-9\-]{1,63}(?:\.[A-Za-z0-9\-]{1,63}){0,8}\.[A-Za-z]{2,24}"
)
# Candidate: groups of digits (optionally in parentheses) joined by one space, dot, dash or slash, not part of
# a longer alphanumeric token; ``_is_phone`` applies the digit-count and look-alike rules. Bounded repetitions.
_PHONE_RE = re.compile(
    r"(?<![A-Za-z0-9_])\+?\(?\d{1,15}\)?(?:[ .\-/]\(?\d{1,15}\)?){0,15}(?![A-Za-z0-9_])"
)
_DIGITS_RE = re.compile(r"\d+")
# Things that look like phone numbers by digit count but are dates, date-times or years.
_DATE_RE = re.compile(r"(?<!\d)(?:(?:19|20)\d{2}[./-]\d{1,2}[./-]\d{1,2}|\d{1,2}[./-]\d{1,2}[./-](?:19|20)\d{2})(?!\d)")
_YEAR_RE = re.compile(r"(?:19|20)\d{2}")
MIN_PHONE_DIGITS = 9
MAX_PHONE_DIGITS = 15
MAX_GROUP_DIGITS = 12
MIN_RANGE_DIGITS = 5


def _is_phone(candidate: str) -> bool:
    groups = _DIGITS_RE.findall(candidate)
    total = sum(len(g) for g in groups)
    if not MIN_PHONE_DIGITS <= total <= MAX_PHONE_DIGITS:
        return False
    if len(groups) == 1:  # a bare run of 9-15 digits
        return True
    if any(len(g) > MAX_GROUP_DIGITS for g in groups):
        return False
    if _DATE_RE.search(candidate) or all(_YEAR_RE.fullmatch(g) for g in groups):
        return False  # a date, a date with a time, or a list of years
    plain = candidate.strip()
    if (
        len(groups) == 2
        and re.fullmatch(r"\d+-\d+", plain)
        and all(len(g) >= MIN_RANGE_DIGITS and not g.startswith("0") for g in groups)
    ):
        return False  # a range such as a salary: 120000-150000
    return True


def mask_personal_data(text: str) -> Tuple[str, int]:
    """``text`` with e-mail addresses as ``[email]`` and phone numbers as ``[phone]``, and the number of
    replacements."""
    count = 0

    def email(match: "re.Match[str]") -> str:
        nonlocal count
        count += 1
        return "[email]"

    def phone(match: "re.Match[str]") -> str:
        nonlocal count
        if not _is_phone(match.group(0)):
            return match.group(0)
        count += 1
        return "[phone]"

    text = _EMAIL_RE.sub(email, text)
    text = _PHONE_RE.sub(phone, text)
    return text, count
