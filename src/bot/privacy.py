"""Consent notice and best-effort masking of personal data before text reaches an LLM.

Masking is local and deliberately conservative: it catches obvious e-mail addresses and phone numbers
and leaves dates, years, salary ranges and identifiers alone. Names and addresses cannot be recognised
reliably, which is why the notice asks the user to leave them out.
"""
import re
from typing import Tuple

CONSENT_VERSION = 2

CONSENT_NOTICE = (
    "Before I use a language model for free text: what you write is sent to NVIDIA NIM and, when that "
    "is unavailable, to free models on OpenRouter, together with your stored profile (persona and definitions, "
    "which are not masked) so that I can propose changes to it. Their logging and training policies may differ, and I "
    "cannot control what they keep. Please leave out names, addresses and phone numbers. Obvious e-mail addresses "
    "and phone numbers are masked here before sending when I recognise them, but that is best effort. "
    "If you send a CV, the text is read on this machine; e-mail addresses, phone numbers, web addresses starting "
    "with http or www, links with a path, links to well-known profile sites and @handles are masked and the "
    "contact header is dropped before anything is sent; the CV text itself is never stored. "
    "The facts I note from your answers and your CV are masked and stored locally in the bot's database for up to "
    "24 hours. Names and addresses cannot be recognised reliably, so please leave them out. "
    "Choose below to continue; the commands (/help) work without any language model."
)

# Every repetition is bounded, so hostile input cannot make the matching quadratic.
_EMAIL_RE = re.compile(
    r"(?<![A-Za-z0-9._%+\-])[A-Za-z0-9._%+\-]{1,64}@[A-Za-z0-9\-]{1,63}(?:\.[A-Za-z0-9\-]{1,63}){0,8}\.[A-Za-z]{2,24}"
)
# Candidate: groups of digits (optionally in parentheses) joined by one space, dot, dash or slash, not part of
# a longer alphanumeric token; ``_is_phone`` applies the digit-count and look-alike rules. Bounded repetitions.
_SCHEME_TAIL_RE = re.compile(r"(?:https?|ftp)$", re.IGNORECASE)
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
        found = match.group(0)
        if match.string[match.end() : match.end() + 3] == "://":
            # the TLD class swallowed the scheme of a link that follows directly: give it back
            scheme = _SCHEME_TAIL_RE.search(found)
            if scheme and re.search(r"\.[A-Za-z]{2,24}$", found[: scheme.start()]):
                return "[email]" + scheme.group(0)
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


# --- links and handles ---------------------------------------------------------------------------------
# Every repetition is bounded or a single character class, so hostile input stays linear.
_URL_RE = re.compile(
    r"(?<![\w@])(?:https?://|ftp://|www\.)[^\s<>\"']*[^\s<>\"'.,;:!?)\]}]|(?<![\w@])(?:https?://|ftp://|www\.)",
    re.IGNORECASE,
)
_SOCIAL_HOSTS = (
    "linkedin.com", "github.com", "gitlab.com", "twitter.com", "x.com", "facebook.com", "instagram.com",
    "t.me", "telegram.me", "medium.com", "behance.net", "dribbble.com", "xing.com", "youtube.com", "youtu.be",
    "stackoverflow.com", "bit.ly", "tiktok.com", "threads.net", "goo.gl", "tinyurl.com", "lnkd.in", "ow.ly",
)
_SOCIAL_RE = re.compile(
    r"(?<![\w@./-])(?:[A-Za-z0-9-]{1,30}\.){0,2}(?:"
    + "|".join(re.escape(h) for h in _SOCIAL_HOSTS)
    + r")(?![\w-])(?:/[^\s<>\"']*[^\s<>\"'.,;:!?)\]}])?",
    re.IGNORECASE,
)
# A dotted host with a letters-only TLD followed by a path (``janedoe.io/work``). A bare ``name.tld`` stays
# readable on purpose (``Vue.js``, ``ASP.NET``, ``node.js`` are not links), and so does a tech name followed by
# a slash (``Node.js/TypeScript``, ``ASP.NET/C#``): a ``.js`` ending, an upper-case ``.NET`` and the known
# ``*.net`` frameworks are never treated as hosts.
_PATH_LINK_RE = re.compile(
    r"(?<![\w@./-])(?:[A-Za-z0-9-]{1,63}\.){1,5}[A-Za-z]{2,24}/[^\s<>\"']*[^\s<>\"'.,;:!?)\]}]"
)
_DOTNET_HOSTS = frozenset({"asp.net", "vb.net", "ado.net"})


def _path_link(match: "re.Match[str]") -> str:
    token = match.group(0)
    host = token.split("/", 1)[0]
    tld = host.rsplit(".", 1)[-1]
    if tld.lower() == "js" or tld == "NET" or host.lower() in _DOTNET_HOSTS:
        return token  # a tech name such as Node.js/TypeScript, not a link
    return "[link]"
_HANDLE_RE = re.compile(r"(?<![\w@.])@\w{2,30}(?:\.\w{1,30}){0,3}")


def mask_links(text: str) -> str:
    """``text`` with URLs and social-media paths as ``[link]`` and ``@handles`` as ``[handle]``."""
    text = _URL_RE.sub("[link]", text)
    text = _SOCIAL_RE.sub("[link]", text)
    text = _PATH_LINK_RE.sub(_path_link, text)
    return _HANDLE_RE.sub("[handle]", text)


def mask_all(text: str) -> str:
    """E-mail addresses, phone numbers, links and handles masked."""
    return mask_links(mask_personal_data(text)[0])


# --- contact header of a CV ----------------------------------------------------------------------------
_HEADINGS = frozenset(
    {
        # English
        "summary", "profile", "about me", "objective", "experience", "work experience", "employment",
        "education", "skills", "languages", "certifications", "projects",
        # Ukrainian
        "резюме", "про себе", "мета", "цілі", "досвід", "досвід роботи", "освіта", "навички", "мови",
        "сертифікати", "проєкти", "проекти",
        # Polish
        "podsumowanie", "profil", "o mnie", "cel zawodowy", "doświadczenie", "doświadczenie zawodowe",
        "wykształcenie", "edukacja", "umiejętności", "języki", "certyfikaty", "projekty",
        # German
        "zusammenfassung", "über mich", "berufsziel", "berufserfahrung", "erfahrung", "ausbildung", "bildung",
        "kenntnisse", "fähigkeiten", "sprachen", "zertifikate", "projekte",
        # Spanish
        "resumen", "perfil", "sobre mí", "objetivo", "experiencia", "experiencia laboral", "educación",
        "formación", "habilidades", "competencias", "idiomas", "certificaciones", "proyectos",
        # French
        "résumé", "à propos", "objectif", "expérience", "expérience professionnelle", "formation", "éducation",
        "compétences", "langues",
    }
)
MAX_HEADING_CHARS = 40
HEADER_FALLBACK_LINES = 6


def _is_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > MAX_HEADING_CHARS:
        return False
    return stripped.casefold().rstrip(":").strip() in _HEADINGS


def strip_contact_header(text: str) -> str:
    """``text`` without its contact header: everything before the first heading line (never the first line);
    with no heading, the first 6 non-empty lines. ``""`` when everything was header."""
    lines = text.splitlines()
    first = next((i for i, line in enumerate(lines) if line.strip()), None)
    if first is None:
        return ""
    for i in range(first + 1, len(lines)):
        if _is_heading(lines[i]):
            return "\n".join(lines[i:])
    seen = 0
    for i in range(first, len(lines)):
        if lines[i].strip():
            seen += 1
            if seen == HEADER_FALLBACK_LINES:
                return "\n".join(lines[i + 1 :]).lstrip("\n")
    return ""
