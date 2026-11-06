import re
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PreFilterResult:
    passed: bool
    reason: Optional[str] = None


# Compiled exclusion regex matching sports, fitness, vocal, and pet training (EN, PL, UK)
EXCLUSION_TITLE_REGEX = re.compile(
    r"\b("
    # Sports & Athletics (EN/PL/UK)
    r"sport[s]?|sportow[a-ząćęłńóśźż]*|спорт[а-яіїє]*|"
    r"athletic[s]?|lekkoatletyk[ai]|"
    r"swim|swimming|pływani[ae]|basen|плаванн[яі]|"
    r"tennis|tenis[a-ząćęłńóśźż]*|padel|squash|теніс[а-яіїє]*|"
    r"gym|fitness|siłowni[a-ząćęłńóśźż]*|kulturystyk[a-ząćęłńóśźż]*|workout|cross-?fit|фітнес[а-яіїє]*|"
    r"soccer|football|piłk[ai]|piłkars[a-ząćęłńóśźż]*|futsal|футбол[а-яіїє]*|"
    r"basketball|koszykówk[ai]|volleyball|siatkówk[ai]|баскетбол[а-яіїє]*|волейбол[а-яіїє]*|"
    r"gymnastics|gimnastyk[ai]|acrobatics|akrobatyk[ai]|гімнастик[а-яіїє]*|"
    r"hockey|hokej|skating|łyżw[a-ząćęłńóśźż]*|rolk[a-ząćęłńóśźż]*|хокей[а-яіїє]*|"
    r"running|biegani[ae]|біг[а-яіїє]*|"
    r"martial\s*arts|karate|judo|boxing|boks|mma|wrestling|zapasy|бокс[а-яіїє]*|карате|"
    r"climbing|wspinaczk[ai]|ski|skiing|snowboard|narciar[a-ząćęłńóśźż]*|лиж[іи]|"
    r"yoga|joga|pilates|йог[аи]|"
    r"horse\s*riding|jeździectw[ao]|hipoterap[a-ząćęłńóśźż]*|кінн[а-яіїє]*|"
    # Arts, Voice & Performance
    r"vocal|singing|śpiew[a-ząćęłńóśźż]*|głos|voice|вокал[а-яіїє]*|спів[а-яіїє]*|"
    r"acting\s+coach|aktors[a-ząćęłńóśźż]*|drama|teatr|theatre|театр[а-яіїє]*|"
    r"dance|taniec|tańca|choreograph[a-z]*|танц[іе][а-яіїє]*|"
    # Esports & Gaming
    r"e-?sport[s]?|gaming|valorant|league\s*of\s*legends|cs:go|dota|"
    # Animal training
    r"animal[s]?|pet[s]?|dog|puppy|pies|(?:treser|trener)[a-ząćęłńóśźż]*\s+ps[aó]w|treser[a-ząćęłńóśźż]*|zwierząt|собак[а-яіїє]*|цуцен[а-яіїє]*"
    r")\b",
    re.IGNORECASE | re.UNICODE,
)

# ---------------------------------------------------------------------------
# Title traps and seniority
#
# All patterns are whole-word, case-insensitive and Unicode-aware. Inflected
# stems use a ``\w*`` suffix (e.g. ``psycholog\w*``); raw substrings are never
# used, so "Transport" does not match ``sport`` and "Business" does not match
# ``bus``.
# ---------------------------------------------------------------------------

_FLAGS = re.IGNORECASE | re.UNICODE


def _word_regex(*alternatives: str) -> "re.Pattern[str]":
    return re.compile(r"\b(?:" + "|".join(alternatives) + r")\b", _FLAGS)


# Pure IT / sales / driving / other roles. Never rescued, even when a coaching
# word is present ("Agile Coach" is a false positive we want to drop).
# Terms are deliberately SINGULAR (role names): plural audience words such as
# "Coach for Developers" or "Product Managers" must reach the LLM.
# Some athletic terms overlap EXCLUSION_TITLE_REGEX, which runs first and so
# supplies the reason string for those.
HARD_TRAP_REGEX = _word_regex(
    r"project\s+manager", r"product\s+manager", r"delivery\s+manager",
    r"scrum", r"agile",
    r"developer", r"devops", r"qa",
    r"account\s+executive", r"sales\s+manager", r"sales\s+director",
    r"financial\s+controller", r"audit\w*",
    r"fitness", r"gym", r"football", r"soccer", r"tennis", r"swim\w*",
    r"athletic\w*", r"vocal",
    r"bus", r"coach\s+builder", r"motor\s?coach", r"coach\s+driver", r"driver",
    # PL / UK equivalents
    r"programista", r"inżynier\s+oprogramowania", r"kierowca",
    r"розробник", r"водій",
)

# Technical-ish words that are dropped unless a coaching/psychology/L&D word
# rescues the title ("Career Coach - Software Engineering Bootcamp").
SOFT_TRAP_REGEX = _word_regex(
    r"software", r"engineering", r"architects?", r"data",
    r"program(?:me)?\s+managers?",
)

# Seniority terms, dropped unless rescued ("Head of Coaching" survives).
SENIORITY_REGEX = _word_regex(
    r"directors?", r"vp", r"vice\s+presidents?", r"head\s+of", r"chief",
    r"ceo", r"cfo", r"coo", r"cto", r"cio", r"senior\s+managers?",
    # PL
    r"dyrektor\w*", r"kierownik\w*", r"prezes\w*", r"członek\s+zarządu",
    # UK
    r"директор\w*", r"керівник\w*", r"голова",
)

# Coaching / psychology / L&D vocabulary that rescues soft traps and seniority.
# Bare "learning" is deliberately NOT here ("Director of Machine Learning" drops).
RESCUE_REGEX = re.compile(
    r"(?<!\w)(?:"
    r"coach\w*|psycholog\w*|counsel\w*|doradc\w*|trener\w*|mentor\w*|"
    r"learning\s+(?:and|&)\s+development|l&d|talent\s+development|people\s+development|"
    r"learning\s+experience|leadership\s+development|organi[sz]ational\s+development|"
    r"szkole\w*|психолог\w*|коуч\w*"
    r")(?!\w)",
    _FLAGS,
)


def classify_title(title: Optional[str]) -> Optional[str]:
    """Return a drop reason for a job title, or None if the title may proceed.

    Single implementation shared by the pre-filter and the evaluator safety net.
    Order: sports/pets/gaming regex, hard traps, soft traps (unless rescued),
    seniority (unless rescued). Empty titles are not classified (None).
    """
    clean_title = (title or "").strip()
    if not clean_title:
        return None

    match = EXCLUSION_TITLE_REGEX.search(clean_title)
    if match:
        return f"EXCLUDED_TITLE_REGEX: '{match.group(0)}'"

    match = HARD_TRAP_REGEX.search(clean_title)
    if match:
        return f"EXCLUDED_TITLE_TRAP: '{match.group(0)}'"

    if RESCUE_REGEX.search(clean_title):
        return None

    match = SOFT_TRAP_REGEX.search(clean_title)
    if match:
        return f"EXCLUDED_TITLE_TRAP: '{match.group(0)}'"

    match = SENIORITY_REGEX.search(clean_title)
    if match:
        return f"EXCLUDED_SENIORITY: '{match.group(0)}'"

    return None


MIN_DESCRIPTION_CHARS = 150
MIN_DESCRIPTION_WORDS = 30


def pre_filter_job(title: str, description: str) -> PreFilterResult:
    """Deterministic pre-filter evaluating title exclusions and description quality.

    Filters out athletic, sports, fitness, vocal, gaming, and animal training positions,
    IT/sales/driving title traps, senior-management titles (unless coaching-related),
    as well as postings with empty titles or descriptions that are too short or empty.
    """
    clean_title = (title or "").strip()
    if not clean_title:
        return PreFilterResult(passed=False, reason="TITLE_EMPTY")

    reason = classify_title(clean_title)
    if reason:
        return PreFilterResult(passed=False, reason=reason)

    clean_desc = (description or "").strip()
    if len(clean_desc) < MIN_DESCRIPTION_CHARS:
        return PreFilterResult(
            passed=False,
            reason=f"DESCRIPTION_TOO_SHORT: length={len(clean_desc)} (min {MIN_DESCRIPTION_CHARS})",
        )

    words = clean_desc.split()
    if len(words) < MIN_DESCRIPTION_WORDS:
        return PreFilterResult(
            passed=False,
            reason=f"DESCRIPTION_TOO_SHORT: words={len(words)} (min {MIN_DESCRIPTION_WORDS})",
        )

    return PreFilterResult(passed=True, reason=None)
