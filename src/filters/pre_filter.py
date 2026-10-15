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
    r"swim|swimming|pływani[ae]|basen|плаванн[яі]|"
    r"tennis|tenis[a-ząćęłńóśźż]*|padel|squash|теніс[а-яіїє]*|"
    r"gym|fitness|siłowni[a-ząćęłńóśźż]*|kulturystyk[a-ząćęłńóśźż]*|workout|crossfit|фітнес[а-яіїє]*|"
    r"soccer|football|piłk[ai]|piłkars[a-ząćęłńóśźż]*|futsal|футбол[а-яіїє]*|"
    r"basketball|koszykówk[ai]|volleyball|siatkówk[ai]|баскетбол[а-яіїє]*|волейбол[а-яіїє]*|"
    r"gymnastics|gimnastyk[ai]|acrobatics|akrobatyk[ai]|гімнастик[а-яіїє]*|"
    r"hockey|hokej|skating|łyżw[a-ząćęłńóśźż]*|rolk[a-ząćęłńóśźż]*|хокей[а-яіїє]*|"
    r"running|biegani[ae]|athletics|lekkoatletyk[ai]|біг[а-яіїє]*|"
    r"martial\s*arts|karate|judo|boxing|boks|mma|wrestling|zapasy|бокс[а-яіїє]*|карате|"
    r"climbing|wspinaczk[ai]|ski|skiing|snowboard|narciar[a-ząćęłńóśźż]*|лиж[іи]|"
    r"yoga|joga|pilates|йог[аи]|"
    r"horse\s*riding|jeździectw[ao]|hipoterap[a-ząćęłńóśźż]*|кінн[а-яіїє]*|"
    # Arts, Voice & Performance
    r"vocal|singing|śpiew[a-ząćęłńóśźż]*|głos|voice|вокал[а-яіїє]*|спів[а-яіїє]*|"
    r"acting|drama|teatr|theatre|театр[а-яіїє]*|"
    r"dance|taniec|tańca|choreograph[a-z]*|танц[іе][а-яіїє]*|"
    # Esports & Gaming
    r"esport[s]?|gaming|valorant|league\s*of\s*legends|cs:go|dota|"
    # Animal training
    r"dog|puppy|pies|psa|treser[a-ząćęłńóśźż]*|zwierząt|собак[а-яіїє]*|цуцен[а-яіїє]*"
    r")\b",
    re.IGNORECASE | re.UNICODE,
)

MIN_DESCRIPTION_CHARS = 150
MIN_DESCRIPTION_WORDS = 30


def pre_filter_job(title: str, description: str) -> PreFilterResult:
    """Deterministic pre-filter evaluating title exclusions and description quality.

    Filters out athletic, sports, fitness, vocal, gaming, and animal training positions,
    as well as postings with descriptions that are too short or empty.
    """
    clean_title = (title or "").strip()
    match = EXCLUSION_TITLE_REGEX.search(clean_title)
    if match:
        return PreFilterResult(
            passed=False,
            reason=f"EXCLUDED_TITLE_REGEX: '{match.group(0)}'",
        )

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
