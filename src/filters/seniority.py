"""Seniority levels and the generated terms that exclude titles above a profile's target level."""
from typing import Dict, Iterable, List, Tuple

from src.filters.rules import CompiledRule, compile_rule

SENIORITY_LEVELS = ("junior", "mid", "senior", "lead", "head", "executive")

# Terms that signal each level, per language (only English ships). All are matched as exact words.
SENIORITY_TERMS: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "en": {
        "senior": ("senior", "sr"),
        "lead": ("team lead", "team leader", "principal"),
        "head": (
            "head of", "director", "directors", "vp", "vice president", "vice presidents",
            "senior manager", "senior managers",
        ),
        "executive": (
            "chief", "ceo", "cfo", "coo", "cto", "cio", "cmo", "chro", "evp", "svp", "c-level",
            "managing director",
        ),
    },
}


def seniority_rules(target_level: "str | None", languages: Iterable[str]) -> Tuple[CompiledRule, ...]:
    """Rules dropping titles of any level strictly above ``target_level`` (none when it is None)."""
    if target_level is None or target_level not in SENIORITY_LEVELS:
        return ()
    above = SENIORITY_LEVELS[SENIORITY_LEVELS.index(target_level) + 1:]
    langs: List[str] = ["en"]
    for lang in languages:
        lang = lang.lower()
        if lang in SENIORITY_TERMS and lang not in langs:
            langs.append(lang)
    rules: List[CompiledRule] = []
    for lang in langs:
        for level in above:
            for term in SENIORITY_TERMS[lang].get(level, ()):
                key = f"level:{level}:{term}"
                rules.append(compile_rule("seniority", term, "exact", "EXCLUDED_SENIORITY", "level", key=key))
    return tuple(rules)
