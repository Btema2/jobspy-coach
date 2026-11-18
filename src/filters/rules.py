"""Compiled pre-filter rules and the single title matcher.

A profile's rules (and the seniority table) compile to ``CompiledRule`` objects; ``match_title`` applies
them. ``rules=None`` means the legacy seed ruleset (today's hard-coded behaviour). This module imports
nothing from ``src.profile.compiler`` (the compiler imports it).

Evaluation order: block rules (in order), then any rescue match returns "pass", then soft rules, then
seniority rules.
"""
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Sequence, Tuple

from src.profile.legacy_seed import seed_rules

RULE_KINDS = frozenset({"block", "soft", "seniority", "rescue"})
RULE_MODES = frozenset({"exact", "prefix", "pattern"})

_FLAGS = re.IGNORECASE | re.UNICODE


@dataclass(frozen=True)
class CompiledRule:
    kind: str
    regex: "re.Pattern[str]"
    reason: str
    source: str
    key: str = ""


@dataclass(frozen=True)
class RuleHit:
    reason: str
    rule_key: str


def rule_key(kind: str, term: str, match_mode: str, reason: str) -> str:
    """Stable identifier of a rule, stored with a pre-filtered job so it can be traced back."""
    if match_mode == "pattern":
        return f"seed:{kind}:{reason}"
    return f"{kind}:{match_mode}:{term.casefold()}"


def _literal(term: str) -> str:
    return r"\s+".join(re.escape(part) for part in term.split())


def compile_rule(
    kind: str, term: str, match_mode: str, reason: str, source: str, key: Optional[str] = None
) -> CompiledRule:
    """Compile one rule; raises ``ValueError`` on an unknown kind or mode or an invalid regex."""
    if kind not in RULE_KINDS:
        raise ValueError(f"unknown kind '{kind}'")
    if match_mode == "exact":
        pattern = rf"(?<!\w){_literal(term)}(?!\w)"
    elif match_mode == "prefix":
        pattern = rf"(?<!\w){_literal(term)}\w*"
    elif match_mode == "pattern":
        pattern = term
    else:
        raise ValueError(f"unknown match mode '{match_mode}'")
    try:
        regex = re.compile(pattern, _FLAGS)
    except re.error as exc:
        raise ValueError(f"invalid regular expression ({exc})") from exc
    if key is None:
        key = rule_key(kind, term, match_mode, reason)
    return CompiledRule(kind, regex, reason, source, key)


@lru_cache(maxsize=1)
def legacy_rules() -> Tuple[CompiledRule, ...]:
    """The legacy seed ruleset, compiled once."""
    return tuple(compile_rule(r.kind, r.term, r.match_mode, r.reason, r.source) for r in seed_rules())


def _first(rules: Sequence[CompiledRule], kind: str, title: str):
    for rule in rules:
        if rule.kind == kind:
            match = rule.regex.search(title)
            if match:
                return rule, match
    return None


def match_title(title: Optional[str], rules: Optional[Sequence[CompiledRule]] = None) -> Optional[RuleHit]:
    """The rule that drops ``title``, or None if the title may proceed (blank titles are not classified)."""
    clean = (title or "").strip()
    if not clean:
        return None
    if rules is None:
        rules = legacy_rules()

    for kind in ("block", "rescue", "soft", "seniority"):
        found = _first(rules, kind, clean)
        if found is None:
            continue
        if kind == "rescue":
            return None
        rule, match = found
        return RuleHit(f"{rule.reason}: '{match.group(0)}'", rule.key)
    return None
