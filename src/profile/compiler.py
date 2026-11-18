"""Pure, network-free compilation of a profile snapshot into what the daemon runs.

``compile_profile`` turns a snapshot into search queries (pinned queries 1:1, then a roles x markets x
boards expansion), compiled rules, the evaluator system prompt and a cost estimate. Board specifics
(Indeed quote stripping and 168 h window, Google term derivation, LinkedIn pacing) stay in the scraper;
only what decides *which* (query, board) calls exist lives here.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from src.evaluator.prompts import DEFAULT_CANDIDATE_PERSONA, render_system_prompt
from src.filters.rules import RULE_KINDS, RULE_MODES, CompiledRule, compile_rule
from src.filters.seniority import seniority_rules
from src.profile.errors import ProfileCompileError
from src.profile.snapshot import ProfileSnapshot
from src.scraper.query_catalog import SearchQuery

LINKEDIN_COOLDOWN_SECONDS = 12.0
PER_CALL_ALLOWANCE_SECONDS = 15.0

EXPANSION_BOARDS = ("linkedin", "google", "zip_recruiter", "indeed", "glassdoor")
# Indeed and Glassdoor search one country at a time and need a country code.
COUNTRY_BOUND_BOARDS = frozenset({"indeed", "glassdoor"})

# python-jobspy 1.4.0 `Site` values (guarded by a test against the installed package).
SUPPORTED_BOARDS = frozenset(
    {"linkedin", "indeed", "glassdoor", "google", "zip_recruiter", "bayt", "bdjobs", "hellowork", "naukri"}
)

# python-jobspy 1.4.0 `Country` values (guarded by a test against the installed package).
KNOWN_COUNTRIES = frozenset({
    "argentina", "australia", "austria", "bahrain", "belgium", "brazil", "canada", "chile", "china",
    "colombia", "costa rica", "czech republic", "denmark", "ecuador", "egypt", "finland", "france",
    "germany", "greece", "hong kong", "hungary", "india", "indonesia", "ireland", "israel", "italy",
    "japan", "kuwait", "luxembourg", "malaysia", "mexico", "morocco", "netherlands", "new zealand",
    "nigeria", "norway", "oman", "pakistan", "panama", "peru", "philippines", "poland", "portugal",
    "qatar", "romania", "saudi arabia", "singapore", "south africa", "south korea", "spain", "sweden",
    "switzerland", "taiwan", "thailand", "türkiye", "uk", "ukraine", "united arab emirates", "uruguay",
    "usa", "venezuela", "vietnam",
})

# python-jobspy 1.4.0 `country_aliases` (guarded by a test against the installed package): spellings the
# library resolves to a canonical `Country` value. They are normalised to the canonical value on write.
COUNTRY_ALIASES = {
    "czechia": "czech republic",
    "turkey": "türkiye",
    "uae": "united arab emirates",
    "united kingdom": "uk",
    "us": "usa",
    "united states": "usa",
}


def canonical_country(value: str) -> str:
    """Trimmed, lower-cased country with python-jobspy's aliases mapped to the canonical value."""
    name = value.strip().lower()
    return COUNTRY_ALIASES.get(name, name)


GLASSDOOR_COUNTRIES = frozenset({
    "argentina", "australia", "austria", "belgium", "brazil", "canada", "france", "germany", "hong kong",
    "india", "ireland", "italy", "malaysia", "mexico", "netherlands", "new zealand", "singapore", "spain",
    "switzerland", "uk", "usa", "vietnam",
})
ZIPRECRUITER_COUNTRIES = frozenset({"usa", "uk", "canada"})

_COUNTRY_LABELS = {"usa": "United States", "uk": "United Kingdom"}


@dataclass(frozen=True)
class CallEstimate:
    total_calls: int
    linkedin_calls: int
    duration_seconds: float


@dataclass(frozen=True)
class CompiledProfile:
    version: int
    queries: Tuple[SearchQuery, ...]
    rules: Tuple[CompiledRule, ...]
    system_prompt: str
    bonus_enabled: bool
    paused: bool
    wow_threshold: float
    digest_threshold: float
    estimate: CallEstimate
    warnings: Tuple[str, ...]


def expanded_query_id(role: str, term: str, market: str, country: Optional[str] = None) -> str:
    """Stable id of an expanded query; the role is parsed back out of it by the stats command."""
    base = f"role={role};term={term};market={market}"
    return f"{base};country={country}" if country else base


def parse_expanded_query_id(query_id: str) -> Optional[Dict[str, str]]:
    """Inverse of ``expanded_query_id``; None for ids that are not expanded (pinned ids)."""
    if not query_id.startswith("role="):
        return None
    parsed: Dict[str, str] = {}
    for part in query_id.split(";"):
        key, sep, value = part.partition("=")
        if not sep or key in parsed:
            return None
        parsed[key] = value
    keys = list(parsed)
    if keys not in (["role", "term", "market"], ["role", "term", "market", "country"]):
        return None
    return parsed


def _country_label(country: str) -> str:
    return _COUNTRY_LABELS.get(country, country.title())


def _compile_rule(kind: str, term: str, mode: str, reason: str, source: str) -> CompiledRule:
    try:
        return compile_rule(kind, term, mode, reason, source)
    except ValueError as exc:
        raise ProfileCompileError(f"rule '{term}': {exc}") from exc


def _expand(snapshot: ProfileSnapshot, warnings: List[str]) -> List[SearchQuery]:
    queries: List[SearchQuery] = []
    warned_markets: Set[str] = set()
    markets = [m for m in snapshot.markets if m.enabled]
    for role in snapshot.roles:
        if not role.enabled:
            continue
        terms: List[Tuple[str, Optional[str]]] = []
        seen: Set[str] = set()
        for term, lang in [(role.name, None)] + [(t.term, t.language) for t in role.terms]:
            if term.lower() not in seen:
                seen.add(term.lower())
                terms.append((term, lang))
        for market in markets:
            market_languages = {x.lower() for x in market.languages}
            remote = market.remote or market.remote_only
            for term, lang in terms:
                if lang and lang.lower() not in market_languages:
                    continue
                base = ["linkedin", "google"]
                if market.countries and set(market.countries) <= ZIPRECRUITER_COUNTRIES:
                    base.append("zip_recruiter")
                queries.append(
                    SearchQuery(expanded_query_id(role.name, term, market.name), term, market.location,
                                base, None, remote)
                )
                if not market.countries and market.name not in warned_markets:
                    warned_markets.add(market.name)
                    warnings.append(f"market '{market.name}': no country, Indeed and Glassdoor skipped")
                for country in market.countries:
                    boards = ["indeed"] + (["glassdoor"] if country in GLASSDOOR_COUNTRIES else [])
                    location = market.location if len(market.countries) == 1 else _country_label(country)
                    queries.append(
                        SearchQuery(expanded_query_id(role.name, term, market.name, country), term, location,
                                    boards, country, remote)
                    )
    return queries


def compile_profile(
    snapshot: ProfileSnapshot, *, scrape_interval_hours: Optional[float] = None
) -> CompiledProfile:
    warnings: List[str] = []
    queries: List[SearchQuery] = []
    for p in snapshot.pinned_queries:
        if not p.enabled:
            continue
        unknown = [b for b in p.boards if b not in SUPPORTED_BOARDS]
        if unknown:
            raise ProfileCompileError(f"pinned query '{p.query_id}': unknown board(s) {', '.join(unknown)}")
        if p.country is not None:
            if "glassdoor" in p.boards and p.country not in GLASSDOOR_COUNTRIES:
                warnings.append(f"pinned query '{p.query_id}': glassdoor does not support country '{p.country}'")
            if "zip_recruiter" in p.boards and p.country not in ZIPRECRUITER_COUNTRIES:
                warnings.append(
                    f"pinned query '{p.query_id}': zip_recruiter does not support country '{p.country}'"
                )
        queries.append(
            SearchQuery(p.query_id, p.term, p.location, list(p.boards), p.country, p.is_remote,
                        p.google_search_term)
        )
    queries.extend(_expand(snapshot, warnings))

    unique: Dict[str, SearchQuery] = {}
    for q in queries:
        unique.setdefault(q.query_id, q)
    final = tuple(unique.values())

    calls = [
        (q, b) for q in final for b in q.site_names
        if not (b in COUNTRY_BOUND_BOARDS and q.country_code is None)
    ]
    linkedin_calls = sum(1 for _, b in calls if b == "linkedin")
    estimate = CallEstimate(
        len(calls),
        linkedin_calls,
        linkedin_calls * LINKEDIN_COOLDOWN_SECONDS + len(calls) * PER_CALL_ALLOWANCE_SECONDS,
    )
    if scrape_interval_hours is not None and estimate.duration_seconds > scrape_interval_hours * 3600:
        warnings.append(
            f"estimated scrape duration {estimate.duration_seconds / 3600:.2f} h exceeds the scrape "
            f"interval of {scrape_interval_hours:g} h"
        )

    f = snapshot.fields
    languages: List[str] = []
    for market in snapshot.markets:
        if market.enabled:
            languages.extend(x.lower() for x in market.languages)
    rules = tuple(
        _compile_rule(r.kind, r.term, r.match_mode, r.reason, r.source) for r in snapshot.rules
    ) + seniority_rules(f.target_level, languages)
    return CompiledProfile(
        version=snapshot.version,
        queries=final,
        rules=rules,
        system_prompt=render_system_prompt(f.persona.strip() or DEFAULT_CANDIDATE_PERSONA),
        bonus_enabled=bool(f.bonus_criteria.strip()),
        paused=f.paused,
        wow_threshold=f.wow_threshold,
        digest_threshold=f.digest_threshold,
        estimate=estimate,
        warnings=tuple(warnings),
    )
