import dataclasses
import re
from pathlib import Path

import pytest

from src.evaluator.prompts import DEFAULT_CANDIDATE_PERSONA, build_system_prompt
from src.profile import compiler
from src.profile.compiler import compile_profile, expanded_query_id
from src.profile.errors import ProfileCompileError
from src.profile.snapshot import (
    Market,
    PinnedQuery,
    ProfileFields,
    ProfileSnapshot,
    Role,
    RoleTerm,
    Rule,
)
from src.scraper.query_catalog import SearchQuery

FIXTURES = Path(__file__).parent / "fixtures"


def _snap(**kw) -> ProfileSnapshot:
    kw.setdefault("fields", ProfileFields())
    return ProfileSnapshot(**kw)


def _usa_market(**kw) -> Market:
    base = dict(name="<MARKET>", location="Remote", countries=("usa",), remote=True)
    base.update(kw)
    return Market(**base)


def test_pinned_queries_compile_one_to_one():
    pinned = (
        PinnedQuery("Q-B", "<T1>", "<LOC>", ("linkedin", "indeed"), country="germany", is_remote=True,
                    google_search_term="custom"),
        PinnedQuery("Q-A", "<T2>", "Europe", ("linkedin", "google")),
        PinnedQuery("Q-OFF", "<T3>", "<LOC>", ("linkedin",), enabled=False),
    )
    compiled = compile_profile(_snap(pinned_queries=pinned))

    assert compiled.queries == (
        SearchQuery("Q-B", "<T1>", "<LOC>", ["linkedin", "indeed"], "germany", True, "custom"),
        SearchQuery("Q-A", "<T2>", "Europe", ["linkedin", "google"], None, False, None),
    )


def test_role_market_expansion():
    compiled = compile_profile(_snap(roles=(Role("<ROLE>"),), markets=(_usa_market(),)))

    base, usa = compiled.queries
    assert base.query_id == "role=<ROLE>;term=<ROLE>;market=<MARKET>"
    assert base.site_names == ["linkedin", "google", "zip_recruiter"]
    assert base.country_code is None and base.is_remote is True and base.location == "Remote"
    assert usa.query_id == "role=<ROLE>;term=<ROLE>;market=<MARKET>;country=usa"
    assert usa.site_names == ["indeed", "glassdoor"]
    assert usa.country_code == "usa" and usa.location == "Remote" and usa.is_remote is True
    assert len(compiled.queries) == 2


def test_expanded_query_id_format():
    assert expanded_query_id("r", "t", "m") == "role=r;term=t;market=m"
    assert expanded_query_id("r", "t", "m", "usa") == "role=r;term=t;market=m;country=usa"


def test_language_tag_limits_markets():
    role = Role("<ROLE>", terms=(RoleTerm("<DE TERM>", "DE"),))
    markets = (
        Market("<M-DE>", "<LOC-DE>", languages=("de",)),
        Market("<M-EN>", "<LOC-EN>", languages=("en",)),
    )
    compiled = compile_profile(_snap(roles=(role,), markets=markets))

    terms_by_market = {(q.search_term, q.location) for q in compiled.queries}
    assert ("<DE TERM>", "<LOC-DE>") in terms_by_market
    assert ("<DE TERM>", "<LOC-EN>") not in terms_by_market
    assert ("<ROLE>", "<LOC-EN>") in terms_by_market


def test_untagged_term_goes_to_all_markets():
    role = Role("<ROLE>", terms=(RoleTerm("<PLAIN>"),))
    markets = (
        Market("<M1>", "<L1>", languages=("de",)),
        Market("<M2>", "<L2>", languages=("en",)),
    )
    compiled = compile_profile(_snap(roles=(role,), markets=markets))

    plain = {q.location for q in compiled.queries if q.search_term == "<PLAIN>"}
    assert plain == {"<L1>", "<L2>"}


def test_term_dedupe_is_case_insensitive_first_wins():
    role = Role("<Role>", terms=(RoleTerm("<ROLE>"), RoleTerm("<other>")))
    compiled = compile_profile(_snap(roles=(role,), markets=(Market("<M>", "<L>"),)))
    assert [q.search_term for q in compiled.queries] == ["<Role>", "<other>"]


def test_country_bound_boards_one_call_per_country():
    market = Market("<MARKET>", "<LOC>", countries=("germany", "poland"))
    compiled = compile_profile(_snap(roles=(Role("<ROLE>"),), markets=(market,)))

    country_queries = [q for q in compiled.queries if q.country_code]
    assert [(q.country_code, q.site_names, q.location) for q in country_queries] == [
        ("germany", ["indeed", "glassdoor"], "Germany"),
        ("poland", ["indeed"], "Poland"),
    ]
    indeed_queries = [q for q in compiled.queries if "indeed" in q.site_names]
    assert len(indeed_queries) == 2


def test_country_label_special_cases():
    market = Market("<M>", "<LOC>", countries=("usa", "uk"))
    compiled = compile_profile(_snap(roles=(Role("<ROLE>"),), markets=(market,)))
    assert [q.location for q in compiled.queries if q.country_code] == ["United States", "United Kingdom"]


def test_market_without_country_skips_indeed_with_warning():
    market = Market("<MARKET>", "Europe")
    compiled = compile_profile(_snap(roles=(Role("<ROLE>", terms=(RoleTerm("<T2>"),)),), markets=(market,)))

    assert all("indeed" not in q.site_names and "glassdoor" not in q.site_names for q in compiled.queries)
    msgs = [w for w in compiled.warnings if "market '<MARKET>'" in w]
    assert msgs == ["market '<MARKET>': no country, Indeed and Glassdoor skipped"]


def test_ziprecruiter_only_for_us_uk_canada():
    markets = (
        Market("<US>", "<L1>", countries=("usa", "canada")),
        Market("<EU>", "<L2>", countries=("germany",)),
        Market("<NONE>", "<L3>"),
    )
    compiled = compile_profile(_snap(roles=(Role("<ROLE>"),), markets=markets))
    base = {q.location: q.site_names for q in compiled.queries if q.country_code is None}
    assert "zip_recruiter" in base["<L1>"]
    assert "zip_recruiter" not in base["<L2>"]
    assert "zip_recruiter" not in base["<L3>"]


def test_disabled_items_excluded():
    roles = (Role("<ON>"), Role("<OFF>", enabled=False))
    markets = (Market("<M-ON>", "<L1>"), Market("<M-OFF>", "<L2>", enabled=False))
    compiled = compile_profile(_snap(roles=roles, markets=markets))

    assert {(q.search_term, q.location) for q in compiled.queries} == {("<ON>", "<L1>")}


def test_dedup_by_query_id():
    pinned = (
        PinnedQuery("role=<ROLE>;term=<ROLE>;market=<M>", "<PINNED>", "<L>", ("linkedin",)),
        PinnedQuery("DUP", "<A>", "<L>", ("linkedin",)),
        PinnedQuery("DUP", "<B>", "<L>", ("google",)),
    )
    compiled = compile_profile(
        _snap(pinned_queries=pinned, roles=(Role("<ROLE>"),), markets=(Market("<M>", "<L>"),))
    )
    assert [q.search_term for q in compiled.queries] == ["<PINNED>", "<A>"]


def test_call_count_and_duration_estimate():
    compiled = compile_profile(_snap(roles=(Role("<ROLE>"),), markets=(_usa_market(),)))

    est = compiled.estimate
    assert est.total_calls == 5
    assert est.linkedin_calls == 1
    assert est.duration_seconds == 87.0


def test_pinned_indeed_without_country_contributes_no_calls():
    pinned = (PinnedQuery("Q", "<T>", "Europe", ("indeed",)),)
    compiled = compile_profile(_snap(pinned_queries=pinned))
    assert compiled.estimate.total_calls == 0
    assert compiled.estimate.duration_seconds == 0.0


def test_estimate_over_interval_warns():
    snap = _snap(roles=(Role("<ROLE>"),), markets=(_usa_market(),))

    assert not any("interval" in w for w in compile_profile(snap).warnings)
    assert not any("interval" in w for w in compile_profile(snap, scrape_interval_hours=1.0).warnings)
    warned = compile_profile(snap, scrape_interval_hours=0.01)
    assert any("interval" in w for w in warned.warnings)


def test_pinned_board_country_warnings():
    pinned = (
        PinnedQuery("G", "<T>", "<L>", ("glassdoor",), country="poland"),
        PinnedQuery("Z", "<T>", "<L>", ("zip_recruiter",), country="germany"),
        PinnedQuery("OK", "<T>", "<L>", ("glassdoor", "zip_recruiter"), country="usa"),
    )
    compiled = compile_profile(_snap(pinned_queries=pinned))
    assert len(compiled.warnings) == 2
    assert any("glassdoor" in w and "poland" in w for w in compiled.warnings)
    assert any("zip_recruiter" in w and "germany" in w for w in compiled.warnings)


def test_unknown_board_raises_compile_error():
    pinned = (PinnedQuery("Q", "<T>", "<L>", ("nosuchboard",)),)
    with pytest.raises(ProfileCompileError):
        compile_profile(_snap(pinned_queries=pinned))


def test_compiled_prompt_equals_todays_prompt():
    persona = (FIXTURES / "legacy_persona.md").read_text(encoding="utf-8")
    compiled = compile_profile(_snap(fields=ProfileFields(persona=persona)))
    assert compiled.system_prompt == build_system_prompt(persona.strip())


def test_empty_persona_matches_frozen_generic_prompt():
    compiled = compile_profile(_snap(fields=ProfileFields(persona="   ")))
    frozen = (FIXTURES / "system_prompt_generic.txt").read_text()
    assert compiled.system_prompt == frozen
    assert DEFAULT_CANDIDATE_PERSONA in compiled.system_prompt


def test_bonus_enabled_follows_bonus_criteria():
    assert compile_profile(_snap()).bonus_enabled is False
    assert compile_profile(_snap(fields=ProfileFields(bonus_criteria="  "))).bonus_enabled is False
    assert compile_profile(_snap(fields=ProfileFields(bonus_criteria="<CRED>"))).bonus_enabled is True


def test_passthrough_fields():
    snap = _snap(
        fields=ProfileFields(paused=True, wow_threshold=9.0, digest_threshold=7.0), version=4
    )
    compiled = compile_profile(snap)
    assert (compiled.version, compiled.paused, compiled.wow_threshold, compiled.digest_threshold) == (
        4, True, 9.0, 7.0,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        compiled.version = 5  # type: ignore[misc]


def _rule(rules, kind_mode_term):
    kind, mode, term = kind_mode_term
    compiled = compile_profile(_snap(rules=(Rule(kind, term, mode, "r", "user"),)))
    return compiled.rules[0].regex


def test_rule_match_modes():
    exact = _rule(None, ("block", "exact", "l&d"))
    assert exact.search("Head of L&D")
    assert not exact.search("l&dx")
    assert not exact.search("xl&d")

    prefix = _rule(None, ("soft", "prefix", "coach"))
    assert prefix.search("Coaching")
    assert not prefix.search("Lifecoach")

    pattern = _rule(None, ("block", "pattern", "cross-?fit"))
    assert pattern.search("Crossfit")
    assert pattern.search("CROSS-FIT")


def test_compiled_rule_fields():
    compiled = compile_profile(_snap(rules=(Rule("seniority", "intern", "exact", "why", "seed"),)))
    rule = compiled.rules[0]
    assert (rule.kind, rule.reason, rule.source) == ("seniority", "why", "seed")
    assert isinstance(rule.regex, re.Pattern)


def test_invalid_pattern_raises_compile_error():
    with pytest.raises(ProfileCompileError):
        compile_profile(_snap(rules=(Rule("block", "(unclosed", "pattern", "r", "seed"),)))


def test_unknown_rule_kind_or_mode_raises():
    with pytest.raises(ProfileCompileError):
        compile_profile(_snap(rules=(Rule("weird", "x", "exact", "r", "user"),)))
    with pytest.raises(ProfileCompileError):
        compile_profile(_snap(rules=(Rule("block", "x", "fuzzy", "r", "user"),)))


def test_code_tables_match_jobspy():
    import jobspy.glassdoor.constant as gd
    import jobspy.model as jm

    assert compiler.KNOWN_COUNTRIES == {c.value for c in jm.Country}
    assert compiler.SUPPORTED_BOARDS == {s.value for s in jm.Site}
    assert compiler.GLASSDOOR_COUNTRIES == {c.value for c in gd.domains}


def test_constants():
    assert compiler.LINKEDIN_COOLDOWN_SECONDS == 12.0
    assert compiler.PER_CALL_ALLOWANCE_SECONDS == 15.0
    assert compiler.EXPANSION_BOARDS == ("linkedin", "google", "zip_recruiter", "indeed", "glassdoor")
    assert compiler.COUNTRY_BOUND_BOARDS == frozenset({"indeed", "glassdoor"})
    assert compiler.ZIPRECRUITER_COUNTRIES == frozenset({"usa", "uk", "canada"})
