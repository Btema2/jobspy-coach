import pytest

from src.filters import classify_title
from src.filters.pre_filter import pre_filter_job
from src.filters.rules import CompiledRule, RuleHit, compile_rule, legacy_rules, match_title, rule_key
from src.filters.seniority import SENIORITY_LEVELS, SENIORITY_TERMS, seniority_rules
from src.profile.compiler import compile_profile
from src.profile.importer import build_imported_snapshot
from tests import test_pre_filter
from tests.test_pre_filter import VALID_DESC


def _imported():
    snap, _ = build_imported_snapshot([], "<PERSONA>", wow_threshold=8.5, digest_threshold=6.5)
    return compile_profile(snap)


def _harvest_titles():
    titles = []
    for name, value in vars(test_pre_filter).items():
        if name.endswith("_TITLES") and isinstance(value, list):
            titles.extend(value)
        marks = getattr(value, "pytestmark", None) if callable(value) else None
        for mark in marks or ():
            if mark.name == "parametrize" and mark.args and mark.args[0] == "title":
                titles.extend(mark.args[1])
    return titles


def test_all_pre_filter_cases_identical_with_seed_and_imported_rules():
    titles = _harvest_titles()
    assert len(titles) > 100
    imported_rules = _imported().rules
    seed = legacy_rules()
    for t in titles:
        base = classify_title(t)
        assert base == classify_title(t, rules=seed) == classify_title(t, rules=imported_rules), t
        a = pre_filter_job(t, VALID_DESC)
        for rules in (seed, imported_rules):
            b = pre_filter_job(t, VALID_DESC, rules=rules)
            assert (a.passed, a.reason) == (b.passed, b.reason), t


def test_reason_prefixes_preserved():
    assert classify_title("Swimming Coach", rules=legacy_rules()) == "EXCLUDED_TITLE_REGEX: 'Swimming'"
    hit = match_title("Swimming Coach", legacy_rules())
    assert isinstance(hit, RuleHit) and hit.reason == "EXCLUDED_TITLE_REGEX: 'Swimming'"


def test_rule_key_recorded():
    assert pre_filter_job("Scrum Master", VALID_DESC).rule_key == "seed:block:EXCLUDED_TITLE_TRAP"
    stub = pre_filter_job("<ROLE> Coach", "too short")
    assert stub.passed is False and stub.rule_key is None
    assert pre_filter_job("<ROLE> Coach", VALID_DESC).rule_key is None


def test_rule_key_function():
    assert rule_key("block", "x", "pattern", "R") == "seed:block:R"
    assert rule_key("soft", "Data", "exact", "R") == "soft:exact:data"
    assert rule_key("block", "Developer", "prefix", "R") == "block:prefix:developer"


def _rule(kind, term, mode="exact", reason="USER"):
    return compile_rule(kind, term, mode, reason, "user")


def test_compile_rule_errors_and_fields():
    rule = _rule("block", "abc")
    assert isinstance(rule, CompiledRule)
    assert (rule.kind, rule.reason, rule.source, rule.key) == ("block", "USER", "user", "block:exact:abc")
    assert compile_rule("block", "abc", "exact", "R", "user", key="k").key == "k"
    with pytest.raises(ValueError):
        compile_rule("nope", "abc", "exact", "R", "user")
    with pytest.raises(ValueError):
        compile_rule("block", "abc", "fuzzy", "R", "user")
    with pytest.raises(ValueError):
        compile_rule("block", "(", "pattern", "R", "seed")


def test_exact_user_rule_is_singular_only():
    exact = (_rule("block", "developer"),)
    assert classify_title("Developer Coach", rules=exact) is not None
    assert classify_title("Coach for Developers", rules=exact) is None
    prefix = (_rule("block", "developer", "prefix"),)
    assert classify_title("Developer Coach", rules=prefix) is not None
    assert classify_title("Coach for Developers", rules=prefix) is not None


def test_ld_term_with_lookaround():
    rules = (_rule("soft", "data"), _rule("rescue", "l&d"))
    assert classify_title("Data L&D Partner", rules=rules) is None
    assert classify_title("Data L&Dx Partner", rules=rules) is not None


def test_multiword_term_matches_any_whitespace():
    rules = (_rule("block", "project manager"),)
    assert classify_title("Project  Manager", rules=rules) is not None
    assert classify_title("Project Manager", rules=rules) is not None
    assert classify_title("Project\tManager", rules=rules) is not None


def test_kind_order_block_beats_rescue():
    rules = (_rule("rescue", "coach", "prefix"), _rule("block", "agile"))
    hit = match_title("Agile Coach", rules)
    assert hit is not None and hit.rule_key == "block:exact:agile"


def test_rescue_saves_soft_and_seniority():
    rules = (
        _rule("soft", "data"),
        _rule("seniority", "director"),
        _rule("rescue", "coach", "prefix"),
    )
    assert match_title("Data Director", rules) is not None
    assert match_title("Data Director Coach", rules) is None


def test_empty_ruleset_drops_nothing_but_stub_check_remains():
    assert classify_title("Swimming Coach", rules=()) is None
    assert pre_filter_job("Swimming Coach", VALID_DESC, rules=()).passed is True
    res = pre_filter_job("Swimming Coach", "too short", rules=())
    assert res.passed is False and res.reason.startswith("DESCRIPTION_TOO_SHORT")


def test_blank_title_matches_nothing():
    assert match_title(None, legacy_rules()) is None
    assert match_title("   ", legacy_rules()) is None


@pytest.mark.parametrize("target,title,dropped", [
    ("mid", "Senior <ROLE>", True),
    ("mid", "Head of <ROLE>", True),
    ("mid", "Chief <ROLE> Officer", True),
    ("mid", "Sr. <ROLE>", True),
    ("mid", "<ROLE> Specialist", False),
    ("senior", "Senior <ROLE>", False),
    ("senior", "Director of <ROLE>", True),
    ("executive", "Chief <ROLE> Officer", False),
    ("executive", "Head of <ROLE>", False),
    ("junior", "Senior <ROLE>", True),
])
def test_seniority_table_levels(target, title, dropped):
    rules = seniority_rules(target, ["en"])
    assert (classify_title(title, rules=rules) is not None) is dropped


def test_seniority_target_none_and_executive_add_nothing():
    assert seniority_rules(None, ["en"]) == ()
    assert seniority_rules("executive", ["en"]) == ()


def test_seniority_rule_shape_and_languages():
    rules = seniority_rules("mid", ["en", "zz"])
    assert rules and all(r.kind == "seniority" and r.source == "level" for r in rules)
    assert all(r.reason == "EXCLUDED_SENIORITY" and r.key.startswith("level:") for r in rules)
    assert "level:senior:senior" in {r.key for r in rules}
    assert len({r.key for r in rules}) == len(rules)  # "en" listed once even if repeated
    assert len(seniority_rules("mid", ["en", "en"])) == len(rules)
    assert SENIORITY_LEVELS == ("junior", "mid", "senior", "lead", "head", "executive")
    assert set(SENIORITY_TERMS) == {"en"}


def test_seniority_rules_rescued():
    rules = seniority_rules("mid", ["en"]) + (_rule("rescue", "<ROLE>"),)
    assert classify_title("Senior <ROLE>", rules=rules) is None


@pytest.mark.parametrize("mode", ["exact", "prefix", "pattern"])
@pytest.mark.parametrize("term", ["", "   ", "\t"])
def test_compile_rule_rejects_an_empty_term(mode, term):
    with pytest.raises(ValueError, match="empty"):
        compile_rule("block", term, mode, "USER", "user")


def test_seniority_rules_have_unique_keys():
    from src.filters.seniority import seniority_rules

    keys = [r.key for r in seniority_rules("junior", ["en", "EN", "xx"])]
    assert keys and len(keys) == len(set(keys))
