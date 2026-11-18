import ast
import re

from src.evaluator.prompts import DEFAULT_CANDIDATE_PERSONA, build_system_prompt
from src.filters import pre_filter
from src.profile import legacy_seed
from src.profile.legacy_seed import SEED_RULES, seed_rules
from src.profile.ops import validate_snapshot
from src.profile.snapshot import ProfileFields, ProfileSnapshot

_PAIRS = (
    ("block", "EXCLUSION_TITLE_REGEX", "EXCLUDED_TITLE_REGEX"),
    ("block", "HARD_TRAP_REGEX", "EXCLUDED_TITLE_TRAP"),
    ("soft", "SOFT_TRAP_REGEX", "EXCLUDED_TITLE_TRAP"),
    ("seniority", "SENIORITY_REGEX", "EXCLUDED_SENIORITY"),
    ("rescue", "RESCUE_REGEX", "RESCUE"),
)


def test_seed_patterns_equal_prefilter_regexes():
    assert len(SEED_RULES) == len(_PAIRS)
    for rule, (_, regex_name, _) in zip(SEED_RULES, _PAIRS):
        regex = getattr(pre_filter, regex_name)
        assert rule.pattern == regex.pattern, regex_name
        compiled = re.compile(rule.pattern, re.IGNORECASE | re.UNICODE)
        assert compiled.flags == regex.flags, regex_name


def test_seed_reason_labels_and_order():
    assert [(r.kind, r.reason) for r in SEED_RULES] == [(k, reason) for k, _, reason in _PAIRS]


def test_seed_rules_are_pattern_rules_from_the_seed_source():
    rules = seed_rules()

    assert [r.term for r in rules] == [r.pattern for r in SEED_RULES]
    assert {(r.match_mode, r.source) for r in rules} == {("pattern", "seed")}
    assert [(r.kind, r.reason) for r in rules] == [(r.kind, r.reason) for r in SEED_RULES]


def test_seed_definitions_are_substrings_of_todays_prompt():
    prompt = build_system_prompt(DEFAULT_CANDIDATE_PERSONA)

    for text in (
        legacy_seed.LEGACY_CORE_ROLE_DEFINITION,
        legacy_seed.LEGACY_ADJACENT_ROLE_DEFINITION,
        legacy_seed.LEGACY_DEALBREAKERS,
    ):
        assert text.strip() and text in prompt
    assert "WOW" in legacy_seed.LEGACY_CORE_ROLE_DEFINITION
    assert "role_is_direct_coaching" in legacy_seed.LEGACY_CORE_ROLE_DEFINITION
    assert "DIGEST" in legacy_seed.LEGACY_ADJACENT_ROLE_DEFINITION
    assert "HARD REJECT" in legacy_seed.LEGACY_DEALBREAKERS


def test_seed_rules_pass_ops_validation():
    snapshot = ProfileSnapshot(ProfileFields(paused=True), rules=seed_rules())

    assert validate_snapshot(snapshot, None) == []


def test_seed_module_does_not_import_the_prefilter():
    tree = ast.parse(open(legacy_seed.__file__, encoding="utf-8").read())
    imported = [
        node.module if isinstance(node, ast.ImportFrom) else alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in (node.names if isinstance(node, ast.Import) else [None])
    ]
    assert not any("pre_filter" in (name or "") or "filters" in (name or "") for name in imported)
