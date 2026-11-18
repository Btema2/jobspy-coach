import ast
import re
from pathlib import Path

from src.filters import pre_filter
from src.profile import legacy_seed
from src.profile.legacy_seed import SEED_RULES, seed_rules
from src.profile.ops import validate_snapshot
from src.profile.snapshot import (
    SNAPSHOT_SCHEMA,
    ProfileFields,
    ProfileSnapshot,
    snapshot_from_dict,
    snapshot_to_dict,
)

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


def test_v1_definitions_are_substrings_of_the_frozen_v1_prompt():
    prompt = (Path(__file__).parent / "fixtures" / "system_prompt_generic.txt").read_text(encoding="utf-8")

    for text in (
        legacy_seed.LEGACY_V1_CORE_ROLE_DEFINITION,
        legacy_seed.LEGACY_V1_ADJACENT_ROLE_DEFINITION,
        legacy_seed.LEGACY_V1_DEALBREAKERS,
    ):
        assert text.strip() and text in prompt
    assert "role_is_direct_coaching" in legacy_seed.LEGACY_V1_CORE_ROLE_DEFINITION
    assert "DIGEST" in legacy_seed.LEGACY_V1_ADJACENT_ROLE_DEFINITION
    assert "HARD REJECT" in legacy_seed.LEGACY_V1_DEALBREAKERS


def test_new_seed_bodies_have_no_band_numbers_or_old_key():
    for text in (
        legacy_seed.LEGACY_CORE_ROLE_DEFINITION,
        legacy_seed.LEGACY_ADJACENT_ROLE_DEFINITION,
        legacy_seed.LEGACY_DEALBREAKERS,
    ):
        assert text.strip()
        for banned in ("8.5", "6.5", "role_is_direct_coaching", "- WOW", "- DIGEST", "- HARD REJECT", "hard_reject"):
            assert banned not in text, banned
    assert "core role" in legacy_seed.LEGACY_CORE_ROLE_DEFINITION


def test_new_seed_bodies_are_declarative_not_commands():
    """The data block is declared data, so its text must describe roles, never instruct the model."""
    bodies = (
        legacy_seed.LEGACY_CORE_ROLE_DEFINITION,
        legacy_seed.LEGACY_ADJACENT_ROLE_DEFINITION,
        legacy_seed.LEGACY_DEALBREAKERS,
    )
    for text in bodies:
        for command in ("Score as", "Set `", "Do not", "Never ", "Always ", "Ignore", "It is not a core role for"):
            assert command not in text, command
    assert "marginal matches" in legacy_seed.LEGACY_ADJACENT_ROLE_DEFINITION
    assert "are not core roles" in legacy_seed.LEGACY_CORE_ROLE_DEFINITION
    # The meaning is kept: the same role families are still named.
    assert "mixed HR / recruitment responsibilities" in legacy_seed.LEGACY_ADJACENT_ROLE_DEFINITION
    assert "administrative L&D focus" in legacy_seed.LEGACY_ADJACENT_ROLE_DEFINITION
    assert "corporate trainer" in legacy_seed.LEGACY_CORE_ROLE_DEFINITION


def test_text_upgrades_map_earlier_texts_to_new_per_field():
    assert legacy_seed.LEGACY_TEXT_UPGRADES == {
        legacy_seed.LEGACY_V1_CORE_ROLE_DEFINITION: legacy_seed.LEGACY_CORE_ROLE_DEFINITION,
        legacy_seed.LEGACY_V2_CORE_ROLE_DEFINITION: legacy_seed.LEGACY_CORE_ROLE_DEFINITION,
        legacy_seed.LEGACY_V1_ADJACENT_ROLE_DEFINITION: legacy_seed.LEGACY_ADJACENT_ROLE_DEFINITION,
        legacy_seed.LEGACY_V2_ADJACENT_ROLE_DEFINITION: legacy_seed.LEGACY_ADJACENT_ROLE_DEFINITION,
        legacy_seed.LEGACY_V1_DEALBREAKERS: legacy_seed.LEGACY_DEALBREAKERS,
    }
    assert legacy_seed.LEGACY_V2_CORE_ROLE_DEFINITION != legacy_seed.LEGACY_CORE_ROLE_DEFINITION
    assert legacy_seed.LEGACY_V2_ADJACENT_ROLE_DEFINITION != legacy_seed.LEGACY_ADJACENT_ROLE_DEFINITION


def test_snapshot_schema_2_upgrades_the_first_branch_3_seed_text_only():
    fields = ProfileFields(
        core_role_definition=legacy_seed.LEGACY_V2_CORE_ROLE_DEFINITION,
        adjacent_role_definition=legacy_seed.LEGACY_V2_ADJACENT_ROLE_DEFINITION + " Edited.",
        dealbreakers=legacy_seed.LEGACY_DEALBREAKERS,
    )
    stored = snapshot_to_dict(ProfileSnapshot(fields))
    assert stored["snapshot_schema"] == 2

    loaded = snapshot_from_dict(stored).fields

    assert loaded.core_role_definition == legacy_seed.LEGACY_CORE_ROLE_DEFINITION
    assert loaded.adjacent_role_definition == legacy_seed.LEGACY_V2_ADJACENT_ROLE_DEFINITION + " Edited."
    assert loaded.dealbreakers == legacy_seed.LEGACY_DEALBREAKERS


def test_snapshot_schema_1_upgrades_seed_text():
    fields = ProfileFields(
        core_role_definition=legacy_seed.LEGACY_V1_CORE_ROLE_DEFINITION,
        adjacent_role_definition=legacy_seed.LEGACY_V1_ADJACENT_ROLE_DEFINITION + " Edited.",
        dealbreakers=legacy_seed.LEGACY_V1_DEALBREAKERS,
        persona=legacy_seed.LEGACY_V1_DEALBREAKERS,  # a non-definition field is never rewritten
    )
    stored = snapshot_to_dict(ProfileSnapshot(fields))
    stored["snapshot_schema"] = 1

    loaded = snapshot_from_dict(stored).fields

    assert SNAPSHOT_SCHEMA == 2
    assert loaded.core_role_definition == legacy_seed.LEGACY_CORE_ROLE_DEFINITION
    assert loaded.dealbreakers == legacy_seed.LEGACY_DEALBREAKERS
    assert loaded.adjacent_role_definition == legacy_seed.LEGACY_V1_ADJACENT_ROLE_DEFINITION + " Edited."
    assert loaded.persona == legacy_seed.LEGACY_V1_DEALBREAKERS
    assert stored["profile"]["core_role_definition"] == legacy_seed.LEGACY_V1_CORE_ROLE_DEFINITION  # input intact


def test_snapshot_schema_2_with_current_or_edited_text_is_untouched():
    for text in (legacy_seed.LEGACY_CORE_ROLE_DEFINITION, legacy_seed.LEGACY_V1_CORE_ROLE_DEFINITION + " Mine."):
        stored = snapshot_to_dict(ProfileSnapshot(ProfileFields(core_role_definition=text)))
        assert stored["snapshot_schema"] == 2
        assert snapshot_from_dict(stored).fields.core_role_definition == text


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
