from dataclasses import replace
from pathlib import Path

import pytest

from src.profile import ops
from src.profile.errors import NothingToUndo, ProfileValidationError, VersionConflict
from src.profile.legacy_seed import SEED_RULES, seed_rules
from src.profile.ops import (
    AddMarket,
    AddPinnedQuery,
    AddRole,
    AddRoleTerm,
    AddRule,
    RemoveMarket,
    RemovePinnedQuery,
    RemoveRole,
    RemoveRoleTerm,
    RemoveRule,
    ReplaceProfile,
    SetEnabled,
    SetField,
    apply_changes,
    apply_operation,
    create_starter_profile,
    diff_snapshots,
    history,
    restore,
    undo,
    validate_snapshot,
)
from src.profile.snapshot import (
    Market,
    PinnedQuery,
    ProfileFields,
    ProfileSnapshot,
    Role,
    RoleTerm,
    Rule,
)
from src.profile.store import ProfileStore
from src.storage.db import Database


@pytest.fixture
def store(tmp_path: Path) -> ProfileStore:
    db = Database(tmp_path / "profile.db")
    db.init_db()
    return ProfileStore(db)


@pytest.fixture
def started(store: ProfileStore) -> ProfileStore:
    create_starter_profile(store, wow_threshold=8.5, digest_threshold=6.5)
    return store


def _market(**kw) -> Market:
    base = dict(name="<MARKET>", location="<LOCATION>", countries=("usa",), remote=True)
    base.update(kw)
    return Market(**base)


def _pinned(**kw) -> PinnedQuery:
    base = dict(query_id="Q-1", term="<TERM>", location="<LOCATION>", boards=("linkedin",))
    base.update(kw)
    return PinnedQuery(**base)


def _snap(**kw) -> ProfileSnapshot:
    kw.setdefault("fields", ProfileFields(paused=True))
    return ProfileSnapshot(**kw)


def _messages(exc_info) -> str:
    return " | ".join(exc_info.value.messages)


# --- versions, diff, apply_changes -------------------------------------------------------------


def test_add_role_bumps_version_snapshots_and_diffs(started: ProfileStore):
    result = apply_changes(started, [AddRole("<ROLE>")], label="add role")

    assert result.version == 2
    assert result.diff == ["+ role <ROLE>"]
    assert [v.version for v in history(started)] == [2, 1]
    assert history(started)[0].label == "add role"
    snap = started.read_snapshot()
    assert snap is not None and [r.name for r in snap.roles] == ["<ROLE>"]


def test_create_starter_profile_is_paused_and_empty(store: ProfileStore):
    result = create_starter_profile(store, wow_threshold=9.0, digest_threshold=7.0)

    assert result.version == 1
    snap = store.read_snapshot()
    assert snap == ProfileSnapshot(
        ProfileFields(paused=True, wow_threshold=9.0, digest_threshold=7.0), version=1
    )
    assert history(store)[0].label == "init starter profile"


def test_stale_expected_version_conflicts(started: ProfileStore):
    with pytest.raises(VersionConflict):
        apply_changes(started, [AddRole("<ROLE>")], label="x", expected_version=0)

    snap = started.read_snapshot()
    assert snap is not None and snap.version == 1 and snap.roles == ()


def test_matching_expected_version_is_accepted(started: ProfileStore):
    assert apply_changes(started, [AddRole("<ROLE>")], label="x", expected_version=1).version == 2


@pytest.mark.parametrize(
    "wow,digest,ok",
    [
        (8.5, 6.5, True),
        (8.5, 8.4, True),
        (8.5, 8.45, False),
        (8.5, 3.0, False),
        (8.5, 3.01, True),
        (10.0, 6.5, True),
        (10.01, 6.5, False),
    ],
)
def test_threshold_invariants(started: ProfileStore, wow, digest, ok):
    ops_ = [SetField("wow_threshold", wow), SetField("digest_threshold", digest)]
    if ok:
        assert apply_changes(started, ops_, label="t").version == 2
    else:
        with pytest.raises(ProfileValidationError):
            apply_changes(started, ops_, label="t")
        assert started.read_snapshot().version == 1  # type: ignore[union-attr]


def test_call_cap_rejected_and_nothing_written(started: ProfileStore):
    apply_changes(started, [SetField("max_calls_per_cycle", 4)], label="cap")

    with pytest.raises(ProfileValidationError) as exc:
        apply_changes(started, [AddMarket(_market()), AddRole("<ROLE>")], label="grow")

    text = _messages(exc)
    assert "5" in text and "4" in text
    snap = started.read_snapshot()
    assert snap is not None and snap.version == 2 and snap.roles == () and snap.markets == ()


def test_max_calls_must_be_positive(started: ProfileStore):
    with pytest.raises(ProfileValidationError):
        apply_changes(started, [SetField("max_calls_per_cycle", 0)], label="x")


def test_unpause_refused_without_queries(started: ProfileStore):
    with pytest.raises(ProfileValidationError) as exc:
        apply_changes(started, [SetField("paused", False)], label="resume")

    assert "no enabled search quer" in _messages(exc)
    snap = started.read_snapshot()
    assert snap is not None and snap.fields.paused is True


def test_unpause_allowed_with_a_query(started: ProfileStore):
    result = apply_changes(
        started, [AddPinnedQuery(_pinned()), SetField("paused", False)], label="resume"
    )

    assert result.version == 2
    snap = started.read_snapshot()
    assert snap is not None and snap.fields.paused is False


def test_unpause_refused_when_only_disabled_queries(started: ProfileStore):
    with pytest.raises(ProfileValidationError):
        apply_changes(
            started,
            [AddPinnedQuery(_pinned(enabled=False)), SetField("paused", False)],
            label="resume",
        )


def test_removing_last_role_while_unpaused_is_allowed(started: ProfileStore):
    apply_changes(
        started, [AddRole("<ROLE>"), AddMarket(_market()), SetField("paused", False)], label="go"
    )

    result = apply_changes(started, [RemoveRole("<ROLE>")], label="drop")

    assert result.diff == ["- role <ROLE>"]
    snap = started.read_snapshot()
    assert snap is not None and snap.fields.paused is False and snap.roles == ()


def test_warnings_are_the_compiled_warnings(started: ProfileStore):
    result = apply_changes(
        started, [AddRole("<ROLE>"), AddMarket(_market(countries=()))], label="x"
    )

    assert any("no country" in w for w in result.warnings)


def test_apply_changes_on_absent_profile_diffs_against_nothing(store: ProfileStore):
    result = apply_changes(
        store, [SetField("paused", True), AddRole("<ROLE>")], label="first", expected_version=0
    )

    assert result.version == 1
    assert result.diff == ["paused: False -> True", "+ role <ROLE>"]


# --- undo / restore ----------------------------------------------------------------------------


def test_undo_and_undo_of_undo_toggle(started: ProfileStore):
    state_a = started.read_snapshot()
    apply_changes(started, [AddRole("<ROLE>")], label="add role")
    state_b = started.read_snapshot()
    assert state_a is not None and state_b is not None

    first = undo(started)
    assert first.version == 3 and first.diff == ["- role <ROLE>"]
    assert replace(started.read_snapshot(), version=1) == state_a  # type: ignore[type-var]
    assert history(started)[0].label == "undo (back to v1)"

    second = undo(started)
    assert second.version == 4 and second.diff == ["+ role <ROLE>"]
    assert replace(started.read_snapshot(), version=2) == state_b  # type: ignore[type-var]
    assert history(started)[0].label == "undo (back to v2)"


def test_nothing_to_undo_on_first_version(started: ProfileStore):
    with pytest.raises(NothingToUndo):
        undo(started)


def test_nothing_to_undo_without_a_profile(store: ProfileStore):
    with pytest.raises(NothingToUndo):
        undo(store)


def test_restore_any_of_last_10(started: ProfileStore):
    for i in range(1, 5):
        apply_changes(started, [SetField("persona", f"<P{i}>")], label=f"p{i}")

    result = restore(started, 3)

    assert result.version == 6
    snap = started.read_snapshot()
    assert snap is not None and snap.fields.persona == "<P2>"
    assert history(started)[0].label == "restore v3"


def test_restore_pruned_version_fails(started: ProfileStore):
    for i in range(12):
        apply_changes(started, [SetField("persona", f"<P{i}>")], label=f"p{i}")

    with pytest.raises(ProfileValidationError) as exc:
        restore(started, 1)

    assert "version 1 is not among the last 10" in _messages(exc)
    assert started.read_snapshot().version == 13  # type: ignore[union-attr]


def test_restore_validates_like_any_change(started: ProfileStore):
    apply_changes(
        started, [AddRole("<ROLE>"), AddMarket(_market()), SetField("paused", False)], label="go"
    )
    apply_changes(started, [RemoveRole("<ROLE>"), SetField("paused", True)], label="pause")
    # v3 is paused and empty; restoring v2 (unpaused, with queries) is valid again.
    assert restore(started, 2).version == 4
    snap = started.read_snapshot()
    assert snap is not None and snap.fields.paused is False and len(snap.roles) == 1


# --- validation --------------------------------------------------------------------------------


def test_persona_over_8000_rejected(started: ProfileStore):
    assert apply_changes(started, [SetField("persona", "x" * 8000)], label="ok").version == 2
    with pytest.raises(ProfileValidationError) as exc:
        apply_changes(started, [SetField("persona", "x" * 8001)], label="too long")
    assert "8000" in _messages(exc)


@pytest.mark.parametrize(
    "name", ["core_role_definition", "adjacent_role_definition", "dealbreakers", "bonus_criteria"]
)
def test_every_text_field_is_capped(started: ProfileStore, name):
    with pytest.raises(ProfileValidationError):
        apply_changes(started, [SetField(name, "x" * 8001)], label="x")


@pytest.mark.parametrize("points,ok", [(0, True), (5, True), (2.5, True), (-0.1, False), (5.1, False)])
def test_bonus_points_range(started: ProfileStore, points, ok):
    if ok:
        apply_changes(started, [SetField("bonus_points", points)], label="x")
    else:
        with pytest.raises(ProfileValidationError):
            apply_changes(started, [SetField("bonus_points", points)], label="x")


@pytest.mark.parametrize("value,ok", [(None, True), ("en", True), ("uk", True), ("fil", True), ("EN", False), ("english", False), ("e", False), ("e1", False), ("", False)])
def test_output_language_is_a_language_code_or_unset(started: ProfileStore, value, ok):
    if ok:
        apply_changes(started, [SetField("output_language", value)], label="x")
    else:
        with pytest.raises(ProfileValidationError):
            apply_changes(started, [SetField("output_language", value)], label="x")


def test_set_field_unknown_name_and_wrong_type(started: ProfileStore):
    with pytest.raises(ProfileValidationError):
        apply_changes(started, [SetField("persnoa", "x")], label="x")
    with pytest.raises(ProfileValidationError):
        apply_changes(started, [SetField("paused", "yes")], label="x")
    with pytest.raises(ProfileValidationError):
        apply_changes(started, [SetField("wow_threshold", True)], label="x")
    with pytest.raises(ProfileValidationError):
        apply_changes(started, [SetField("max_calls_per_cycle", 2.5)], label="x")


def test_set_field_int_becomes_float(started: ProfileStore):
    apply_changes(started, [SetField("wow_threshold", 9)], label="x")
    snap = started.read_snapshot()
    assert snap is not None and snap.fields.wow_threshold == 9.0
    assert isinstance(snap.fields.wow_threshold, float)


@pytest.mark.parametrize("term", ["ab", "x" * 61, "  ab  "])
def test_rule_term_length_rejected(started: ProfileStore, term):
    with pytest.raises(ProfileValidationError):
        apply_changes(started, [AddRule(Rule("block", term, "exact", "r", "user"))], label="x")


@pytest.mark.parametrize("n", [3, 60])
def test_rule_term_length_edges_accepted(started: ProfileStore, n):
    apply_changes(started, [AddRule(Rule("block", "x" * n, "prefix", "r", "user"))], label="x")


def test_rule_term_validation(started: ProfileStore):
    with pytest.raises(ProfileValidationError):
        apply_changes(
            started, [AddRule(Rule("block", "(a|b)", "pattern", "r", "user"))], label="x"
        )
    snapshot = _snap(
        rules=tuple(Rule("block", f"term{i:04d}", "exact", "r", "user") for i in range(501))
    )
    with pytest.raises(ProfileValidationError) as exc:
        apply_changes(started, [ReplaceProfile(snapshot)], label="x")
    assert "500" in _messages(exc)
    ok = _snap(rules=tuple(Rule("block", f"term{i:04d}", "exact", "r", "user") for i in range(500)))
    assert apply_changes(started, [ReplaceProfile(ok)], label="x").version == 2


@pytest.mark.parametrize(
    "rule",
    [
        Rule("weird", "abc", "exact", "r", "user"),
        Rule("block", "abc", "fuzzy", "r", "user"),
        Rule("block", "abc", "exact", "r", "robot"),
    ],
)
def test_rule_kind_mode_source_validated(started: ProfileStore, rule):
    with pytest.raises(ProfileValidationError):
        apply_changes(started, [AddRule(rule)], label="x")


def test_duplicate_rule_rejected(started: ProfileStore):
    rule = Rule("block", "abc", "exact", "r", "user")
    apply_changes(started, [AddRule(rule)], label="x")
    with pytest.raises(ProfileValidationError):
        apply_changes(started, [AddRule(replace(rule, term="ABC"))], label="x")


def test_add_and_remove_rule(started: ProfileStore):
    rule = Rule("soft", "abc", "prefix", "r", "user")
    assert apply_changes(started, [AddRule(rule)], label="a").diff == ["+ rule soft abc"]
    assert apply_changes(started, [RemoveRule("soft", "ABC")], label="b").diff == ["- rule soft abc"]
    with pytest.raises(ProfileValidationError):
        apply_changes(started, [RemoveRule("soft", "abc")], label="c")


def test_pattern_rule_must_equal_seed_entry(started: ProfileStore):
    seed = seed_rules()[0]
    assert apply_changes(started, [AddRule(seed)], label="seed").version == 2

    edited = replace(seed_rules()[1], term=SEED_RULES[1].pattern + "|extra")
    with pytest.raises(ProfileValidationError) as exc:
        apply_changes(started, [AddRule(edited)], label="edited")
    assert "seed" in _messages(exc)

    other_reason = replace(seed_rules()[1], reason="OTHER")
    with pytest.raises(ProfileValidationError):
        apply_changes(started, [AddRule(other_reason)], label="reason")


def test_invalid_regex_in_a_seed_sourced_pattern_is_reported(started: ProfileStore):
    with pytest.raises(ProfileValidationError):
        apply_changes(
            started, [AddRule(Rule("block", "(unclosed", "pattern", "r", "seed"))], label="x"
        )


def test_role_and_market_names_validated(started: ProfileStore):
    for bad in ("", "   ", "x" * 81):
        with pytest.raises(ProfileValidationError):
            apply_changes(started, [AddRole(bad)], label="x")
        with pytest.raises(ProfileValidationError):
            apply_changes(started, [AddMarket(_market(name=bad))], label="x")
    apply_changes(started, [AddRole("x" * 80)], label="x")


def test_role_names_unique_case_insensitively(started: ProfileStore):
    apply_changes(started, [AddRole("<ROLE>")], label="x")
    with pytest.raises(ProfileValidationError):
        apply_changes(started, [AddRole("<role>")], label="x")
    apply_changes(started, [AddMarket(_market())], label="x")
    with pytest.raises(ProfileValidationError):
        apply_changes(started, [AddMarket(_market(name="<market>"))], label="x")


def test_role_term_length_language_and_duplicates(started: ProfileStore):
    with pytest.raises(ProfileValidationError):
        apply_changes(started, [AddRole("<ROLE>", (RoleTerm("a"),))], label="x")
    with pytest.raises(ProfileValidationError):
        apply_changes(started, [AddRole("<ROLE>", (RoleTerm("x" * 81),))], label="x")
    with pytest.raises(ProfileValidationError):
        apply_changes(started, [AddRole("<ROLE>", (RoleTerm("<T1>", "polish"),))], label="x")
    with pytest.raises(ProfileValidationError):
        apply_changes(
            started, [AddRole("<ROLE>", (RoleTerm("<T1>"), RoleTerm("<t1>")))], label="x"
        )
    apply_changes(started, [AddRole("<ROLE>", (RoleTerm("<T1>", "EN "),))], label="ok")
    snap = started.read_snapshot()
    assert snap is not None and snap.roles[0].terms == (RoleTerm("<T1>", "en"),)


def test_add_and_remove_role_term(started: ProfileStore):
    apply_changes(started, [AddRole("<ROLE>")], label="x")
    added = apply_changes(started, [AddRoleTerm("<role>", "<TERM>", "de")], label="t")
    assert added.diff == ["+ term <TERM> [de] (role <ROLE>)"]
    removed = apply_changes(started, [RemoveRoleTerm("<ROLE>", "<term>", "de")], label="t")
    assert removed.diff == ["- term <TERM> [de] (role <ROLE>)"]
    with pytest.raises(ProfileValidationError):
        apply_changes(started, [RemoveRoleTerm("<ROLE>", "<TERM>")], label="t")
    with pytest.raises(ProfileValidationError):
        apply_changes(started, [AddRoleTerm("<NOPE>", "<TERM>")], label="t")


def test_remove_role_term_without_language_matches_any_tag(started: ProfileStore):
    apply_changes(started, [AddRole("<ROLE>", (RoleTerm("<TERM>", "de"),))], label="x")

    result = apply_changes(started, [RemoveRoleTerm("<ROLE>", "<TERM>")], label="t")

    assert result.diff == ["- term <TERM> [de] (role <ROLE>)"]


def test_market_field_validation(started: ProfileStore):
    bad = [
        _market(remote=False, remote_only=True),
        _market(location="  "),
        _market(languages=("english",)),
        _market(languages=("e",)),
    ]
    for market in bad:
        with pytest.raises(ProfileValidationError):
            apply_changes(started, [AddMarket(market)], label="x")
    ok = _market(remote_only=True, languages=("EN ", "de"), countries=())
    apply_changes(started, [AddMarket(ok)], label="x")
    snap = started.read_snapshot()
    assert snap is not None and snap.markets[0].languages == ("en", "de")


def test_unknown_country_in_market_rejected(started: ProfileStore):
    with pytest.raises(ProfileValidationError) as exc:
        apply_changes(started, [AddMarket(_market(countries=("polnad",)))], label="x")
    assert "polnad" in _messages(exc)
    assert started.read_snapshot().markets == ()  # type: ignore[union-attr]


def test_market_countries_are_normalised(started: ProfileStore):
    apply_changes(started, [AddMarket(_market(countries=("USA ", " Poland")))], label="x")

    snap = started.read_snapshot()
    assert snap is not None and snap.markets[0].countries == ("usa", "poland")


def test_validate_snapshot_does_not_normalise(started: ProfileStore):
    problems = validate_snapshot(_snap(markets=(_market(countries=("USA ",)),)), None)
    assert any("USA" in p for p in problems)


def test_pinned_query_validation(started: ProfileStore):
    bad = [
        _pinned(query_id=""),
        _pinned(term=""),
        _pinned(boards=()),
        _pinned(boards=("nosuchboard",)),
        _pinned(country="polnad"),
        _pinned(country=""),
    ]
    for q in bad:
        with pytest.raises(ProfileValidationError):
            apply_changes(started, [AddPinnedQuery(q)], label="x")
    apply_changes(started, [AddPinnedQuery(_pinned())], label="ok")
    with pytest.raises(ProfileValidationError):
        apply_changes(started, [AddPinnedQuery(_pinned())], label="dup")


def test_disabled_pinned_query_still_validates_boards(started: ProfileStore):
    with pytest.raises(ProfileValidationError) as exc:
        apply_changes(
            started, [AddPinnedQuery(_pinned(boards=("nosuchboard",), enabled=False))], label="x"
        )
    assert "nosuchboard" in _messages(exc)


def test_pinned_country_and_boards_are_normalised(started: ProfileStore):
    q = _pinned(country=" POLAND ", boards=("LinkedIn ", "Indeed"))

    apply_changes(started, [AddPinnedQuery(q)], label="x")

    snap = started.read_snapshot()
    assert snap is not None
    assert snap.pinned_queries[0].country == "poland"
    assert snap.pinned_queries[0].boards == ("linkedin", "indeed")


def test_add_remove_and_toggle_items(started: ProfileStore):
    apply_changes(
        started, [AddRole("<ROLE>"), AddMarket(_market()), AddPinnedQuery(_pinned())], label="x"
    )
    assert apply_changes(started, [SetEnabled("role", "<role>", False)], label="a").diff == [
        "~ role <ROLE> disabled"
    ]
    assert apply_changes(started, [SetEnabled("market", "<MARKET>", False)], label="b").diff == [
        "~ market <MARKET> disabled"
    ]
    assert apply_changes(started, [SetEnabled("pinned_query", "Q-1", False)], label="c").diff == [
        "~ pinned Q-1 disabled"
    ]
    assert apply_changes(started, [SetEnabled("pinned_query", "q-1", True)], label="d").diff == [
        "~ pinned Q-1 enabled"
    ]
    assert apply_changes(started, [RemoveMarket("<market>")], label="e").diff == ["- market <MARKET>"]
    assert apply_changes(started, [RemovePinnedQuery("Q-1")], label="f").diff == ["- pinned Q-1"]


@pytest.mark.parametrize(
    "op",
    [
        RemoveRole("<NOPE>"),
        RemoveMarket("<NOPE>"),
        RemovePinnedQuery("<NOPE>"),
        SetEnabled("role", "<NOPE>", False),
        SetEnabled("market", "<NOPE>", False),
        SetEnabled("pinned_query", "<NOPE>", False),
        RemoveRule("block", "nope"),
    ],
)
def test_remove_unknown_target_is_validation_error(started: ProfileStore, op):
    with pytest.raises(ProfileValidationError):
        apply_changes(started, [op], label="x")
    assert started.read_snapshot().version == 1  # type: ignore[union-attr]


def test_remove_unknown_role_is_validation_error(started: ProfileStore):
    with pytest.raises(ProfileValidationError) as exc:
        apply_operation(_snap(), RemoveRole("<NOPE>"))
    assert "<NOPE>" in _messages(exc)


def test_apply_operation_is_pure(started: ProfileStore):
    base = started.read_snapshot()
    assert base is not None

    new = apply_operation(base, AddRole("<ROLE>"))

    assert base.roles == () and [r.name for r in new.roles] == ["<ROLE>"]


def test_replace_profile_keeps_current_version_number(started: ProfileStore):
    base = started.read_snapshot()
    assert base is not None
    new = apply_operation(base, ReplaceProfile(ProfileSnapshot(ProfileFields(persona="<P>"), version=99)))
    assert new.version == base.version and new.fields.persona == "<P>"


# --- diff --------------------------------------------------------------------------------------


def test_diff_lines_are_single_line():
    new_persona = "a\nb" * 46 + "\r\n" * 4
    old = _snap(fields=ProfileFields(persona="a" * 120))
    new = _snap(
        fields=ProfileFields(persona=new_persona),
        roles=(Role("two\nlines", terms=(RoleTerm("t\nx", "en"),)),),
        markets=(_market(name="m\nm"),),
        pinned_queries=(_pinned(query_id="q\nq"),),
        rules=(Rule("block", "r\nr", "exact", "r", "user"),),
    )

    lines = diff_snapshots(old, new)

    assert f"persona: 120 -> {len(new_persona)} chars" in lines
    assert len(lines) >= 6
    assert all("\n" not in line and "\r" not in line for line in lines)


def test_diff_line_forms():
    old = _snap(
        fields=ProfileFields(wow_threshold=8.5),
        roles=(Role("<A>"), Role("<B>")),
        markets=(_market(),),
    )
    new = _snap(
        fields=ProfileFields(wow_threshold=8.0, output_language="en"),
        roles=(Role("<A>", enabled=False), Role("<C>", terms=(RoleTerm("<T>", "de"), RoleTerm("<U>")))),
        markets=(_market(), _market(name="<M2>")),
        pinned_queries=(_pinned(),),
        rules=(Rule("block", "abc", "exact", "r", "user"),),
    )

    assert diff_snapshots(old, new) == [
        "output_language: (unset) -> en",
        "wow_threshold: 8.5 -> 8.0",
        "- role <B>",
        "~ role <A> disabled",
        "+ role <C>",
        "+ term <T> [de] (role <C>)",
        "+ term <U> (role <C>)",
        "+ market <M2>",
        "+ pinned Q-1",
        "+ rule block abc",
    ]


def test_diff_of_market_field_changes():
    old = _snap(markets=(_market(),))
    new = _snap(markets=(_market(location="<OTHER>", languages=("en",)),))

    assert diff_snapshots(old, new) == [
        "~ market <MARKET> location: <LOCATION> -> <OTHER>",
        "~ market <MARKET> languages: (none) -> en",
    ]


def test_diff_against_nothing_renders_every_item_as_added():
    new = _snap(
        fields=ProfileFields(),
        roles=(Role("<A>", terms=(RoleTerm("<T>"),)),),
        markets=(_market(),),
        pinned_queries=(_pinned(),),
        rules=(Rule("soft", "abc", "exact", "r", "user"),),
    )

    assert diff_snapshots(None, new) == [
        "+ role <A>",
        "+ term <T> (role <A>)",
        "+ market <MARKET>",
        "+ pinned Q-1",
        "+ rule soft abc",
    ]


def test_diff_of_identical_snapshots_is_empty():
    assert diff_snapshots(_snap(), _snap()) == []


def test_module_exposes_operation_union():
    names = {t.__name__ for t in ops.Operation.__args__}  # type: ignore[attr-defined]
    assert names == {
        "SetField", "AddRole", "RemoveRole", "AddRoleTerm", "RemoveRoleTerm", "AddMarket",
        "RemoveMarket", "AddPinnedQuery", "RemovePinnedQuery", "SetEnabled", "AddRule",
        "RemoveRule", "ReplaceProfile",
    }


# --- review fixes ------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["wow_threshold", "digest_threshold", "bonus_points"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_numbers_are_validation_errors(started: ProfileStore, name, value):
    with pytest.raises(ProfileValidationError):
        apply_changes(started, [SetField(name, value)], label="x")

    assert started.read_snapshot().version == 1  # type: ignore[union-attr]


def test_threshold_problems_reject_non_finite():
    assert ops.threshold_problems(float("nan"), 6.5)
    assert ops.threshold_problems(8.5, float("nan"))
    assert ops.threshold_problems(float("inf"), 6.5)


def test_rule_reason_is_capped(started: ProfileStore):
    ok = Rule("block", "abc", "exact", "r" * 60, "user")
    apply_changes(started, [AddRule(ok)], label="x")
    with pytest.raises(ProfileValidationError):
        apply_changes(started, [AddRule(Rule("block", "abd", "exact", "r" * 61, "user"))], label="x")


@pytest.mark.parametrize("term", ["a\nb", "ab\rc", "a\tb", "ab\x00c", "ab\x1bc"])
@pytest.mark.parametrize("mode", ["exact", "prefix"])
def test_rule_terms_reject_control_characters(started: ProfileStore, term, mode):
    with pytest.raises(ProfileValidationError):
        apply_changes(started, [AddRule(Rule("block", term, mode, "r", "user"))], label="x")


def test_pinned_query_ids_unique_case_insensitively(started: ProfileStore):
    apply_changes(started, [AddPinnedQuery(_pinned(query_id="Q-A"))], label="x")
    with pytest.raises(ProfileValidationError):
        apply_changes(started, [AddPinnedQuery(_pinned(query_id="q-a"))], label="x")


def test_restore_only_maps_a_missing_version_to_a_validation_error(started: ProfileStore, monkeypatch):
    def broken(*args, **kwargs):
        raise KeyError("unrelated")

    monkeypatch.setattr(ops, "apply_changes", broken)

    with pytest.raises(KeyError):
        restore(started, 1)


# --- country aliases accepted by python-jobspy -------------------------------------------------


def _jobspy_country_inputs():
    from jobspy.model import Country, country_aliases

    canonical = [c.value for c in Country]
    return canonical, dict(country_aliases)


def test_every_jobspy_country_alias_and_value_is_accepted_and_normalised(started: ProfileStore):
    canonical, aliases = _jobspy_country_inputs()
    from jobspy.model import Country

    spellings = list(canonical) + list(aliases)
    assert {"us", "united states", "united kingdom", "czechia", "turkey", "uae"} <= set(aliases)
    for spelling in spellings:
        for variant in (spelling, spelling.upper(), f" {spelling} "):
            expected = Country(spelling).value  # what the scraper / python-jobspy resolves it to
            apply_changes(
                started,
                [AddPinnedQuery(_pinned(query_id="Q-ALIAS", country=variant)),
                 AddMarket(_market(name="<M-ALIAS>", countries=(variant,)))],
                label="alias",
            )
            snap = started.read_snapshot()
            assert snap.pinned_queries[0].country == expected, variant
            assert snap.markets[0].countries == (expected,), variant
            apply_changes(started, [RemovePinnedQuery("Q-ALIAS"), RemoveMarket("<M-ALIAS>")], label="clean")


def test_alias_table_equals_the_installed_jobspy_aliases():
    from jobspy.model import country_aliases
    from src.profile.compiler import COUNTRY_ALIASES

    assert COUNTRY_ALIASES == country_aliases


def test_query_id_is_not_stripped_by_normalisation(started: ProfileStore):
    apply_changes(started, [AddPinnedQuery(_pinned(query_id=" Q-SP "))], label="x")

    snap = started.read_snapshot()
    assert snap is not None and snap.pinned_queries[0].query_id == " Q-SP "


def test_starter_profile_with_invalid_thresholds_falls_back_with_a_note(store: ProfileStore):
    result = create_starter_profile(store, wow_threshold=5.0, digest_threshold=9.0)

    snap = store.read_snapshot()
    assert snap is not None
    assert (snap.fields.wow_threshold, snap.fields.digest_threshold) == (8.5, 6.5)
    assert any("8.5" in w and "6.5" in w for w in result.warnings)


def test_target_level_must_be_known(started: ProfileStore):
    with pytest.raises(ProfileValidationError) as exc:
        apply_changes(started, [SetField("target_level", "boss")], label="x")
    assert "target_level" in _messages(exc)
    apply_changes(started, [SetField("target_level", "senior")], label="x")
    apply_changes(started, [SetField("target_level", None)], label="x")


def test_names_reject_semicolon_and_equals(started: ProfileStore):
    with pytest.raises(ProfileValidationError) as exc:
        apply_changes(started, [AddRole("a;b")], label="x")
    assert "a;b" in _messages(exc)
    apply_changes(started, [AddRole("<ROLE>")], label="x")
    with pytest.raises(ProfileValidationError) as exc:
        apply_changes(started, [AddRoleTerm("<ROLE>", "x=y")], label="x")
    assert "x=y" in _messages(exc)
    with pytest.raises(ProfileValidationError) as exc:
        apply_changes(started, [AddMarket(_market(name="m;n"))], label="x")
    assert "m;n" in _messages(exc)
