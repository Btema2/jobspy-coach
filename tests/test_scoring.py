import dataclasses

import pytest

from src.evaluator.scoring import (
    BONUS_MIN_BASE_SCORE,
    DIGEST_MAX_SCORE,
    REJECT_CAP,
    ScoringPolicy,
    decide_wow,
    final_score,
    found_with_remote_filter,
    remote_label,
)

NO_PROFILE = ScoringPolicy(system_prompt="P", bonus_enabled=True)


def score(raw, *, hard_reject=False, bonus_flag=False, is_core_role=False, title_reason=None, policy=NO_PROFILE):
    return final_score(
        raw, hard_reject=hard_reject, bonus_flag=bonus_flag, is_core_role=is_core_role,
        title_reason=title_reason, policy=policy,
    )


def test_constants_keep_their_values():
    assert (BONUS_MIN_BASE_SCORE, REJECT_CAP, DIGEST_MAX_SCORE) == (6.0, 2.0, 8.4)


def test_policy_defaults_and_frozen():
    policy = ScoringPolicy(system_prompt="P")
    assert (policy.wow_threshold, policy.digest_threshold) == (8.5, 6.5)
    assert (policy.bonus_enabled, policy.bonus_points, policy.core_cap) == (False, 1.5, 8.4)
    assert policy.rules is None and policy.remote_filter_query_ids == frozenset()
    with pytest.raises(dataclasses.FrozenInstanceError):
        policy.wow_threshold = 9.0  # type: ignore[misc]


@pytest.mark.parametrize(
    "raw,kwargs,expected",
    [
        (14, dict(is_core_role=True), 10.0),
        (-3, {}, 1.0),
        (9.0, dict(), 8.4),
        (8.0, dict(bonus_flag=True), 8.4),
        (7.0, dict(), 7.0),
        (6.0, dict(bonus_flag=True), 7.5),
        (8.0, dict(bonus_flag=True, is_core_role=True), 9.5),
        (9.0, dict(is_core_role=True), 9.0),
        (9.0, dict(bonus_flag=True, is_core_role=True), 10.0),
        (5.0, dict(bonus_flag=True), 5.0),
        (5.9, dict(bonus_flag=True), 5.9),
        (8.0, dict(hard_reject=True, bonus_flag=True), 2.0),
        (9.0, dict(hard_reject=True, is_core_role=True, bonus_flag=True), 2.0),
        (2.0, dict(bonus_flag=True), 2.0),
        (1.5, dict(bonus_flag=True), 1.5),
        (7.26, {}, 7.3),
        (6.04, dict(bonus_flag=True), 7.5),
        (9.0, dict(title_reason="EXCLUDED_SENIORITY: 'x'"), 2.0),
        (9.0, dict(title_reason="EXCLUDED_SENIORITY: 'x'", bonus_flag=True), 2.0),
        (1.4, dict(title_reason="EXCLUDED_SENIORITY: 'x'"), 1.4),
    ],
)
def test_final_score_matches_legacy_postprocess(raw, kwargs, expected):
    assert score(raw, **kwargs) == expected


def test_cap_applied_after_rounding_keeps_digest():
    policy = ScoringPolicy(system_prompt="P", wow_threshold=8.25, digest_threshold=8.15, core_cap=8.15)
    result = score(9.0, policy=policy)
    assert result == 8.15
    assert result >= policy.digest_threshold
    assert decide_wow(result, hard_reject=False, is_core_role=False, policy=policy) is False


def test_bonus_disabled_policy_ignores_flag():
    policy = ScoringPolicy(system_prompt="P", bonus_enabled=False)
    assert score(7.0, bonus_flag=True, policy=policy) == 7.0


def test_bonus_points_from_policy():
    policy = ScoringPolicy(system_prompt="P", bonus_enabled=True, bonus_points=2.0)
    assert score(6.5, bonus_flag=True, policy=policy) == 8.4  # capped: not a core role
    assert score(6.5, bonus_flag=True, is_core_role=True, policy=policy) == 8.5


def test_cap_off_allows_wow_without_core_flag():
    policy = ScoringPolicy(system_prompt="P", core_cap=None)
    result = score(9.0, policy=policy)
    assert result == 9.0
    assert decide_wow(result, hard_reject=False, is_core_role=False, policy=policy) is True


def test_decide_wow_needs_core_role_when_capped():
    assert decide_wow(9.0, hard_reject=False, is_core_role=False, policy=NO_PROFILE) is False
    assert decide_wow(9.0, hard_reject=False, is_core_role=True, policy=NO_PROFILE) is True
    assert decide_wow(9.0, hard_reject=True, is_core_role=True, policy=NO_PROFILE) is False
    assert decide_wow(8.4, hard_reject=False, is_core_role=True, policy=NO_PROFILE) is False
    assert decide_wow(8.5, hard_reject=False, is_core_role=True, policy=NO_PROFILE) is True


def test_remote_label_values():
    assert remote_label("verified", False) == "yes"
    assert remote_label("verified", True) == "yes"
    assert remote_label("unverified", True) == "unverified (found with the remote filter)"
    assert remote_label("unverified", False) == "unverified"
    assert remote_label(None, False) == "unverified"


def test_found_with_remote_filter_uses_query_id_of_source_query():
    policy = ScoringPolicy(system_prompt="P", remote_filter_query_ids=frozenset({"role=<R>;term=<T>;market=<M>"}))
    assert found_with_remote_filter("role=<R>;term=<T>;market=<M>|linkedin", policy) is True
    assert found_with_remote_filter("role=<R>;term=<T>;market=<M>|indeed", policy) is True
    assert found_with_remote_filter("other|linkedin", policy) is False
    assert found_with_remote_filter(None, policy) is False
