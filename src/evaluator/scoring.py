"""Score post-processing as pure functions over a per-call ``ScoringPolicy``.

The policy carries everything that used to be frozen into the chain or read from ``.env``: the system
prompt, thresholds, bonus, the non-core cap, the title rules and the ids of remote-only queries. The same
``final_score`` re-buckets a stored row (from its raw score and flags) without another LLM call.

The constants live here (not in ``chain.py``) so that ``chain`` can import this module; ``chain`` re-exports
them under their old names.
"""
from dataclasses import dataclass
from typing import FrozenSet, Optional, Tuple

from src.filters.rules import CompiledRule

BONUS_POINTS = 1.5
BONUS_MIN_BASE_SCORE = 6.0
REJECT_CAP = 2.0
DIGEST_MAX_SCORE = 8.4  # no-profile default cap for non-core roles (WOW needs >= 8.5)


@dataclass(frozen=True)
class ScoringPolicy:
    system_prompt: str
    wow_threshold: float = 8.5
    digest_threshold: float = 6.5
    bonus_enabled: bool = False
    bonus_points: float = BONUS_POINTS
    core_cap: Optional[float] = DIGEST_MAX_SCORE
    rules: Optional[Tuple[CompiledRule, ...]] = None
    remote_filter_query_ids: FrozenSet[str] = frozenset()


def final_score(
    raw: float,
    *,
    hard_reject: bool,
    bonus_flag: bool,
    is_core_role: bool,
    title_reason: Optional[str],
    policy: ScoringPolicy,
) -> float:
    """The stored score for a model score and its flags.

    The non-core cap is applied after rounding and is not itself rounded, so a cap such as ``8.15`` never
    falls below an equal digest threshold.
    """
    score = min(10.0, max(1.0, float(raw)))
    if hard_reject or score <= REJECT_CAP:
        score = min(score, REJECT_CAP)
    elif bonus_flag and score >= BONUS_MIN_BASE_SCORE and policy.bonus_enabled:
        score = min(10.0, score + policy.bonus_points)
    if title_reason and score > REJECT_CAP:
        score = REJECT_CAP
    score = round(score, 1)
    if policy.core_cap is not None and not is_core_role and score > policy.core_cap:
        score = policy.core_cap
    return score


def decide_wow(score: float, *, hard_reject: bool, is_core_role: bool, policy: ScoringPolicy) -> bool:
    """WOW needs the threshold, no hard reject, and a core role whenever the cap is on."""
    return (
        score >= policy.wow_threshold
        and not hard_reject
        and (is_core_role or policy.core_cap is None)
    )


def remote_label(remote_status: Optional[str], found_with_remote_filter: bool) -> str:
    """The value of the job prompt's ``Remote:`` line."""
    if remote_status == "verified":
        return "yes"
    if found_with_remote_filter:
        return "unverified (found with the remote filter)"
    return "unverified"


def found_with_remote_filter(source_query: Optional[str], policy: ScoringPolicy) -> bool:
    """True when ``source_query`` (``<query id>|<board>``) names a query of a remote-only market."""
    if not source_query:
        return False
    return source_query.rsplit("|", 1)[0] in policy.remote_filter_query_ids
