"""Revive pre-filtered rows and re-bucket evaluated rows after a rule or threshold change (no LLM call)."""
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from src.evaluator.scoring import ScoringPolicy, final_score
from src.filters.pre_filter import pre_filter_job
from src.filters.rules import CompiledRule, match_title
from src.storage.db import STALE_PENDING_DAYS, Database

SAMPLE_LIMIT = 5


@dataclass(frozen=True)
class RequeueResult:
    checked: int
    requeued: int
    still_dropped: int
    too_old: int
    samples: Tuple[str, ...]


@dataclass(frozen=True)
class RebucketResult:
    checked: int
    changed: int
    skipped_missing: int
    to_digest: int
    to_dismissed: int
    samples: Tuple[str, ...]
    too_old: int = 0


def requeue_prefiltered(
    db: Database,
    rules: Sequence[CompiledRule],
    *,
    dry_run: bool = False,
    max_age_days: int = STALE_PENDING_DAYS,
) -> RequeueResult:
    """Send pre-filtered rows that pass ``rules`` back to ``unprocessed``.

    Rows older than ``max_age_days`` are only counted: they would expire before any digest, after the
    evaluation had already been paid for. Stub descriptions stay dropped.
    """
    rows = db.get_prefiltered_for_requeue(max_age_days)
    too_old = db.count_prefiltered_older_than(max_age_days)
    revive: List[int] = []
    samples: List[str] = []
    for job in rows:
        if pre_filter_job(job.title, job.description, rules).passed:
            revive.append(job.id)
            if len(samples) < SAMPLE_LIMIT:
                samples.append(job.title)
    requeued = len(revive) if dry_run else db.requeue_jobs(revive)
    return RequeueResult(len(rows), requeued, len(rows) - len(revive), too_old, tuple(samples))


def rebucket_evaluated(
    db: Database,
    policy: ScoringPolicy,
    *,
    dry_run: bool = False,
    max_age_days: int = STALE_PENDING_DAYS,
) -> RebucketResult:
    """Recompute the final score and bucket of evaluated rows from their stored model output.

    Only ``pending`` and ``dismissed`` rows are looked at; a row that now reaches the WOW level simply joins
    the digest (no instant alert is ever sent from here). Rows without the stored fields are skipped, and rows
    older than ``max_age_days`` are only counted (``too_old``): a revived one would expire at the next digest.
    """
    rows = db.get_rebucket_candidates(max_age_days)
    too_old = db.count_rebucket_older_than(max_age_days)
    changes: List[Tuple[int, float, str]] = []
    samples: List[str] = []
    skipped = to_digest = to_dismissed = 0
    for job in rows:
        if None in (job.eval_raw_score, job.eval_is_core_role, job.eval_bonus_flag, job.eval_hard_reject):
            skipped += 1
            continue
        hit = match_title(job.title, policy.rules)
        score = final_score(
            job.eval_raw_score,
            hard_reject=job.eval_hard_reject,
            bonus_flag=job.eval_bonus_flag,
            is_core_role=job.eval_is_core_role,
            title_reason=hit.reason if hit else None,
            policy=policy,
        )
        status = "pending" if score >= policy.digest_threshold else "dismissed"
        if score == job.eval_score and status == job.notification_status:
            continue
        changes.append((job.id, score, status))
        if status == "pending":
            to_digest += 1
        else:
            to_dismissed += 1
        if len(samples) < SAMPLE_LIMIT:
            samples.append(job.title)
    changed = len(changes) if dry_run else db.apply_rebucket(changes)
    return RebucketResult(len(rows), changed, skipped, to_digest, to_dismissed, tuple(samples), too_old)
