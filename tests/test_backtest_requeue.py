"""Rule backtest, requeue of pre-filtered rows and threshold re-bucketing (no network, no LLM)."""
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock

import pytest

from src.evaluator.scoring import ScoringPolicy
from src.filters.rules import compile_rule
from src.profile.backtest import (
    MIN_BACKTEST_JOBS,
    BacktestResult,
    backtest_rules,
    backtest_snapshots,
    render_backtest,
)
from src.profile.requeue import rebucket_evaluated, requeue_prefiltered
from src.profile.snapshot import ProfileFields, ProfileSnapshot, Rule
from src.storage.db import Database, compute_dedup_hash
from src.storage.models import JobPosting

LONG = " ".join(f"word{n}" for n in range(60))  # >= 150 characters and >= 30 words


def _rule(term: str, kind: str = "block"):
    return compile_rule(kind, term, "exact", "USER_RULE", "user")


@pytest.fixture
def db(tmp_path: Path) -> Database:
    database = Database(tmp_path / "bt.db")
    database.init_db()
    return database


def _job(n: int, title: Optional[str] = None, description: str = LONG, **kw) -> JobPosting:
    title = title if title is not None else f"Title {n}"
    return JobPosting(
        dedup_hash=compute_dedup_hash("Example Co", title, f"City {n}"),
        site="linkedin",
        job_url=f"https://example.test/{n}",
        title=title,
        company="Example Co",
        location=f"City {n}",
        description=description,
        **kw,
    )


def _set(db: Database, job_id: int, **columns) -> None:
    assignments = ", ".join(f"{k} = ?" for k in columns)
    with db._get_connection() as conn:
        conn.execute(f"UPDATE jobs SET {assignments} WHERE id = ?", (*columns.values(), job_id))
        conn.commit()


def _row(db: Database, job_id: int) -> JobPosting:
    return db.get_job_by_id(job_id)


# --- backtest ---------------------------------------------------------------------------------


def _titles(total: int, marked: int = 3):
    titles = [f"Plain role {n}" for n in range(total)]
    for n in range(marked):
        titles[n * 7] = f"<TERM> position {n}"
    return titles


def test_backtest_not_meaningful_under_200():
    small = backtest_rules(_titles(MIN_BACKTEST_JOBS - 1), (), (_rule("<TERM>"),))
    assert small.meaningful is False
    assert (small.newly_dropped, small.newly_passing) == (0, 0)
    assert small.dropped_samples == () and small.passing_samples == ()
    assert render_backtest(small) == ["backtest not meaningful: only 199 stored jobs (needs 200)"]

    enough = backtest_rules(_titles(MIN_BACKTEST_JOBS), (), (_rule("<TERM>"),))
    assert enough.meaningful is True


def test_backtest_counts_and_samples():
    titles = _titles(200)
    added = backtest_rules(titles, (), (_rule("<TERM>"),))
    assert (added.total_jobs, added.newly_dropped, added.newly_passing) == (200, 3, 0)
    assert set(added.dropped_samples) == {f"<TERM> position {n}" for n in range(3)}

    removed = backtest_rules(titles, (_rule("<TERM>"),), ())
    assert (removed.newly_dropped, removed.newly_passing) == (0, 3)
    assert set(removed.passing_samples) == {f"<TERM> position {n}" for n in range(3)}

    lines = render_backtest(added)
    assert lines[0] == "backtest: 3 of 200 stored jobs would now be dropped, 0 would now pass"
    assert all(line.startswith("  - ") for line in lines[1:]) and len(lines) == 4


def test_backtest_sample_limit():
    result = backtest_rules(_titles(200, marked=20), (), (_rule("<TERM>"),), sample=5)
    assert result.newly_dropped == 20 and len(result.dropped_samples) == 5


def test_backtest_none_when_rules_unchanged(db):
    snap = ProfileSnapshot(ProfileFields(), rules=(Rule("block", "<TERM>", "exact", "USER_RULE", "user"),))
    assert backtest_snapshots(db, snap, snap) is None


def test_backtest_snapshots_uses_stored_titles(db):
    for n, title in enumerate(_titles(200)):
        db.save_job(_job(n, title=title))
    old = ProfileSnapshot(ProfileFields())
    new = ProfileSnapshot(ProfileFields(), rules=(Rule("block", "<TERM>", "exact", "USER_RULE", "user"),))

    result = backtest_snapshots(db, old, new)

    assert isinstance(result, BacktestResult)
    assert (result.meaningful, result.total_jobs, result.newly_dropped) == (True, 200, 3)


def test_backtest_snapshots_without_old_profile_compares_to_the_legacy_ruleset(db):
    db.save_job(_job(1))
    new = ProfileSnapshot(ProfileFields(), rules=(Rule("block", "<TERM>", "exact", "USER_RULE", "user"),))
    result = backtest_snapshots(db, None, new)
    assert result is not None and result.meaningful is False


def test_get_titles_for_backtest_newest_first_and_non_empty(db):
    db.save_job(_job(1, title="First"))
    db.save_job(_job(2, title="Second"))
    db.save_job(_job(3, title=""))
    assert db.get_titles_for_backtest() == ["Second", "First"]
    assert db.get_titles_for_backtest(limit=1) == ["Second"]


# --- requeue ----------------------------------------------------------------------------------


def _dropped(db: Database, n: int, title: str, description: str = LONG) -> int:
    job_id, _ = db.save_job(
        _job(
            n, title=title, description=description, pre_filtered=True,
            pre_filter_reason="USER_RULE: 'x'", pre_filter_rule="block:exact:<term>",
            eval_status="skipped", notification_status="dismissed",
        )
    )
    return job_id


def test_requeue_revives_rows_that_now_pass(db):
    job_id = _dropped(db, 1, "<TERM> coach")
    stored_hash = _row(db, job_id).dedup_hash

    result = requeue_prefiltered(db, ())  # the rule was removed

    assert (result.checked, result.requeued, result.still_dropped, result.too_old) == (1, 1, 0, 0)
    assert result.samples == ("<TERM> coach",)
    row = _row(db, job_id)
    assert (row.pre_filtered, row.pre_filter_reason, row.pre_filter_rule) == (False, None, None)
    assert (row.eval_status, row.notification_status, row.eval_attempts) == ("unprocessed", "pending", 0)
    assert row.dedup_hash == stored_hash
    assert [j.id for j in db.get_unprocessed_jobs()] == [job_id]


def test_requeue_keeps_rows_a_rule_still_drops(db):
    job_id = _dropped(db, 1, "<TERM> coach")
    result = requeue_prefiltered(db, (_rule("<TERM>"),))
    assert (result.requeued, result.still_dropped) == (0, 1)
    assert _row(db, job_id).pre_filtered is True


def test_requeue_keeps_stub_descriptions_dropped(db):
    job_id = _dropped(db, 1, "Fine title", description="too short")
    result = requeue_prefiltered(db, ())
    assert (result.requeued, result.still_dropped) == (0, 1)
    assert _row(db, job_id).pre_filtered is True


def test_requeue_dry_run_writes_nothing(db):
    job_id = _dropped(db, 1, "<TERM> coach")
    result = requeue_prefiltered(db, (), dry_run=True)
    assert result.requeued == 1
    row = _row(db, job_id)
    assert row.pre_filtered is True and row.pre_filter_rule == "block:exact:<term>"
    assert db.get_unprocessed_jobs() == []


def test_requeue_skips_rows_older_than_stale_window(db):
    old_id = _dropped(db, 1, "<TERM> old")
    fresh_id = _dropped(db, 2, "<TERM> fresh")
    _set(db, old_id, created_at="2000-01-01 00:00:00")
    with db._get_connection() as conn:
        conn.execute("UPDATE jobs SET created_at = datetime('now', '-8 days') WHERE id = ?", (old_id,))
        conn.commit()

    result = requeue_prefiltered(db, ())

    assert (result.requeued, result.too_old) == (1, 1)
    assert _row(db, old_id).pre_filtered is True
    assert _row(db, fresh_id).pre_filtered is False


def test_requeue_jobs_only_touches_prefiltered_rows(db):
    dropped = _dropped(db, 1, "<TERM> coach")
    live_id, _ = db.save_job(_job(2, eval_status="evaluated", notification_status="digest_sent"))
    assert db.requeue_jobs([dropped, live_id]) == 1
    assert _row(db, live_id).notification_status == "digest_sent"
    assert db.requeue_jobs([]) == 0


# --- re-bucket --------------------------------------------------------------------------------


def _policy(**kw) -> ScoringPolicy:
    values = dict(
        system_prompt="x", wow_threshold=8.5, digest_threshold=6.5, bonus_enabled=False,
        bonus_points=1.5, core_cap=8.4, rules=(),
    )
    values.update(kw)
    return ScoringPolicy(**values)


def _evaluated(
    db: Database, n: int, *, raw: Optional[float] = 7.0, score: float = 7.0, core: Optional[bool] = True,
    bonus: Optional[bool] = False, reject: Optional[bool] = False, status: str = "pending", title: Optional[str] = None,
) -> int:
    job_id, _ = db.save_job(
        _job(n, title=title, eval_status="evaluated", eval_score=score, notification_status=status)
    )
    as_int = lambda v: None if v is None else int(v)  # noqa: E731
    _set(  # save_job does not store the raw evaluation fields
        db, job_id, eval_raw_score=raw, eval_is_core_role=as_int(core), eval_bonus_flag=as_int(bonus),
        eval_hard_reject=as_int(reject),
    )
    return job_id


def test_rebucket_recomputes_under_new_thresholds(db):
    job_id = _evaluated(db, 1, raw=7.0, score=7.0)

    result = rebucket_evaluated(db, _policy(digest_threshold=7.5))
    assert (result.checked, result.changed, result.to_dismissed, result.to_digest) == (1, 1, 1, 0)
    row = _row(db, job_id)
    assert (row.notification_status, row.eval_score) == ("dismissed", 7.0)

    result = rebucket_evaluated(db, _policy(digest_threshold=6.5))
    assert (result.changed, result.to_digest) == (1, 1)
    assert _row(db, job_id).notification_status == "pending"

    again = rebucket_evaluated(db, _policy(digest_threshold=6.5))
    assert again.changed == 0 and again.samples == ()


def test_rebucket_skips_rows_without_stored_fields(db):
    job_id = _evaluated(db, 1, raw=None, score=7.0)
    result = rebucket_evaluated(db, _policy(digest_threshold=9.0))
    assert (result.checked, result.skipped_missing, result.changed) == (1, 1, 0)
    assert _row(db, job_id).notification_status == "pending"
    for n, missing in enumerate(({"core": None}, {"bonus": None}, {"reject": None}), start=2):
        _evaluated(db, n, **missing)
    assert rebucket_evaluated(db, _policy(), dry_run=True).skipped_missing == 4


def test_rebucket_never_alerts_and_leaves_sent_rows(db):
    notifier = MagicMock()
    job_id = _evaluated(db, 1, raw=9.5, score=8.4)
    sent = {s: _evaluated(db, n, raw=9.5, score=8.4, status=s) for n, s in
            enumerate(("instant_sent", "digest_sent", "expired_dismissed"), start=2)}

    result = rebucket_evaluated(db, _policy(wow_threshold=9.0))

    assert result.changed == 1 and result.to_digest == 1
    row = _row(db, job_id)
    assert (row.eval_score, row.eval_is_wow, row.notification_status) == (9.5, False, "pending")
    for status, sid in sent.items():
        other = _row(db, sid)
        assert (other.notification_status, other.eval_score) == (status, 8.4)
    notifier.send_message.assert_not_called()


def test_rebucket_applies_bonus_and_cap_from_policy(db):
    bonus_id = _evaluated(db, 1, raw=7.0, score=7.0, bonus=True)
    capped_id = _evaluated(db, 2, raw=9.0, score=9.0, core=False)

    rebucket_evaluated(db, _policy(bonus_enabled=True, bonus_points=2.0, wow_threshold=8.5, core_cap=8.4))

    assert _row(db, bonus_id).eval_score == 9.0
    assert _row(db, capped_id).eval_score == 8.4

    rebucket_evaluated(db, _policy(wow_threshold=8.0, core_cap=7.9))
    assert _row(db, capped_id).eval_score == 7.9


def test_rebucket_hard_reject_stays_capped(db):
    job_id = _evaluated(db, 1, raw=8.0, score=8.0, reject=True)
    result = rebucket_evaluated(db, _policy())
    assert result.to_dismissed == 1
    row = _row(db, job_id)
    assert (row.eval_score, row.notification_status) == (2.0, "dismissed")


def test_rebucket_dry_run_writes_nothing(db):
    job_id = _evaluated(db, 1, raw=7.0, score=7.0)
    result = rebucket_evaluated(db, _policy(digest_threshold=7.5), dry_run=True)
    assert result.changed == 1 and result.samples
    assert _row(db, job_id).notification_status == "pending"


def test_rebucket_applies_a_title_rule_that_now_matches(db):
    job_id = _evaluated(db, 1, raw=8.0, score=8.0, title="<TERM> coach")
    rebucket_evaluated(db, _policy(rules=(_rule("<TERM>"),)))
    assert _row(db, job_id).eval_score == 2.0


def test_apply_rebucket_guards_status(db):
    sent_id = _evaluated(db, 1, status="digest_sent")
    pending_id = _evaluated(db, 2)
    assert db.apply_rebucket([(sent_id, 1.0, "dismissed"), (pending_id, 1.0, "dismissed")]) == 1
    assert _row(db, sent_id).notification_status == "digest_sent"
    assert _row(db, pending_id).notification_status == "dismissed"
    assert db.apply_rebucket([]) == 0
