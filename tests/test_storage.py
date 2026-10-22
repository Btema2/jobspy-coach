import pytest
from datetime import datetime, timezone
from pathlib import Path
from src.storage.db import Database, compute_dedup_hash
from src.storage.models import JobPosting, JobEvaluationResult


def test_compute_dedup_hash_normalization():
    # Identical after normalizing casing, corporate suffixes and whitespace
    h1 = compute_dedup_hash("Allegro Sp. z o.o.", "Executive Coach ", "Warsaw, Poland")
    h2 = compute_dedup_hash("allegro", "executive coach", "warsaw, poland")
    h3 = compute_dedup_hash("Allegro LLC", "EXECUTIVE  COACH", "Warsaw Poland")
    assert h1 == h2 == h3

    # Different title should produce different hash
    h_other = compute_dedup_hash("Allegro", "Career Coach", "Warsaw")
    assert h1 != h_other


def test_database_wal_and_busy_timeout_pragmas(tmp_path: Path):
    db_file = tmp_path / "pragma_test.db"
    db = Database(db_file)
    db.init_db()
    with db._get_connection() as conn:
        journal_mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        busy_timeout = conn.execute("PRAGMA busy_timeout;").fetchone()[0]
        assert journal_mode.lower() == "wal"
        assert busy_timeout == 5000


def test_database_init_and_job_lifecycle(tmp_path: Path):
    db_file = tmp_path / "test_jobs.db"
    db = Database(db_file)
    db.init_db()

    job = JobPosting(
        dedup_hash=compute_dedup_hash("Tech Corp", "Leadership Coach", "Remote"),
        site="linkedin",
        job_url="https://linkedin.com/jobs/view/123",
        title="Leadership Coach",
        company="Tech Corp",
        location="Remote",
        description="A great coaching opportunity for executive leaders.",
        is_remote=True,
    )

    # 1. Save new job
    job_id, is_new = db.save_job(job)
    assert job_id > 0
    assert is_new is True

    # 2. Saving duplicate should return existing ID and is_new=False
    dup_id, dup_is_new = db.save_job(job)
    assert dup_id == job_id
    assert dup_is_new is False

    # 3. Retrieve unprocessed
    unprocessed = db.get_unprocessed_jobs()
    assert len(unprocessed) == 1
    assert unprocessed[0].id == job_id
    assert unprocessed[0].eval_status == "unprocessed"

    # 4. Update LLM Evaluation
    eval_res = JobEvaluationResult(
        score=9.2,
        is_wow=True,
        summary_uk="Чудова вакансія для лідерського коучингу.",
        reasoning="Strong match for executive coach persona",
        language_requirements=["English", "Polish"],
        pros=["High autonomy"],
        cons=["Strict timezone"],
    )
    db.update_evaluation(job_id=job_id, eval_result=eval_res)

    evaluated_job = db.get_job_by_hash(job.dedup_hash)
    assert evaluated_job is not None
    assert evaluated_job.eval_status == "evaluated"
    assert evaluated_job.eval_score == 9.2
    assert evaluated_job.eval_is_wow is True
    assert evaluated_job.notification_status == "pending"

    # 5. Mark instant sent
    db.mark_instant_sent(job_id)
    sent_job = db.get_job_by_hash(job.dedup_hash)
    assert sent_job.notification_status == "instant_sent"
    assert sent_job.instant_sent_at is not None

    # 6. Verify stats
    stats = db.get_stats()
    assert stats["total_jobs"] == 1
    assert stats["evaluated_jobs"] == 1
    assert stats["instant_sent"] == 1


def test_database_eval_failure(tmp_path: Path):
    db_file = tmp_path / "test_failure.db"
    db = Database(db_file)
    db.init_db()

    job = JobPosting(
        dedup_hash=compute_dedup_hash("Fail Corp", "Scrum Coach", "Remote"),
        site="indeed",
        job_url="https://indeed.com/view/fail",
        title="Scrum Coach",
        company="Fail Corp",
        location="Remote",
        description="Testing failure handling",
    )
    job_id, _ = db.save_job(job)
    db.update_eval_failure(job_id, "API timeout communicating with OpenRouter")

    failed_job = db.get_job_by_hash(job.dedup_hash)
    assert failed_job is not None
    assert failed_job.eval_status == "failed"
    assert failed_job.eval_error == "API timeout communicating with OpenRouter"

    stats = db.get_stats()
    assert stats["failed_jobs"] == 1


def test_database_pending_digest_jobs(tmp_path: Path):
    db_file = tmp_path / "test_digest.db"
    db = Database(db_file)
    db.init_db()

    # Job 1: Score 8.0, pending -> should appear in digest
    j1 = JobPosting(
        dedup_hash=compute_dedup_hash("Comp A", "Career Coach", "Warsaw"),
        site="indeed",
        job_url="https://indeed.com/view/1",
        title="Career Coach",
        company="Comp A",
        location="Warsaw",
        description="Career coaching role in Warsaw.",
        is_remote=False,
    )
    id1, _ = db.save_job(j1)
    db.update_evaluation(
        id1,
        eval_result=JobEvaluationResult(
            score=8.0,
            is_wow=False,
            summary_uk="Робота в Варшаві.",
            reasoning="Good",
            language_requirements=["Polish"],
            pros=[],
            cons=[],
        ),
    )

    # Job 2: Score 4.5, pending -> should NOT appear (below 6.0)
    j2 = JobPosting(
        dedup_hash=compute_dedup_hash("Comp B", "Junior Trainer", "Remote"),
        site="glassdoor",
        job_url="https://glassdoor.com/view/2",
        title="Junior Trainer",
        company="Comp B",
        location="Remote",
        description="Junior support trainer.",
        is_remote=True,
    )
    id2, _ = db.save_job(j2)
    db.update_evaluation(
        id2,
        eval_result=JobEvaluationResult(
            score=4.5,
            is_wow=False,
            summary_uk="Низький бал.",
            reasoning="Poor fit",
            language_requirements=[],
            pros=[],
            cons=[],
        ),
    )

    pending_digest = db.get_pending_digest_jobs(min_score=6.0, limit=10)
    assert len(pending_digest) == 1
    assert pending_digest[0].id == id1

    db.mark_digest_sent([id1])
    assert len(db.get_pending_digest_jobs(min_score=6.0, limit=10)) == 0


def test_database_expire_stale_pending_jobs(tmp_path: Path):
    db_file = tmp_path / "test_expire.db"
    db = Database(db_file)
    db.init_db()

    job = JobPosting(
        dedup_hash=compute_dedup_hash("Old Co", "Coach", "Warsaw"),
        site="indeed",
        job_url="https://indeed.com/old",
        title="Coach",
        company="Old Co",
        location="Warsaw",
        description="Old vacancy description for testing expiration.",
        eval_status="evaluated",
        notification_status="pending",
    )
    job_id, _ = db.save_job(job)

    # Artificially age the job record past 7 days (10 days old)
    with db._get_connection() as conn:
        conn.execute(
            "UPDATE jobs SET created_at = DATETIME('now', '-10 days'), eval_status = 'evaluated', notification_status = 'pending' WHERE id = ?",
            (job_id,),
        )
        conn.commit()

    expired_count = db.expire_stale_pending_jobs(days=7)
    assert expired_count == 1

    expired_job = db.get_job_by_hash(job.dedup_hash)
    assert expired_job is not None
    assert expired_job.notification_status == "expired_dismissed"


def test_database_record_scrape_run(tmp_path: Path):
    db_file = tmp_path / "test_scrape_runs.db"
    db = Database(db_file)
    db.init_db()

    run_id = db.record_scrape_run(
        board="linkedin",
        query="Executive Coach",
        jobs_found=12,
        new_jobs_saved=4,
        duration_sec=3.45,
        status="completed",
    )
    assert run_id > 0

    with db._get_connection() as conn:
        row = conn.execute("SELECT * FROM scrape_runs WHERE id = ?", (run_id,)).fetchone()
        assert row["board"] == "linkedin"
        assert row["query"] == "Executive Coach"
        assert row["jobs_found"] == 12
        assert row["new_jobs_saved"] == 4
        assert row["duration_sec"] == 3.45
        assert row["status"] == "completed"
        assert row["error_message"] is None


def test_database_get_job_by_id_and_not_found(tmp_path: Path):
    db_file = tmp_path / "test_get_by_id.db"
    db = Database(db_file)
    db.init_db()

    assert db.get_job_by_id(9999) is None
    assert db.get_job_by_hash("nonexistent_hash") is None

    job = JobPosting(
        dedup_hash=compute_dedup_hash("Corp", "Title", "Location"),
        site="linkedin",
        job_url="https://linkedin.com/jobs/view/1",
        title="Title",
        company="Corp",
        location="Location",
        description="Description",
    )
    job_id, _ = db.save_job(job)
    fetched = db.get_job_by_id(job_id)
    assert fetched is not None
    assert fetched.id == job_id
    assert fetched.company == "Corp"


def test_database_unprocessed_jobs_prefiltered_exclusion(tmp_path: Path):
    db_file = tmp_path / "test_prefilter.db"
    db = Database(db_file)
    db.init_db()

    job1 = JobPosting(
        dedup_hash=compute_dedup_hash("Comp 1", "Coach 1", "Remote"),
        site="indeed",
        job_url="https://indeed.com/1",
        title="Coach 1",
        company="Comp 1",
        location="Remote",
        description="Normal coach",
        pre_filtered=False,
    )
    job2 = JobPosting(
        dedup_hash=compute_dedup_hash("Comp 2", "Coach 2", "Remote"),
        site="indeed",
        job_url="https://indeed.com/2",
        title="Coach 2",
        company="Comp 2",
        location="Remote",
        description="Pre-filtered coach",
        pre_filtered=True,
        pre_filter_reason="german_language_required",
    )
    db.save_job(job1)
    db.save_job(job2)

    unprocessed = db.get_unprocessed_jobs()
    assert len(unprocessed) == 1
    assert unprocessed[0].title == "Coach 1"

    stats = db.get_stats()
    assert stats["total_jobs"] == 2
    assert stats["prefiltered_jobs"] == 1


def test_database_mark_digest_sent_empty(tmp_path: Path):
    db_file = tmp_path / "test_empty_digest.db"
    db = Database(db_file)
    db.init_db()

    # Calling mark_digest_sent with empty list should be safe no-op
    db.mark_digest_sent([])


def test_compute_dedup_hash_empty_and_fallback(tmp_path: Path):
    h = compute_dedup_hash("", "", "")
    assert isinstance(h, str) and len(h) == 64

    # Test invalid date handling in row_to_job
    db_file = tmp_path / "test_dates.db"
    db = Database(db_file)
    db.init_db()

    job = JobPosting(
        dedup_hash=compute_dedup_hash("Co", "Role", "Loc"),
        site="indeed",
        job_url="https://indeed.com/r",
        title="Role",
        company="Co",
        location="Loc",
        description="Desc",
    )
    job_id, _ = db.save_job(job)
    with db._get_connection() as conn:
        conn.execute("UPDATE jobs SET evaluated_at = 'not-a-valid-date' WHERE id = ?", (job_id,))
        conn.commit()

    fetched = db.get_job_by_id(job_id)
    assert fetched is not None
    assert fetched.evaluated_at is None


def test_database_update_evaluation_with_notification_status(tmp_path: Path):
    db_file = tmp_path / "test_notif_status.db"
    db = Database(db_file)
    db.init_db()

    job = JobPosting(
        dedup_hash=compute_dedup_hash("Co", "Low Score Role", "Loc"),
        site="indeed",
        job_url="https://indeed.com/low",
        title="Low Score Role",
        company="Co",
        location="Loc",
        description="Desc",
    )
    job_id, _ = db.save_job(job)

    eval_res = JobEvaluationResult(
        score=4.0,
        is_wow=False,
        summary_uk="Низька оцінка.",
        reasoning="Irrelevant",
    )
    db.update_evaluation(job_id=job_id, eval_result=eval_res, notification_status="dismissed")

    reloaded = db.get_job_by_id(job_id)
    assert reloaded is not None
    assert reloaded.eval_status == "evaluated"
    assert reloaded.eval_score == 4.0
    assert reloaded.notification_status == "dismissed"


def test_database_mark_dismissed(tmp_path: Path):
    db_file = tmp_path / "test_mark_dismissed.db"
    db = Database(db_file)
    db.init_db()

    job = JobPosting(
        dedup_hash=compute_dedup_hash("Co", "Role", "Loc"),
        site="indeed",
        job_url="https://indeed.com/dismiss",
        title="Role",
        company="Co",
        location="Loc",
        description="Desc",
    )
    job_id, _ = db.save_job(job)
    assert job.notification_status == "pending"

    db.mark_dismissed(job_id)
    reloaded = db.get_job_by_id(job_id)
    assert reloaded is not None
    assert reloaded.notification_status == "dismissed"


def test_database_delete_empty_description_jobs(tmp_path: Path):
    db_file = tmp_path / "test_delete_empty.db"
    db = Database(db_file)
    db.init_db()

    # 1. Job with empty description on linkedin -> should be deleted
    j1 = JobPosting(
        dedup_hash=compute_dedup_hash("Comp A", "Coach 1", "Remote"),
        site="linkedin",
        job_url="https://linkedin.com/1",
        title="Coach 1",
        company="Comp A",
        location="Remote",
        description="",
    )
    # 2. Job with whitespace description on linkedin (mixed casing site) -> should be deleted
    j2 = JobPosting(
        dedup_hash=compute_dedup_hash("Comp B", "Coach 2", "Remote"),
        site="LinkedIn",
        job_url="https://linkedin.com/2",
        title="Coach 2",
        company="Comp B",
        location="Remote",
        description="   ",
    )
    # 3. Job with valid description on linkedin -> should NOT be deleted
    j3 = JobPosting(
        dedup_hash=compute_dedup_hash("Comp C", "Coach 3", "Remote"),
        site="linkedin",
        job_url="https://linkedin.com/3",
        title="Coach 3",
        company="Comp C",
        location="Remote",
        description="A detailed coaching job description.",
    )
    # 4. Job with empty description on indeed -> should NOT be deleted when site="linkedin"
    j4 = JobPosting(
        dedup_hash=compute_dedup_hash("Comp D", "Coach 4", "Remote"),
        site="indeed",
        job_url="https://indeed.com/4",
        title="Coach 4",
        company="Comp D",
        location="Remote",
        description="",
    )

    db.save_job(j1)
    db.save_job(j2)
    db.save_job(j3)
    db.save_job(j4)

    # Act: delete empty description jobs for linkedin
    deleted_count = db.delete_empty_description_jobs("linkedin")

    # Assert
    assert deleted_count == 2
    assert db.get_job_by_hash(j1.dedup_hash) is None
    assert db.get_job_by_hash(j2.dedup_hash) is None
    assert db.get_job_by_hash(j3.dedup_hash) is not None
    assert db.get_job_by_hash(j4.dedup_hash) is not None




