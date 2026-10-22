import pytest
from unittest.mock import MagicMock
from src.evaluator.schemas import JobEvaluationResult
from src.filters.pre_filter import pre_filter_job
from src.scheduler.daemon import DaemonService
from src.scraper.query_catalog import SearchQuery
from src.storage.db import Database, compute_dedup_hash
from src.storage.models import JobPosting


def test_full_pipeline_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr("src.scheduler.daemon.check_jobspy_update", MagicMock(return_value=None))
    """End-to-end integration test verifying the complete autonomous coaching monitor lifecycle:
    1. Scraping cycle simulation with noisy raw postings (sports noise, short stubs, valid roles, cross-board duplicates).
    2. Deterministic pre-filter rejection of athletic/sports titles and short stubs without invoking LLM.
    3. Cross-board SHA-256 deduplication dropping duplicate listings.
    4. LLM evaluation triggering instant WOW alerts for high-scoring roles (score >= 9.0 / is_wow == True).
    5. Queuing of high-relevance roles (score 7.5) for daily digest.
    6. Low-relevance roles (score 5.0) evaluated but excluded from digest.
    7. Expiration of stale pending postings (> 7 days) to 'expired_dismissed'.
    8. Morning daily digest delivery dispatching qualifying pending postings and updating status to 'digest_sent'.
    """
    db_file = tmp_path / "e2e_jobs.db"

    # 1. Prepare simulated noisy scraped jobs:
    # - Job A: Swimming coach (athletic noise -> pre-filter regex rejection)
    # - Job B: Marginal HR role (evaluated score 5.0 -> dismissed from digest)
    # - Job C: Career Coach (evaluated score 7.5 -> queued for daily digest)
    # - Job D: Executive Coach WOW role (evaluated score 9.6, is_wow=True -> instant alert)
    # - Job E: Cross-board duplicate of Job D from Indeed (must be deduplicated)
    # - Job F: Short stub role with insufficient description (pre-filter length rejection)

    long_desc = "Develop executive leadership strategy and coach corporate professionals. " * 5

    raw_jobs = [
        JobPosting(
            dedup_hash=compute_dedup_hash("City Pool", "Swimming Coach", "Warsaw"),
            site="indeed",
            job_url="https://indeed.com/pool",
            title="Swimming Coach",
            company="City Pool",
            location="Warsaw",
            description="Teach swimming techniques to children and adults at city pool." * 5,
        ),
        JobPosting(
            dedup_hash=compute_dedup_hash("HR Agency", "Recruiter & Coach", "Warsaw"),
            site="glassdoor",
            job_url="https://glassdoor.com/hr",
            title="Recruiter & Coach",
            company="HR Agency",
            location="Warsaw",
            description="Recruitment specialist with some coaching of junior sourcers." * 5,
        ),
        JobPosting(
            dedup_hash=compute_dedup_hash("Tech Corp", "Career Transition Coach", "Remote Poland"),
            site="linkedin",
            job_url="https://linkedin.com/career",
            title="Career Transition Coach",
            company="Tech Corp",
            location="Remote Poland",
            description="Provide one-on-one career coaching and transition workshops." * 5,
            is_remote=True,
        ),
        JobPosting(
            dedup_hash=compute_dedup_hash("Global Bank", "Executive Leadership Coach", "Remote EU"),
            site="linkedin",
            job_url="https://linkedin.com/exec",
            title="Executive Leadership Coach",
            company="Global Bank",
            location="Remote EU",
            description=long_desc,
            is_remote=True,
        ),
        JobPosting(
            dedup_hash=compute_dedup_hash("Global Bank Sp. z o.o.", "Executive  Leadership Coach", "Remote EU"),
            site="indeed",
            job_url="https://indeed.com/exec-dup",
            title="Executive Leadership Coach",
            company="Global Bank Sp. z o.o.",
            location="Remote EU",
            description=long_desc,
            is_remote=True,
        ),
        JobPosting(
            dedup_hash=compute_dedup_hash("Fast Track", "Agile Coach", "Krakow"),
            site="glassdoor",
            job_url="https://glassdoor.com/stub",
            title="Agile Coach",
            company="Fast Track",
            location="Krakow",
            description="Short stub description.",  # Under 150 chars / under 30 words
        ),
    ]

    def mock_scrape_board(site, query):
        results = []
        for j in raw_jobs:
            if j.site == site:
                pf = pre_filter_job(j.title, j.description)
                if not pf.passed:
                    j.pre_filtered = True
                    j.pre_filter_reason = pf.reason
                    j.notification_status = "dismissed"
                else:
                    j.pre_filtered = False
                    j.notification_status = "pending"
                results.append(j)
        return results

    mock_scraper = MagicMock()
    mock_scraper.scrape_board.side_effect = mock_scrape_board

    def mock_evaluate(job):
        if "Recruiter" in job.title:
            return JobEvaluationResult(
                score=5.0,
                is_wow=False,
                summary_uk="Змішана роль рекрутера з мінімальним коучингом.",
                reasoning="Heavy recruitment focus",
                language_requirements=["Polish"],
                pros=[],
                cons=["Low coaching relevance"],
            )
        elif "Career" in job.title:
            return JobEvaluationResult(
                score=7.5,
                is_wow=False,
                summary_uk="Індивідуальний кар'єрний коучинг для ІТ спеціалістів.",
                reasoning="Strong alignment with career coaching domain",
                language_requirements=["Polish", "English"],
                pros=["Remote", "Good culture"],
                cons=[],
            )
        elif "Executive" in job.title:
            return JobEvaluationResult(
                score=9.6,
                is_wow=True,
                summary_uk="Топ-вакансія executive коуча для роботи з керівниками банку.",
                reasoning="Dream role with international exposure and professional growth opportunity",
                language_requirements=["English"],
                pros=["High salary", "C-suite client base"],
                cons=[],
            )
        raise ValueError(f"Unexpected job title {job.title}")

    mock_evaluator = MagicMock()
    mock_evaluator.evaluate_job.side_effect = mock_evaluate

    mock_notifier = MagicMock()
    mock_notifier.send_wow_alert.return_value = True
    mock_notifier.send_daily_digest.return_value = 1

    daemon = DaemonService(
        db_path=db_file,
        scraper=mock_scraper,
        evaluator=mock_evaluator,
        notifier=mock_notifier,
    )

    test_queries = [
        SearchQuery(
            query_id="Q-TEST-E2E",
            search_term="Coaching",
            location="Remote",
            site_names=["indeed", "glassdoor", "linkedin"],
        )
    ]

    # Execute scrape and evaluate cycle
    new_saved, wow_sent = daemon.run_pipeline_cycle(queries=test_queries, dry_run=False)

    # 5 distinct jobs saved (duplicate dropped by SHA-256 dedup)
    # 6 raw jobs: Job A (saved), Job E (saved), Job B (saved), Job F (saved), Job C (saved), Job D (dedup dropped)
    assert new_saved == 5
    # Exactly 1 WOW alert dispatched (Executive Coach)
    assert wow_sent == 1
    assert mock_notifier.send_wow_alert.call_count == 1

    # Verify evaluator was called only for non-prefiltered jobs (Recruiter, Career, Executive)
    assert mock_evaluator.evaluate_job.call_count == 3

    db = Database(db_file)
    stats = db.get_stats()
    assert stats["total_jobs"] == 5
    assert stats["prefiltered_jobs"] == 2  # Swimming Coach (sports) + Agile Coach (short stub)
    assert stats["evaluated_jobs"] == 3    # Recruiter, Career, Executive
    assert stats["instant_sent"] == 1      # Executive WOW role
    assert stats["pending_digest"] == 1    # Career Coach (score 7.5 >= 6.0)

    # Verify pre-filter rejection reasons stored in DB
    swimming_job = db.get_job_by_hash(compute_dedup_hash("City Pool", "Swimming Coach", "Warsaw"))
    assert swimming_job is not None
    assert swimming_job.pre_filtered is True
    assert "EXCLUDED_TITLE_REGEX" in swimming_job.pre_filter_reason
    assert swimming_job.notification_status == "dismissed"

    stub_job = db.get_job_by_hash(compute_dedup_hash("Fast Track", "Agile Coach", "Krakow"))
    assert stub_job is not None
    assert stub_job.pre_filtered is True
    assert "DESCRIPTION_TOO_SHORT" in stub_job.pre_filter_reason
    assert stub_job.notification_status == "dismissed"

    # Verify Recruiter job (score 5.0 < 6.0 threshold, not wow) was immediately dismissed
    recruiter_job = db.get_job_by_hash(compute_dedup_hash("HR Agency", "Recruiter & Coach", "Warsaw"))
    assert recruiter_job is not None
    assert recruiter_job.eval_status == "evaluated"
    assert recruiter_job.eval_score == 5.0
    assert recruiter_job.notification_status == "dismissed"

    # Verify scrape run audit records were written for each board
    with db._get_connection() as conn:
        scrape_runs = conn.execute("SELECT board, jobs_found, new_jobs_saved, status FROM scrape_runs ORDER BY id ASC").fetchall()
        assert len(scrape_runs) == 3
        boards = [r["board"] for r in scrape_runs]
        assert boards == ["indeed", "glassdoor", "linkedin"]
        for r in scrape_runs:
            assert r["status"] == "completed"

    # Simulate a stale pending job in the database (>7 days old)
    stale_job = JobPosting(
        dedup_hash=compute_dedup_hash("Old Corp", "Leadership Mentor", "Warsaw"),
        site="indeed",
        job_url="https://indeed.com/stale-mentor",
        title="Leadership Mentor",
        company="Old Corp",
        location="Warsaw",
        description="Mentoring senior tech leads in communication and organizational growth." * 5,
        eval_status="evaluated",
        eval_score=7.5,
        notification_status="pending",
    )
    stale_id, _ = db.save_job(stale_job)
    with db._get_connection() as conn:
        conn.execute("UPDATE jobs SET created_at = DATETIME('now', '-10 days') WHERE id = ?", (stale_id,))
        conn.commit()

    # Pre-digest check: 2 pending digest candidates (Career Coach + Stale Leadership Mentor)
    stats_pre_digest = db.get_stats()
    assert stats_pre_digest["pending_digest"] == 2

    # Execute daily digest cycle
    digest_processed = daemon.run_daily_digest_cycle(dry_run=False)

    # Only Career Coach (score 7.5) is delivered; Stale job expired, WOW role was instant_sent, Recruiter was 5.0
    assert digest_processed == 1
    assert mock_notifier.send_daily_digest.call_count == 1
    # Verify the job sent to daily digest is indeed the Career Coach
    digest_call_args = mock_notifier.send_daily_digest.call_args[0]
    sent_digest_jobs = digest_call_args[0]
    assert len(sent_digest_jobs) == 1
    assert sent_digest_jobs[0].title == "Career Transition Coach"
    assert sent_digest_jobs[0].eval_score == 7.5

    # Verify stale job transitioned to expired_dismissed
    reloaded_stale = db.get_job_by_id(stale_id)
    assert reloaded_stale is not None
    assert reloaded_stale.notification_status == "expired_dismissed"

    # Verify updated DB stats
    updated_stats = db.get_stats()
    assert updated_stats["total_jobs"] == 6
    assert updated_stats["prefiltered_jobs"] == 2
    assert updated_stats["evaluated_jobs"] == 4
    assert updated_stats["instant_sent"] == 1
    assert updated_stats["digest_sent"] == 1
    assert updated_stats["pending_digest"] == 0


def test_e2e_dry_run_pipeline_cycle(tmp_path):
    """Verify that in dry-run mode, scraping, evaluation, and digest cycles produce no persistent side-effects."""
    db_file = tmp_path / "e2e_dry_run.db"

    mock_scraper = MagicMock()
    mock_job = JobPosting(
        dedup_hash=compute_dedup_hash("Enterprise Co", "Executive Coach", "Remote"),
        site="linkedin",
        job_url="https://linkedin.com/dry-exec",
        title="Executive Coach",
        company="Enterprise Co",
        location="Remote",
        description="Executive coaching for senior leaders." * 5,
        is_remote=True,
    )
    mock_scraper.scrape_board.return_value = [mock_job]

    mock_evaluator = MagicMock()
    mock_evaluator.evaluate_job.return_value = JobEvaluationResult(
        score=9.5,
        is_wow=True,
        summary_uk="Тестова dry-run вакансія.",
        reasoning="Dry-run fit",
        language_requirements=["English"],
        pros=[],
        cons=[],
    )

    mock_notifier = MagicMock()

    daemon = DaemonService(
        db_path=db_file,
        scraper=mock_scraper,
        evaluator=mock_evaluator,
        notifier=mock_notifier,
    )

    test_queries = [
        SearchQuery(
            query_id="Q-DRY",
            search_term="Coach",
            location="Remote",
            site_names=["linkedin"],
        )
    ]

    # Execute scrape cycle in dry-run mode
    new_saved, wow_sent = daemon.run_pipeline_cycle(queries=test_queries, dry_run=True)
    assert new_saved == 0
    assert wow_sent == 0
    mock_notifier.send_wow_alert.assert_not_called()

    # DB should remain empty
    db = Database(db_file)
    stats = db.get_stats()
    assert stats["total_jobs"] == 0
    assert stats["prefiltered_jobs"] == 0
    assert stats["evaluated_jobs"] == 0

    # Execute digest cycle in dry-run mode
    digest_count = daemon.run_daily_digest_cycle(dry_run=True)
    assert digest_count == 0
    mock_notifier.send_daily_digest.assert_not_called()


def test_e2e_evaluator_retry_and_recovery(tmp_path):
    """Verify that transient evaluation failures leave postings in 'unprocessed' state
    and subsequent pipeline cycles successfully evaluate and dispatch alerts."""
    db_file = tmp_path / "e2e_retry.db"

    mock_scraper = MagicMock()
    job = JobPosting(
        dedup_hash=compute_dedup_hash("Resilient Corp", "Executive Leadership Coach", "Warsaw"),
        site="indeed",
        job_url="https://indeed.com/resilient",
        title="Executive Leadership Coach",
        company="Resilient Corp",
        location="Warsaw",
        description="Senior executive coach for cross-functional leadership and enterprise transformation. " * 5,
    )
    mock_scraper.scrape_board.return_value = [job]

    mock_evaluator = MagicMock()
    # First call: transient failure (HTTP 503)
    # Second call: success
    mock_evaluator.evaluate_job.side_effect = [
        RuntimeError("HTTP 503 Service Unavailable"),
        JobEvaluationResult(
            score=9.2,
            is_wow=True,
            summary_uk="Executive коучинг після успішного retry.",
            reasoning="Passed on retry",
            language_requirements=["English"],
            pros=["Great role"],
            cons=[],
        ),
    ]

    mock_notifier = MagicMock()
    mock_notifier.send_wow_alert.return_value = True

    daemon = DaemonService(
        db_path=db_file,
        scraper=mock_scraper,
        evaluator=mock_evaluator,
        notifier=mock_notifier,
    )

    test_queries = [
        SearchQuery(
            query_id="Q-RETRY",
            search_term="Coach",
            location="Warsaw",
            site_names=["indeed"],
        )
    ]

    # Cycle 1: Job scraped, evaluation fails transiently
    new_saved, wow_sent = daemon.run_pipeline_cycle(queries=test_queries, dry_run=False)
    assert new_saved == 1
    assert wow_sent == 0
    mock_notifier.send_wow_alert.assert_not_called()

    db = Database(db_file)
    saved = db.get_job_by_hash(job.dedup_hash)
    assert saved is not None
    assert saved.eval_status == "unprocessed"

    # Cycle 2: Scraper returns same job (dedup skips insert), evaluation succeeds
    mock_scraper.scrape_board.return_value = [job]
    new_saved_2, wow_sent_2 = daemon.run_pipeline_cycle(queries=test_queries, dry_run=False)
    assert new_saved_2 == 0  # Deduplicated
    assert wow_sent_2 == 1  # Successfully evaluated and alert sent
    assert mock_notifier.send_wow_alert.call_count == 1

    saved_after = db.get_job_by_hash(job.dedup_hash)
    assert saved_after.eval_status == "evaluated"
    assert saved_after.eval_score == 9.2
    assert saved_after.notification_status == "instant_sent"
