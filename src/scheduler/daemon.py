import logging
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import httpx
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from src.config import get_settings
from src.evaluator.llm_evaluator import OpenRouterEvaluator
from src.filters.pre_filter import pre_filter_job
from src.notifier.telegram_notifier import TelegramNotifier
from src.scraper.jobspy_scraper import JobSpyScraper
from src.scraper.query_catalog import DEFAULT_QUERY_CATALOG, SearchQuery
from src.storage.db import Database
from src.storage.models import JobPosting

logger = logging.getLogger(__name__)


class DaemonService:
    def __init__(
        self,
        db_path: Optional[Path] = None,
        scraper: Optional[JobSpyScraper] = None,
        evaluator: Optional[OpenRouterEvaluator] = None,
        notifier: Optional[TelegramNotifier] = None,
    ):
        settings = get_settings()
        self.db = Database(db_path or settings.db_path)
        self.db.init_db()

        self.scraper = scraper or JobSpyScraper(
            jitter_min=settings.scraper_jitter_min_seconds,
            jitter_max=settings.scraper_jitter_max_seconds,
            results_per_query=settings.scraper_results_per_query,
            hours_old=settings.hours_old_window,
        )
        self.evaluator = evaluator or OpenRouterEvaluator(
            api_key=settings.openrouter_api_key,
            model=settings.openrouter_model,
            base_url=settings.openrouter_base_url,
            rpm_limit=settings.openrouter_rpm_limit,
            timeout=settings.openrouter_timeout_seconds,
        )
        self.notifier = notifier or TelegramNotifier(
            token=settings.telegram_bot_token,
            chat_id=settings.telegram_chat_id,
        )
        self.scheduler: Optional[BlockingScheduler] = None

    def evaluate_unprocessed(self, limit: int = 50, dry_run: bool = False) -> Tuple[int, int]:
        unprocessed = self.db.get_unprocessed_jobs(limit=limit)
        logger.info(f"Evaluating {len(unprocessed)} unprocessed postings with LLM...")

        evaluated_count = 0
        wow_sent = 0

        for job in unprocessed:
            if job.id is None:
                continue
            try:
                eval_res = self.evaluator.evaluate_job(job)
                if dry_run:
                    logger.info(
                        f"[DRY-RUN] Evaluated job {job.id} ({job.title}): Score={eval_res.score}, WOW={eval_res.is_wow}"
                    )
                    evaluated_count += 1
                    if eval_res.score >= get_settings().wow_score_threshold or eval_res.is_wow:
                        wow_sent += 1
                    continue

                self.db.update_evaluation(job_id=job.id, eval_result=eval_res)
                evaluated_count += 1

                settings = get_settings()
                if eval_res.score >= settings.wow_score_threshold or eval_res.is_wow:
                    job.eval_score = eval_res.score
                    job.eval_is_wow = eval_res.is_wow
                    job.eval_summary_uk = eval_res.summary_uk
                    job.eval_reasoning = eval_res.reasoning
                    job.eval_languages = eval_res.language_requirements
                    job.eval_pros = eval_res.pros
                    job.eval_cons = eval_res.cons
                    sent = self.notifier.send_wow_alert(job)
                    if sent:
                        self.db.mark_instant_sent(job.id)
                        wow_sent += 1
            except (httpx.TimeoutException, httpx.NetworkError) as e:
                # Transient network/timeout errors leave status unprocessed to retry on next cycle
                logger.warning(
                    f"Transient network/timeout error evaluating job {job.id} ('{job.title}'): {e}. "
                    "Leaving in 'unprocessed' status for subsequent cycle retry."
                )
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (429, 500, 502, 503, 504):
                    logger.warning(
                        f"Transient HTTP {e.response.status_code} evaluating job {job.id} ('{job.title}'): {e}. "
                        "Leaving in 'unprocessed' status for subsequent cycle retry."
                    )
                else:
                    logger.error(f"Non-recoverable HTTP {e.response.status_code} evaluating job {job.id}: {e}")
                    if not dry_run:
                        self.db.update_eval_failure(job.id, str(e))
            except Exception as e:
                if any(k in str(e).lower() for k in ["503", "502", "504", "429", "timeout", "connection", "network"]):
                    logger.warning(
                        f"Transient error evaluating job {job.id} ('{job.title}'): {e}. "
                        "Leaving in 'unprocessed' status for subsequent cycle retry."
                    )
                else:
                    logger.error(f"Permanent evaluation failure for job {job.id}: {e}")
                    if not dry_run:
                        self.db.update_eval_failure(job.id, str(e))

        return evaluated_count, wow_sent

    def run_pipeline_cycle(
        self, queries: Optional[List[SearchQuery]] = None, dry_run: bool = False
    ) -> Tuple[int, int]:
        active_queries = queries or DEFAULT_QUERY_CATALOG
        logger.info("Starting pipeline scrape cycle across queries...")
        total_new_saved = 0

        for q in active_queries:
            for site in q.site_names:
                t0 = time.time()
                try:
                    board_postings = self.scraper.scrape_board(site, q)
                    duration = time.time() - t0
                    board_new_saved = 0

                    for posting in board_postings:
                        pf_res = pre_filter_job(posting.title, posting.description)
                        if not pf_res.passed:
                            posting.pre_filtered = True
                            posting.pre_filter_reason = pf_res.reason
                            posting.notification_status = "dismissed"
                        else:
                            posting.pre_filtered = False

                        if dry_run:
                            logger.info(
                                f"[DRY-RUN] Discovered: '{posting.title}' at '{posting.company}' (Pre-filtered: {posting.pre_filtered})"
                            )
                            continue

                        _, is_new = self.db.save_job(posting)
                        if is_new:
                            board_new_saved += 1
                            total_new_saved += 1

                    if not dry_run:
                        self.db.record_scrape_run(
                            board=site,
                            query=q.search_term,
                            jobs_found=len(board_postings),
                            new_jobs_saved=board_new_saved,
                            duration_sec=duration,
                            status="completed",
                        )
                except Exception as e:
                    duration = time.time() - t0
                    logger.error(f"Failed scrape cycle for board '{site}' query '{q.query_id}': {e}")
                    if not dry_run:
                        self.db.record_scrape_run(
                            board=site,
                            query=q.search_term,
                            jobs_found=0,
                            new_jobs_saved=0,
                            duration_sec=duration,
                            status="failed",
                            error_message=str(e),
                        )

        if dry_run:
            return 0, 0

        # Process evaluations
        _, wow_sent = self.evaluate_unprocessed(limit=50, dry_run=False)
        logger.info(f"Pipeline cycle completed. New: {total_new_saved}, WOW alerts: {wow_sent}")
        return total_new_saved, wow_sent

    def run_daily_digest_cycle(self, dry_run: bool = False) -> int:
        if not dry_run:
            expired = self.db.expire_stale_pending_jobs(days=7)
            if expired > 0:
                logger.info(f"Transitioned {expired} stale pending jobs (>7 days) to 'expired_dismissed'.")

        settings = get_settings()
        pending = self.db.get_pending_digest_jobs(
            min_score=settings.digest_score_threshold,
            limit=settings.digest_max_items,
        )
        if not pending:
            logger.info("No pending qualifying jobs for daily digest.")
            return 0

        date_str = datetime.now().strftime("%d.%m.%Y")
        if dry_run:
            logger.info(f"[DRY-RUN] Daily digest would contain {len(pending)} jobs.")
            return len(pending)

        sent_count = self.notifier.send_daily_digest(pending, date_str=date_str)
        if sent_count > 0:
            self.db.mark_digest_sent([j.id for j in pending if j.id is not None])
        return len(pending)

    def start(self) -> None:
        settings = get_settings()
        self.scheduler = BlockingScheduler(timezone=settings.timezone)

        # Scrape interval
        self.scheduler.add_job(
            self.run_pipeline_cycle,
            trigger=IntervalTrigger(hours=settings.scrape_interval_hours),
            id="scrape_pipeline",
            name="Scrape & Evaluate Pipeline",
        )

        # Daily digest cron at 09:00 Europe/Warsaw
        hour, minute = settings.digest_time_cet.split(":")
        self.scheduler.add_job(
            self.run_daily_digest_cycle,
            trigger=CronTrigger(hour=int(hour), minute=int(minute)),
            id="daily_digest",
            name="Morning Digest",
        )

        def shutdown(signum, frame):
            logger.info("Received termination signal. Shutting down daemon gracefully...")
            if self.scheduler and self.scheduler.running:
                self.scheduler.shutdown(wait=False)
            sys.exit(0)

        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGTERM, shutdown)

        logger.info("Running initial startup scraping cycle...")
        self.run_pipeline_cycle()

        logger.info(
            f"Daemon started. Scrape every {settings.scrape_interval_hours}h, "
            f"Digest at {settings.digest_time_cet} {settings.timezone}."
        )
        self.scheduler.start()
