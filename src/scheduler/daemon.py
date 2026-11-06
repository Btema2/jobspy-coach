import logging
import random
import signal
import sys
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, CancelledError, Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple
from zoneinfo import ZoneInfo

import httpx
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from src.config import get_settings
from src.evaluator.chain import ProviderChainEvaluator, build_chain_from_settings
from src.evaluator.errors import ConfigurationError, EvaluationRejected, ProvidersUnavailable, StopRequested
from src.notifier.telegram_notifier import TelegramNotifier
from src.notifier.templates import format_version_update_notice
from src.scraper.jobspy_scraper import JobSpyScraper
from src.scraper.query_catalog import DEFAULT_QUERY_CATALOG, SearchQuery, get_active_queries
from src.scraper.version_checker import check_jobspy_update
from src.storage.db import Database
from src.storage.models import JobPosting

logger = logging.getLogger(__name__)


def _iter_completed(futures: Dict[Future, Any], poll_seconds: float = 0.2) -> Iterator[Future]:
    """Yield futures as they finish, including ones cancelled by ``shutdown(cancel_futures=True)``.

    ``concurrent.futures.as_completed`` never reports a future that was cancelled while still
    pending (the executor's shutdown does not notify its waiters), so it would block forever
    after a batch abort or a shutdown signal. Cancelled futures are therefore polled for.
    """
    pending = set(futures)
    while pending:
        done, pending = wait(pending, timeout=poll_seconds, return_when=FIRST_COMPLETED)
        yield from done
        cancelled = {f for f in pending if f.cancelled()}
        pending -= cancelled
        yield from cancelled


class DaemonService:
    def __init__(
        self,
        db_path: Optional[Path] = None,
        scraper: Optional[JobSpyScraper] = None,
        evaluator: Optional[ProviderChainEvaluator] = None,
        notifier: Optional[TelegramNotifier] = None,
    ):
        settings = get_settings()
        self.settings = settings
        self.db = Database(db_path or settings.db_path)
        self.db.init_db()

        self.scraper = scraper or JobSpyScraper(
            jitter_min=settings.scraper_jitter_min_seconds,
            jitter_max=settings.scraper_jitter_max_seconds,
            results_per_query=settings.scraper_results_per_query,
            hours_old=settings.hours_old_window,
            enabled_boards=settings.scraper_enabled_boards,
            proxies=settings.scraper_proxies,
        )
        # The provider chain is built lazily (first evaluation): constructing the daemon must
        # never require API keys. ``_stop_event`` is shared with the chain once it is built.
        self.evaluator: Optional[ProviderChainEvaluator] = evaluator
        self._stop_event = threading.Event()
        self._shutdown_requested = threading.Event()
        self._chain_lock = threading.Lock()
        # Futures of the batch in flight (snapshot-cancelled by request_shutdown).
        self._futures: List[Future] = []
        self.notifier = notifier or TelegramNotifier(
            token=settings.telegram_bot_token,
            chat_id=settings.telegram_chat_id,
        )
        self.scheduler: Optional[BlockingScheduler] = None

    # ----- evaluation engine -----

    def _get_evaluator(self) -> Any:
        """Return the provider chain, building it on first use (raises ConfigurationError)."""
        with self._chain_lock:
            if self.evaluator is None:
                self.evaluator = build_chain_from_settings(self.settings, stop_event=self._stop_event)
            return self.evaluator

    def _stop_chain(self) -> None:
        stop = getattr(self.evaluator, "stop", None)
        if callable(stop):
            stop()

    def request_shutdown(self) -> None:
        """Signal-handler hook: wake blocked workers and drop queued evaluations.

        Safe to call at any time (before the chain/batch exist, or repeatedly). It never
        calls ``executor.shutdown()``: that takes the executor's internal lock, which the
        interrupted main thread may already hold (self-deadlock). Queued futures are
        cancelled instead; the batch's own caller thread shuts the executor down. Workers
        mid-HTTP finish or time out on their own; nothing joins them.
        """
        self._shutdown_requested.set()
        self._stop_event.set()
        self._stop_chain()
        for future in list(self._futures):
            future.cancel()

    def _rearm_stop_event(self, chain: Any) -> bool:
        """Undo a previous batch abort so a new batch can run.

        Returns False (leaving the stop events set) if a shutdown was requested, including
        one that landed between the caller's earlier check and the clear below.
        """
        self._stop_event.clear()
        event = getattr(chain, "stop_event", None)
        if isinstance(event, threading.Event):
            event.clear()
        if self._shutdown_requested.is_set():
            self._stop_event.set()
            if isinstance(event, threading.Event):
                event.set()
            return False
        return True

    @staticmethod
    def _provider_for_model(chain: Any, model: Optional[str]) -> str:
        for provider in getattr(chain, "providers", None) or []:
            models = [getattr(provider, "model", None), *(getattr(provider, "models", None) or [])]
            if model and model in models:
                return str(getattr(provider, "name", "unknown"))
        return "unknown"

    def evaluate_unprocessed(self, limit: Optional[int] = None, dry_run: bool = False) -> Tuple[int, int]:
        """Evaluate queued jobs concurrently; return ``(evaluated_count, wow_sent)``.

        Invariant: the thread that calls this method is the only one that touches the
        database or Telegram. Worker threads only run ``chain.evaluate_job`` (LLM calls).

        Database or Telegram errors raised while handling a result abort the whole batch
        (the exception propagates); jobs not yet handled stay ``unprocessed`` and are
        retried next cycle.
        """
        if self._shutdown_requested.is_set():
            return 0, 0
        if limit is None:
            limit = self.settings.eval_batch_limit
        jobs = [j for j in self.db.get_unprocessed_jobs(limit=limit) if j.id is not None]
        if not jobs:
            return 0, 0

        chain = self._get_evaluator()
        if not self._rearm_stop_event(chain) or self._shutdown_requested.is_set():
            return 0, 0
        total = len(jobs)
        settings = self.settings
        logger.info(
            "Evaluating %d unprocessed postings with %d workers...", total, settings.eval_max_workers
        )

        evaluated_count = 0
        wow_sent = 0
        failures = 0
        done = 0
        per_provider: Counter = Counter()
        batch_stopped = False
        t0 = time.monotonic()

        def timed_eval(job: JobPosting) -> Tuple[Any, float]:
            started = time.monotonic()
            return chain.evaluate_job(job), time.monotonic() - started

        # No ``with``: its __exit__ would shutdown(wait=True) and block shutdown for minutes.
        executor = ThreadPoolExecutor(max_workers=max(1, settings.eval_max_workers), thread_name_prefix="eval")
        futures: Dict[Future, JobPosting] = {}
        self._futures = []
        try:
            for job in jobs:
                if self._shutdown_requested.is_set():
                    break
                future = executor.submit(timed_eval, job)
                futures[future] = job
                self._futures.append(future)
            if self._shutdown_requested.is_set():  # a signal landed while submitting
                for future in futures:
                    future.cancel()
            for future in _iter_completed(futures):
                job = futures[future]
                try:
                    eval_res, latency = future.result()
                except (CancelledError, StopRequested):
                    continue
                except ProvidersUnavailable as e:
                    # Provider-wide outage: job stays unprocessed with attempts unchanged.
                    if not batch_stopped:
                        batch_stopped = True
                        logger.error(
                            "All evaluation providers unavailable (%s); stopping this batch. "
                            "Remaining jobs stay 'unprocessed' for the next cycle.",
                            e,
                        )
                        executor.shutdown(wait=False, cancel_futures=True)
                        self._stop_chain()
                    continue
                except EvaluationRejected as e:
                    failures += 1
                    done += 1
                    logger.warning(
                        "[%d/%d] job %s ('%s') rejected by every provider: %s", done, total, job.id, job.title, e
                    )
                    if not dry_run:
                        self.db.record_eval_attempt(job.id, str(e), settings.eval_max_attempts)
                    continue
                except Exception as e:  # keep the pre-existing exception -> outcome mapping
                    failures += 1
                    done += 1
                    self._handle_legacy_failure(job, e, dry_run)
                    continue

                done += 1
                model = eval_res.eval_model if isinstance(eval_res.eval_model, str) else None
                provider_name = self._provider_for_model(chain, model)
                per_provider[provider_name] += 1
                logger.info(
                    "[%d/%d] %s %s %.1fs %s -> %.1f",
                    done, total, provider_name, model or "unknown-model", latency, job.title, eval_res.score,
                )

                # The alert decision lives here, from settings, once.
                eval_res.is_wow = bool(
                    eval_res.score >= settings.wow_score_threshold and not eval_res.hard_reject
                )
                if dry_run:
                    evaluated_count += 1
                    if eval_res.is_wow:
                        wow_sent += 1
                    continue

                notification_status = None
                if eval_res.score < settings.digest_score_threshold and not eval_res.is_wow:
                    notification_status = "dismissed"

                self.db.update_evaluation(
                    job_id=job.id, eval_result=eval_res, notification_status=notification_status
                )
                evaluated_count += 1
                if notification_status:
                    job.notification_status = notification_status

                if eval_res.is_wow:
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
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
            self._futures = []

        wall = time.monotonic() - t0
        rpm = (done / wall * 60.0) if wall > 0 else 0.0
        provider_counts = ", ".join(f"{k}={v}" for k, v in sorted(per_provider.items())) or "none"
        logger.info(
            "Evaluation batch finished: processed=%d/%d evaluated=%d wow=%d failures=%d providers[%s] "
            "wall=%.1fs effective=%.1f rpm%s",
            done, total, evaluated_count, wow_sent, failures, provider_counts, wall, rpm,
            " (batch aborted: providers unavailable)" if batch_stopped else "",
        )
        return evaluated_count, wow_sent

    def _handle_legacy_failure(self, job: JobPosting, e: Exception, dry_run: bool) -> None:
        """Pre-existing mapping: transient errors leave the job unprocessed, others fail it."""
        if isinstance(e, (httpx.TimeoutException, httpx.NetworkError)):
            logger.warning(
                f"Transient network/timeout error evaluating job {job.id} ('{job.title}'): {e}. "
                "Leaving in 'unprocessed' status for subsequent cycle retry."
            )
        elif isinstance(e, httpx.HTTPStatusError):
            if e.response.status_code in (429, 500, 502, 503, 504):
                logger.warning(
                    f"Transient HTTP {e.response.status_code} evaluating job {job.id} ('{job.title}'): {e}. "
                    "Leaving in 'unprocessed' status for subsequent cycle retry."
                )
            else:
                logger.error(f"Non-recoverable HTTP {e.response.status_code} evaluating job {job.id}: {e}")
                if not dry_run:
                    self.db.update_eval_failure(job.id, str(e))
        elif any(k in str(e).lower() for k in ["503", "502", "504", "429", "timeout", "connection", "network"]):
            logger.warning(
                f"Transient error evaluating job {job.id} ('{job.title}'): {e}. "
                "Leaving in 'unprocessed' status for subsequent cycle retry."
            )
        else:
            logger.error(f"Permanent evaluation failure for job {job.id}: {e}")
            if not dry_run:
                self.db.update_eval_failure(job.id, str(e))

    def run_pipeline_cycle(
        self, queries: Optional[List[SearchQuery]] = None, dry_run: bool = False
    ) -> Tuple[int, int]:
        active_queries = queries or get_active_queries()
        logger.info("Starting pipeline scrape cycle across queries...")
        total_new_saved = 0

        for q_idx, q in enumerate(active_queries):
            for s_idx, site in enumerate(q.site_names):
                jitter_max = getattr(self.scraper, "jitter_max", 0.0)
                if (q_idx > 0 or s_idx > 0) and isinstance(jitter_max, (int, float)) and jitter_max > 0:
                    jitter_min = getattr(self.scraper, "jitter_min", 0.0)
                    min_val = jitter_min if isinstance(jitter_min, (int, float)) else 0.0
                    time.sleep(random.uniform(min_val, jitter_max))

                t0 = time.time()
                try:
                    board_postings = self.scraper.scrape_board(site, q)
                    duration = time.time() - t0
                    board_new_saved = 0

                    for posting in board_postings:
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
        _, wow_sent = self.evaluate_unprocessed(limit=self.settings.eval_batch_limit, dry_run=False)
        logger.info(f"Pipeline cycle completed. New: {total_new_saved}, WOW alerts: {wow_sent}")
        return total_new_saved, wow_sent

    def run_daily_digest_cycle(self, dry_run: bool = False) -> int:
        if not dry_run:
            expired = self.db.expire_stale_pending_jobs(days=7)
            if expired > 0:
                logger.info(f"Transitioned {expired} stale pending jobs (>7 days) to 'expired_dismissed'.")

        pending = self.db.get_pending_digest_jobs(
            min_score=self.settings.digest_score_threshold,
            limit=self.settings.digest_max_items,
        )
        if not pending:
            logger.info("No pending qualifying jobs for daily digest.")
            return 0

        date_str = datetime.now(ZoneInfo(self.settings.timezone)).strftime("%d.%m.%Y")
        if dry_run:
            logger.info(f"[DRY-RUN] Daily digest would contain {len(pending)} jobs.")
            return len(pending)

        update_notice = None
        try:
            version_info = check_jobspy_update()
            if version_info and version_info.has_update:
                update_notice = format_version_update_notice(version_info)
        except Exception as e:
            logger.debug("Failed checking JobSpy update for daily digest: %s", e)

        sent_count = self.notifier.send_daily_digest(pending, date_str=date_str, update_notice=update_notice)
        if sent_count > 0:
            self.db.mark_digest_sent([j.id for j in pending if j.id is not None])
        return len(pending)

    def start(self) -> None:
        self.scheduler = BlockingScheduler(timezone=self.settings.timezone)

        # Scrape interval
        self.scheduler.add_job(
            self.run_pipeline_cycle,
            trigger=IntervalTrigger(hours=self.settings.scrape_interval_hours),
            id="scrape_pipeline",
            name="Scrape & Evaluate Pipeline",
        )

        # Daily digest cron at the configured local time (settings.timezone)
        hour, minute = self.settings.digest_time_cet.split(":")
        self.scheduler.add_job(
            self.run_daily_digest_cycle,
            trigger=CronTrigger(hour=int(hour), minute=int(minute)),
            id="daily_digest",
            name="Morning Digest",
        )

        def shutdown(signum, frame):
            logger.info("Received termination signal. Shutting down daemon gracefully...")
            self.request_shutdown()
            if self.scheduler and self.scheduler.running:
                self.scheduler.shutdown(wait=False)
            sys.exit(0)

        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGTERM, shutdown)

        try:
            version_info = check_jobspy_update()
            if version_info and version_info.has_update:
                logger.info(
                    "New JobSpy version available: v%s (installed: v%s). Run 'pip install -U python-jobspy' to update.",
                    version_info.latest_version,
                    version_info.installed_version,
                )
        except Exception as e:
            logger.debug("Failed startup JobSpy version check: %s", e)

        if self.evaluator is None and not (self.settings.nim_enabled or self.settings.openrouter_enabled):
            logger.warning(
                "No LLM provider is configured (set NVIDIA_API_KEY and/or OPENROUTER_API_KEY with "
                "OPENROUTER_MODELS): scraping will run but evaluation will fail until one is set."
            )

        logger.info("Running initial startup scraping cycle...")
        try:
            self.run_pipeline_cycle()
        except ConfigurationError as exc:
            logger.error(
                "Startup cycle could not evaluate jobs: %s. Scraping continues on schedule; "
                "queued jobs stay unprocessed until an LLM provider is configured.",
                exc,
            )

        logger.info(
            f"Daemon started. Scrape every {self.settings.scrape_interval_hours}h, "
            f"Digest at {self.settings.digest_time_cet} {self.settings.timezone}."
        )
        self.scheduler.start()
