"""Concurrency, outcome-mapping and shutdown tests for DaemonService.evaluate_unprocessed.

All evaluators are in-process fakes; synchronisation uses events/barriers with
generous timeouts that are only reached when a test would otherwise fail.
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from src.config import Settings
from src.evaluator.chain import ProviderChainEvaluator
from src.evaluator.errors import (
    ConfigurationError,
    EvaluationRejected,
    ProvidersUnavailable,
    StopRequested,
)
from src.evaluator.providers import CompletionResult
from src.evaluator.rate_limiter import SlidingWindowRateLimiter
from src.evaluator.schemas import JobEvaluationResult
from src.scheduler.daemon import DaemonService
from src.storage.models import JobPosting

WAIT = 5.0  # upper bound for any blocking wait in these tests


def make_settings(**overrides) -> Settings:
    base = dict(
        EVAL_MAX_WORKERS=3,
        EVAL_MAX_ATTEMPTS=2,
        EVAL_BATCH_LIMIT=50,
        WOW_SCORE_THRESHOLD=8.5,
        DIGEST_SCORE_THRESHOLD=6.5,
    )
    base.update(overrides)
    return Settings(_env_file=None, **base)


def make_result(score: float, model: str = "vendor/model", **extra) -> JobEvaluationResult:
    return JobEvaluationResult(
        score=score,
        summary_uk="A sufficiently long summary.",
        reasoning="Because.",
        eval_model=model,
        **extra,
    )


def make_job(n: int, title: Optional[str] = None) -> JobPosting:
    return JobPosting(
        dedup_hash=f"hash_{n}",
        site="indeed",
        job_url=f"https://example.test/{n}",
        title=title or f"Career Coach {n}",
        company="Example Co",
        location="Remote",
        description="A coaching role description that is long enough to be meaningful.",
    )


class FakeEvaluator:
    """Stand-in for ProviderChainEvaluator: scripted per job title, thread-aware."""

    def __init__(self, behaviour: Callable[[JobPosting], JobEvaluationResult]):
        self.behaviour = behaviour
        self.stop_event = threading.Event()
        self.stop_calls = 0
        self.calls: List[str] = []
        self.threads: List[int] = []
        self._lock = threading.Lock()

    def stop(self) -> None:
        self.stop_calls += 1
        self.stop_event.set()

    def evaluate_job(self, job: JobPosting) -> JobEvaluationResult:
        with self._lock:
            self.calls.append(job.title)
            self.threads.append(threading.get_ident())
        return self.behaviour(job)


def make_daemon(tmp_path, evaluator, notifier=None, **settings_overrides) -> DaemonService:
    daemon = DaemonService(
        db_path=tmp_path / "conc.db",
        evaluator=evaluator,
        notifier=notifier if notifier is not None else MagicMock(),
    )
    daemon.settings = make_settings(**settings_overrides)
    return daemon


def seed(daemon: DaemonService, count: int) -> List[int]:
    return [daemon.db.save_job(make_job(n))[0] for n in range(1, count + 1)]


# ----- concurrency bounds / thread identity -----


def test_max_simultaneous_never_exceeds_workers(tmp_path):
    state = {"now": 0, "max": 0}
    lock = threading.Lock()
    barrier = threading.Barrier(3, timeout=WAIT)  # forces 3 simultaneous in-flight calls

    def behaviour(job):
        with lock:
            state["now"] += 1
            state["max"] = max(state["max"], state["now"])
        try:
            barrier.wait()
            time.sleep(0.005)
        finally:
            with lock:
                state["now"] -= 1
        return make_result(7.0)

    daemon = make_daemon(tmp_path, FakeEvaluator(behaviour), EVAL_MAX_WORKERS=3)
    seed(daemon, 6)

    evaluated, wow = daemon.evaluate_unprocessed(limit=10)

    assert evaluated == 6
    assert wow == 0
    assert state["max"] == 3  # reached the cap (barrier) and never exceeded it


def test_only_caller_thread_touches_db_and_telegram(tmp_path):
    fake = FakeEvaluator(lambda job: make_result(9.0))
    notifier = MagicMock()
    daemon = make_daemon(tmp_path, fake, notifier=notifier)
    seed(daemon, 4)

    touched: List[int] = []
    real_update = daemon.db.update_evaluation
    real_mark = daemon.db.mark_instant_sent

    def update(*args, **kwargs):
        touched.append(threading.get_ident())
        return real_update(*args, **kwargs)

    def mark(*args, **kwargs):
        touched.append(threading.get_ident())
        return real_mark(*args, **kwargs)

    def alert(job):
        touched.append(threading.get_ident())
        return True

    daemon.db.update_evaluation = update
    daemon.db.mark_instant_sent = mark
    notifier.send_wow_alert.side_effect = alert

    caller: Dict[str, int] = {}

    def run():
        caller["id"] = threading.get_ident()
        caller["result"] = daemon.evaluate_unprocessed(limit=10)

    t = threading.Thread(target=run)  # deliberately not the main thread
    t.start()
    t.join(WAIT)
    assert not t.is_alive()

    assert caller["result"] == (4, 4)
    assert len(touched) == 12  # 4 x (update, alert, mark)
    assert set(touched) == {caller["id"]}
    assert caller["id"] not in set(fake.threads)  # workers are different threads
    assert len(set(fake.threads)) >= 1


def test_results_handled_as_completed_not_submission_order(tmp_path):
    release_slow = threading.Event()

    def behaviour(job):
        if job.title == "Career Coach 1":  # first submitted, finishes last
            assert release_slow.wait(WAIT)
        return make_result(7.0)

    daemon = make_daemon(tmp_path, FakeEvaluator(behaviour), EVAL_MAX_WORKERS=2)
    ids = seed(daemon, 2)

    order: List[int] = []
    real_update = daemon.db.update_evaluation

    def update(job_id, *args, **kwargs):
        order.append(job_id)
        if job_id == ids[1]:
            release_slow.set()
        return real_update(job_id, *args, **kwargs)

    daemon.db.update_evaluation = update

    assert daemon.evaluate_unprocessed(limit=10) == (2, 0)
    assert order == [ids[1], ids[0]]


def test_tracked_futures_are_released_after_batch(tmp_path):
    daemon = make_daemon(tmp_path, FakeEvaluator(lambda job: make_result(7.0)))
    seed(daemon, 2)
    daemon.evaluate_unprocessed(limit=10)
    assert daemon._futures == []


# ----- outcome mapping -----


def test_providers_unavailable_stops_batch_and_keeps_jobs_unprocessed(tmp_path):
    def behaviour(job):
        if job.title == "Career Coach 1":
            raise ProvidersUnavailable("all down")
        # the other in-flight workers block until the batch is stopped
        assert fake.stop_event.wait(WAIT)
        raise StopRequested("stop")

    fake = FakeEvaluator(behaviour)
    daemon = make_daemon(tmp_path, fake, EVAL_MAX_WORKERS=2)
    ids = seed(daemon, 4)

    evaluated, wow = daemon.evaluate_unprocessed(limit=10)

    assert (evaluated, wow) == (0, 0)
    assert fake.stop_calls >= 1
    assert "Career Coach 4" not in fake.calls  # still queued -> cancelled, never started
    for job_id in ids:
        job = daemon.db.get_job_by_id(job_id)
        assert job.eval_status == "unprocessed"
        assert job.eval_attempts == 0
        assert job.eval_error is None


def test_next_batch_works_after_providers_unavailable_stop(tmp_path):
    mode = {"down": True}

    def behaviour(job):
        if mode["down"]:
            raise ProvidersUnavailable("down")
        return make_result(7.0)

    fake = FakeEvaluator(behaviour)
    daemon = make_daemon(tmp_path, fake, EVAL_MAX_WORKERS=1)
    seed(daemon, 2)

    assert daemon.evaluate_unprocessed(limit=10) == (0, 0)
    assert fake.stop_event.is_set()

    mode["down"] = False
    assert daemon.evaluate_unprocessed(limit=10) == (2, 0)  # stop event re-armed for the next cycle
    assert not fake.stop_event.is_set()


def test_evaluation_rejected_increments_attempts_until_failed(tmp_path):
    def behaviour(job):
        raise EvaluationRejected("invalid for this job")

    daemon = make_daemon(tmp_path, FakeEvaluator(behaviour), EVAL_MAX_ATTEMPTS=2)
    (job_id,) = seed(daemon, 1)

    assert daemon.evaluate_unprocessed(limit=10) == (0, 0)
    job = daemon.db.get_job_by_id(job_id)
    assert (job.eval_status, job.eval_attempts) == ("unprocessed", 1)
    assert "invalid for this job" in (job.eval_error or "")

    assert daemon.evaluate_unprocessed(limit=10) == (0, 0)
    job = daemon.db.get_job_by_id(job_id)
    assert (job.eval_status, job.eval_attempts) == ("failed", 2)

    assert daemon.evaluate_unprocessed(limit=10) == (0, 0)  # failed jobs are no longer fetched


def test_stop_requested_is_ignored(tmp_path):
    def behaviour(job):
        raise StopRequested("stop")

    daemon = make_daemon(tmp_path, FakeEvaluator(behaviour))
    (job_id,) = seed(daemon, 1)

    assert daemon.evaluate_unprocessed(limit=10) == (0, 0)
    job = daemon.db.get_job_by_id(job_id)
    assert (job.eval_status, job.eval_attempts, job.eval_error) == ("unprocessed", 0, None)


# ----- alert decision (thresholds live in the daemon) -----


@pytest.mark.parametrize(
    "score, extra, expect_wow, expect_status",
    [
        (9.5, {}, True, "instant_sent"),
        (8.5, {}, True, "instant_sent"),
        (8.4, {}, False, "pending"),
        (6.5, {}, False, "pending"),
        (6.4, {}, False, "dismissed"),
        (9.5, {"hard_reject": True}, False, "pending"),  # hard_reject can never be WOW
        (3.0, {"hard_reject": True}, False, "dismissed"),
    ],
)
def test_threshold_decisions(tmp_path, score, extra, expect_wow, expect_status):
    notifier = MagicMock()
    notifier.send_wow_alert.return_value = True
    # the model's own is_wow claim must be ignored in favour of the daemon's decision
    fake = FakeEvaluator(lambda job: make_result(score, is_wow=True, **extra))
    daemon = make_daemon(tmp_path, fake, notifier=notifier)
    (job_id,) = seed(daemon, 1)

    evaluated, wow = daemon.evaluate_unprocessed(limit=10)

    assert evaluated == 1
    assert wow == (1 if expect_wow else 0)
    assert notifier.send_wow_alert.called == expect_wow
    job = daemon.db.get_job_by_id(job_id)
    assert job.eval_status == "evaluated"
    assert job.eval_score == score
    assert bool(job.eval_is_wow) == expect_wow
    assert job.notification_status == expect_status


def test_thresholds_come_from_settings(tmp_path):
    notifier = MagicMock()
    notifier.send_wow_alert.return_value = True
    fake = FakeEvaluator(lambda job: make_result(7.0))
    daemon = make_daemon(tmp_path, fake, notifier=notifier, WOW_SCORE_THRESHOLD=7.0, DIGEST_SCORE_THRESHOLD=7.5)
    seed(daemon, 1)
    assert daemon.evaluate_unprocessed(limit=10) == (1, 1)


def test_wow_alert_failure_leaves_job_pending(tmp_path):
    notifier = MagicMock()
    notifier.send_wow_alert.return_value = False
    daemon = make_daemon(tmp_path, FakeEvaluator(lambda job: make_result(9.0)), notifier=notifier)
    (job_id,) = seed(daemon, 1)

    assert daemon.evaluate_unprocessed(limit=10) == (1, 0)
    job = daemon.db.get_job_by_id(job_id)
    assert job.eval_status == "evaluated"
    assert job.notification_status == "pending"


def test_dry_run_same_flow_without_writes_or_alerts(tmp_path):
    notifier = MagicMock()
    scores = {"Career Coach 1": 9.0, "Career Coach 2": 4.0, "Career Coach 3": 7.0}
    fake = FakeEvaluator(lambda job: make_result(scores[job.title]))
    daemon = make_daemon(tmp_path, fake, notifier=notifier)
    ids = seed(daemon, 3)

    evaluated, wow = daemon.evaluate_unprocessed(limit=10, dry_run=True)

    assert (evaluated, wow) == (3, 1)
    notifier.send_wow_alert.assert_not_called()
    for job_id in ids:
        job = daemon.db.get_job_by_id(job_id)
        assert job.eval_status == "unprocessed"
        assert job.eval_score is None


def test_dry_run_failures_do_not_touch_db(tmp_path):
    def behaviour(job):
        if job.title == "Career Coach 1":
            raise EvaluationRejected("bad")
        raise ValueError("permanent")

    daemon = make_daemon(tmp_path, FakeEvaluator(behaviour), EVAL_MAX_WORKERS=1)
    ids = seed(daemon, 2)
    assert daemon.evaluate_unprocessed(limit=10, dry_run=True) == (0, 0)
    for job_id in ids:
        job = daemon.db.get_job_by_id(job_id)
        assert (job.eval_status, job.eval_attempts) == ("unprocessed", 0)


# ----- legacy exception mapping is preserved -----


def test_other_exceptions_keep_existing_semantics(tmp_path):
    def behaviour(job):
        if job.title == "Career Coach 1":
            raise RuntimeError("HTTP 503 Service Unavailable")
        raise ValueError("Invalid prompt format syntax")

    daemon = make_daemon(tmp_path, FakeEvaluator(behaviour))
    transient_id, permanent_id = seed(daemon, 2)

    assert daemon.evaluate_unprocessed(limit=10) == (0, 0)
    assert daemon.db.get_job_by_id(transient_id).eval_status == "unprocessed"
    permanent = daemon.db.get_job_by_id(permanent_id)
    assert permanent.eval_status == "failed"
    assert "Invalid prompt format" in (permanent.eval_error or "")


def test_default_limit_comes_from_settings(tmp_path):
    fake = FakeEvaluator(lambda job: make_result(7.0))
    daemon = make_daemon(tmp_path, fake, EVAL_BATCH_LIMIT=2)
    seed(daemon, 5)
    assert daemon.evaluate_unprocessed() == (2, 0)


# ----- logging -----


def test_logs_per_job_progress_and_batch_summary_without_descriptions(tmp_path, caplog):
    fake = FakeEvaluator(lambda job: make_result(7.0, model="vendor/model-x"))
    provider = MagicMock()
    provider.name = "nim"
    provider.model = "vendor/model-x"
    fake.providers = [provider]
    daemon = make_daemon(tmp_path, fake, EVAL_MAX_WORKERS=1)
    seed(daemon, 2)

    with caplog.at_level(logging.INFO, logger="src.scheduler.daemon"):
        daemon.evaluate_unprocessed(limit=10)

    text = caplog.text
    assert "[1/2]" in text and "[2/2]" in text
    assert "Career Coach 1" in text and "vendor/model-x" in text and "nim" in text
    assert "7.0" in text
    assert "processed=2" in text
    assert "nim=2" in text
    assert "rpm" in text
    assert "coaching role description" not in text  # descriptions never logged


# ----- lazy chain construction -----


def test_constructor_does_not_build_chain_or_require_keys(tmp_path, monkeypatch):
    builder = MagicMock(side_effect=ConfigurationError("no provider"))
    monkeypatch.setattr("src.scheduler.daemon.build_chain_from_settings", builder)
    daemon = DaemonService(db_path=tmp_path / "lazy.db", notifier=MagicMock())
    builder.assert_not_called()
    assert daemon.evaluator is None


def test_evaluate_builds_chain_lazily_once_and_surfaces_config_error(tmp_path, monkeypatch):
    builder = MagicMock(side_effect=ConfigurationError("No LLM provider is configured"))
    monkeypatch.setattr("src.scheduler.daemon.build_chain_from_settings", builder)
    daemon = DaemonService(db_path=tmp_path / "lazy2.db", notifier=MagicMock())
    daemon.settings = make_settings()
    seed(daemon, 1)

    with pytest.raises(ConfigurationError, match="No LLM provider"):
        daemon.evaluate_unprocessed(limit=10)

    fake = FakeEvaluator(lambda job: make_result(7.0))
    builder.side_effect = None
    builder.return_value = fake
    assert daemon.evaluate_unprocessed(limit=10) == (1, 0)
    assert daemon.evaluate_unprocessed(limit=10) == (0, 0)
    assert builder.call_count == 2  # failed attempt + one successful build, then reused
    assert builder.call_args.kwargs["stop_event"] is daemon._stop_event


def test_no_unprocessed_jobs_does_not_need_a_chain(tmp_path, monkeypatch):
    builder = MagicMock(side_effect=ConfigurationError("no provider"))
    monkeypatch.setattr("src.scheduler.daemon.build_chain_from_settings", builder)
    daemon = DaemonService(db_path=tmp_path / "lazy3.db", notifier=MagicMock())
    assert daemon.evaluate_unprocessed(limit=10) == (0, 0)
    builder.assert_not_called()


# ----- shutdown -----


class BlockingProvider:
    """Real-limiter provider: the first call passes, later calls block in acquire()."""

    name = "blocking"
    model = "vendor/blocking"
    max_tokens = 100

    def __init__(self, stop_event: threading.Event):
        from src.evaluator.circuit_breaker import CircuitBreaker

        self.breaker = CircuitBreaker(failure_threshold=3, cooldown_seconds=600)
        self.limiter = SlidingWindowRateLimiter(max_calls=1, period=3600.0, stop_event=stop_event)
        self.waiting = threading.Semaphore(0)

    def is_available(self) -> bool:
        return True

    def complete(self, system, user, max_tokens=None):
        self.waiting.release()
        self.limiter.acquire()
        return CompletionResult(
            content='{"score": 7.0, "summary_uk": "A sufficiently long summary.", "reasoning": "r"}',
            model_used=self.model,
            latency=0.0,
            finish_reason="stop",
        )


def _blocked_batch(tmp_path):
    stop_event = threading.Event()
    provider = BlockingProvider(stop_event)
    provider.limiter.acquire()  # exhaust the only slot: every worker now blocks in acquire()
    chain = ProviderChainEvaluator([provider], stop_event=stop_event, system_prompt="x")
    daemon = make_daemon(tmp_path, chain, EVAL_MAX_WORKERS=2)
    seed(daemon, 4)

    result: Dict[str, object] = {}

    def run():
        result["value"] = daemon.evaluate_unprocessed(limit=10)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    for _ in range(2):  # both workers are inside acquire()
        assert provider.waiting.acquire(timeout=WAIT)
    return daemon, chain, t, result


def test_stop_releases_workers_blocked_in_rate_limiter(tmp_path):
    daemon, chain, t, result = _blocked_batch(tmp_path)
    assert t.is_alive()

    started = time.monotonic()
    chain.stop()
    t.join(WAIT)

    assert not t.is_alive()
    assert time.monotonic() - started < 2.0
    assert result["value"] == (0, 0)
    for job in daemon.db.get_unprocessed_jobs(limit=10):
        assert job.eval_attempts == 0


def test_sigterm_handler_stops_chain_and_cancels_queued_futures(monkeypatch, tmp_path):
    import signal

    mock_scheduler = MagicMock()
    monkeypatch.setattr("src.scheduler.daemon.BlockingScheduler", MagicMock(return_value=mock_scheduler))
    monkeypatch.setattr("src.scheduler.daemon.check_jobspy_update", MagicMock(return_value=None))
    handlers = {}
    monkeypatch.setattr("signal.signal", lambda sig, handler: handlers.__setitem__(sig, handler))
    exit_mock = MagicMock()
    monkeypatch.setattr("sys.exit", exit_mock)

    shutdown_threads: List[int] = []

    class RecordingExecutor(ThreadPoolExecutor):
        def shutdown(self, *args, **kwargs):
            shutdown_threads.append(threading.get_ident())
            return super().shutdown(*args, **kwargs)

    monkeypatch.setattr("src.scheduler.daemon.ThreadPoolExecutor", RecordingExecutor)

    daemon, chain, t, result = _blocked_batch(tmp_path)
    daemon.run_pipeline_cycle = MagicMock(return_value=(0, 0))
    daemon.start()
    mock_scheduler.running = True

    started = time.monotonic()
    handlers[signal.SIGTERM](signal.SIGTERM, None)  # runs on this (signal-handling) thread
    t.join(WAIT)

    assert chain.stop_event.is_set()
    assert not t.is_alive()
    assert time.monotonic() - started < 2.0
    assert result["value"] == (0, 0)
    # the handler must never touch the executor's shutdown lock: only the batch thread shuts it down
    assert shutdown_threads and set(shutdown_threads) == {t.ident}
    mock_scheduler.shutdown.assert_called()
    exit_mock.assert_called_once_with(0)


def test_shutdown_handler_is_safe_without_chain_or_executor(monkeypatch, tmp_path):
    import signal

    mock_scheduler = MagicMock()
    monkeypatch.setattr("src.scheduler.daemon.BlockingScheduler", MagicMock(return_value=mock_scheduler))
    monkeypatch.setattr("src.scheduler.daemon.check_jobspy_update", MagicMock(return_value=None))
    handlers = {}
    monkeypatch.setattr("signal.signal", lambda sig, handler: handlers.__setitem__(sig, handler))
    exit_mock = MagicMock()
    monkeypatch.setattr("sys.exit", exit_mock)

    daemon = DaemonService(db_path=tmp_path / "noeval.db", notifier=MagicMock())
    daemon.run_pipeline_cycle = MagicMock(return_value=(0, 0))
    daemon.start()
    assert daemon.evaluator is None and daemon._futures == []

    handlers[signal.SIGINT](signal.SIGINT, None)
    exit_mock.assert_called_once_with(0)
    assert daemon._stop_event.is_set()  # a chain built later would already be stopped


def test_shutdown_requested_during_job_fetch_is_not_undone(tmp_path):
    fake = FakeEvaluator(lambda job: make_result(7.0))
    daemon = make_daemon(tmp_path, fake, EVAL_MAX_WORKERS=1)
    ids = seed(daemon, 3)
    real_fetch = daemon.db.get_unprocessed_jobs

    def fetch(limit):
        jobs = real_fetch(limit=limit)
        daemon.request_shutdown()  # a signal lands while the caller thread is fetching
        return jobs

    daemon.db.get_unprocessed_jobs = fetch

    assert daemon.evaluate_unprocessed(limit=10) == (0, 0)
    assert fake.calls == []
    assert daemon._stop_event.is_set() and fake.stop_event.is_set()
    for job_id in ids:
        assert daemon.db.get_job_by_id(job_id).eval_status == "unprocessed"


def test_shutdown_requested_while_submitting_cancels_the_batch(tmp_path, monkeypatch):
    holder = {}

    class SignalingExecutor(ThreadPoolExecutor):
        def submit(self, *args, **kwargs):
            future = super().submit(*args, **kwargs)
            if not holder.get("fired"):
                holder["fired"] = True
                holder["daemon"].request_shutdown()  # signal lands right after the first submit
            return future

    monkeypatch.setattr("src.scheduler.daemon.ThreadPoolExecutor", SignalingExecutor)

    def behaviour(job):
        assert fake.stop_event.wait(WAIT)
        raise StopRequested("stop")

    fake = FakeEvaluator(behaviour)
    daemon = make_daemon(tmp_path, fake, EVAL_MAX_WORKERS=1)
    holder["daemon"] = daemon
    ids = seed(daemon, 3)

    assert daemon.evaluate_unprocessed(limit=10) == (0, 0)
    assert "Career Coach 3" not in fake.calls
    for job_id in ids:
        assert daemon.db.get_job_by_id(job_id).eval_status == "unprocessed"


def test_no_provider_configured_logs_warning_at_start_without_raising(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr("src.scheduler.daemon.BlockingScheduler", MagicMock())
    monkeypatch.setattr("src.scheduler.daemon.check_jobspy_update", MagicMock(return_value=None))
    monkeypatch.setattr("signal.signal", lambda sig, handler: None)
    builder = MagicMock()
    monkeypatch.setattr("src.scheduler.daemon.build_chain_from_settings", builder)

    daemon = DaemonService(db_path=tmp_path / "warn.db", notifier=MagicMock())
    daemon.settings = make_settings()  # no NVIDIA / OpenRouter keys
    daemon.run_pipeline_cycle = MagicMock(return_value=(0, 0))

    with caplog.at_level(logging.WARNING, logger="src.scheduler.daemon"):
        daemon.start()

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING and "provider" in r.getMessage().lower()]
    assert len(warnings) == 1
    builder.assert_not_called()


def test_start_survives_config_error_in_real_startup_cycle(monkeypatch, tmp_path, caplog):
    """No provider + an unprocessed job: the REAL startup cycle must not kill the daemon."""
    scheduler_cls = MagicMock()
    monkeypatch.setattr("src.scheduler.daemon.BlockingScheduler", scheduler_cls)
    monkeypatch.setattr("src.scheduler.daemon.check_jobspy_update", MagicMock(return_value=None))
    monkeypatch.setattr("src.scheduler.daemon.get_active_queries", lambda: [])
    monkeypatch.setattr("signal.signal", lambda sig, handler: None)
    builder = MagicMock(side_effect=ConfigurationError("No LLM provider is configured"))
    monkeypatch.setattr("src.scheduler.daemon.build_chain_from_settings", builder)

    daemon = DaemonService(db_path=tmp_path / "startcfg.db", scraper=MagicMock(), notifier=MagicMock())
    daemon.settings = make_settings()  # no NVIDIA / OpenRouter keys
    seed(daemon, 1)

    with caplog.at_level(logging.ERROR, logger="src.scheduler.daemon"):
        daemon.start()  # must not raise

    builder.assert_called()
    scheduler_cls.return_value.start.assert_called_once()
    errors = [r for r in caplog.records if r.levelno == logging.ERROR and "provider" in r.getMessage().lower()]
    assert len(errors) == 1


def test_configured_provider_logs_no_warning_at_start(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr("src.scheduler.daemon.BlockingScheduler", MagicMock())
    monkeypatch.setattr("src.scheduler.daemon.check_jobspy_update", MagicMock(return_value=None))
    monkeypatch.setattr("signal.signal", lambda sig, handler: None)

    daemon = DaemonService(db_path=tmp_path / "nowarn.db", notifier=MagicMock())
    daemon.settings = make_settings(NVIDIA_API_KEY="k")
    daemon.run_pipeline_cycle = MagicMock(return_value=(0, 0))

    with caplog.at_level(logging.WARNING, logger="src.scheduler.daemon"):
        daemon.start()

    assert not [r for r in caplog.records if "provider" in r.getMessage().lower()]


def test_scheduler_job_defaults_are_single_instance_and_coalesced():
    from apscheduler.schedulers.blocking import BlockingScheduler

    scheduler = BlockingScheduler()
    assert scheduler._job_defaults["max_instances"] == 1
    assert scheduler._job_defaults["coalesce"] is True
