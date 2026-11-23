import threading
import time

import pytest

from src.evaluator.errors import RateLimited, RateLimitWaitExceeded, StopRequested
from src.evaluator.rate_limiter import (
    BOUNDARY_MARGIN_SECONDS,
    PRIORITY_YIELD_SECONDS,
    SlidingWindowRateLimiter,
)


class FakeClock:
    """Deterministic clock; ``wait`` advances time instead of sleeping."""

    def __init__(self, start: float = 1000.0):
        self.now = start
        self.waits = []

    def __call__(self) -> float:
        return self.now

    def wait(self, seconds: float) -> bool:
        self.waits.append(seconds)
        self.now += seconds
        return False


def make(max_calls=30, period=60.0, clock=None, **kwargs):
    clock = clock or FakeClock()
    limiter = SlidingWindowRateLimiter(
        max_calls, period, time_fn=clock, wait_fn=clock.wait, **kwargs
    )
    return limiter, clock


@pytest.mark.parametrize("bad", [0, -1])
def test_rejects_max_calls_below_one(bad):
    with pytest.raises(ValueError):
        SlidingWindowRateLimiter(bad)


def test_rejects_non_positive_period():
    with pytest.raises(ValueError):
        SlidingWindowRateLimiter(5, period=0)


def test_calls_under_the_cap_do_not_wait():
    limiter, clock = make(max_calls=5)
    for _ in range(5):
        limiter.acquire()
    assert clock.waits == []


def test_hundred_acquires_never_exceed_cap_in_any_trailing_window():
    limiter, clock = make(max_calls=30, period=60.0)
    stamps = []
    for _ in range(100):
        stamps.append(limiter.acquire())
        clock.now += 0.125  # exactly representable; callers arrive faster than the cap allows
    assert len(stamps) == 100
    for i in range(30, len(stamps)):
        # the 31st call after any call must be at least a full period later
        assert stamps[i] - stamps[i - 30] > 60.0
    # and the invariant, stated directly over every trailing window
    for t in stamps:
        in_window = [s for s in stamps if t - 60.0 <= s <= t]  # CLOSED trailing window
        assert len(in_window) <= 30


def test_acquire_waits_only_as_long_as_needed():
    limiter, clock = make(max_calls=2, period=10.0)
    limiter.acquire()
    clock.now += 4.0
    limiter.acquire()
    limiter.acquire()  # must wait until the first call leaves the window
    assert clock.waits == [pytest.approx(6.0 + BOUNDARY_MARGIN_SECONDS)]


def test_entry_exactly_one_period_old_still_counts():
    """Strict boundary: a stamp exactly `period` old is not pruned until the margin has passed."""
    limiter, clock = make(max_calls=1, period=10.0)
    first = limiter.acquire()
    clock.now = first + 10.0  # exactly one period later: the closed window still holds `first`
    second = limiter.acquire()
    assert clock.waits, "must wait for the safety margin instead of reusing the slot at the boundary"
    assert second - first == pytest.approx(10.0 + BOUNDARY_MARGIN_SECONDS)
    assert len(clock.waits) == 1  # no spinning: the computed wait includes the margin


def test_pause_for_delays_acquire():
    limiter, clock = make(max_calls=5)
    limiter.pause_for(15.0)
    start = clock.now
    limiter.acquire()
    assert clock.now - start == pytest.approx(15.0)
    assert clock.waits == [pytest.approx(15.0)]


def test_pause_for_never_shortens_an_existing_pause():
    limiter, clock = make(max_calls=5)
    limiter.pause_for(30.0)
    limiter.pause_for(5.0)
    start = clock.now
    limiter.acquire()
    assert clock.now - start == pytest.approx(30.0)


def test_pause_for_expires():
    limiter, clock = make(max_calls=5)
    limiter.pause_for(5.0)
    clock.now += 6.0
    limiter.acquire()
    assert clock.waits == []


def test_acquire_raises_if_stop_already_set_and_consumes_no_slot():
    stop = threading.Event()
    stop.set()
    limiter, clock = make(max_calls=1, stop_event=stop)
    with pytest.raises(StopRequested):
        limiter.acquire()
    stop.clear()
    limiter.acquire()
    assert clock.waits == []  # the slot was still free


def test_acquire_raises_if_stop_set_during_wait():
    stop = threading.Event()
    clock = FakeClock()

    def waiting_sets_stop(seconds):
        clock.waits.append(seconds)
        stop.set()
        return True

    limiter = SlidingWindowRateLimiter(
        1, 60.0, time_fn=clock, stop_event=stop, wait_fn=waiting_sets_stop
    )
    limiter.acquire()
    with pytest.raises(StopRequested):
        limiter.acquire()
    assert len(clock.waits) == 1


def test_lock_is_not_held_while_waiting():
    """pause_for from another thread must be possible while a waiter is blocked."""
    clock = FakeClock()
    in_wait = threading.Event()
    release = threading.Event()

    def blocking_wait(seconds):
        in_wait.set()
        release.wait(timeout=5)
        clock.now += seconds
        return False

    limiter = SlidingWindowRateLimiter(1, 60.0, time_fn=clock, wait_fn=blocking_wait)
    limiter.acquire()
    worker = threading.Thread(target=limiter.acquire)
    worker.start()
    assert in_wait.wait(timeout=5)
    pauser = threading.Thread(target=limiter.pause_for, args=(1.0,))
    pauser.start()
    pauser.join(timeout=2)
    assert not pauser.is_alive(), "pause_for blocked while a waiter held the lock"
    release.set()
    worker.join(timeout=5)
    assert not worker.is_alive()


def test_real_threads_respect_window_invariant():
    max_calls, period, n_threads = 5, 0.2, 12
    limiter = SlidingWindowRateLimiter(max_calls, period)
    stamps = []
    stamps_lock = threading.Lock()

    def worker():
        stamp = limiter.acquire()
        with stamps_lock:
            stamps.append(stamp)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert all(not t.is_alive() for t in threads)
    assert len(stamps) == n_threads
    stamps.sort()
    for i in range(max_calls, len(stamps)):
        assert stamps[i] - stamps[i - max_calls] > period


def test_blocked_waiter_wakes_promptly_when_stop_is_set():
    stop = threading.Event()
    limiter = SlidingWindowRateLimiter(1, period=3600.0, stop_event=stop)
    limiter.acquire()  # fill the window; next acquire would wait an hour
    outcome = {}

    def waiter():
        started = time.monotonic()
        try:
            limiter.acquire()
        except StopRequested:
            outcome["stopped_after"] = time.monotonic() - started

    thread = threading.Thread(target=waiter)
    thread.start()
    time.sleep(0.02)  # let it block; correctness does not depend on this
    stop.set()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert outcome["stopped_after"] < 2.0


def test_default_waits_use_stop_event_not_sleep(monkeypatch):
    def forbidden(_):
        raise AssertionError("time.sleep must never be used")

    monkeypatch.setattr(time, "sleep", forbidden)
    limiter = SlidingWindowRateLimiter(1, period=0.05)
    limiter.acquire()
    limiter.acquire()  # waits ~50 ms via Event.wait


# ---------- priority and per-call stop event (chat path) ----------

def test_acquire_without_arguments_unchanged():
    limiter, clock = make(max_calls=2, period=10.0)
    limiter.acquire()
    limiter.acquire()
    limiter.acquire()
    assert clock.waits == [pytest.approx(10.0 + BOUNDARY_MARGIN_SECONDS)]
    assert limiter.priority_waiters == 0


class Gates:
    """Per-thread blocking ``wait_fn``: a thread in ``wait`` stays blocked until the test releases it."""

    def __init__(self):
        self._sems = {}
        self._lock = threading.Lock()
        self.waits = {}

    def _sem(self, name):
        with self._lock:
            return self._sems.setdefault(name, threading.Semaphore(0))

    def wait(self, seconds):
        name = threading.current_thread().name
        with self._lock:
            self.waits.setdefault(name, []).append(seconds)
        self._sem(name).acquire(timeout=5)
        return False

    def release(self, name):
        self._sem(name).release()

    def wait_until_waiting(self, name, count=1):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            with self._lock:
                if len(self.waits.get(name, [])) >= count:
                    return
            time.sleep(0.005)
        raise AssertionError(f"{name} never started waiting")


def test_priority_acquire_beats_saturated_batch():
    clock = FakeClock()
    gates = Gates()
    limiter = SlidingWindowRateLimiter(2, 10.0, time_fn=clock, wait_fn=gates.wait)
    limiter.acquire()  # stamp at 1000
    clock.now = 1005.0
    limiter.acquire()  # stamp at 1005; the window is now full
    clock.now = 1006.0
    order = []
    order_lock = threading.Lock()

    def run(priority):
        stamp = limiter.acquire(priority=priority)
        with order_lock:
            order.append(("priority" if priority else "batch", stamp))

    batch = threading.Thread(target=run, args=(False,), name="batch", daemon=True)
    batch.start()
    gates.wait_until_waiting("batch")
    chat = threading.Thread(target=run, args=(True,), name="chat", daemon=True)
    chat.start()
    gates.wait_until_waiting("chat")
    assert limiter.priority_waiters == 1

    clock.now = 1010.1  # the oldest stamp has aged out: exactly one slot is free
    gates.release("batch")  # the batch caller wakes first, while the priority caller is still registered
    gates.wait_until_waiting("batch", count=2)
    assert gates.waits["batch"][1] >= PRIORITY_YIELD_SECONDS  # it yielded instead of taking the free slot
    assert order == []
    assert limiter.priority_waiters == 1

    gates.release("chat")  # the priority caller takes the free slot
    chat.join(timeout=5)
    assert not chat.is_alive()
    assert [name for name, _ in order] == ["priority"]

    clock.now = 1015.2  # the second stamp ages out: the following slot goes to the batch caller
    gates.release("batch")
    batch.join(timeout=5)
    assert not batch.is_alive()
    assert [name for name, _ in order] == ["priority", "batch"]
    assert order[0][1] == pytest.approx(1010.1)
    assert order[1][1] == pytest.approx(1015.2)
    assert limiter.priority_waiters == 0


def test_batch_yields_while_priority_waits():
    clock = FakeClock()
    limiter = SlidingWindowRateLimiter(5, 60.0, time_fn=clock)

    def waiting(seconds):
        clock.waits.append(seconds)
        clock.now += seconds
        with limiter._lock:
            limiter._priority_waiters = 0  # the priority caller finished while the batch caller waited
        return False

    limiter._wait = waiting
    with limiter._lock:
        limiter._priority_waiters = 1  # simulate a blocked priority caller; the slot itself is free
    assert limiter.priority_waiters == 1
    limiter.acquire()
    assert clock.waits and clock.waits[0] >= PRIORITY_YIELD_SECONDS


def test_priority_waiters_restored_after_stop():
    stop = threading.Event()
    stop.set()
    limiter, _ = make(max_calls=1, stop_event=stop)
    with pytest.raises(StopRequested):
        limiter.acquire(priority=True)
    assert limiter.priority_waiters == 0


def test_per_call_stop_event_independent_of_limiter_event():
    limiter_stop = threading.Event()
    limiter_stop.set()
    limiter = SlidingWindowRateLimiter(1, 3600.0, stop_event=limiter_stop)
    with pytest.raises(StopRequested):
        limiter.acquire()
    clear = threading.Event()
    limiter.acquire(stop_event=clear)  # the batch event is set but this caller's own event is not
    outcome = {}

    def waiter():
        try:
            limiter.acquire(stop_event=clear)
        except StopRequested:
            outcome["stopped"] = True

    thread = threading.Thread(target=waiter)
    thread.start()
    time.sleep(0.02)
    clear.set()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert outcome == {"stopped": True}


def test_max_wait_raises_rate_limit_wait_exceeded():
    limiter, clock = make(max_calls=1, period=50.0)
    limiter.acquire()
    with pytest.raises(RateLimitWaitExceeded) as exc_info:
        limiter.acquire(max_wait=30.0)
    assert isinstance(exc_info.value, RateLimited)
    assert exc_info.value.retry_after is None
    assert clock.waits == []  # no wait that would cross the limit was started


def test_max_wait_allows_a_wait_inside_the_limit():
    limiter, clock = make(max_calls=1, period=10.0)
    limiter.acquire()
    limiter.acquire(max_wait=30.0)
    assert len(clock.waits) == 1


def test_pause_for_applies_to_priority_callers():
    limiter, clock = make(max_calls=5)
    limiter.pause_for(15.0)
    start = clock.now
    limiter.acquire(priority=True)
    assert clock.now - start == pytest.approx(15.0)
    limiter.pause_for(40.0)
    with pytest.raises(RateLimitWaitExceeded):
        limiter.acquire(priority=True, max_wait=30.0)
    assert limiter.priority_waiters == 0
