"""Thread-safe sliding-window rate limiter.

Guarantees at most ``max_calls`` acquisitions in any closed trailing window of ``period`` seconds
within one process. Waiting is done with ``stop_event.wait`` (never ``time.sleep``)
and without holding the lock, so shutdown wakes blocked workers immediately and
other threads can still ``pause_for`` while someone waits.
"""

import threading
import time
from collections import deque
from typing import Callable, Deque, Optional

from src.evaluator.errors import RateLimitWaitExceeded, StopRequested


# A stamp is pruned only once it is strictly older than ``period`` by this margin, so the
# guarantee holds for every CLOSED trailing window of length ``period`` (clock jitter safe).
BOUNDARY_MARGIN_SECONDS = 0.05
# A non-priority caller that finds a priority caller waiting does not take a slot; it re-checks after this long.
PRIORITY_YIELD_SECONDS = 0.25


class SlidingWindowRateLimiter:
    def __init__(
        self,
        max_calls: int,
        period: float = 60.0,
        time_fn: Callable[[], float] = time.monotonic,
        stop_event: Optional[threading.Event] = None,
        wait_fn: Optional[Callable[[float], object]] = None,
    ):
        if max_calls < 1:
            raise ValueError("max_calls must be >= 1")
        if period <= 0:
            raise ValueError("period must be > 0")
        self.max_calls = max_calls
        self.period = period
        self._time = time_fn
        self.stop_event = stop_event if stop_event is not None else threading.Event()
        self._wait_fn = wait_fn
        self._wait = wait_fn if wait_fn is not None else self.stop_event.wait
        self._lock = threading.Lock()
        self._stamps: Deque[float] = deque()
        self._paused_until = 0.0
        self._priority_waiters = 0

    @property
    def priority_waiters(self) -> int:
        """Number of priority callers currently inside ``acquire`` (batch callers yield while it is above 0)."""
        with self._lock:
            return self._priority_waiters

    def pause_for(self, seconds: float) -> None:
        """Block all acquisitions for ``seconds`` (never shortens an existing pause)."""
        with self._lock:
            self._paused_until = max(self._paused_until, self._time() + max(0.0, seconds))

    def acquire(
        self,
        priority: bool = False,
        stop_event: Optional[threading.Event] = None,
        max_wait: Optional[float] = None,
    ) -> float:
        """Block until a slot is free; return the timestamp recorded for it.

        ``stop_event`` (when given) replaces the limiter's own event for this call, both for the stop
        check and for the wait (an injected ``wait_fn`` still wins, for tests). A ``priority`` caller
        is counted as a waiter while it is inside; non-priority callers never take a slot meanwhile.
        ``max_wait`` bounds the total wait from entry: a wait that would cross it raises
        ``RateLimitWaitExceeded`` instead of starting.

        Raises StopRequested if the stop event is set before or after waiting.
        """
        event = stop_event if stop_event is not None else self.stop_event
        if self._wait_fn is not None:
            wait = self._wait_fn
        elif stop_event is not None:
            wait = event.wait
        else:
            wait = self._wait
        entered = self._time()
        if priority:
            with self._lock:
                self._priority_waiters += 1
        try:
            while True:
                if event.is_set():
                    raise StopRequested("stop requested while waiting for a rate-limit slot")
                with self._lock:
                    now = self._time()
                    while self._stamps and self._stamps[0] + self.period + BOUNDARY_MARGIN_SECONDS <= now:
                        self._stamps.popleft()
                    if self._paused_until > now:
                        delay = self._paused_until - now
                    elif len(self._stamps) >= self.max_calls:
                        delay = self._stamps[0] + self.period + BOUNDARY_MARGIN_SECONDS - now
                    elif not priority and self._priority_waiters > 0:
                        delay = PRIORITY_YIELD_SECONDS  # the slot is free but a priority caller is ahead of us
                    else:
                        self._stamps.append(now)
                        return now
                    if not priority and self._priority_waiters > 0:
                        delay = max(delay, PRIORITY_YIELD_SECONDS)
                if max_wait is not None and (now - entered) + delay > max_wait:
                    raise RateLimitWaitExceeded(None, "limiter wait exceeded")
                # Lock released: wait, then re-evaluate from scratch.
                wait(delay)
        finally:
            if priority:
                with self._lock:
                    self._priority_waiters -= 1
