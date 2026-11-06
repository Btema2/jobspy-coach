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

from src.evaluator.errors import StopRequested


# A stamp is pruned only once it is strictly older than ``period`` by this margin, so the
# guarantee holds for every CLOSED trailing window of length ``period`` (clock jitter safe).
BOUNDARY_MARGIN_SECONDS = 0.05


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
        self._wait = wait_fn if wait_fn is not None else self.stop_event.wait
        self._lock = threading.Lock()
        self._stamps: Deque[float] = deque()
        self._paused_until = 0.0

    def pause_for(self, seconds: float) -> None:
        """Block all acquisitions for ``seconds`` (never shortens an existing pause)."""
        with self._lock:
            self._paused_until = max(self._paused_until, self._time() + max(0.0, seconds))

    def acquire(self) -> float:
        """Block until a slot is free; return the timestamp recorded for it.

        Raises StopRequested if the stop event is set before or after waiting.
        """
        while True:
            if self.stop_event.is_set():
                raise StopRequested("stop requested while waiting for a rate-limit slot")
            with self._lock:
                now = self._time()
                while self._stamps and self._stamps[0] + self.period + BOUNDARY_MARGIN_SECONDS <= now:
                    self._stamps.popleft()
                if self._paused_until > now:
                    delay = self._paused_until - now
                elif len(self._stamps) >= self.max_calls:
                    delay = self._stamps[0] + self.period + BOUNDARY_MARGIN_SECONDS - now
                else:
                    self._stamps.append(now)
                    return now
            # Lock released: wait, then re-evaluate from scratch.
            self._wait(delay)
