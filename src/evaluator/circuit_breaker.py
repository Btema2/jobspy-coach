"""Thread-safe circuit breaker with a single half-open probe.

States: ``closed`` (calls allowed), ``open`` (calls refused until the cooldown
elapses), ``half_open`` (cooldown over; exactly one probe call is allowed).

Usage contract: ``allow()`` may CLAIM the single half-open probe. Whoever gets
``True`` must settle it with ``record_success()``, ``record_failure()`` or
``release_probe()`` in a ``try/finally`` so that an exception or an
InvalidResponse can never leave the breaker stuck half-open with a probe nobody
will ever finish::

    if not breaker.allow():
        skip_provider()
    settled = False
    try:
        result = call_provider()
        breaker.record_success(); settled = True
    except TransientProviderError:
        breaker.record_failure(); settled = True
        raise
    finally:
        if not settled:
            breaker.release_probe()
"""

import threading
import time
from typing import Callable

CLOSED = "closed"
OPEN = "open"
HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 3,
        cooldown_seconds: float = 600,
        time_fn: Callable[[], float] = time.monotonic,
    ):
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self._time = time_fn
        self._lock = threading.Lock()
        self._failures = 0
        self._opened_at = 0.0
        self._open = False
        self._probe_in_flight = False

    def _cooldown_elapsed(self) -> bool:
        return self._time() - self._opened_at >= self.cooldown_seconds

    @property
    def state(self) -> str:
        """Current state (read-only; does not claim the probe)."""
        with self._lock:
            if not self._open:
                return CLOSED
            if self._probe_in_flight or self._cooldown_elapsed():
                return HALF_OPEN
            return OPEN

    def allow(self) -> bool:
        """Return True if a call may proceed. In half-open, atomically claims the one probe."""
        with self._lock:
            if not self._open:
                return True
            if self._probe_in_flight or not self._cooldown_elapsed():
                return False
            self._probe_in_flight = True
            return True

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._open = False
            self._probe_in_flight = False

    def record_failure(self) -> None:
        with self._lock:
            if self._open:
                if self._probe_in_flight:
                    # Failed half-open probe: reopen with a fresh cooldown.
                    self._probe_in_flight = False
                    self._opened_at = self._time()
                # Straggler failure while plainly open: ignore, keep the cooldown.
                return
            self._failures += 1
            if self._failures >= self.failure_threshold:
                self._trip()

    def release_probe(self) -> None:
        """Give back a claimed probe without recording an outcome (state stays half-open)."""
        with self._lock:
            self._probe_in_flight = False

    def open_immediately(self) -> None:
        """Open now regardless of the failure count (e.g. 401/403 bad key)."""
        with self._lock:
            self._trip()

    def _trip(self) -> None:
        self._open = True
        self._opened_at = self._time()
        self._probe_in_flight = False
        self._failures = 0
