import threading
import time
from contextlib import contextmanager
from typing import Iterator


class ScrapeCoordinator:
    """Serialises scrape requests and owns the LinkedIn pacing, so that every caller sharing one
    coordinator (the scheduled cycle and any on-demand search) observes the same cooldown.

    ``call(board)`` holds a plain (non re-entrant) lock for the whole ``with`` block; code inside it must
    not start another locked scrape. For LinkedIn the cooldown since the previous LinkedIn request is
    waited out first and the request time is recorded."""

    def __init__(self, linkedin_cooldown_seconds: float = 12.0):
        self.linkedin_cooldown_seconds = linkedin_cooldown_seconds
        self.last_linkedin_call_time = 0.0
        self._lock = threading.Lock()

    @contextmanager
    def call(self, board: str) -> Iterator[None]:
        with self._lock:
            if board.lower() == "linkedin":
                elapsed = time.time() - self.last_linkedin_call_time
                if elapsed < self.linkedin_cooldown_seconds:
                    time.sleep(self.linkedin_cooldown_seconds - elapsed)
                self.last_linkedin_call_time = time.time()
            yield
