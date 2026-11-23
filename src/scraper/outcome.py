from dataclasses import dataclass
from typing import List, Literal, Optional

from src.storage.models import JobPosting

ScrapeStatus = Literal["ok", "empty", "failed", "skipped"]
STATUS_OK: ScrapeStatus = "ok"
STATUS_EMPTY: ScrapeStatus = "empty"
STATUS_FAILED: ScrapeStatus = "failed"
STATUS_SKIPPED: ScrapeStatus = "skipped"


@dataclass
class ScrapeOutcome:
    """Result of one (query, board) scrape call.

    ``rows`` is what the board returned (before the pre-filter and before the LinkedIn missing-description
    skip), ``prefiltered`` how many of ``postings`` were flagged by the pre-filter and ``skipped_empty`` how
    many LinkedIn rows were skipped for a missing description.
    """

    postings: List[JobPosting]
    rows: int
    status: ScrapeStatus
    error: Optional[str] = None
    skip_reason: Optional[str] = None
    prefiltered: int = 0
    skipped_empty: int = 0
