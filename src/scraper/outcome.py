from dataclasses import dataclass
from typing import List, Optional

from src.storage.models import JobPosting

# status values: "ok", "empty", "failed", "skipped"


@dataclass
class ScrapeOutcome:
    """Result of one (query, board) scrape call.

    ``rows`` is what the board returned (before the pre-filter and before the LinkedIn missing-description
    skip), ``prefiltered`` how many of ``postings`` were flagged by the pre-filter and ``skipped_empty`` how
    many LinkedIn rows were skipped for a missing description.
    """

    postings: List[JobPosting]
    rows: int
    status: str
    error: Optional[str] = None
    skip_reason: Optional[str] = None
    prefiltered: int = 0
    skipped_empty: int = 0
