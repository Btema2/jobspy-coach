from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class JobEvaluationResult:
    score: float
    is_wow: bool
    summary_uk: str
    reasoning: str
    language_requirements: List[str] = field(default_factory=list)
    pros: List[str] = field(default_factory=list)
    cons: List[str] = field(default_factory=list)
    eval_model: Optional[str] = None


@dataclass
class JobPosting:
    dedup_hash: str
    site: str
    job_url: str
    title: str
    company: str
    location: str
    description: str
    id: Optional[int] = None
    external_id: Optional[str] = None
    job_url_direct: Optional[str] = None
    job_type: Optional[str] = None
    date_posted: Optional[str] = None
    salary_min: Optional[float] = None
    salary_max: Optional[float] = None
    salary_currency: Optional[str] = None
    salary_interval: Optional[str] = None
    is_remote: bool = False

    # Pre-filter
    pre_filtered: bool = False
    pre_filter_reason: Optional[str] = None

    # LLM Evaluation
    eval_status: str = "unprocessed"  # unprocessed, evaluated, failed, skipped
    eval_score: Optional[float] = None
    eval_is_wow: bool = False
    eval_summary_uk: Optional[str] = None
    eval_reasoning: Optional[str] = None
    eval_languages: List[str] = field(default_factory=list)
    eval_pros: List[str] = field(default_factory=list)
    eval_cons: List[str] = field(default_factory=list)
    eval_error: Optional[str] = None
    evaluated_at: Optional[datetime] = None

    # Notification
    notification_status: str = "pending"  # pending, instant_sent, digest_sent, dismissed, expired_dismissed
    instant_sent_at: Optional[datetime] = None
    digest_sent_at: Optional[datetime] = None

    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class ScrapeRun:
    board: str
    query: str
    jobs_found: int
    new_jobs_saved: int
    duration_sec: float
    status: str
    id: Optional[int] = None
    error_message: Optional[str] = None
    run_time: Optional[datetime] = None
