import hashlib
import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

from src.storage.models import JobEvaluationResult, JobPosting, ScrapeRun


def compute_dedup_hash(company: str, title: str, location: str) -> str:
    """Computes deterministic SHA-256 hash over normalized canonical fields."""
    def normalize(text: Optional[str]) -> str:
        if not text:
            return ""
        s = text.lower()
        # Strip corporate entity suffixes
        s = re.sub(r"\b(sp\.\s*z\s*o\.?\s*o\.?|sp\s*z\s*o\s*o|inc\.?|llc|ltd|gmbh|sa|corp)(?=\b|\s|$)", "", s)
        # Strip non-alphanumeric punctuation except spaces
        s = re.sub(r"[^\w\s]", " ", s)
        # Condense whitespace
        return " ".join(s.split())

    norm_company = normalize(company)
    norm_title = normalize(title)
    norm_location = normalize(location)

    payload = f"{norm_company}:{norm_title}:{norm_location}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _parse_dt(val: Any) -> Optional[datetime]:
    if not val:
        return None
    if isinstance(val, datetime):
        return val
    try:
        return datetime.fromisoformat(val)
    except (ValueError, TypeError):
        return None


class Database:
    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA busy_timeout = 5000;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dedup_hash TEXT UNIQUE NOT NULL,
                site TEXT NOT NULL,
                external_id TEXT,
                job_url TEXT NOT NULL,
                job_url_direct TEXT,
                title TEXT NOT NULL,
                company TEXT NOT NULL,
                location TEXT NOT NULL,
                job_type TEXT,
                date_posted TEXT,
                salary_min REAL,
                salary_max REAL,
                salary_currency TEXT,
                salary_interval TEXT,
                description TEXT NOT NULL,
                is_remote INTEGER NOT NULL DEFAULT 0,

                pre_filtered INTEGER NOT NULL DEFAULT 0,
                pre_filter_reason TEXT,

                eval_status TEXT NOT NULL DEFAULT 'unprocessed',
                eval_score REAL,
                eval_is_wow INTEGER DEFAULT 0,
                eval_summary_uk TEXT,
                eval_reasoning TEXT,
                eval_languages TEXT,
                eval_pros TEXT,
                eval_cons TEXT,
                eval_error TEXT,
                evaluated_at TEXT,

                notification_status TEXT NOT NULL DEFAULT 'pending',
                instant_sent_at TEXT,
                digest_sent_at TEXT,

                created_at TEXT NOT NULL DEFAULT (DATETIME('now')),
                updated_at TEXT NOT NULL DEFAULT (DATETIME('now'))
            );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_dedup_hash ON jobs(dedup_hash);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_eval_status ON jobs(eval_status);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_notif_status ON jobs(notification_status);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_eval_score ON jobs(eval_score);")

            conn.execute("""
            CREATE TABLE IF NOT EXISTS scrape_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_time TEXT NOT NULL DEFAULT (DATETIME('now')),
                board TEXT NOT NULL,
                query TEXT NOT NULL,
                jobs_found INTEGER NOT NULL DEFAULT 0,
                new_jobs_saved INTEGER NOT NULL DEFAULT 0,
                duration_sec REAL NOT NULL DEFAULT 0.0,
                status TEXT NOT NULL,
                error_message TEXT
            );
            """)
            conn.commit()

    def save_job(self, job: JobPosting) -> Tuple[int, bool]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM jobs WHERE dedup_hash = ?", (job.dedup_hash,))
            row = cursor.fetchone()
            if row:
                job.id = row["id"]
                return row["id"], False

            cursor.execute("""
            INSERT INTO jobs (
                dedup_hash, site, external_id, job_url, job_url_direct,
                title, company, location, job_type, date_posted,
                salary_min, salary_max, salary_currency, salary_interval,
                description, is_remote, pre_filtered, pre_filter_reason,
                eval_status, eval_score, eval_is_wow, eval_summary_uk,
                eval_reasoning, eval_languages, eval_pros, eval_cons,
                eval_error, evaluated_at,
                notification_status, instant_sent_at, digest_sent_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                job.dedup_hash, job.site, job.external_id, job.job_url, job.job_url_direct,
                job.title, job.company, job.location, job.job_type, job.date_posted,
                job.salary_min, job.salary_max, job.salary_currency, job.salary_interval,
                job.description, int(job.is_remote), int(job.pre_filtered), job.pre_filter_reason,
                job.eval_status, job.eval_score, int(job.eval_is_wow), job.eval_summary_uk,
                job.eval_reasoning,
                json.dumps(job.eval_languages, ensure_ascii=False) if job.eval_languages else None,
                json.dumps(job.eval_pros, ensure_ascii=False) if job.eval_pros else None,
                json.dumps(job.eval_cons, ensure_ascii=False) if job.eval_cons else None,
                job.eval_error,
                job.evaluated_at.isoformat() if job.evaluated_at else None,
                job.notification_status,
                job.instant_sent_at.isoformat() if job.instant_sent_at else None,
                job.digest_sent_at.isoformat() if job.digest_sent_at else None,
            ))
            conn.commit()
            new_id = cursor.lastrowid
            job.id = new_id
            return new_id, True

    def get_job_by_id(self, job_id: int) -> Optional[JobPosting]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_job(row)

    def get_job_by_hash(self, dedup_hash: str) -> Optional[JobPosting]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM jobs WHERE dedup_hash = ?", (dedup_hash,))
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_job(row)

    def get_unprocessed_jobs(self, limit: int = 50) -> List[JobPosting]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM jobs WHERE eval_status = 'unprocessed' AND pre_filtered = 0 ORDER BY id ASC LIMIT ?",
                (limit,)
            )
            return [self._row_to_job(r) for r in cursor.fetchall()]

    def update_evaluation(self, job_id: int, eval_result: JobEvaluationResult) -> None:
        now_str = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute("""
            UPDATE jobs SET
                eval_status = 'evaluated',
                eval_score = ?,
                eval_is_wow = ?,
                eval_summary_uk = ?,
                eval_reasoning = ?,
                eval_languages = ?,
                eval_pros = ?,
                eval_cons = ?,
                evaluated_at = ?,
                updated_at = ?
            WHERE id = ?
            """, (
                eval_result.score, int(eval_result.is_wow), eval_result.summary_uk, eval_result.reasoning,
                json.dumps(eval_result.language_requirements, ensure_ascii=False),
                json.dumps(eval_result.pros, ensure_ascii=False),
                json.dumps(eval_result.cons, ensure_ascii=False),
                now_str, now_str, job_id
            ))
            conn.commit()

    def update_eval_failure(self, job_id: int, error_message: str) -> None:
        now_str = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute("""
            UPDATE jobs SET
                eval_status = 'failed',
                eval_error = ?,
                updated_at = ?
            WHERE id = ?
            """, (error_message, now_str, job_id))
            conn.commit()

    def get_pending_digest_jobs(self, min_score: float = 6.0, limit: int = 10) -> List[JobPosting]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT * FROM jobs
            WHERE eval_status = 'evaluated'
              AND notification_status = 'pending'
              AND eval_score >= ?
            ORDER BY eval_score DESC, id ASC
            LIMIT ?
            """, (min_score, limit))
            return [self._row_to_job(r) for r in cursor.fetchall()]

    def mark_instant_sent(self, job_id: int) -> None:
        now_str = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute("""
            UPDATE jobs SET
                notification_status = 'instant_sent',
                instant_sent_at = ?,
                updated_at = ?
            WHERE id = ?
            """, (now_str, now_str, job_id))
            conn.commit()

    def mark_digest_sent(self, job_ids: List[int]) -> None:
        if not job_ids:
            return
        now_str = datetime.now(timezone.utc).isoformat()
        placeholders = ",".join("?" for _ in job_ids)
        with self._get_connection() as conn:
            conn.execute(f"""
            UPDATE jobs SET
                notification_status = 'digest_sent',
                digest_sent_at = ?,
                updated_at = ?
            WHERE id IN ({placeholders})
            """, [now_str, now_str] + job_ids)
            conn.commit()

    def record_scrape_run(
        self,
        board: str,
        query: str,
        jobs_found: int,
        new_jobs_saved: int,
        duration_sec: float,
        status: str,
        error_message: Optional[str] = None
    ) -> int:
        now_str = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO scrape_runs (
                run_time, board, query, jobs_found, new_jobs_saved,
                duration_sec, status, error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                now_str, board, query, jobs_found, new_jobs_saved,
                duration_sec, status, error_message
            ))
            conn.commit()
            return cursor.lastrowid

    def expire_stale_pending_jobs(self, days: int = 7) -> int:
        now_str = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            UPDATE jobs SET
                notification_status = 'expired_dismissed',
                updated_at = ?
            WHERE notification_status = 'pending'
              AND eval_status = 'evaluated'
              AND datetime(created_at) < datetime('now', '-' || ? || ' days')
            """, (now_str, days))
            conn.commit()
            return cursor.rowcount

    def get_stats(self) -> Dict[str, int]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            total = cursor.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
            prefiltered = cursor.execute("SELECT COUNT(*) FROM jobs WHERE pre_filtered = 1").fetchone()[0]
            evaluated = cursor.execute("SELECT COUNT(*) FROM jobs WHERE eval_status = 'evaluated'").fetchone()[0]
            failed = cursor.execute("SELECT COUNT(*) FROM jobs WHERE eval_status = 'failed'").fetchone()[0]
            instant = cursor.execute("SELECT COUNT(*) FROM jobs WHERE notification_status = 'instant_sent'").fetchone()[0]
            digest = cursor.execute("SELECT COUNT(*) FROM jobs WHERE notification_status = 'digest_sent'").fetchone()[0]
            pending_digest = cursor.execute("SELECT COUNT(*) FROM jobs WHERE eval_status = 'evaluated' AND notification_status = 'pending' AND eval_score >= 6.0").fetchone()[0]
            return {
                "total_jobs": total,
                "prefiltered_jobs": prefiltered,
                "evaluated_jobs": evaluated,
                "failed_jobs": failed,
                "instant_sent": instant,
                "digest_sent": digest,
                "pending_digest": pending_digest
            }

    def _row_to_job(self, row: sqlite3.Row) -> JobPosting:
        return JobPosting(
            id=row["id"],
            dedup_hash=row["dedup_hash"],
            site=row["site"],
            external_id=row["external_id"],
            job_url=row["job_url"],
            job_url_direct=row["job_url_direct"],
            title=row["title"],
            company=row["company"],
            location=row["location"],
            job_type=row["job_type"],
            date_posted=row["date_posted"],
            salary_min=row["salary_min"],
            salary_max=row["salary_max"],
            salary_currency=row["salary_currency"],
            salary_interval=row["salary_interval"],
            description=row["description"],
            is_remote=bool(row["is_remote"]),
            pre_filtered=bool(row["pre_filtered"]),
            pre_filter_reason=row["pre_filter_reason"],
            eval_status=row["eval_status"],
            eval_score=row["eval_score"],
            eval_is_wow=bool(row["eval_is_wow"]),
            eval_summary_uk=row["eval_summary_uk"],
            eval_reasoning=row["eval_reasoning"],
            eval_languages=json.loads(row["eval_languages"]) if row["eval_languages"] else [],
            eval_pros=json.loads(row["eval_pros"]) if row["eval_pros"] else [],
            eval_cons=json.loads(row["eval_cons"]) if row["eval_cons"] else [],
            eval_error=row["eval_error"],
            evaluated_at=_parse_dt(row["evaluated_at"]),
            notification_status=row["notification_status"],
            instant_sent_at=_parse_dt(row["instant_sent_at"]),
            digest_sent_at=_parse_dt(row["digest_sent_at"]),
            created_at=_parse_dt(row["created_at"]),
            updated_at=_parse_dt(row["updated_at"]),
        )
