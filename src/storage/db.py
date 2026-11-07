import hashlib
import html
import json
import re
import sqlite3
import unicodedata
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

from src.storage.models import JobEvaluationResult, JobPosting, ScrapeRun


# --- Dedup normalization -----------------------------------------------------------------------
#
# Design: exact equality of *normalized* fields, never fuzzy matching. Each rule below only removes
# text that is noise for identity (casing, diacritics, punctuation, legal forms, regional-branch
# words, connector words, location granularity) and never text that distinguishes two roles
# (title specialisations, seniority, language variants, other cities).

_EXTRA_FOLD = str.maketrans({"ł": "l", "đ": "d", "ø": "o", "æ": "ae", "œ": "oe", "ß": "ss", "ı": "i"})

# Gender markers such as "(m/w/d)", "(K/M)", "(all genders)"; only these parentheticals are dropped from titles.
_GENDER_PARTS = {"m", "f", "w", "d", "k", "x", "h", "gn", "nb", "inni", "divers", "diverse"}
_GENDER_PAREN_RE = re.compile(r"\(\s*([a-z]+(?:\s*[/,]\s*[a-z]+)*|all genders)\s*\)")

# Legal-form suffixes, matched only at the END of a company name (as token sequences).
_LEGAL_SUFFIXES = (
    ("sp", "z", "oo"), ("sp", "z", "o", "o"), ("sp", "k"), ("sp", "j"), ("spolka", "akcyjna"),
    ("sa",), ("inc",), ("llc",), ("ltd",), ("limited",), ("gmbh",), ("ag",), ("plc",), ("corp",),
    ("bv",), ("nv",), ("srl",), ("spa",), ("co",),
)
# Regional-branch / group descriptor words, stripped from the END of a company name only
# (never when they would be the whole name).
_COMPANY_DESCRIPTORS = {"polska", "poland", "international", "global", "group", "europe", "deutschland"}

# Connector words dropped from titles. "i" (Polish "and") is only dropped BETWEEN two other words so a
# trailing level such as "Analyst I" keeps its meaning.
_TITLE_CONNECTORS = {"and", "und"}

_REMOTE_MARKERS = ("remote", "hybrid", "anywhere", "worldwide")
_LOCATION_AREA_RE = re.compile(r"\s+(?:(?:metropolitan|metro)(?:\s+(?:area|region))?|area)$")


def _fold(text: Optional[str]) -> str:
    """Lowercase, unescape HTML entities and fold diacritics."""
    if not text:
        return ""
    s = html.unescape(text).lower().translate(_EXTRA_FOLD)
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def _tokens(s: str) -> List[str]:
    """Collapse dotted abbreviations (s.a. -> sa), turn punctuation into spaces, split."""
    s = re.sub(r"\b(\w)\.", r"\1", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return s.split()


def _normalize_company(text: Optional[str]) -> str:
    s = re.sub(r"\([^)]*\)", " ", _fold(text))
    tokens = _tokens(s)
    changed = True
    while changed and len(tokens) > 1:
        changed = False
        for suffix in _LEGAL_SUFFIXES:
            n = len(suffix)
            if len(tokens) > n and tuple(tokens[-n:]) == suffix:
                tokens = tokens[:-n]
                changed = True
                break
        if not changed and len(tokens) > 1 and tokens[-1] in _COMPANY_DESCRIPTORS:
            tokens = tokens[:-1]
            changed = True
    return " ".join(tokens)


def _drop_gender_markers(match: "re.Match[str]") -> str:
    body = match.group(1)
    if body == "all genders":
        return " "
    parts = [p.strip() for p in re.split(r"[/,]", body)]
    return " " if all(p in _GENDER_PARTS for p in parts) else match.group(0)


def _normalize_title(text: Optional[str]) -> str:
    s = _GENDER_PAREN_RE.sub(_drop_gender_markers, _fold(text))
    tokens = _tokens(s)
    kept = [
        t for i, t in enumerate(tokens)
        if t not in _TITLE_CONNECTORS and not (t == "i" and 0 < i < len(tokens) - 1)
    ]
    return " ".join(kept)


def _normalize_location(text: Optional[str]) -> str:
    s = _fold(text)
    if any(marker in s for marker in _REMOTE_MARKERS):
        # "Remote Poland" vs "Remote Germany" are different postings: keep the whole string.
        return " ".join(_tokens(s))
    city = s.split(",")[0].strip()
    stripped = _LOCATION_AREA_RE.sub("", city)
    return " ".join(_tokens(stripped or city))


def compute_dedup_hash(company: str, title: str, location: str) -> str:
    """Deterministic SHA-256 over normalized company, title and location.

    Same posting scraped twice with a regional company suffix, a connector word in the title or a
    different location granularity (city vs "City Metropolitan Area" vs "City, Region, Country")
    hashes identically; different roles or different cities never do.
    """
    payload = (
        f"{_normalize_company(company)}:{_normalize_title(title)}:{_normalize_location(location)}"
    ).encode("utf-8")
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


def _ensure_column(conn: sqlite3.Connection, table: str, name: str, ddl: str) -> None:
    """Idempotently add a column; tolerant of a concurrent init_db winning the race."""
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if name in existing:
        return
    try:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")
    except sqlite3.OperationalError as exc:
        if "duplicate column" not in str(exc).lower():
            raise


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
                eval_model TEXT,
                eval_attempts INTEGER NOT NULL DEFAULT 0,

                notification_status TEXT NOT NULL DEFAULT 'pending',
                instant_sent_at TEXT,
                digest_sent_at TEXT,

                created_at TEXT NOT NULL DEFAULT (DATETIME('now')),
                updated_at TEXT NOT NULL DEFAULT (DATETIME('now'))
            );
            """)
            _ensure_column(conn, "jobs", "eval_model", "TEXT")
            _ensure_column(conn, "jobs", "eval_attempts", "INTEGER NOT NULL DEFAULT 0")
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
                eval_error, evaluated_at, eval_model, eval_attempts,
                notification_status, instant_sent_at, digest_sent_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                job.eval_model if isinstance(job.eval_model, str) else None,
                int(job.eval_attempts or 0),
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

    def update_evaluation(
        self,
        job_id: int,
        eval_result: JobEvaluationResult,
        notification_status: Optional[str] = None,
    ) -> None:
        now_str = datetime.now(timezone.utc).isoformat()
        eval_model = eval_result.eval_model if isinstance(eval_result.eval_model, str) else None
        with self._get_connection() as conn:
            if notification_status is not None:
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
                    eval_model = ?,
                    notification_status = ?,
                    evaluated_at = ?,
                    updated_at = ?
                WHERE id = ?
                """, (
                    eval_result.score, int(eval_result.is_wow), eval_result.summary_uk, eval_result.reasoning,
                    json.dumps(eval_result.language_requirements, ensure_ascii=False),
                    json.dumps(eval_result.pros, ensure_ascii=False),
                    json.dumps(eval_result.cons, ensure_ascii=False),
                    eval_model,
                    notification_status,
                    now_str, now_str, job_id
                ))
            else:
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
                    eval_model = ?,
                    evaluated_at = ?,
                    updated_at = ?
                WHERE id = ?
                """, (
                    eval_result.score, int(eval_result.is_wow), eval_result.summary_uk, eval_result.reasoning,
                    json.dumps(eval_result.language_requirements, ensure_ascii=False),
                    json.dumps(eval_result.pros, ensure_ascii=False),
                    json.dumps(eval_result.cons, ensure_ascii=False),
                    eval_model,
                    now_str, now_str, job_id
                ))
            conn.commit()

    def mark_dismissed(self, job_id: int) -> None:
        now_str = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute("""
            UPDATE jobs SET
                notification_status = 'dismissed',
                updated_at = ?
            WHERE id = ?
            """, (now_str, job_id))
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

    def record_eval_attempt(self, job_id: int, error: str, max_attempts: int) -> int:
        """Record a job-specific failed evaluation attempt.

        Increments ``eval_attempts`` and stores the last error. The job stays
        ``unprocessed`` (retried next cycle) until attempts reach ``max_attempts``,
        at which point it becomes ``failed``. Only ``unprocessed`` jobs are touched
        (an already evaluated/failed job is left unchanged). Returns the job's
        attempt count (0 if the job does not exist).
        """
        now_str = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE jobs SET
                    eval_attempts = eval_attempts + 1,
                    eval_error = ?,
                    eval_status = CASE WHEN eval_attempts + 1 >= ? THEN 'failed' ELSE eval_status END,
                    updated_at = ?
                WHERE id = ? AND eval_status = 'unprocessed'
                """,
                (error, max_attempts, now_str, job_id),
            )
            row = conn.execute("SELECT eval_attempts FROM jobs WHERE id = ?", (job_id,)).fetchone()
            return int(row["eval_attempts"]) if row else 0

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

    def delete_empty_description_jobs(self, site: str = "linkedin") -> int:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM jobs WHERE lower(site) = lower(?) AND (description IS NULL OR trim(description) = '')",
                (site,),
            )
            conn.commit()
            return cursor.rowcount

    def rehash_all(self, dry_run: bool = False) -> Dict[str, int]:
        """Recompute ``dedup_hash`` for every row with the current normalization (idempotent).

        Rows whose new hash collides are one posting seen twice. Per group the keeper is the row that
        already sent an alert (``instant_sent``/``digest_sent``), else an evaluated row, else a row that was not
        pre-filtered (so a legacy stub never beats a full posting), else the oldest (lowest id). The other rows are deleted, EXCEPT further alert-sent rows: those are never deleted;
        they keep a unique filler hash derived from the canonical one ("twins"). Returns counters:
        ``total`` rows scanned, ``rehashed`` rows whose stored hash changed, ``merged`` rows deleted,
        ``kept_twins`` alert-sent rows kept as twins. With ``dry_run`` nothing is written.
        """
        alerted = ("instant_sent", "digest_sent")
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT id, dedup_hash, company, title, location, eval_status, notification_status, pre_filtered "
                "FROM jobs ORDER BY id"
            ).fetchall()
            groups: Dict[str, List[sqlite3.Row]] = {}
            for row in rows:
                groups.setdefault(
                    compute_dedup_hash(row["company"], row["title"], row["location"]), []
                ).append(row)

            deletions: List[int] = []
            targets: Dict[int, str] = {}
            twins = 0
            for canonical, members in groups.items():
                members.sort(key=lambda r: (
                    r["notification_status"] not in alerted,
                    r["eval_status"] != "evaluated",
                    bool(r["pre_filtered"]),
                    r["id"],
                ))
                targets[members[0]["id"]] = canonical
                for extra in members[1:]:
                    if extra["notification_status"] in alerted:
                        twins += 1
                        targets[extra["id"]] = hashlib.sha256(
                            f"{canonical}:twin:{extra['id']}".encode("utf-8")
                        ).hexdigest()
                    else:
                        deletions.append(extra["id"])

            current = {r["id"]: r["dedup_hash"] for r in rows}
            changes = {i: h for i, h in targets.items() if current[i] != h}
            stats = {
                "total": len(rows),
                "rehashed": len(changes),
                "merged": len(deletions),
                "kept_twins": twins,
            }
            if dry_run:
                return stats

            for job_id in deletions:
                conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            # Two phases so a new hash can never collide with a not-yet-rewritten old hash (UNIQUE).
            for job_id in changes:
                conn.execute("UPDATE jobs SET dedup_hash = ? WHERE id = ?", (f"rehash-tmp-{job_id}", job_id))
            for job_id, new_hash in changes.items():
                conn.execute("UPDATE jobs SET dedup_hash = ? WHERE id = ?", (new_hash, job_id))
            return stats

    def get_stats(self, pending_threshold: float = 6.0) -> Dict[str, int]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            total = cursor.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
            prefiltered = cursor.execute("SELECT COUNT(*) FROM jobs WHERE pre_filtered = 1").fetchone()[0]
            evaluated = cursor.execute("SELECT COUNT(*) FROM jobs WHERE eval_status = 'evaluated'").fetchone()[0]
            failed = cursor.execute("SELECT COUNT(*) FROM jobs WHERE eval_status = 'failed'").fetchone()[0]
            instant = cursor.execute("SELECT COUNT(*) FROM jobs WHERE notification_status = 'instant_sent'").fetchone()[0]
            digest = cursor.execute("SELECT COUNT(*) FROM jobs WHERE notification_status = 'digest_sent'").fetchone()[0]
            pending_digest = cursor.execute("SELECT COUNT(*) FROM jobs WHERE eval_status = 'evaluated' AND notification_status = 'pending' AND eval_score >= ?", (pending_threshold,)).fetchone()[0]
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
            eval_model=row["eval_model"],
            eval_attempts=row["eval_attempts"] or 0,
            notification_status=row["notification_status"],
            instant_sent_at=_parse_dt(row["instant_sent_at"]),
            digest_sent_at=_parse_dt(row["digest_sent_at"]),
            created_at=_parse_dt(row["created_at"]),
            updated_at=_parse_dt(row["updated_at"]),
        )
