import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import jobspy
import pandas as pd

from src.filters.pre_filter import classify_title, pre_filter_job
from src.scraper.query_catalog import DEFAULT_QUERY_CATALOG, SearchQuery, get_active_queries
from src.storage.db import compute_dedup_hash
from src.storage.models import JobPosting

logger = logging.getLogger(__name__)

TRACKING_QUERY_PARAMS = {
    "utm",
    "refid",
    "trackingid",
    "trk",
    "trkinfo",
    "source",
    "from",
    "ref",
    "midtoken",
    "originalsubdomain",
    "fbclid",
    "gclid",
}


# Boards whose python-jobspy scraper is bound to a country (the ``country_indeed`` argument).
COUNTRY_BOUND_BOARDS = {"indeed", "glassdoor"}


def build_google_search_term(query: SearchQuery, hours_old: int) -> str:
    """Builds the Google Jobs query string (same shape python-jobspy documents:
    ``<term> jobs near <location> <time range> [remote]``), or returns the per-query override."""
    if query.google_search_term:
        return query.google_search_term
    if hours_old <= 24:
        time_filter = "since yesterday"
    elif hours_old <= 72:
        time_filter = "in the last 3 days"
    elif hours_old <= 168:
        time_filter = "in the last week"
    else:
        time_filter = "in the last month"
    term = f"{query.search_term} jobs near {query.location} {time_filter}"
    if query.is_remote:
        term += " remote"
    return term


def scrape_jobs(*args, **kwargs):
    """Wrapper delegating to jobspy.scrape_jobs for dynamic monkeypatching compatibility."""
    return jobspy.scrape_jobs(*args, **kwargs)


def clean_job_url(raw_url: Optional[str]) -> str:
    """Strips ephemeral tracking query parameters and fragments while preserving functional parameters."""
    if not raw_url:
        return ""
    s = str(raw_url).strip()
    if not s or s.lower() in ("nan", "none", "null"):
        return ""
    try:
        parsed = urlparse(s)
    except ValueError:
        return ""
    if not parsed.scheme and not parsed.netloc:
        return ""
    try:
        qsl = parse_qsl(parsed.query, keep_blank_values=True)
    except Exception:
        qsl = []
    clean_qsl = [
        (k, v)
        for k, v in qsl
        if not (k.lower().startswith("utm_") or k.lower() in TRACKING_QUERY_PARAMS)
    ]
    new_query = urlencode(clean_qsl) if clean_qsl else ""
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, ""))



def _clean_str(val: Any) -> Optional[str]:
    """Cleans a string field, handling pandas NaN/None and stripping whitespace."""
    if val is None or pd.isna(val):
        return None
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", "null"):
        return None
    return s


def _clean_float(val: Any) -> Optional[float]:
    """Cleans a numeric field, handling pandas NaN/None and invalid types."""
    if val is None or pd.isna(val):
        return None
    try:
        f = float(val)
        return None if pd.isna(f) else f
    except (ValueError, TypeError):
        return None


@dataclass
class _BoardTotals:
    calls: int = 0
    rows: int = 0
    kept: int = 0
    prefiltered: int = 0
    skipped_empty: int = 0
    failed: int = 0
    seconds: float = 0.0


@dataclass
class _Cycle:
    """Progress state of one scrape cycle (counter for the ``[i/N]`` prefix plus per-board totals)."""

    total: int
    index: int = 0
    started: float = field(default_factory=time.time)
    skipped: Dict[Tuple[str, str], int] = field(default_factory=dict)
    totals: Dict[str, _BoardTotals] = field(default_factory=dict)


class JobSpyScraper:
    def __init__(
        self,
        jitter_min: float = 4.0,
        jitter_max: float = 9.0,
        results_per_query: int = 10,
        hours_old: int = 48,
        apply_prefilter: bool = True,
        enabled_boards: Optional[List[str]] = None,
        proxies: Optional[List[str]] = None,
        linkedin_cooldown_seconds: float = 12.0,
        fetch_description: bool = True,
    ):
        self.jitter_min = jitter_min
        self.jitter_max = jitter_max
        self.results_per_query = results_per_query
        self.hours_old = hours_old
        self.apply_prefilter = apply_prefilter
        self.enabled_boards = enabled_boards
        self.proxies = proxies
        self.linkedin_cooldown_seconds = linkedin_cooldown_seconds
        self.last_linkedin_call_time = 0.0
        self.fetch_description = fetch_description
        self._logged_skips: set = set()
        self._logged_board_errors: set = set()
        self._cycle: Optional[_Cycle] = None

    def _build_scrape_kwargs(self, site_name: str, query: SearchQuery) -> dict:
        board = site_name.lower()
        search_term = query.search_term
        if board == "indeed" and query.is_remote and '"' in search_term:
            # Indeed returns 0 rows for a quoted phrase combined with the remote filter (verified live).
            search_term = " ".join(search_term.replace('"', " ").split())

        kwargs: dict = {
            "site_name": [site_name],
            "search_term": search_term,
            "location": query.location,
            "results_wanted": self.results_per_query,
            "hours_old": self.hours_old,
            "is_remote": query.is_remote,
            "proxies": self.proxies,
            "fetch_description": self.fetch_description,
        }
        if board in COUNTRY_BOUND_BOARDS:
            kwargs["country_indeed"] = query.country_code
        if board == "google":
            kwargs["google_search_term"] = build_google_search_term(query, self.hours_old)
        return kwargs

    def _warn_board_errors_once(self, df: pd.DataFrame) -> None:
        """Surfaces ``df.attrs["errors"]`` (python-jobspy's per-board error) once per (board, message)."""
        errors = df.attrs.get("errors") or {}
        for board, err in errors.items():
            key = (str(board).lower(), str(err))
            if not err or key in self._logged_board_errors:
                continue
            self._logged_board_errors.add(key)
            hint = ""
            if key[0] == "google":
                hint = (
                    " Google may be serving a JavaScript/bot challenge page to this client or IP, so Google "
                    "results will stay empty until that changes; remove 'google' from site_names to stop trying."
                )
            logger.warning(
                "%s reported an error (python-jobspy: '%s').%s Logged once per board and message.",
                str(board).title(), err, hint,
            )

    # ----- planning and progress -----

    def board_skip_reason(self, site_name: str, query: SearchQuery) -> Optional[str]:
        """Why ``scrape_board`` will not request this (query, board), or None if it will."""
        board = site_name.lower()
        if self.enabled_boards is not None and board not in [b.lower() for b in self.enabled_boards]:
            return "disabled"
        if board in COUNTRY_BOUND_BOARDS and not query.country_code:
            return "no_country"
        return None

    def plan_calls(self, queries: List[SearchQuery]) -> List[Tuple[SearchQuery, str]]:
        """The (query, board) calls a scrape cycle will actually request, in order."""
        return [
            (q, site)
            for q in queries
            for site in q.site_names
            if self.board_skip_reason(site, q) is None
        ]

    def begin_cycle(self, queries: List[SearchQuery]) -> List[Tuple[SearchQuery, str]]:
        """Starts progress tracking for a cycle and returns its call plan (N for the ``[i/N]`` prefix)."""
        plan = self.plan_calls(queries)
        cycle = _Cycle(total=len(plan))
        # Boards that are configured on a query but cannot be requested never reach scrape_board on the
        # daemon/CLI path (they are not in the plan), so surface them here. Disabled boards are deliberate.
        affected: Dict[str, List[SearchQuery]] = {}
        for q in queries:
            for site in q.site_names:
                if self.board_skip_reason(site, q) == "no_country":
                    affected.setdefault(site.lower(), []).append(q)
        for board, qs in affected.items():
            cycle.skipped[(board, "no_country")] = len(qs)
            self._warn_missing_country_once(board, qs[0], len(qs))
        self._cycle = cycle
        return plan

    def _warn_missing_country_once(self, site_name: str, query: SearchQuery, n_queries: int) -> None:
        key = (site_name.lower(), "no_country")
        if key in self._logged_skips:
            logger.debug("Skipping %s for query '%s': no country_code", site_name, query.query_id)
            return
        self._logged_skips.add(key)
        # Indeed/Glassdoor search one country at a time; a pan-regional location such as "Europe"
        # returns nothing (verified live), and silently falling back to "usa" searches the wrong market.
        logger.warning(
            "Skipping %s for %d quer%s without a country_code (first: '%s', location '%s'): %s searches one "
            "country at a time. Set country_code on the query to enable it. "
            "Logged once per board; later skips are logged at DEBUG.",
            site_name, n_queries, "y" if n_queries == 1 else "ies", query.query_id, query.location, site_name,
        )

    def end_cycle(self) -> None:
        """Logs per-board totals for the cycle and stops progress tracking."""
        cycle, self._cycle = self._cycle, None
        if cycle is None:
            return
        parts = []
        for board, t in cycle.totals.items():
            part = f"{board}: {t.calls} calls, {t.rows} rows, {t.kept} kept, {t.prefiltered} pre-filtered"
            if t.skipped_empty:
                part += f", {t.skipped_empty} skipped (empty description)"
            if t.failed:
                part += f", {t.failed} failed"
            parts.append(part)
        summary = "; ".join(parts) if parts else "no boards requested"
        if cycle.skipped:
            summary += "; skipped: " + ", ".join(
                f"{board} (no country_code) x{n}" for (board, _), n in cycle.skipped.items()
            )
        logger.info("Scrape summary (%d calls, %.0fs): %s", cycle.total, time.time() - cycle.started, summary)

    def _call_prefix(self) -> str:
        c = self._cycle
        return f"[{c.index}/{c.total}] " if c else ""

    def _totals(self, board: str) -> Optional[_BoardTotals]:
        return self._cycle.totals.setdefault(board, _BoardTotals()) if self._cycle else None

    def scrape_board(self, site_name: str, query: SearchQuery) -> List[JobPosting]:
        board = site_name.lower()
        reason = self.board_skip_reason(site_name, query)
        if reason == "disabled":
            logger.debug("Skipping board '%s' as it is not in enabled_boards", site_name)
            return []
        if reason == "no_country":
            self._warn_missing_country_once(site_name, query, 1)
            return []

        # Every real request is logged here (not in a caller) so that the CLI `scrape`, the daemon and
        # scrape_query/scrape_all_queries, which all end up in scrape_board, show the same progress.
        if self._cycle:
            self._cycle.index += 1
        prefix = self._call_prefix()
        logger.info('%s%s %s "%s" %s', prefix, query.query_id, site_name, query.search_term, query.location)
        totals = self._totals(board)

        if board == "linkedin":
            elapsed = time.time() - self.last_linkedin_call_time
            if elapsed < self.linkedin_cooldown_seconds:
                time.sleep(self.linkedin_cooldown_seconds - elapsed)
            self.last_linkedin_call_time = time.time()

        t0 = time.time()
        df: Optional[pd.DataFrame] = None
        failed = False
        for attempt in range(1, 4):
            try:
                df = scrape_jobs(**self._build_scrape_kwargs(site_name, query))
                break
            except Exception as e:
                if attempt < 3:
                    backoff = (2 ** attempt) * 3
                    logger.warning(
                        f"Attempt {attempt}/3 failed scraping '{site_name}' for query '{query.query_id}': {e}. "
                        f"Retrying in {backoff}s..."
                    )
                    time.sleep(backoff)
                else:
                    logger.warning(
                        f"Error scraping board '{site_name}' for query '{query.query_id}' after 3 attempts: {e}"
                    )
                    failed = True

        if failed:
            seconds = time.time() - t0
            logger.info("%s%s %s failed after 3 attempts in %.1fs", prefix, query.query_id, site_name, seconds)
            if totals:
                totals.calls += 1
                totals.failed += 1
                totals.seconds += seconds
            return []

        if isinstance(df, pd.DataFrame):
            self._warn_board_errors_once(df)

        rows = 0 if df is None else len(df)
        postings: List[JobPosting] = []
        skipped_empty = 0
        if df is not None and not df.empty:
            postings, skipped_empty = self._rows_to_postings(df, site_name, query)

        seconds = time.time() - t0
        prefiltered = sum(1 for p in postings if p.pre_filtered)
        msg = (
            f"{prefix}{query.query_id} {site_name} done: {rows} rows returned, {len(postings)} kept after URL "
            f"cleaning ({prefiltered} pre-filtered, {len(postings) - prefiltered} to evaluate)"
        )
        if skipped_empty:
            msg += f", {skipped_empty} skipped (empty description)"
        logger.info("%s in %.1fs", msg, seconds)
        if skipped_empty:
            logger.warning(
                "LinkedIn: skipped %d posting(s) with an empty description for query '%s'. The detail fetch most "
                "likely failed (e.g. HTTP 429). They are not saved, so they are retried in a later cycle "
                "instead of being stored as dismissed and suppressed by the dedup hash.",
                skipped_empty, query.query_id,
            )
        if totals:
            totals.calls += 1
            totals.rows += rows
            totals.kept += len(postings)
            totals.prefiltered += prefiltered
            totals.skipped_empty += skipped_empty
            totals.seconds += seconds
        return postings

    def _rows_to_postings(
        self, df: pd.DataFrame, site_name: str, query: SearchQuery
    ) -> Tuple[List[JobPosting], int]:
        """Converts a python-jobspy frame to postings; returns them plus the count of LinkedIn rows skipped
        because their description is empty."""
        board = site_name.lower()
        skipped_empty = 0
        postings: List[JobPosting] = []
        for _, row in df.iterrows():
            title = _clean_str(row.get("title"))
            raw_url = _clean_str(row.get("job_url"))

            if not title or not raw_url:
                continue

            clean_url = clean_job_url(raw_url)
            if not clean_url:
                continue

            company = _clean_str(row.get("company")) or ""
            location = _clean_str(row.get("location")) or query.location
            desc = _clean_str(row.get("description")) or ""

            # LinkedIn only returns a description from a second per-job request. python-jobspy logs a failed
            # one (HTTP 429, redirect, ...) and returns the row with description=None. Persisting that row
            # would pre-filter it as DESCRIPTION_TOO_SHORT and its dedup hash would then suppress the job in
            # every later scrape, so it is skipped (retried next cycle) instead. Conservative: only LinkedIn,
            # only when details were requested, only a fully empty description (short stubs still go to the
            # pre-filter), and not when the title alone already makes the pre-filter drop it for good.
            if (
                board == "linkedin"
                and self.fetch_description
                and not desc
                and not (self.apply_prefilter and classify_title(title))
            ):
                skipped_empty += 1
                continue

            dedup_hash = compute_dedup_hash(company, title, location)

            if self.apply_prefilter:
                pf_res = pre_filter_job(title, desc)
                is_prefiltered = not pf_res.passed
                pre_filter_reason = pf_res.reason if is_prefiltered else None
                notification_status = "dismissed" if is_prefiltered else "pending"
            else:
                is_prefiltered = False
                pre_filter_reason = None
                notification_status = "pending"

            raw_direct_url = _clean_str(row.get("job_url_direct"))
            job_url_direct = clean_job_url(raw_direct_url) if raw_direct_url else None

            is_remote_val = row.get("is_remote")
            if is_remote_val is None or pd.isna(is_remote_val):
                is_remote = query.is_remote
            else:
                is_remote = bool(is_remote_val)

            date_posted_val = _clean_str(row.get("date_posted")) or ""
            external_id_val = _clean_str(row.get("id")) or ""

            postings.append(
                JobPosting(
                    dedup_hash=dedup_hash,
                    site=site_name,
                    job_url=clean_url,
                    external_id=external_id_val,
                    job_url_direct=job_url_direct,
                    job_type=_clean_str(row.get("job_type")),
                    title=title,
                    company=company,
                    location=location,
                    description=desc,
                    salary_min=_clean_float(row.get("min_amount")),
                    salary_max=_clean_float(row.get("max_amount")),
                    salary_currency=_clean_str(row.get("currency")),
                    salary_interval=_clean_str(row.get("interval")),
                    is_remote=is_remote,
                    date_posted=date_posted_val,
                    pre_filtered=is_prefiltered,
                    pre_filter_reason=pre_filter_reason,
                    notification_status=notification_status,
                )
            )

        return postings, skipped_empty

    def scrape_query(self, query: SearchQuery) -> List[JobPosting]:
        own_cycle = self._cycle is None
        if own_cycle:
            self.begin_cycle([query])
        all_postings: List[JobPosting] = []
        try:
            requested = 0
            for site in query.site_names:
                if self.board_skip_reason(site, query) is not None:
                    # Disabled / country-less boards make no request, so they get no inter-board delay either.
                    all_postings.extend(self.scrape_board(site, query))
                    continue
                if requested > 0 and self.jitter_max > 0:
                    time.sleep(random.uniform(self.jitter_min, self.jitter_max))
                requested += 1
                all_postings.extend(self.scrape_board(site, query))
        finally:
            if own_cycle:
                self.end_cycle()
        return all_postings

    def scrape_all_queries(self, queries: Optional[List[SearchQuery]] = None) -> List[JobPosting]:
        active_queries = queries or get_active_queries()
        self.begin_cycle(active_queries)
        aggregated: List[JobPosting] = []
        try:
            for q in active_queries:
                aggregated.extend(self.scrape_query(q))
        finally:
            self.end_cycle()
        return aggregated

    run_scraping_cycle = scrape_all_queries
