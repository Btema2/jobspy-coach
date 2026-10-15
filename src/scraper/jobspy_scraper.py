import logging
import random
import time
from typing import List, Optional
from urllib.parse import urlparse, urlunparse

import pandas as pd
import jobspy

from src.filters.pre_filter import pre_filter_job
from src.scraper.query_catalog import DEFAULT_QUERY_CATALOG, SearchQuery
from src.storage.db import compute_dedup_hash
from src.storage.models import JobPosting

logger = logging.getLogger(__name__)


def scrape_jobs(*args, **kwargs):
    """Wrapper delegating to jobspy.scrape_jobs for dynamic monkeypatching compatibility."""
    return jobspy.scrape_jobs(*args, **kwargs)



def clean_job_url(raw_url: Optional[str]) -> str:
    """Strips query tracking parameters and fragments from job URLs."""
    if not raw_url:
        return ""
    parsed = urlparse(raw_url)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))


class JobSpyScraper:
    def __init__(
        self,
        jitter_min: float = 4.0,
        jitter_max: float = 9.0,
        results_per_query: int = 25,
        hours_old: int = 48,
        apply_prefilter: bool = True,
    ):
        self.jitter_min = jitter_min
        self.jitter_max = jitter_max
        self.results_per_query = results_per_query
        self.hours_old = hours_old
        self.apply_prefilter = apply_prefilter

    def scrape_board(self, site_name: str, query: SearchQuery) -> List[JobPosting]:
        for attempt in range(1, 4):
            try:
                df: pd.DataFrame = scrape_jobs(
                    site_name=[site_name],
                    search_term=query.search_term,
                    location=query.location,
                    results_wanted=self.results_per_query,
                    hours_old=self.hours_old,
                    country_indeed=query.country_code or "poland",
                    is_remote=query.is_remote,
                )
                if df is None or df.empty:
                    return []

                postings: List[JobPosting] = []
                for _, row in df.iterrows():
                    title = str(row.get("title") or "").strip()
                    company = str(row.get("company") or "").strip()
                    location = str(row.get("location") or query.location).strip()
                    raw_url = str(row.get("job_url") or "").strip()
                    clean_url = clean_job_url(raw_url)
                    desc = str(row.get("description") or "").strip()

                    if not title or not clean_url:
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

                    def _clean_str(val) -> Optional[str]:
                        if val is None or pd.isna(val):
                            return None
                        s = str(val).strip()
                        return s if s else None

                    def _clean_float(val) -> Optional[float]:
                        if val is None or pd.isna(val):
                            return None
                        try:
                            return float(val)
                        except (ValueError, TypeError):
                            return None

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
                return postings
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
                    return []
        return []

    def scrape_query(self, query: SearchQuery) -> List[JobPosting]:
        all_postings: List[JobPosting] = []
        for site in query.site_names:
            postings = self.scrape_board(site, query)
            all_postings.extend(postings)

        if self.jitter_max > 0:
            delay = random.uniform(self.jitter_min, self.jitter_max)
            time.sleep(delay)

        return all_postings

    def run_scraping_cycle(self, queries: Optional[List[SearchQuery]] = None) -> List[JobPosting]:
        active_queries = queries or DEFAULT_QUERY_CATALOG
        aggregated: List[JobPosting] = []
        for q in active_queries:
            logger.info(f"Running scraper query '{q.query_id}' ({q.search_term}) in '{q.location}'...")
            res = self.scrape_query(q)
            aggregated.extend(res)
        return aggregated
