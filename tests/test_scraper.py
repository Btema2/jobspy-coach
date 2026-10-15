import pytest
from unittest.mock import MagicMock
import pandas as pd
from src.scraper.query_catalog import DEFAULT_QUERY_CATALOG, SearchQuery
from src.scraper.jobspy_scraper import JobSpyScraper, clean_job_url


def test_clean_job_url():
    dirty = "https://www.linkedin.com/jobs/view/123456789/?refId=abcd&trackingId=xyz&utm_source=feed"
    clean = clean_job_url(dirty)
    assert clean == "https://www.linkedin.com/jobs/view/123456789/"

    indeed_dirty = "https://indeed.com/view/100?utm=1&trk=abc#frag"
    assert clean_job_url(indeed_dirty) == "https://indeed.com/view/100"

    assert clean_job_url("") == ""
    assert clean_job_url(None) == ""


def test_jobspy_scraper_per_board_isolation(monkeypatch):
    # Mock jobspy scrape_jobs to fail on linkedin but succeed on indeed
    def mock_scrape_jobs(site_name, **kwargs):
        if "linkedin" in site_name:
            raise RuntimeError("HTTP 429 Too Many Requests from LinkedIn")
        if "indeed" in site_name:
            return pd.DataFrame([{
                "id": "ind-1",
                "site": "indeed",
                "job_url": "https://indeed.com/view/100?utm=1",
                "title": "Executive Coach",
                "company": "Leadership Lab",
                "location": "Warsaw, Poland",
                "description": "Corporate coaching role in Warsaw.",
                "is_remote": False,
                "date_posted": "2026-10-15"
            }])
        return pd.DataFrame()

    monkeypatch.setattr("jobspy.scrape_jobs", mock_scrape_jobs)
    monkeypatch.setattr("time.sleep", MagicMock())

    scraper = JobSpyScraper(jitter_min=0.0, jitter_max=0.0)
    query = SearchQuery(
        query_id="Q-TEST",
        search_term="Executive Coach",
        location="Warsaw",
        site_names=["linkedin", "indeed"],
        country_code="poland"
    )

    results = scraper.scrape_query(query)
    # LinkedIn failed gracefully without halting execution; Indeed returned 1 job
    assert len(results) == 1
    assert results[0].site == "indeed"
    assert results[0].title == "Executive Coach"
    assert results[0].job_url == "https://indeed.com/view/100"


def test_jobspy_scraper_retries_transient_failures(monkeypatch):
    attempts = 0

    def mock_scrape_jobs(site_name, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RuntimeError("503 Service Unavailable")
        return pd.DataFrame([{
            "id": "ind-retry",
            "site": "indeed",
            "job_url": "https://indeed.com/view/200",
            "title": "Corporate Coach",
            "company": "Leadership Lab",
            "location": "Warsaw, Poland",
            "description": "Corporate coaching role in Warsaw.",
            "is_remote": False,
            "date_posted": "2026-10-15"
        }])

    monkeypatch.setattr("jobspy.scrape_jobs", mock_scrape_jobs)
    mock_sleep = MagicMock()
    monkeypatch.setattr("time.sleep", mock_sleep)

    scraper = JobSpyScraper(jitter_min=0.0, jitter_max=0.0)
    query = SearchQuery(
        query_id="Q-TEST",
        search_term="Corporate Coach",
        location="Warsaw",
        site_names=["indeed"],
        country_code="poland"
    )

    results = scraper.scrape_board("indeed", query)
    assert len(results) == 1
    assert attempts == 3
    # Verify exponential backoff sleep was called twice: (2**1)*3 = 6s, (2**2)*3 = 12s
    assert mock_sleep.call_count == 2
    mock_sleep.assert_any_call(6)
    mock_sleep.assert_any_call(12)


def test_jobspy_scraper_canonicalizes_urls_and_applies_prefilter(monkeypatch):
    def mock_scrape_jobs(site_name, **kwargs):
        return pd.DataFrame([
            {
                "id": "job-exec-1",
                "site": "linkedin",
                "job_url": "https://linkedin.com/jobs/view/12345?refId=feed_123&trackingId=tk_987&utm_medium=email",
                "job_url_direct": "https://company.com/apply/12345?source=linkedin",
                "title": "Executive Leadership Coach",
                "company": "Global Talent Partners Sp. z o.o.",
                "location": "Warsaw, Poland",
                "description": (
                    "We are seeking an experienced Executive Leadership Coach to mentor top-tier "
                    "business leaders across the European region. The ideal candidate will design "
                    "tailored coaching journeys, drive leadership capability transformations, and partner "
                    "with cross-functional executives to foster an empowering leadership culture."
                ),
                "is_remote": False,
                "date_posted": "2026-10-15",
                "job_type": "fulltime",
                "min_amount": 25000.0,
                "max_amount": 35000.0,
                "currency": "PLN",
                "interval": "monthly"
            },
            {
                "id": "job-sports-2",
                "site": "linkedin",
                "job_url": "https://linkedin.com/jobs/view/67890?trackingId=sports_tk",
                "job_url_direct": None,
                "title": "Head Football & Fitness Coach",
                "company": "Warsaw Athletic Academy",
                "location": "Warsaw, Poland",
                "description": (
                    "Join our elite sports club as Head Football & Fitness Coach. You will oversee daily "
                    "pitch drills, conditioning programs, match tactics, and athletic physical training "
                    "for our competitive youth and adult football squads throughout the competitive season."
                ),
                "is_remote": False,
                "date_posted": "2026-10-15",
                "job_type": "fulltime"
            },
            {
                "id": "job-short-3",
                "site": "linkedin",
                "job_url": "https://linkedin.com/jobs/view/11111?utm_source=feed",
                "title": "Business Coach",
                "company": "Tiny Agency",
                "location": "Warsaw, Poland",
                "description": "Short stub description.",
                "is_remote": True,
                "date_posted": "2026-10-15"
            }
        ])

    monkeypatch.setattr("jobspy.scrape_jobs", mock_scrape_jobs)

    scraper = JobSpyScraper(jitter_min=0.0, jitter_max=0.0)
    query = SearchQuery(
        query_id="Q-TEST-PREFILTER",
        search_term="Coach",
        location="Warsaw",
        site_names=["linkedin"],
        country_code="poland"
    )

    results = scraper.scrape_board("linkedin", query)
    assert len(results) == 3

    # 1. Executive Leadership Coach -> Valid, url stripped, pre_filtered False
    exec_job = results[0]
    assert exec_job.job_url == "https://linkedin.com/jobs/view/12345"
    assert exec_job.job_url_direct == "https://company.com/apply/12345"
    assert exec_job.title == "Executive Leadership Coach"
    assert exec_job.company == "Global Talent Partners Sp. z o.o."
    assert exec_job.pre_filtered is False
    assert exec_job.pre_filter_reason is None
    assert exec_job.notification_status == "pending"
    assert exec_job.salary_min == 25000.0
    assert exec_job.salary_max == 35000.0
    assert exec_job.salary_currency == "PLN"
    assert exec_job.salary_interval == "monthly"
    assert exec_job.job_type == "fulltime"
    assert len(exec_job.dedup_hash) == 64

    # 2. Football & Fitness Coach -> Excluded by title regex, pre_filtered True
    sports_job = results[1]
    assert sports_job.job_url == "https://linkedin.com/jobs/view/67890"
    assert sports_job.pre_filtered is True
    assert "EXCLUDED_TITLE_REGEX" in sports_job.pre_filter_reason
    assert sports_job.notification_status == "dismissed"

    # 3. Short description -> Excluded by description length, pre_filtered True
    short_job = results[2]
    assert short_job.job_url == "https://linkedin.com/jobs/view/11111"
    assert short_job.pre_filtered is True
    assert "DESCRIPTION_TOO_SHORT" in short_job.pre_filter_reason
    assert short_job.notification_status == "dismissed"


def test_jobspy_scraper_empty_and_invalid_data(monkeypatch):
    def mock_scrape_empty(site_name, **kwargs):
        return pd.DataFrame()

    monkeypatch.setattr("jobspy.scrape_jobs", mock_scrape_empty)
    scraper = JobSpyScraper(jitter_min=0.0, jitter_max=0.0)
    query = SearchQuery(
        query_id="Q-EMPTY",
        search_term="Coach",
        location="Warsaw",
        site_names=["linkedin"]
    )
    assert scraper.scrape_board("linkedin", query) == []

    # Test DataFrame with missing title or missing URL rows (should be skipped)
    def mock_scrape_missing(site_name, **kwargs):
        return pd.DataFrame([
            {"id": "bad-1", "title": "", "job_url": "https://example.com/1"},
            {"id": "bad-2", "title": "Valid Title", "job_url": ""},
            {"id": "bad-3", "title": "   ", "job_url": "https://example.com/3"},
            {"id": "good-4", "title": "Valid Coach", "job_url": "https://example.com/good", "company": "Co", "description": "A" * 160}
        ])

    monkeypatch.setattr("jobspy.scrape_jobs", mock_scrape_missing)
    results = scraper.scrape_board("linkedin", query)
    assert len(results) == 1
    assert results[0].title == "Valid Coach"


def test_jobspy_scraper_exhausted_retries_returns_empty(monkeypatch):
    def mock_scrape_always_fails(site_name, **kwargs):
        raise RuntimeError("Permanent error from site")

    monkeypatch.setattr("jobspy.scrape_jobs", mock_scrape_always_fails)
    monkeypatch.setattr("time.sleep", MagicMock())

    scraper = JobSpyScraper(jitter_min=0.0, jitter_max=0.0)
    query = SearchQuery(
        query_id="Q-FAIL",
        search_term="Coach",
        location="Warsaw",
        site_names=["linkedin"]
    )
    # Should attempt 3 times, fail gracefully, return []
    res = scraper.scrape_board("linkedin", query)
    assert res == []


def test_query_catalog_and_cycle(monkeypatch):
    assert len(DEFAULT_QUERY_CATALOG) == 5
    for q in DEFAULT_QUERY_CATALOG:
        assert q.query_id.startswith("Q-")
        assert len(q.search_term) > 0
        assert len(q.location) > 0
        assert len(q.site_names) > 0

    # Test run_scraping_cycle calls scrape_query for each query
    mock_postings = []
    scraper = JobSpyScraper(jitter_min=0.0, jitter_max=0.0)
    mock_scrape_query = MagicMock(return_value=[])
    scraper.scrape_query = mock_scrape_query

    queries = [
        SearchQuery(query_id="Q1", search_term="Coach", location="Warsaw", site_names=["indeed"]),
        SearchQuery(query_id="Q2", search_term="Trainer", location="Krakow", site_names=["glassdoor"]),
    ]
    res = scraper.run_scraping_cycle(queries)
    assert res == []
    assert mock_scrape_query.call_count == 2
