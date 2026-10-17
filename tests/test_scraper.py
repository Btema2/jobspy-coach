import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from src.scraper.jobspy_scraper import JobSpyScraper, clean_job_url
from src.scraper.query_catalog import DEFAULT_QUERY_CATALOG, SearchQuery


def test_clean_job_url():
    # LinkedIn tracking stripping
    dirty_li = "https://www.linkedin.com/jobs/view/123456789/?refId=abcd&trackingId=xyz&utm_source=feed"
    assert clean_job_url(dirty_li) == "https://www.linkedin.com/jobs/view/123456789/"

    # Indeed functional parameter (jk) preservation
    indeed_with_tracking = "https://pl.indeed.com/viewjob?jk=12345&utm_source=test&from=serp"
    assert clean_job_url(indeed_with_tracking) == "https://pl.indeed.com/viewjob?jk=12345"

    # Glassdoor functional parameter (jl) preservation
    glassdoor_with_tracking = "https://www.glassdoor.com/job-listing/j?jl=67890&refId=abc&source=publisher"
    assert clean_job_url(glassdoor_with_tracking) == "https://www.glassdoor.com/job-listing/j?jl=67890"

    # Fragment removal
    indeed_dirty = "https://indeed.com/view/100?utm=1&trk=abc#frag"
    assert clean_job_url(indeed_dirty) == "https://indeed.com/view/100"

    # Edge cases: empty, None, and literal nan
    assert clean_job_url("") == ""
    assert clean_job_url(None) == ""
    assert clean_job_url("nan") == ""
    assert clean_job_url("   ") == ""


def test_jobspy_scraper_per_board_isolation(monkeypatch):
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


def test_jobspy_scraper_nan_and_missing_handling(monkeypatch):
    """Verifies that pandas NaN values are properly converted and not stringified to 'nan'."""
    def mock_scrape_nan(site_name, **kwargs):
        return pd.DataFrame([
            # Missing title (None/NaN) -> Must be skipped
            {"id": "skip-1", "title": None, "job_url": "https://example.com/1"},
            {"id": "skip-2", "title": np.nan, "job_url": "https://example.com/2"},
            {"id": "skip-3", "title": "nan", "job_url": "https://example.com/3"},
            # Missing job_url (None/NaN) -> Must be skipped
            {"id": "skip-4", "title": "Coach A", "job_url": None},
            {"id": "skip-5", "title": "Coach B", "job_url": np.nan},
            {"id": "skip-6", "title": "Coach C", "job_url": "nan"},
            # Valid row with NaN location -> Must fall back to query.location
            {
                "id": "valid-1",
                "title": "Corporate Coach",
                "job_url": "https://example.com/coach-1",
                "company": np.nan,
                "location": np.nan,
                "description": np.nan,
                "job_type": np.nan,
                "job_url_direct": np.nan,
            }
        ])

    monkeypatch.setattr("jobspy.scrape_jobs", mock_scrape_nan)

    scraper = JobSpyScraper(jitter_min=0.0, jitter_max=0.0)
    query = SearchQuery(
        query_id="Q-NAN-TEST",
        search_term="Coach",
        location="Warsaw, Poland",
        site_names=["indeed"]
    )

    results = scraper.scrape_board("indeed", query)
    assert len(results) == 1
    job = results[0]
    assert job.title == "Corporate Coach"
    assert job.location == "Warsaw, Poland"  # Fell back to query.location, NOT 'nan'
    assert job.company == ""  # Cleaned from np.nan to '', NOT 'nan'
    assert job.description == ""  # Cleaned from np.nan to '', NOT 'nan'
    assert job.job_type is None
    assert job.job_url_direct is None


def test_jobspy_scraper_inter_board_jitter(monkeypatch):
    """Verifies that jitter delay is applied between boards (for i > 0) in scrape_query."""
    def mock_scrape(site_name, **kwargs):
        site_str = site_name[0] if isinstance(site_name, list) else site_name
        return pd.DataFrame([{
            "id": f"{site_str}-1",
            "title": f"Coach on {site_str}",
            "job_url": f"https://{site_str}.com/1",
            "company": "Company",
            "description": "Leadership role"
        }])


    monkeypatch.setattr("jobspy.scrape_jobs", mock_scrape)
    mock_sleep = MagicMock()
    monkeypatch.setattr("time.sleep", mock_sleep)

    scraper = JobSpyScraper(jitter_min=1.5, jitter_max=3.0)
    query = SearchQuery(
        query_id="Q-MULTI-BOARD",
        search_term="Coach",
        location="Warsaw",
        site_names=["linkedin", "indeed", "glassdoor"]
    )

    results = scraper.scrape_query(query)
    assert len(results) == 3
    # Jitter delay must be applied between boards: exactly 2 sleeps for 3 boards
    assert mock_sleep.call_count == 2
    for call in mock_sleep.call_args_list:
        delay = call[0][0]
        assert 1.5 <= delay <= 3.0


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


def test_jobspy_scraper_without_prefilter(monkeypatch):
    def mock_scrape_jobs(site_name, **kwargs):
        return pd.DataFrame([{
            "id": "job-1",
            "site": "linkedin",
            "job_url": "https://linkedin.com/jobs/view/9999",
            "title": "Football Coach",
            "company": "FC Club",
            "location": "Warsaw",
            "description": "Short"
        }])

    monkeypatch.setattr("jobspy.scrape_jobs", mock_scrape_jobs)
    scraper = JobSpyScraper(jitter_min=0.0, jitter_max=0.0, apply_prefilter=False)
    query = SearchQuery(query_id="Q-NO-PF", search_term="Coach", location="Warsaw", site_names=["linkedin"])
    postings = scraper.scrape_board("linkedin", query)
    assert len(postings) == 1
    assert postings[0].pre_filtered is False
    assert postings[0].pre_filter_reason is None
    assert postings[0].notification_status == "pending"


def test_jobspy_scraper_corrupted_salary(monkeypatch):
    def mock_scrape_jobs(site_name, **kwargs):
        return pd.DataFrame([{
            "id": "job-salary",
            "site": "indeed",
            "job_url": "https://indeed.com/view/777",
            "title": "Executive Coach",
            "company": "Corp",
            "location": "Warsaw",
            "description": "Leadership coaching role with executives.",
            "min_amount": "invalid-salary-string",
            "max_amount": "invalid-salary-string"
        }])

    monkeypatch.setattr("jobspy.scrape_jobs", mock_scrape_jobs)
    scraper = JobSpyScraper(jitter_min=0.0, jitter_max=0.0)
    query = SearchQuery(query_id="Q-SALARY", search_term="Coach", location="Warsaw", site_names=["indeed"])
    results = scraper.scrape_board("indeed", query)
    assert len(results) == 1
    assert results[0].salary_min is None
    assert results[0].salary_max is None


def test_query_catalog_incompatible_boards_excluded():
    queries_by_id = {q.query_id: q for q in DEFAULT_QUERY_CATALOG}

    # 1. For every query where country_code == "poland", glassdoor is NOT in site_names
    # 3. Polish queries have at least ["linkedin", "indeed", "google"]
    polish_queries = [q for q in DEFAULT_QUERY_CATALOG if q.country_code == "poland"]
    assert len(polish_queries) > 0
    for q in polish_queries:
        assert "glassdoor" not in q.site_names, f"Query {q.query_id} should not include glassdoor"
        assert set(["linkedin", "indeed", "google"]).issubset(set(q.site_names)), (
            f"Query {q.query_id} must have at least linkedin, indeed, google"
        )

    # 2. For query Q-EU-REM, zip_recruiter is NOT in site_names
    assert "Q-EU-REM" in queries_by_id
    q_eu_rem = queries_by_id["Q-EU-REM"]
    assert "zip_recruiter" not in q_eu_rem.site_names, "Q-EU-REM should not include zip_recruiter"

    # 4. Q-GL-REM retains ["linkedin", "indeed", "glassdoor", "google"]
    assert "Q-GL-REM" in queries_by_id
    q_gl_rem = queries_by_id["Q-GL-REM"]
    assert q_gl_rem.site_names == ["linkedin", "indeed", "glassdoor", "google"]


def test_jobspy_scraper_enabled_boards_filter(monkeypatch):
    mock_scrape = MagicMock()
    monkeypatch.setattr("jobspy.scrape_jobs", mock_scrape)

    scraper = JobSpyScraper(enabled_boards=["indeed"], jitter_min=0.0, jitter_max=0.0)
    query = SearchQuery(
        query_id="Q-TEST-FILTER",
        search_term="Executive Coach",
        location="Warsaw",
        site_names=["linkedin", "indeed"],
    )

    # Calling scrape_board on non-enabled board (linkedin) returns [] without calling scrape_jobs
    res_li = scraper.scrape_board("linkedin", query)
    assert res_li == []
    assert mock_scrape.call_count == 0

    # Calling scrape_board on enabled board (indeed) calls scrape_jobs
    mock_scrape.return_value = pd.DataFrame([{
        "id": "ind-1",
        "site": "indeed",
        "job_url": "https://indeed.com/view/100",
        "title": "Executive Coach",
        "company": "Corp",
        "location": "Warsaw",
        "description": "Coaching executives",
    }])
    res_ind = scraper.scrape_board("indeed", query)
    assert len(res_ind) == 1
    assert mock_scrape.call_count == 1


def test_jobspy_scraper_passes_proxies(monkeypatch):
    mock_scrape = MagicMock(return_value=pd.DataFrame())
    monkeypatch.setattr("jobspy.scrape_jobs", mock_scrape)

    proxies_list = ["http://proxy1:8080", "http://proxy2:8080"]
    scraper = JobSpyScraper(proxies=proxies_list, jitter_min=0.0, jitter_max=0.0)
    query = SearchQuery(
        query_id="Q-TEST-PROXY",
        search_term="Executive Coach",
        location="Warsaw",
        site_names=["indeed"],
    )

    scraper.scrape_board("indeed", query)
    assert mock_scrape.call_count == 1
    _, kwargs = mock_scrape.call_args
    assert kwargs.get("proxies") == proxies_list


def test_jobspy_scraper_linkedin_pacing(monkeypatch):
    mock_scrape = MagicMock(return_value=pd.DataFrame())
    monkeypatch.setattr("jobspy.scrape_jobs", mock_scrape)
    mock_sleep = MagicMock()
    monkeypatch.setattr("time.sleep", mock_sleep)

    current_time = 1000.0

    def fake_time():
        nonlocal current_time
        return current_time

    monkeypatch.setattr("time.time", fake_time)

    scraper = JobSpyScraper(jitter_min=0.0, jitter_max=0.0, linkedin_cooldown_seconds=12.0)
    query = SearchQuery(
        query_id="Q-PACING",
        search_term="Executive Coach",
        location="Warsaw",
        site_names=["linkedin"],
    )

    # First call: elapsed since 0 is large -> no pacing sleep
    scraper.scrape_board("linkedin", query)
    mock_sleep.assert_not_called()
    assert scraper.last_linkedin_call_time == 1000.0

    # Advance clock by 3.0 seconds
    current_time = 1003.0

    # Second call: elapsed is 3.0s < 12.0s -> should sleep for 9.0s
    scraper.scrape_board("linkedin", query)
    mock_sleep.assert_called_once_with(9.0)
    assert scraper.last_linkedin_call_time == 1003.0

    # Non-linkedin boards should not trigger linkedin pacing
    mock_sleep.reset_mock()
    current_time = 1004.0
    scraper.scrape_board("indeed", query)
    mock_sleep.assert_not_called()

