"""ScrapeOutcome statuses from scrape_board_outcome (no network: scrape_jobs is monkeypatched)."""
import enum
from typing import get_args
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.scraper.jobspy_scraper import JobSpyScraper
from src.scraper.outcome import (
    STATUS_EMPTY,
    STATUS_FAILED,
    STATUS_OK,
    STATUS_SKIPPED,
    ScrapeOutcome,
    ScrapeStatus,
)
from src.scraper.query_catalog import SearchQuery, query_key

LONG_DESC = "We are hiring a career coach to support clients one to one in their professional growth. " * 4


@pytest.fixture
def mock_scrape(monkeypatch):
    mock = MagicMock(return_value=pd.DataFrame())
    monkeypatch.setattr("src.scraper.jobspy_scraper.scrape_jobs", mock)
    monkeypatch.setattr("time.sleep", MagicMock())
    return mock


def _scraper(**kw):
    return JobSpyScraper(jitter_min=0.0, jitter_max=0.0, linkedin_cooldown_seconds=0.0, **kw)


def _query(**kw):
    base = dict(
        query_id="Q-T",
        search_term="Career Coach",
        location="Testland",
        site_names=["linkedin"],
        country_code="germany",
    )
    base.update(kw)
    return SearchQuery(**base)


def _frame(rows, attrs=None):
    df = pd.DataFrame(rows)
    if attrs is not None:
        df.attrs = attrs
    return df


def _row(i, title="Career Coach", desc=LONG_DESC):
    return {
        "title": title,
        "job_url": f"https://example.com/job/{i}",
        "company": f"Co{i}",
        "location": "Testland",
        "description": desc,
    }


def test_exception_after_retries_is_failed(mock_scrape):
    mock_scrape.side_effect = RuntimeError("boom")
    out = _scraper().scrape_board_outcome("linkedin", _query())
    assert isinstance(out, ScrapeOutcome)
    assert out.status == "failed"
    assert out.error == "boom"
    assert out.postings == []
    assert out.rows == 0
    assert mock_scrape.call_count == 3


def test_exception_with_empty_message_falls_back_to_type_name(mock_scrape):
    mock_scrape.side_effect = TimeoutError()
    out = _scraper().scrape_board_outcome("linkedin", _query())
    assert out.status == "failed"
    assert out.error == "TimeoutError"
    assert mock_scrape.call_count == 3


def test_block_in_attrs_errors_is_failed(mock_scrape):
    msg = "unexpected page with no jobs (blocked?)"
    mock_scrape.return_value = _frame([], {"errors": {"linkedin": msg}})
    out = _scraper().scrape_board_outcome("linkedin", _query())
    assert out.status == "failed"
    assert out.error == msg
    assert out.rows == 0
    assert out.postings == []


def test_enum_like_attrs_key_maps_to_board(mock_scrape):
    class Site(enum.Enum):
        LINKEDIN = "linkedin"

    mock_scrape.return_value = _frame([], {"errors": {Site.LINKEDIN: "blocked"}})
    out = _scraper().scrape_board_outcome("LinkedIn", _query())
    assert out.status == "failed"
    assert out.error == "blocked"


def test_error_with_rows_is_ok_with_error_note(mock_scrape):
    mock_scrape.return_value = _frame([_row(1)], {"errors": {"linkedin": "partial failure"}})
    out = _scraper().scrape_board_outcome("linkedin", _query())
    assert out.status == "ok"
    assert out.error == "partial failure"
    assert out.rows == 1
    assert len(out.postings) == 1


def test_no_error_no_rows_is_empty(mock_scrape):
    mock_scrape.return_value = pd.DataFrame()
    out = _scraper().scrape_board_outcome("linkedin", _query())
    assert out.status == "empty"
    assert out.error is None
    assert out.rows == 0
    assert out.postings == []


def test_disabled_board_is_skipped(mock_scrape):
    out = _scraper(enabled_boards=["indeed"]).scrape_board_outcome("linkedin", _query())
    assert out.status == "skipped"
    assert out.skip_reason == "disabled"
    assert out.postings == []
    mock_scrape.assert_not_called()


def test_missing_country_is_skipped(mock_scrape):
    out = _scraper().scrape_board_outcome("indeed", _query(site_names=["indeed"], country_code=None))
    assert out.status == "skipped"
    assert out.skip_reason == "no_country"
    mock_scrape.assert_not_called()


def test_ok_outcome_counts_rows_and_prefiltered(mock_scrape):
    mock_scrape.return_value = _frame(
        [
            _row(1),
            _row(2),
            _row(3, title="Head Football & Fitness Coach", desc=LONG_DESC),  # prefiltered
            _row(4, desc=None),  # LinkedIn row with no description -> skipped_empty
        ]
    )
    out = _scraper().scrape_board_outcome("linkedin", _query())
    assert out.status == "ok"
    assert out.error is None
    assert out.rows == 4
    assert out.skipped_empty == 1
    assert len(out.postings) == 3
    assert out.prefiltered == sum(1 for p in out.postings if p.pre_filtered)
    assert out.prefiltered >= 1


def test_scrape_board_still_returns_list(mock_scrape):
    mock_scrape.return_value = _frame([_row(1), _row(2)])
    s = _scraper()
    posts = s.scrape_board("linkedin", _query())
    outcome_posts = s.scrape_board_outcome("linkedin", _query()).postings
    assert isinstance(posts, list)
    assert [p.dedup_hash for p in posts] == [p.dedup_hash for p in outcome_posts]
    assert len(posts) == 2


def test_query_key_format():
    q = _query(query_id="Q-ABC")
    assert query_key(q, "LinkedIn") == "Q-ABC|linkedin"


def test_scrape_outcome_status_constants():
    assert get_args(ScrapeStatus) == ("ok", "empty", "failed", "skipped")
    assert (STATUS_OK, STATUS_EMPTY, STATUS_FAILED, STATUS_SKIPPED) == get_args(ScrapeStatus)


def test_scraper_records_pre_filter_rule(mock_scrape):
    mock_scrape.return_value = _frame([_row(1, title="Scrum Master"), _row(2)])
    out = _scraper().scrape_board_outcome("linkedin", _query())
    trapped = next(p for p in out.postings if p.title == "Scrum Master")
    clean = next(p for p in out.postings if p.title == "Career Coach")
    assert trapped.pre_filtered and trapped.pre_filter_rule == "seed:block:EXCLUDED_TITLE_TRAP"
    assert clean.pre_filter_rule is None

    out = _scraper().scrape_board_outcome("linkedin", _query(), rules=())
    passed = next(p for p in out.postings if p.title == "Scrum Master")
    assert not passed.pre_filtered and passed.pre_filter_rule is None


def test_rules_reach_the_linkedin_empty_description_check(mock_scrape):
    mock_scrape.return_value = _frame([_row(1, title="Scrum Master", desc=None)])
    # legacy seed: the title is dropped for good, so the row is kept (pre-filtered), not skipped
    out = _scraper().scrape_board_outcome("linkedin", _query())
    assert out.skipped_empty == 0 and out.postings[0].pre_filtered
    # without rules the title is fine, so the missing description means the detail fetch failed
    out = _scraper().scrape_board_outcome("linkedin", _query(), rules=())
    assert out.skipped_empty == 1 and out.postings == []
