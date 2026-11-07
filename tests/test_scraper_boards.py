"""Per-board kwargs passed to python-jobspy (no network: scrape_jobs is monkeypatched)."""
import json
import logging
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.scraper.jobspy_scraper import JobSpyScraper, build_google_search_term
from src.scraper.query_catalog import SearchQuery, load_custom_queries


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
        site_names=["indeed"],
        country_code="germany",
    )
    base.update(kw)
    return SearchQuery(**base)


# ---------------------------------------------------------------- Google


@pytest.mark.parametrize(
    "hours_old,phrase",
    [
        (12, "since yesterday"),
        (24, "since yesterday"),
        (48, "in the last 3 days"),
        (72, "in the last 3 days"),
        (168, "in the last week"),
        (200, "in the last month"),
    ],
)
def test_build_google_search_term_time_windows(hours_old, phrase):
    q = _query(search_term="Career Coach", location="Testland")
    assert build_google_search_term(q, hours_old) == f"Career Coach jobs near Testland {phrase}"


def test_build_google_search_term_remote_suffix():
    q = _query(is_remote=True, location="Europe")
    assert build_google_search_term(q, 48) == "Career Coach jobs near Europe in the last 3 days remote"


def test_build_google_search_term_override_wins():
    q = _query(google_search_term="custom coach jobs remote")
    assert build_google_search_term(q, 48) == "custom coach jobs remote"


def test_google_board_receives_derived_google_search_term(mock_scrape):
    _scraper(hours_old=48).scrape_board("google", _query(site_names=["google"], country_code=None))
    kwargs = mock_scrape.call_args.kwargs
    assert kwargs["site_name"] == ["google"]
    assert kwargs["google_search_term"] == "Career Coach jobs near Testland in the last 3 days"
    assert kwargs["search_term"] == "Career Coach"
    assert "country_indeed" not in kwargs


def test_google_board_uses_per_query_override(mock_scrape):
    q = _query(site_names=["google"], country_code=None, google_search_term="my exact google query")
    _scraper().scrape_board("google", q)
    assert mock_scrape.call_args.kwargs["google_search_term"] == "my exact google query"


@pytest.mark.parametrize("board", ["indeed", "linkedin"])
def test_non_google_boards_get_no_google_search_term(mock_scrape, board):
    _scraper().scrape_board(board, _query(site_names=[board]))
    assert "google_search_term" not in mock_scrape.call_args.kwargs


def test_google_error_is_logged_once_not_silent(mock_scrape, caplog):
    df = pd.DataFrame()
    df.attrs["errors"] = {"google": "no job data returned"}
    mock_scrape.return_value = df
    scraper = _scraper()
    q = _query(site_names=["google"], country_code=None)
    with caplog.at_level(logging.WARNING, logger="src.scraper.jobspy_scraper"):
        assert scraper.scrape_board("google", q) == []
        assert scraper.scrape_board("google", q) == []
    msgs = [r.getMessage() for r in caplog.records if "Google" in r.getMessage()]
    assert len(msgs) == 1
    assert "no job data returned" in msgs[0]


def test_google_success_logs_no_warning(mock_scrape, caplog):
    mock_scrape.return_value = pd.DataFrame(
        [{"id": "g1", "title": "Coach", "job_url": "https://g.example/1", "company": "Acme", "description": "x" * 200}]
    )
    with caplog.at_level(logging.WARNING, logger="src.scraper.jobspy_scraper"):
        res = _scraper().scrape_board("google", _query(site_names=["google"], country_code=None))
    assert len(res) == 1
    assert not [r for r in caplog.records if "Google" in r.getMessage()]


# ---------------------------------------------------------------- Indeed / Glassdoor country


def test_indeed_uses_query_country(mock_scrape):
    _scraper().scrape_board("indeed", _query(country_code="poland"))
    assert mock_scrape.call_args.kwargs["country_indeed"] == "poland"


def test_indeed_without_country_is_skipped_not_usa(mock_scrape, caplog):
    q = _query(location="Europe", country_code=None, query_id="Q-REGION")
    with caplog.at_level(logging.WARNING, logger="src.scraper.jobspy_scraper"):
        res = _scraper().scrape_board("indeed", q)
    assert res == []
    mock_scrape.assert_not_called()
    assert any("country_code" in r.getMessage() and "indeed" in r.getMessage().lower() for r in caplog.records)


def test_indeed_skip_warning_is_logged_once(mock_scrape, caplog):
    scraper = _scraper()
    with caplog.at_level(logging.WARNING, logger="src.scraper.jobspy_scraper"):
        scraper.scrape_board("indeed", _query(country_code=None, query_id="Q-A"))
        scraper.scrape_board("indeed", _query(country_code=None, query_id="Q-B"))
    skips = [r for r in caplog.records if r.levelno == logging.WARNING and "country_code" in r.getMessage()]
    assert len(skips) == 1


def test_glassdoor_requires_country_too(mock_scrape):
    scraper = _scraper()
    assert scraper.scrape_board("glassdoor", _query(site_names=["glassdoor"], country_code=None)) == []
    mock_scrape.assert_not_called()
    scraper.scrape_board("glassdoor", _query(site_names=["glassdoor"], country_code="usa"))
    assert mock_scrape.call_args.kwargs["country_indeed"] == "usa"


@pytest.mark.parametrize("board", ["linkedin", "google"])
def test_country_agnostic_boards_run_without_country_and_get_no_country_kwarg(mock_scrape, board):
    _scraper().scrape_board(board, _query(site_names=[board], country_code=None, location="Europe"))
    assert mock_scrape.call_count == 1
    assert "country_indeed" not in mock_scrape.call_args.kwargs


# ---------------------------------------------------------------- Indeed filters / quoting


def test_indeed_passes_hours_old_and_remote_together(mock_scrape):
    """python-jobspy >= 1.4 ANDs the Indeed filters (verified live), so no filter group is dropped."""
    _scraper(hours_old=48).scrape_board("indeed", _query(is_remote=True))
    kwargs = mock_scrape.call_args.kwargs
    assert kwargs["hours_old"] == 48
    assert kwargs["is_remote"] is True


def test_indeed_remote_strips_double_quotes_from_search_term(mock_scrape):
    _scraper().scrape_board("indeed", _query(search_term='"Career Coach"', is_remote=True))
    assert mock_scrape.call_args.kwargs["search_term"] == "Career Coach"


def test_indeed_non_remote_keeps_quoted_phrase(mock_scrape):
    _scraper().scrape_board("indeed", _query(search_term='"Career Coach"', is_remote=False))
    assert mock_scrape.call_args.kwargs["search_term"] == '"Career Coach"'


def test_other_boards_keep_quoted_search_term_with_remote(mock_scrape):
    _scraper().scrape_board("linkedin", _query(search_term='"Career Coach"', is_remote=True, site_names=["linkedin"]))
    assert mock_scrape.call_args.kwargs["search_term"] == '"Career Coach"'


# ---------------------------------------------------------------- loader


def test_loader_reads_optional_google_search_term(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "queries.json").write_text(
        json.dumps(
            [
                {"query_id": "A", "search_term": "t", "location": "l", "site_names": ["google"],
                 "google_search_term": "override"},
                {"query_id": "B", "search_term": "t", "location": "l", "site_names": ["google"]},
            ]
        ),
        encoding="utf-8",
    )
    a, b = load_custom_queries()
    assert a.google_search_term == "override"
    assert b.google_search_term is None


def test_example_queries_json_is_loadable_and_generic(tmp_path, monkeypatch):
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent / "queries.example.json"
    monkeypatch.chdir(tmp_path)
    (tmp_path / "queries.json").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    qs = load_custom_queries()
    assert qs and all(q.query_id and q.search_term for q in qs)
    assert any(q.google_search_term for q in qs)
