"""Scrape progress logging, failed LinkedIn detail fetches and board error surfacing.

No network: ``scrape_jobs`` is monkeypatched, ``time.sleep`` is a mock."""
import logging
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.scheduler.daemon import DaemonService
from src.scraper.jobspy_scraper import JobSpyScraper
from src.scraper.query_catalog import DEFAULT_QUERY_CATALOG, SearchQuery

LOGGER = "src.scraper.jobspy_scraper"
LONG_DESC = "Coaching role working with clients one to one on their careers. " * 5


@pytest.fixture
def sleeps(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr("time.sleep", mock)
    return mock


def _scraper(**kw):
    kw.setdefault("jitter_min", 0.0)
    kw.setdefault("jitter_max", 0.0)
    kw.setdefault("linkedin_cooldown_seconds", 0.0)
    return JobSpyScraper(**kw)


def _query(**kw):
    base = dict(
        query_id="Q-T",
        search_term="Career Coach",
        location="Testland",
        site_names=["linkedin"],
        country_code="testland",
    )
    base.update(kw)
    return SearchQuery(**base)


def _row(n, desc=LONG_DESC, title=None, **extra):
    row = {
        "id": f"id-{n}",
        "title": title or f"Career Coach {n}",
        "job_url": f"https://jobs.example/{n}",
        "company": f"Acme Relief Org {n}",
        "location": "Testland",
        "description": desc,
    }
    row.update(extra)
    return row


def _patch_scrape(monkeypatch, frames):
    """frames: dict board -> DataFrame (mutable, looked up on every call)."""
    def fake(**kwargs):
        return frames.get(kwargs["site_name"][0], pd.DataFrame())

    mock = MagicMock(side_effect=fake)
    monkeypatch.setattr("src.scraper.jobspy_scraper.scrape_jobs", mock)
    return mock


def _info(caplog):
    return [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]


# ------------------------------------------------------------------ progress logging


def test_progress_lines_before_and_after_each_call(monkeypatch, sleeps, caplog):
    _patch_scrape(
        monkeypatch,
        {
            "linkedin": pd.DataFrame([_row(1), _row(2), _row(3, desc="short stub")]),
            "indeed": pd.DataFrame([_row(4)]),
        },
    )
    queries = [
        _query(query_id="Q-A", site_names=["linkedin", "indeed"]),
        _query(query_id="Q-B", site_names=["linkedin"]),
    ]
    with caplog.at_level(logging.INFO, logger=LOGGER):
        _scraper().scrape_all_queries(queries)
    msgs = _info(caplog)

    starts = [m for m in msgs if m.startswith("[") and '"Career Coach"' in m]
    assert starts == [
        '[1/3] Q-A linkedin "Career Coach" Testland',
        '[2/3] Q-A indeed "Career Coach" Testland',
        '[3/3] Q-B linkedin "Career Coach" Testland',
    ]

    done = [m for m in msgs if m.startswith("[1/3]") and "rows returned" in m]
    assert len(done) == 1
    assert "3 rows returned" in done[0]
    assert "3 kept" in done[0]
    assert "1 pre-filtered" in done[0]
    assert "2 to evaluate" in done[0]
    assert done[0].endswith("s")


def test_total_counts_only_calls_that_will_be_requested(monkeypatch, sleeps, caplog):
    _patch_scrape(monkeypatch, {})
    queries = [
        # glassdoor is disabled, indeed has no country: only linkedin is requested.
        _query(query_id="Q-A", site_names=["linkedin", "indeed", "glassdoor"], country_code=None),
        _query(query_id="Q-B", site_names=["linkedin", "indeed"], country_code="testland"),
    ]
    scraper = _scraper(enabled_boards=["linkedin", "indeed"])
    with caplog.at_level(logging.INFO, logger=LOGGER):
        plan = scraper.plan_calls(queries)
        scraper.scrape_all_queries(queries)
    assert [(q.query_id, s) for q, s in plan] == [("Q-A", "linkedin"), ("Q-B", "linkedin"), ("Q-B", "indeed")]
    starts = [m for m in _info(caplog) if m.startswith("[") and "Career Coach" in m]
    assert [m.split(" ")[0] for m in starts] == ["[1/3]", "[2/3]", "[3/3]"]


def test_cycle_summary_has_per_board_totals(monkeypatch, sleeps, caplog):
    _patch_scrape(
        monkeypatch,
        {
            "linkedin": pd.DataFrame([_row(1), _row(2, desc="")]),
            "indeed": pd.DataFrame([_row(3), _row(4, desc="stub")]),
        },
    )
    queries = [_query(site_names=["linkedin", "indeed"])]
    with caplog.at_level(logging.INFO, logger=LOGGER):
        _scraper().scrape_all_queries(queries)
    summary = [m for m in _info(caplog) if m.startswith("Scrape summary")]
    assert len(summary) == 1
    s = summary[0]
    assert "linkedin: 1 calls, 2 rows, 1 kept" in s
    assert "1 skipped (empty description)" in s
    assert "indeed: 1 calls, 2 rows, 2 kept, 1 pre-filtered" in s


def test_progress_lines_appear_on_the_daemon_path(tmp_path, monkeypatch, sleeps, caplog):
    """The CLI `scrape` and the daemon call DaemonService.run_pipeline_cycle (scrape_board per call),
    never scrape_all_queries, so the progress must come from scrape_board itself."""
    _patch_scrape(monkeypatch, {"linkedin": pd.DataFrame([_row(1), _row(2)])})
    scraper = _scraper()
    daemon = DaemonService(db_path=tmp_path / "p.db", scraper=scraper, evaluator=MagicMock(), notifier=MagicMock())
    daemon.evaluate_unprocessed = MagicMock(return_value=(0, 0))
    queries = [
        _query(query_id="Q-A", site_names=["linkedin", "indeed"], country_code=None),  # indeed skipped
        _query(query_id="Q-B", site_names=["linkedin"]),
    ]
    with caplog.at_level(logging.INFO):
        new, _ = daemon.run_pipeline_cycle(queries=queries)
    msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    starts = [m for m in msgs if m.startswith("[") and '"Career Coach"' in m]
    assert [m.split(" ")[0] for m in starts] == ["[1/2]", "[2/2]"]
    assert any(m.startswith("[1/2]") and "2 rows returned" in m for m in msgs)
    assert any(m.startswith("[1/2]") and "saved 2 new" in m for m in msgs)
    assert any(m.startswith("[2/2]") and "saved 0 new" in m for m in msgs)  # same hashes already stored
    assert any(m.startswith("Scrape summary") and "linkedin: 2 calls" in m for m in msgs)
    assert new == 2


def test_daemon_does_not_sleep_jitter_for_skipped_boards(tmp_path, monkeypatch, sleeps):
    _patch_scrape(monkeypatch, {})
    scraper = _scraper(jitter_min=2.0, jitter_max=3.0)
    daemon = DaemonService(db_path=tmp_path / "j.db", scraper=scraper, evaluator=MagicMock(), notifier=MagicMock())
    daemon.evaluate_unprocessed = MagicMock(return_value=(0, 0))
    queries = [_query(site_names=["indeed", "linkedin", "glassdoor"], country_code=None)]
    daemon.run_pipeline_cycle(queries=queries)
    # Only linkedin is requested: no inter-call jitter at all.
    assert sleeps.call_count == 0


def test_scrape_query_jitter_only_between_requested_boards(monkeypatch, sleeps):
    _patch_scrape(monkeypatch, {})
    scraper = _scraper(jitter_min=2.0, jitter_max=3.0, enabled_boards=["linkedin", "indeed"])
    # glassdoor disabled, indeed requested (country set): 2 requested boards -> exactly 1 jitter sleep.
    scraper.scrape_query(_query(site_names=["glassdoor", "linkedin", "indeed"]))
    assert sleeps.call_count == 1
    sleeps.reset_mock()
    # Country-less indeed + disabled glassdoor: only linkedin requested -> no jitter.
    scraper.scrape_query(_query(site_names=["indeed", "linkedin", "glassdoor"], country_code=None))
    assert sleeps.call_count == 0


# -------------------------------------------------- failed LinkedIn detail fetch


def test_linkedin_empty_description_is_skipped_and_counted(monkeypatch, sleeps, caplog):
    df = pd.DataFrame([_row(1), _row(2, desc=None), _row(3, desc="   "), _row(4, desc=float("nan"))])
    _patch_scrape(monkeypatch, {"linkedin": df})
    with caplog.at_level(logging.INFO, logger=LOGGER):
        res = _scraper().scrape_board("linkedin", _query())
    assert [p.title for p in res] == ["Career Coach 1"]
    warns = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warns) == 1
    assert "skipped 3 posting(s) with an empty description" in warns[0] and "retried" in warns[0]
    assert any("3 skipped (empty description)" in m for m in _info(caplog))


def test_linkedin_short_nonempty_description_still_prefiltered(monkeypatch, sleeps):
    _patch_scrape(monkeypatch, {"linkedin": pd.DataFrame([_row(1, desc="Too short to evaluate")])})
    res = _scraper().scrape_board("linkedin", _query())
    assert len(res) == 1
    assert res[0].pre_filtered is True
    assert res[0].notification_status == "dismissed"
    assert res[0].pre_filter_reason.startswith("DESCRIPTION_TOO_SHORT")


@pytest.mark.parametrize("board", ["indeed", "glassdoor", "google"])
def test_non_linkedin_stub_descriptions_reach_prefilter(monkeypatch, sleeps, caplog, board):
    _patch_scrape(monkeypatch, {board: pd.DataFrame([_row(1, desc=""), _row(2, desc="stub")])})
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        res = _scraper().scrape_board(board, _query(site_names=[board]))
    assert len(res) == 2
    assert all(p.pre_filtered and p.pre_filter_reason.startswith("DESCRIPTION_TOO_SHORT") for p in res)
    assert not [r for r in caplog.records if "empty description" in r.getMessage()]


def test_linkedin_empty_description_kept_when_details_not_fetched(monkeypatch, sleeps):
    """With fetch_description=False every LinkedIn description is empty by design: nothing is skipped."""
    _patch_scrape(monkeypatch, {"linkedin": pd.DataFrame([_row(1, desc=None)])})
    res = _scraper(fetch_description=False).scrape_board("linkedin", _query())
    assert len(res) == 1 and res[0].pre_filtered is True


def test_linkedin_empty_description_with_trap_title_is_still_dismissed(monkeypatch, sleeps):
    """A title the pre-filter rejects on its own is final whatever the description: persist it dismissed."""
    df = pd.DataFrame([_row(1, desc=None, title="Professional Football Coach")])
    _patch_scrape(monkeypatch, {"linkedin": df})
    res = _scraper().scrape_board("linkedin", _query())
    assert len(res) == 1
    assert res[0].pre_filtered is True
    assert not res[0].pre_filter_reason.startswith("DESCRIPTION_TOO_SHORT")


def test_skipped_empty_linkedin_job_is_not_persisted_and_is_retried(tmp_path, monkeypatch, sleeps):
    frames = {"linkedin": pd.DataFrame([_row(1, desc=None)])}
    _patch_scrape(monkeypatch, frames)
    daemon = DaemonService(db_path=tmp_path / "r.db", scraper=_scraper(), evaluator=MagicMock(), notifier=MagicMock())
    daemon.evaluate_unprocessed = MagicMock(return_value=(0, 0))
    q = _query()
    assert daemon.run_pipeline_cycle(queries=[q])[0] == 0
    with daemon.db._get_connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
    # Next cycle the detail fetch works: the job is stored as a normal pending job.
    frames["linkedin"] = pd.DataFrame([_row(1)])
    assert daemon.run_pipeline_cycle(queries=[q])[0] == 1


# ------------------------------------------------------ board errors from python-jobspy


def _df_with_errors(errors):
    df = pd.DataFrame()
    df.attrs["errors"] = errors
    return df


def test_non_google_board_errors_logged_once_per_board_and_message(monkeypatch, sleeps, caplog):
    frames = {"linkedin": _df_with_errors({"linkedin": "unexpected page with no jobs (blocked?)"})}
    _patch_scrape(monkeypatch, frames)
    scraper = _scraper()
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        scraper.scrape_board("linkedin", _query())
        scraper.scrape_board("linkedin", _query())
        frames["linkedin"] = _df_with_errors({"linkedin": "HTTP 429"})
        scraper.scrape_board("linkedin", _query())
    msgs = [r.getMessage() for r in caplog.records if "blocked?" in r.getMessage() or "429" in r.getMessage()]
    assert len(msgs) == 2
    assert "unexpected page with no jobs (blocked?)" in msgs[0] and msgs[0].startswith("Linkedin reported an error")
    assert "HTTP 429" in msgs[1]


# --------------------------------------------------------------- catalog invariant


@pytest.mark.parametrize("query", DEFAULT_QUERY_CATALOG, ids=lambda q: q.query_id)
def test_default_catalog_country_bound_boards_have_country_code(query):
    if {"indeed", "glassdoor"} & {s.lower() for s in query.site_names}:
        assert query.country_code


# ------------------------------------------- skipped boards stay visible on the daemon path


def _daemon(tmp_path, scraper):
    daemon = DaemonService(db_path=tmp_path / "s.db", scraper=scraper, evaluator=MagicMock(), notifier=MagicMock())
    daemon.evaluate_unprocessed = MagicMock(return_value=(0, 0))
    return daemon


def test_daemon_path_warns_once_about_country_less_indeed(tmp_path, monkeypatch, sleeps, caplog):
    _patch_scrape(monkeypatch, {})
    daemon = _daemon(tmp_path, _scraper())
    queries = [
        _query(query_id="Q-A", site_names=["linkedin", "indeed"], country_code=None),
        _query(query_id="Q-B", site_names=["indeed", "glassdoor"], country_code=None),
        _query(query_id="Q-C", site_names=["indeed"], country_code="testland"),
    ]
    with caplog.at_level(logging.INFO):
        daemon.run_pipeline_cycle(queries=queries)
        daemon.run_pipeline_cycle(queries=queries)
    warns = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING and "country_code" in r.getMessage()]
    # One warning per (board, reason) per scraper instance, even across cycles.
    assert len(warns) == 2
    indeed = [w for w in warns if "indeed" in w.lower()]
    glassdoor = [w for w in warns if "glassdoor" in w.lower()]
    assert len(indeed) == 1 and "2 queries" in indeed[0]
    assert len(glassdoor) == 1 and "1 query" in glassdoor[0]
    summary = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Scrape summary")]
    assert all("skipped: " in m and "indeed (no country_code) x2" in m for m in summary)


def test_disabled_boards_are_not_warned_about(tmp_path, monkeypatch, sleeps, caplog):
    _patch_scrape(monkeypatch, {})
    daemon = _daemon(tmp_path, _scraper(enabled_boards=["linkedin"]))
    queries = [_query(site_names=["linkedin", "indeed", "glassdoor"], country_code=None)]
    with caplog.at_level(logging.INFO):
        daemon.run_pipeline_cycle(queries=queries)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    summary = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Scrape summary")]
    assert len(summary) == 1 and "skipped" not in summary[0]


def test_scrape_all_queries_warns_country_less_once_not_twice(monkeypatch, sleeps, caplog):
    _patch_scrape(monkeypatch, {})
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        _scraper().scrape_all_queries([_query(site_names=["indeed"], country_code=None)])
    assert len([r for r in caplog.records if "country_code" in r.getMessage()]) == 1
