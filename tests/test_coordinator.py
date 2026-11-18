"""ScrapeCoordinator: per-call lock and shared LinkedIn pacing (no network, no real sleeping)."""
import threading
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.scraper.coordinator import ScrapeCoordinator
from src.scraper.jobspy_scraper import JobSpyScraper
from src.scraper.query_catalog import SearchQuery


def _query(**kw):
    base = dict(
        query_id="Q-C",
        search_term="Career Coach",
        location="Testland",
        site_names=["linkedin"],
        country_code="testland",
    )
    base.update(kw)
    return SearchQuery(**base)


def _scraper(**kw):
    kw.setdefault("jitter_min", 0.0)
    kw.setdefault("jitter_max", 0.0)
    return JobSpyScraper(**kw)


@pytest.fixture
def clock(monkeypatch):
    """Fake time.time / time.sleep; sleep records the request and does not advance the clock."""
    state = {"now": 1000.0}
    sleep = MagicMock()
    monkeypatch.setattr("time.time", lambda: state["now"])
    monkeypatch.setattr("time.sleep", sleep)
    state["sleep"] = sleep
    return state


def test_linkedin_cooldown_sleeps_remaining_time(clock):
    coord = ScrapeCoordinator(linkedin_cooldown_seconds=12.0)

    with coord.call("linkedin"):
        pass
    clock["sleep"].assert_not_called()
    assert coord.last_linkedin_call_time == 1000.0

    clock["now"] = 1003.0
    with coord.call("LinkedIn"):
        pass
    clock["sleep"].assert_called_once_with(9.0)
    assert coord.last_linkedin_call_time == 1003.0


def test_linkedin_cooldown_not_applied_once_elapsed(clock):
    coord = ScrapeCoordinator(linkedin_cooldown_seconds=12.0)
    with coord.call("linkedin"):
        pass
    clock["now"] = 1100.0
    with coord.call("linkedin"):
        pass
    clock["sleep"].assert_not_called()
    assert coord.last_linkedin_call_time == 1100.0


def test_non_linkedin_board_is_not_paced(clock):
    coord = ScrapeCoordinator(linkedin_cooldown_seconds=12.0)
    with coord.call("linkedin"):
        pass
    clock["now"] = 1001.0
    for board in ("indeed", "glassdoor", "google"):
        with coord.call(board):
            pass
    clock["sleep"].assert_not_called()
    assert coord.last_linkedin_call_time == 1000.0


def test_lock_released_after_exception():
    coord = ScrapeCoordinator(linkedin_cooldown_seconds=0.0)
    with pytest.raises(RuntimeError):
        with coord.call("indeed"):
            raise RuntimeError("boom")
    assert coord._lock.acquire(blocking=False)
    coord._lock.release()


def test_calls_are_mutually_exclusive(monkeypatch):
    """Two threads scrape at once. If the lock works, the barrier (needs both inside the fake request at
    the same time) can never fill and times out; if it is broken, both get in and the peak is 2."""
    barrier = threading.Barrier(2, timeout=0.3)
    guard = threading.Lock()
    state = {"current": 0, "peak": 0}

    def fake_scrape_jobs(**kwargs):
        with guard:
            state["current"] += 1
            state["peak"] = max(state["peak"], state["current"])
        try:
            barrier.wait()
        except threading.BrokenBarrierError:
            pass
        finally:
            with guard:
                state["current"] -= 1
        return pd.DataFrame()

    monkeypatch.setattr("src.scraper.jobspy_scraper.scrape_jobs", fake_scrape_jobs)
    scraper = _scraper(linkedin_cooldown_seconds=0.0)
    query = _query(site_names=["indeed"])

    threads = [
        threading.Thread(target=scraper.scrape_board_outcome, args=("indeed", query), kwargs={"track_cycle": False})
        for _ in range(2)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert not any(t.is_alive() for t in threads)
    assert state["peak"] == 1


def test_two_scrapers_sharing_a_coordinator_share_pacing(clock, monkeypatch):
    monkeypatch.setattr("src.scraper.jobspy_scraper.scrape_jobs", MagicMock(return_value=pd.DataFrame()))
    coord = ScrapeCoordinator(linkedin_cooldown_seconds=12.0)
    first = _scraper(coordinator=coord)
    second = _scraper(coordinator=coord)
    query = _query()

    first.scrape_board_outcome("linkedin", query)
    clock["sleep"].assert_not_called()

    clock["now"] = 1005.0
    second.scrape_board_outcome("linkedin", query)
    clock["sleep"].assert_called_once_with(7.0)
    assert first.last_linkedin_call_time == second.last_linkedin_call_time == 1005.0


def test_scraper_properties_delegate_to_coordinator():
    coord = ScrapeCoordinator(linkedin_cooldown_seconds=12.0)
    scraper = _scraper(coordinator=coord)
    scraper.linkedin_cooldown_seconds = 3.0
    scraper.last_linkedin_call_time = 42.0
    assert coord.linkedin_cooldown_seconds == 3.0
    assert coord.last_linkedin_call_time == 42.0

    own = _scraper(linkedin_cooldown_seconds=7.5)
    assert own.linkedin_cooldown_seconds == 7.5
    assert own.last_linkedin_call_time == 0.0


def test_track_cycle_false_leaves_progress_untouched(monkeypatch, caplog):
    monkeypatch.setattr("src.scraper.jobspy_scraper.scrape_jobs", MagicMock(return_value=pd.DataFrame()))
    monkeypatch.setattr("time.sleep", MagicMock())
    scraper = _scraper(linkedin_cooldown_seconds=0.0)
    query = _query()
    scraper.begin_cycle([query])
    assert scraper._cycle.index == 0

    with caplog.at_level("INFO", logger="src.scraper.jobspy_scraper"):
        scraper.scrape_board_outcome("linkedin", query, track_cycle=False)
    assert scraper._cycle.index == 0
    assert scraper._cycle.totals == {}
    assert not any("[" in r.getMessage().split(" ")[0] for r in caplog.records)

    scraper.scrape_board_outcome("linkedin", query)
    assert scraper._cycle.index == 1
    assert scraper._cycle.totals["linkedin"].calls == 1
