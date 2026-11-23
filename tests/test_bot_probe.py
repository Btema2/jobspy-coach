"""/test: a small, read-only trial search (planning, limits, worker thread, coordinator sharing)."""
import threading
import time
from pathlib import Path
from typing import List

import pandas as pd
import pytest

from src.bot.client import TelegramBotClient
from src.bot.commands import TRIGGER_STARTED, BotContext, CommandRouter
from src.bot.probe import (
    TEST_COOLDOWN_SECONDS,
    TEST_MAX_CALLS,
    TEST_MAX_RESULTS,
    ProbeResult,
    ProbeRunner,
    plan_probe,
    run_probe,
)
from src.bot.updates import parse_update
from src.config import Settings
from src.profile.compiler import compile_profile, parse_expanded_query_id
from src.profile.ops import AddMarket, AddRole, SetField, apply_changes, create_starter_profile
from src.profile.snapshot import Market, ProfileFields, ProfileSnapshot, RoleTerm
from src.profile.store import ProfileStore
from src.scraper.jobspy_scraper import JobSpyScraper
from src.scraper.outcome import STATUS_FAILED, STATUS_OK, ScrapeOutcome
from src.scraper.query_catalog import SearchQuery
from src.storage.db import Database, compute_dedup_hash
from src.storage.models import JobPosting
from tests.fake_telegram import FakeTelegram, message_update

LONG = "word " * 60


def _posting(n: int, title=None, pre_filtered=False) -> JobPosting:
    title = title or f"Title {n}"
    return JobPosting(
        dedup_hash=compute_dedup_hash("Example Co", title, f"City {n}"),
        site="google", job_url=f"https://example.test/{n}", title=title, company="Example Co",
        location=f"City {n}", description=LONG, pre_filtered=pre_filtered,
    )


class DoubleScraper:
    """Records every call; returns ``per_call`` postings (distinct per call) as rows."""

    def __init__(self, per_call=4, prefiltered=0, fail=False):
        self.calls: List[dict] = []
        self.per_call = per_call
        self.prefiltered = prefiltered
        self.fail = fail
        self.thread_names: List[str] = []
        self._n = 0

    def scrape_board_outcome(self, site, query, **kw):
        self.calls.append({"site": site, "query": query, **kw})
        self.thread_names.append(threading.current_thread().name)
        if self.fail:
            raise RuntimeError("<TOKEN> boom")
        postings = []
        for _ in range(self.per_call):
            self._n += 1
            postings.append(_posting(self._n, pre_filtered=len(postings) < self.prefiltered))
        return ScrapeOutcome(
            postings=postings, rows=self.per_call, status=STATUS_OK,
            prefiltered=sum(p.pre_filtered for p in postings),
        )


@pytest.fixture
def db(tmp_path: Path) -> Database:
    database = Database(tmp_path / "probe.db")
    database.init_db()
    return database


@pytest.fixture
def store(db) -> ProfileStore:
    profile_store = ProfileStore(db)
    create_starter_profile(profile_store, wow_threshold=8.5, digest_threshold=6.5, output_language="en")
    apply_changes(
        profile_store,
        [
            AddRole("<ROLE>", (RoleTerm("<SYN>", None),)),
            AddMarket(Market("<MARKET>", "<LOCATION>", ("poland",), languages=("pl", "en"))),
            SetField("paused", False),
        ],
        label="fixture",
    )
    return profile_store


def _snapshot(store) -> ProfileSnapshot:
    return store.read_snapshot()


def _counts(db: Database):
    with db._get_connection() as conn:
        return tuple(
            conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("jobs", "scrape_runs", "job_hits")
        )


# --- planning -----------------------------------------------------------------------------


def test_probe_excludes_linkedin_unless_named(store):
    snap = _snapshot(store)
    default = plan_probe(snap, "<ROLE>", include_linkedin=False)
    assert default and all(site != "linkedin" for _, site in default)
    assert len(default) <= TEST_MAX_CALLS
    named = plan_probe(snap, "<ROLE>", include_linkedin=True)
    assert named[0][1] == "linkedin"
    assert len(named) <= TEST_MAX_CALLS


def test_probe_plan_matches_role_case_insensitively_and_only_that_role(store, db):
    apply_changes(store, [AddRole("<OTHER>")], label="other")
    plan = plan_probe(_snapshot(store), "<role>", include_linkedin=True)
    assert plan
    assert {parse_expanded_query_id(q.query_id)["role"] for q, _ in plan} == {"<ROLE>"}


def test_probe_plan_drops_country_bound_boards_without_country(store):
    apply_changes(store, [AddMarket(Market("<REGION>", "<REGION>", ()))], label="region")
    snap = ProfileSnapshot(
        _snapshot(store).fields, _snapshot(store).roles, (Market("<REGION>", "<REGION>", ()),)
    )
    plan = plan_probe(snap, "<ROLE>", include_linkedin=True)
    assert plan and all(site not in ("indeed", "glassdoor") for _, site in plan)


def test_probe_unknown_role_uses_in_memory_role_and_writes_nothing(store, db):
    snap = _snapshot(store)
    before_counts, before_version = _counts(db), snap.version
    plan = plan_probe(snap, "<NEW ROLE>", include_linkedin=False)
    assert plan
    assert {parse_expanded_query_id(q.query_id)["role"] for q, _ in plan} == {"<NEW ROLE>"}
    assert all(q.search_term == "<NEW ROLE>" for q, _ in plan)

    scraper = DoubleScraper()
    run_probe(scraper, db, plan, rules=())
    assert _counts(db) == before_counts
    assert _snapshot(store).version == before_version
    assert snap == _snapshot(store)


# --- running ------------------------------------------------------------------------------


def test_probe_stops_at_5_results_and_uses_results_wanted(store, db):
    plan = plan_probe(_snapshot(store), "<ROLE>", include_linkedin=False)
    assert len(plan) == TEST_MAX_CALLS
    scraper = DoubleScraper(per_call=4)
    result = run_probe(scraper, db, plan, rules=())
    assert len(scraper.calls) == 2 and result.calls == 2
    assert all(c["results_wanted"] == TEST_MAX_RESULTS == 5 for c in scraper.calls)
    assert all(c["track_cycle"] is False for c in scraper.calls)
    assert result.rows == 8


def test_probe_counts_new_jobs_against_db_and_titles(store, db):
    plan = plan_probe(_snapshot(store), "<ROLE>", include_linkedin=False)
    known = _posting(1, "Known Title")

    class Fixed(DoubleScraper):
        def scrape_board_outcome(self, site, query, **kw):
            self.calls.append(kw)
            return ScrapeOutcome(postings=[known, _posting(2, "Fresh Title")], rows=2, status=STATUS_OK)

    db.save_job(known)
    before = _counts(db)
    result = run_probe(Fixed(), db, plan[:1], rules=())
    assert (result.rows, result.new_jobs, result.prefiltered) == (2, 1, 0)
    assert result.titles == ("Known Title", "Fresh Title")
    assert _counts(db) == before


def test_probe_counts_failed_calls_and_continues(store, db):
    plan = plan_probe(_snapshot(store), "<ROLE>", include_linkedin=False)

    class Failing(DoubleScraper):
        def scrape_board_outcome(self, site, query, **kw):
            self.calls.append(kw)
            return ScrapeOutcome(postings=[], rows=0, status=STATUS_FAILED, error="x")

    scraper = Failing()
    result = run_probe(scraper, db, plan, rules=())
    assert result.calls == len(plan) and result.failed == len(plan) and result.rows == 0


# --- the runner ---------------------------------------------------------------------------


class Sink:
    """Records what the worker sends; the first ``fail_first`` sends are reported as undelivered (None)."""

    def __init__(self, fail_first=0):
        self.sent: List[str] = []
        self.attempts = 0
        self.fail_first = fail_first
        self.done = threading.Event()

    def send_message(self, chat_id, text, reply_markup=None):
        self.attempts += 1
        if self.attempts <= self.fail_first:
            return None
        self.sent.append(text)
        self.done.set()
        return 1


def _runner(tmp_path, db, scraper, clock=time.time, sink=None, stop_event=None):
    ctx = BotContext(
        db_path=db.db_path, env_path=tmp_path / ".env", settings=Settings(_env_file=None),
        trigger_cycle=lambda **_: TRIGGER_STARTED, is_cycle_running=lambda: False, scraper=scraper, clock=clock,
        stop_event=stop_event,
    )
    sink = sink or Sink()
    return ProbeRunner(ctx, sink, clock=clock), sink


def _update(text: str, update_id=1):
    return parse_update(message_update(update_id, text))


def _wait(runner: ProbeRunner):
    thread = runner._thread
    assert thread is not None
    thread.join(timeout=10)
    assert not thread.is_alive()


def test_probe_runs_off_the_poller_thread(tmp_path, db, store):
    scraper = DoubleScraper(per_call=2)
    runner, sink = _runner(tmp_path, db, scraper)
    reply = runner.request(_update("/test <ROLE>"))
    assert reply[0].text.startswith("Searching")
    _wait(runner)
    assert scraper.thread_names and set(scraper.thread_names) == {"bot-test"}
    assert runner._thread.daemon


def test_probe_reports_prefiltered_share_and_disclaimer(tmp_path, db, store):
    scraper = DoubleScraper(per_call=4, prefiltered=2)
    runner, sink = _runner(tmp_path, db, scraper)
    runner.request(_update("/test <ROLE>"))
    _wait(runner)
    text = "\n".join(sink.sent)
    assert "rows 8, new 8, pre-filtered 4 of 8 (50%)" in text
    assert "This does not measure relevance." in text
    assert text.count("- ") == 5  # at most 5 titles


def test_probe_reply_escapes_titles(tmp_path, db, store):
    class Evil(DoubleScraper):
        def scrape_board_outcome(self, site, query, **kw):
            return ScrapeOutcome(postings=[_posting(1, "<b>&amp; R&D")], rows=1, status=STATUS_OK)

    runner, sink = _runner(tmp_path, db, Evil())
    runner.request(_update("/test <ROLE>"))
    _wait(runner)
    assert "&lt;b&gt;&amp;amp; R&amp;D" in "\n".join(sink.sent)


def test_probe_cooldown_one_per_5_minutes(tmp_path, db, store):
    now = [1000.0]
    runner, sink = _runner(tmp_path, db, DoubleScraper(per_call=1), clock=lambda: now[0])
    assert TEST_COOLDOWN_SECONDS == 300
    assert runner.request(_update("/test <ROLE>"))[0].text.startswith("Searching")
    _wait(runner)
    now[0] += 100
    assert runner.request(_update("/test <ROLE>"))[0].text == "One /test per 5 minutes; try again in 200 s."
    now[0] += 201
    assert runner.request(_update("/test <ROLE>"))[0].text.startswith("Searching")
    _wait(runner)


def test_probe_usage_and_no_market(tmp_path, db, store):
    runner, _ = _runner(tmp_path, db, DoubleScraper())
    assert runner.request(_update("/test"))[0].text.startswith("Usage: /test")
    apply_changes(store, [SetField("paused", True)], label="p")
    with db._get_connection() as conn:
        conn.execute("DELETE FROM markets")
    reply = runner.request(_update("/test <ROLE>"))
    assert reply[0].text == "Add a market first (/add_market)."
    assert runner._thread is None  # no cooldown consumed
    assert runner.request(_update("/test <ROLE>"))[0].text == "Add a market first (/add_market)."


def test_probe_named_linkedin_flag(tmp_path, db, store):
    scraper = DoubleScraper(per_call=1)
    runner, _ = _runner(tmp_path, db, scraper)
    runner.request(_update("/test <ROLE> LinkedIn"))
    _wait(runner)
    assert scraper.calls[0]["site"] == "linkedin"


def test_probe_worker_exception_reported_by_type_name(tmp_path, db, store):
    runner, sink = _runner(tmp_path, db, DoubleScraper(fail=True))
    runner.request(_update("/test <ROLE>"))
    _wait(runner)
    text = "\n".join(sink.sent)
    assert "RuntimeError" in text and "<TOKEN>" not in text and "boom" not in text


def test_probe_via_router_registers_command(tmp_path, db, store):
    scraper = DoubleScraper(per_call=1)
    ctx = BotContext(
        db_path=db.db_path, env_path=tmp_path / ".env", settings=Settings(_env_file=None),
        trigger_cycle=lambda **_: TRIGGER_STARTED, is_cycle_running=lambda: False, scraper=scraper,
    )
    fake = FakeTelegram()
    router = CommandRouter(ctx, client=TelegramBotClient("<TOKEN>", transport=fake.transport))
    reply = router.handle(_update("/test <ROLE>"))
    assert reply[0].text.startswith("Searching")
    deadline = time.time() + 10
    while not fake.calls("sendMessage") and time.time() < deadline:
        time.sleep(0.02)
    assert "This does not measure relevance." in fake.calls("sendMessage")[0]["text"]


# --- scraper integration ------------------------------------------------------------------


def _df(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"title": f"Title {i}", "job_url": f"https://example.test/{i}", "company": "Example Co",
             "location": "City", "description": LONG}
            for i in range(n)
        ]
    )


def test_build_scrape_kwargs_results_wanted_override():
    scraper = JobSpyScraper(results_per_query=10)
    query = SearchQuery("Q", "term", "loc", ["google"], None)
    assert scraper._build_scrape_kwargs("google", query)["results_wanted"] == 10
    assert scraper._build_scrape_kwargs("google", query, results_wanted=5)["results_wanted"] == 5
    assert scraper._build_scrape_kwargs("google", query, None)["results_wanted"] == 10


def test_scrape_board_outcome_passes_results_wanted(monkeypatch):
    seen = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return _df(1)

    monkeypatch.setattr("src.scraper.jobspy_scraper.scrape_jobs", fake)
    scraper = JobSpyScraper(jitter_min=0, jitter_max=0)
    query = SearchQuery("Q", "term", "loc", ["google"], None)
    scraper.scrape_board_outcome("google", query, track_cycle=False, results_wanted=5)
    assert seen["results_wanted"] == 5
    scraper.scrape_board_outcome("google", query, track_cycle=False)
    assert seen["results_wanted"] == 10


def test_probe_interleaves_with_cycle_calls(monkeypatch, tmp_path, db, store):
    """A real coordinator: the probe's call runs between the cycle's calls, not after all of them."""
    events: List[tuple] = []
    lock = threading.Lock()

    def slow_scrape_jobs(**kwargs):
        with lock:
            events.append(("start", kwargs["search_term"], time.monotonic()))
        time.sleep(0.2)
        with lock:
            events.append(("end", kwargs["search_term"], time.monotonic()))
        return _df(1)

    monkeypatch.setattr("src.scraper.jobspy_scraper.scrape_jobs", slow_scrape_jobs)
    scraper = JobSpyScraper(jitter_min=0, jitter_max=0, linkedin_cooldown_seconds=0)
    cycle_started = threading.Event()

    def cycle():
        for n in range(4):
            scraper.scrape_board_outcome("google", SearchQuery(f"C{n}", f"cycle {n}", "loc", ["google"]))
            cycle_started.set()
            time.sleep(0.03)  # stands in for the jitter between cycle calls

    cycle_thread = threading.Thread(target=cycle)
    cycle_thread.start()
    assert cycle_started.wait(5)

    plan = [(SearchQuery("P", "probe", "loc", ["google"]), "google")]
    probe_thread = threading.Thread(target=run_probe, args=(scraper, db, plan, ()))
    probe_thread.start()
    cycle_thread.join(timeout=10)
    probe_thread.join(timeout=10)
    assert not cycle_thread.is_alive() and not probe_thread.is_alive()

    ends = [e for e in events if e[0] == "end"]
    names = [e[1] for e in ends]
    assert "probe" in names
    assert names.index("probe") < names.index("cycle 3")


# --- review fixes -------------------------------------------------------------------------


def _skip_google(site, query):
    return "disabled" if site.lower() == "google" else None


def test_plan_drops_scraper_skipped_boards_before_the_call_budget(store):
    snap = _snapshot(store)
    plain = plan_probe(snap, "<ROLE>", include_linkedin=False)
    assert len(plain) == TEST_MAX_CALLS and "google" in {site for _, site in plain}
    plan = plan_probe(snap, "<ROLE>", include_linkedin=False, board_skip_reason=_skip_google)
    assert plan and all(site == "indeed" for _, site in plan)
    assert len(plan) == 2  # indeed for the name and the synonym: nothing is padded with skipped calls


def test_probe_reply_names_skipped_boards(tmp_path, db, store):
    scraper = DoubleScraper(per_call=1)
    scraper.board_skip_reason = _skip_google
    runner, sink = _runner(tmp_path, db, scraper)
    runner.request(_update("/test <ROLE>"))
    _wait(runner)
    assert all(c["site"] != "google" for c in scraper.calls)
    assert "Skipped boards: google (disabled in SCRAPER_ENABLED_BOARDS)" in "\n".join(sink.sent)


def test_probe_with_everything_skipped_says_so_and_keeps_cooldown(tmp_path, db, store):
    scraper = DoubleScraper()
    scraper.board_skip_reason = lambda site, query: "disabled"
    runner, sink = _runner(tmp_path, db, scraper)
    reply = runner.request(_update("/test <ROLE>"))[0].text
    assert "SCRAPER_ENABLED_BOARDS" in reply and "google" in reply
    assert runner._thread is None and scraper.calls == []


def test_probe_with_real_scraper_enabled_boards(tmp_path, db, store, monkeypatch):
    monkeypatch.setattr("src.scraper.jobspy_scraper.scrape_jobs", lambda **kw: _df(1))
    scraper = JobSpyScraper(jitter_min=0, jitter_max=0, enabled_boards=["indeed"])
    runner, sink = _runner(tmp_path, db, scraper)
    runner.request(_update("/test <ROLE>"))
    _wait(runner)
    text = "\n".join(sink.sent)
    assert "rows 2" in text and "Skipped boards: google" in text


def test_probe_retries_an_undelivered_result_once(tmp_path, db, store, caplog):
    sink = Sink(fail_first=1)
    runner, _ = _runner(tmp_path, db, DoubleScraper(per_call=1), sink=sink, stop_event=threading.Event())
    runner.RETRY_WAIT_SECONDS = 0.01
    runner.request(_update("/test <ROLE>"))
    _wait(runner)
    assert sink.attempts == 2 and "This does not measure relevance." in "\n".join(sink.sent)


def test_probe_gives_up_after_one_retry_without_logging_content(tmp_path, db, store, caplog):
    sink = Sink(fail_first=99)
    runner, _ = _runner(tmp_path, db, DoubleScraper(per_call=1), sink=sink, stop_event=threading.Event())
    runner.RETRY_WAIT_SECONDS = 0.01
    with caplog.at_level("WARNING"):
        runner.request(_update("/test <ROLE>"))
        _wait(runner)
    assert sink.attempts == 2 and sink.sent == []
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert warnings and not any("Title" in w or "<ROLE>" in w for w in warnings)


def test_probe_retry_wait_stops_on_shutdown(tmp_path, db, store):
    stop = threading.Event()
    stop.set()
    sink = Sink(fail_first=99)
    runner, _ = _runner(tmp_path, db, DoubleScraper(per_call=1), sink=sink, stop_event=stop)
    runner.RETRY_WAIT_SECONDS = 30
    started = time.monotonic()
    runner.request(_update("/test <ROLE>"))
    _wait(runner)
    assert time.monotonic() - started < 5 and sink.attempts == 1


def test_probe_validates_the_role_name_before_searching(tmp_path, db, store):
    scraper = DoubleScraper()
    runner, _ = _runner(tmp_path, db, scraper)
    reply = runner.request(_update("/test " + "x" * 200))[0].text
    assert reply.startswith("That role name is not valid")
    assert runner._thread is None and scraper.calls == []
    assert runner.request(_update("/test a=b"))[0].text.startswith("That role name is not valid")
