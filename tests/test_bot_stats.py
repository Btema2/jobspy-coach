"""/stats: per-role scrape statistics over 7 days (db readers, grouping, rendering, command)."""
import re
import sqlite3
from pathlib import Path

import pytest

from src.bot.client import TelegramBotClient
from src.bot.commands import TRIGGER_STARTED, BotContext, CommandRouter
from src.bot.render import utf16_len
from src.bot.stats import RoleStats, build_role_stats, group_label, render_stats
from src.bot.updates import parse_update
from src.config import Settings
from src.profile.compiler import expanded_query_id
from src.profile.ops import (
    AddMarket,
    AddPinnedQuery,
    AddRole,
    SetField,
    apply_changes,
    create_starter_profile,
)
from src.profile.snapshot import Market, PinnedQuery
from src.profile.store import ProfileStore
from src.storage.db import Database, QueryKeyRuns, compute_dedup_hash
from src.storage.models import JobPosting
from tests.fake_telegram import FakeTelegram, message_update


@pytest.fixture
def db(tmp_path: Path) -> Database:
    database = Database(tmp_path / "stats.db")
    database.init_db()
    return database


def _key(role: str, board: str = "linkedin", market: str = "<MARKET>", country=None, term=None) -> str:
    return expanded_query_id(role, term or role, market, country) + "|" + board


def _run(db: Database, key: str, *, outcome="ok", status="completed", new=0) -> None:
    db.record_scrape_run(
        board=key.rsplit("|", 1)[1], query="q", jobs_found=0, new_jobs_saved=new, duration_sec=0.1,
        status=status, outcome=outcome, query_key=key,
    )


def _job(db: Database, n: int, score, status="evaluated") -> int:
    job = JobPosting(
        dedup_hash=compute_dedup_hash("Example Co", f"Title {n}", f"City {n}"),
        site="linkedin", job_url=f"https://example.test/{n}", title=f"Title {n}", company="Example Co",
        location=f"City {n}", description="word " * 60,
    )
    job_id, _ = db.save_job(job)
    with db._get_connection() as conn:
        conn.execute("UPDATE jobs SET eval_status = ?, eval_score = ? WHERE id = ?", (status, score, job_id))
    return job_id


def _age(db: Database, table: str, days: int, column: str) -> None:
    with db._get_connection() as conn:
        conn.execute(f"UPDATE {table} SET {column} = datetime('now', '-{days} days')")


# --- db readers ---------------------------------------------------------------------------


def test_failed_separate_from_empty(db):
    key = _key("<ROLE>")
    _run(db, key, outcome="ok", new=3)
    _run(db, key, outcome="empty")
    _run(db, key, outcome="failed", status="failed")
    _run(db, key, outcome="skipped")  # excluded
    db.record_scrape_run("linkedin", "q", 0, 0, 0.1, "completed", outcome="ok")  # no key: excluded
    runs = db.get_query_key_runs(7)
    assert runs == [QueryKeyRuns(key, calls=3, ok=1, empty=1, failed=1, new_jobs=3)]
    rows = build_role_stats(runs, [], set(), {"<ROLE>"})
    text = "\n".join(m.text for m in render_stats(rows, 6.5, 1))
    assert "calls 3 (ok 1, empty 1, failed 1)" in text


def test_failed_counts_status_failed_even_without_outcome(db):
    key = _key("<ROLE>")
    _run(db, key, outcome="empty", status="failed")
    assert db.get_query_key_runs(7)[0].failed == 1


def test_window_is_7_days(db):
    old_key = _key("<OLD>")
    _run(db, old_key)
    _age(db, "scrape_runs", 8, "run_time")
    _run(db, _key("<NEW>"))
    runs = db.get_query_key_runs(7)
    assert [r.query_key for r in runs] == [_key("<NEW>")]

    old_job, new_job = _job(db, 1, 9.0), _job(db, 2, 9.0)
    db.record_job_hit(old_job, old_key)
    _age(db, "job_hits", 8, "last_seen")
    db.record_job_hit(new_job, _key("<NEW>"))
    assert db.get_relevant_hits(7, 6.5) == [(_key("<NEW>"), new_job)]


def test_relevant_hits_only_evaluated_and_at_or_above_threshold(db):
    key = _key("<ROLE>")
    for n, (score, status) in enumerate(
        [(6.4, "evaluated"), (6.6, "evaluated"), (9.0, "unprocessed"), (None, "evaluated")], start=1
    ):
        db.record_job_hit(_job(db, n, score, status), key)
    assert [job for _, job in db.get_relevant_hits(7, 6.5)] == [2]
    assert len(db.get_relevant_hits(7, 6.4)) == 2


# --- grouping -----------------------------------------------------------------------------


def test_role_grouping_and_removed_and_pinned(db):
    pinned_id = "role=<PIN>;term=x;market=y"  # looks expanded, is pinned
    keys = [
        _key("<ROLE>", "linkedin", "<M1>"),
        _key("<ROLE>", "indeed", "<M2>", country="poland"),
        _key("<GONE>", "google"),
        pinned_id + "|linkedin",
        "Q-LEGACY|google",
    ]
    for key in keys:
        _run(db, key, new=1)
    rows = build_role_stats(db.get_query_key_runs(7), [], {pinned_id}, {"<ROLE>"})
    labels = {r.label: r for r in rows}
    assert set(labels) == {"<ROLE>", "<GONE> (removed)", f"pinned {pinned_id}", "query Q-LEGACY"}
    assert labels["<ROLE>"].calls == 2 and labels["<ROLE>"].new_jobs == 2


def test_group_label_cases():
    pinned = "role=<PIN>;term=a;market=b"
    assert group_label(pinned + "|google", {pinned}, set()) == f"pinned {pinned}"
    assert group_label(_key("<Role>"), set(), {"<role>"}) == "<Role>"  # casefold match
    assert group_label(_key("<GONE>"), set(), {"<ROLE>"}) == "<GONE> (removed)"
    assert group_label("Q-X|indeed", set(), set()) == "query Q-X"


def test_distinct_jobs_across_keys(db):
    k1, k2 = _key("<ROLE>", "linkedin"), _key("<ROLE>", "google")
    _run(db, k1)
    _run(db, k2)
    job_id = _job(db, 1, 8.0)
    db.record_job_hit(job_id, k1)
    db.record_job_hit(job_id, k2)
    other = _job(db, 2, 8.0)
    db.record_job_hit(other, k2)
    rows = build_role_stats(db.get_query_key_runs(7), db.get_relevant_hits(7, 6.5), set(), {"<ROLE>"})
    assert rows[0].relevant == 2


def test_rows_sorted_by_relevant_then_calls_then_label():
    rows = build_role_stats(
        [
            QueryKeyRuns(_key("B"), 5, 5, 0, 0, 0),
            QueryKeyRuns(_key("A"), 5, 5, 0, 0, 0),
            QueryKeyRuns(_key("C"), 1, 1, 0, 0, 0),
        ],
        [(_key("C"), 1)],
        set(),
        {"A", "B", "C"},
    )
    assert [r.label for r in rows] == ["C", "A", "B"]


# --- rendering ----------------------------------------------------------------------------


def test_render_empty_and_header():
    assert render_stats([], 6.5, 1)[0].text == "No scrape runs in the last 7 days."
    text = render_stats([RoleStats("<ROLE>", 3, 2, 1, 0, 4, 2)], 6.5, 1)[0].text
    assert "7 days" in text and "relevant = evaluated, score &gt;= 6.5" in text
    assert "&lt;ROLE&gt;: calls 3 (ok 2, empty 1, failed 0), new 4, relevant 2" in text


def test_stats_reply_escaped_and_paginated():
    rows = [RoleStats(f"<b>&amp; R&D {n}" + "x" * 80, 1, 1, 0, 0, 0, 0) for n in range(70)]
    pages = [render_stats(rows, 6.5, page) for page in (1, 2, 3)]
    for messages in pages:
        for m in messages:
            assert utf16_len(m.text) <= 3800
            assert "<b>" not in m.text.replace("<b>Stats", "")
            assert "&amp;amp; R&amp;D" in m.text or "page" in m.text
    assert "page 1/3" in pages[0][-1].text and "/stats 2" in pages[0][-1].text
    assert "/stats 4" not in pages[2][-1].text
    assert render_stats(rows, 6.5, 99)[-1].text.count("page 3/3") == 1  # clamped


# --- the command --------------------------------------------------------------------------


@pytest.fixture
def router(tmp_path, db) -> CommandRouter:
    store = ProfileStore(db)
    create_starter_profile(store, wow_threshold=8.5, digest_threshold=6.5, output_language="en")
    apply_changes(
        store,
        [AddRole("<ROLE>"), AddMarket(Market("<MARKET>", "<LOCATION>", ("poland",))), SetField("paused", False)],
        label="fixture",
    )
    ctx = BotContext(
        db_path=db.db_path, env_path=tmp_path / ".env", settings=Settings(_env_file=None),
        trigger_cycle=lambda **_: TRIGGER_STARTED, is_cycle_running=lambda: False, scraper=None,
    )
    return CommandRouter(ctx, client=TelegramBotClient("<TOKEN>", transport=FakeTelegram().transport))


def _stats(router, args="") -> str:
    update = parse_update(message_update(1, f"/stats {args}".strip()))
    return "\n".join(m.text for m in router.handle(update))


def test_relevant_uses_current_digest_threshold(router, db):
    key = _key("<ROLE>")
    _run(db, key)
    db.record_job_hit(_job(db, 1, 6.4), key)
    db.record_job_hit(_job(db, 2, 6.6), key)
    assert re.search(r"calls 1 .*relevant 1\b", _stats(router))

    apply_changes(ProfileStore(db), [SetField("digest_threshold", 6.0)], label="lower")
    text = _stats(router)
    assert "score &gt;= 6" in text and re.search(r"relevant 2\b", text)


def test_stats_command_pinned_and_removed(router, db):
    apply_changes(
        ProfileStore(db),
        [AddPinnedQuery(PinnedQuery("role=<P>;term=t;market=m", "t", "loc", ("linkedin",)))],
        label="pin",
    )
    _run(db, "role=<P>;term=t;market=m|linkedin")
    _run(db, _key("<GONE>"))
    text = _stats(router)
    assert "pinned role=&lt;P&gt;;term=t;market=m" in text
    assert "&lt;GONE&gt; (removed)" in text


def test_stats_command_usage_and_no_profile(tmp_path, db):
    ctx = BotContext(
        db_path=db.db_path, env_path=tmp_path / ".env", settings=Settings(_env_file=None, DIGEST_SCORE_THRESHOLD=7.0),
        trigger_cycle=lambda **_: TRIGGER_STARTED, is_cycle_running=lambda: False, scraper=None,
    )
    router = CommandRouter(ctx, client=TelegramBotClient("<TOKEN>", transport=FakeTelegram().transport))
    _run(db, _key("<ROLE>"))
    text = _stats(router)  # no profile: the .env digest threshold is used
    assert "score &gt;= 7" in text
    assert "Usage" in _stats(router, "abc")
