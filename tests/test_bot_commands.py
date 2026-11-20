"""CommandRouter: read, change and safety commands through ops (no network, no real .env)."""
import re
from dataclasses import replace
from pathlib import Path
from typing import List

import pytest

from src.bot.client import TelegramBotClient
from src.bot.commands import (
    COMMANDS,
    MAX_ARGS_CHARS,
    MUTATING_COMMANDS,
    TRIGGER_BUSY,
    TRIGGER_NOT_READY,
    TRIGGER_STARTED,
    USER_RULE_REASON,
    BotContext,
    CommandRouter,
)
from src.bot.render import utf16_len
from src.bot.updates import OutgoingMessage, parse_update
from src.config import Settings
from src.profile.ops import (
    AddMarket,
    AddRole,
    AddRule,
    SetField,
    apply_changes,
    create_starter_profile,
)
from src.profile.snapshot import Market, RoleTerm, Rule
from src.profile.store import ProfileStore
from src.storage.db import Database, compute_dedup_hash
from src.storage.models import JobPosting
from tests.fake_telegram import FakeTelegram, callback_update, message_update

LONG = " ".join(f"word{n}" for n in range(60))


class Control:
    def __init__(self):
        self.trigger_result = TRIGGER_STARTED
        self.trigger_calls = 0
        self.notify_flags = []
        self.running = False

    def trigger(self, notify: bool = False) -> str:
        self.trigger_calls += 1
        self.notify_flags.append(notify)
        return self.trigger_result

    def is_running(self) -> bool:
        return self.running


@pytest.fixture
def control():
    return Control()


@pytest.fixture
def db(tmp_path: Path) -> Database:
    database = Database(tmp_path / "bot.db")
    database.init_db()
    return database


@pytest.fixture
def store(db: Database) -> ProfileStore:
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


@pytest.fixture
def telegram() -> FakeTelegram:
    return FakeTelegram()


@pytest.fixture
def router(tmp_path, db, store, control, telegram) -> CommandRouter:
    ctx = BotContext(
        db_path=db.db_path,
        env_path=tmp_path / ".env",
        settings=Settings(_env_file=None),
        trigger_cycle=control.trigger,
        is_cycle_running=control.is_running,
        scraper=None,
    )
    client = TelegramBotClient("<TOKEN>", transport=telegram.transport)
    return CommandRouter(ctx, client=client)


def send(router: CommandRouter, text: str, update_id: int = 1) -> str:
    messages = router.handle(parse_update(message_update(update_id, text)))
    for message in messages:
        assert utf16_len(message.text) <= 3800
    return "\n".join(m.text for m in messages)


def version(store: ProfileStore) -> int:
    return store.read_snapshot().version


def _job(n: int, title: str, **kw) -> JobPosting:
    return JobPosting(
        dedup_hash=compute_dedup_hash("Example Co", title, f"City {n}"),
        site="linkedin",
        job_url=f"https://example.test/{n}",
        title=title,
        company="Example Co",
        location=f"City {n}",
        description=LONG,
        **kw,
    )


# --- change commands ----------------------------------------------------------------------


def test_add_role_with_synonyms_bumps_version_and_shows_diff(router, store):
    before = version(store)
    versions = len(store.list_versions())
    reply = send(router, "/add_role Career Coach = life coach, mentor")
    assert version(store) == before + 1
    assert len(store.list_versions()) == versions + 1
    assert "+ role Career Coach" in reply
    assert "+ term life coach" in reply
    assert "+ term mentor" in reply
    role = [r for r in store.read_snapshot().roles if r.name == "Career Coach"][0]
    assert [t.term for t in role.terms] == ["life coach", "mentor"]


def test_add_role_existing_adds_only_new_synonyms(router, store):
    reply = send(router, "/add_role <role> = <syn>, other term")
    snapshot = store.read_snapshot()
    role = [r for r in snapshot.roles if r.name == "<ROLE>"][0]
    assert [t.term for t in role.terms] == ["<SYN>", "other term"]
    assert "+ term other term" in reply and "<SYN>" not in reply.replace("&lt;SYN&gt;", "")
    before = version(store)
    again = send(router, "/add_role <ROLE> = other TERM")
    assert "already exists" in again
    assert version(store) == before
    assert "&lt;ROLE&gt;" in again


def test_add_role_rejects_separator(router, store):
    before = version(store)
    reply = send(router, "/add_role a;b")
    assert reply.startswith("Not applied")
    assert version(store) == before


def test_add_role_usage(router):
    assert "Usage: /add_role" in send(router, "/add_role")
    assert "Usage: /add_role" in send(router, "/add_role = x")


def test_remove_role(router, store):
    reply = send(router, "/remove_role <role>")
    assert "- role &lt;ROLE&gt;" in reply
    assert not store.read_snapshot().roles
    assert send(router, "/remove_role nothing").startswith("Not applied")
    assert "Usage: /remove_role" in send(router, "/remove_role")


def test_add_market_ambiguous_asks_and_changes_nothing(router, store):
    before = version(store)
    reply = send(router, "/add_market united")
    assert reply.startswith("Which one did you mean?")
    assert "United Kingdom" in reply and "United States" in reply
    assert version(store) == before


def test_add_market_unknown_country(router, store):
    reply = send(router, "/add_market <atlantis>")
    assert reply == "Unknown country '&lt;atlantis&gt;'. Try the English name, e.g. Germany."


def test_add_market_adds_and_replaces_same_name(router, store):
    reply = send(router, "/add_market Germany remote")
    assert "+ market Germany" in reply
    market = [m for m in store.read_snapshot().markets if m.name == "Germany"][0]
    assert market.countries == ("germany",) and market.remote and not market.remote_only
    assert market.location == "Germany" and market.languages == ("de", "en")
    before = version(store)
    reply = send(router, "/add_market germany")
    assert version(store) == before + 1
    market = [m for m in store.read_snapshot().markets if m.name == "Germany"][0]
    assert not market.remote
    assert len([m for m in store.read_snapshot().markets if m.name == "Germany"]) == 1
    assert "Profile v" in reply


def test_add_market_remote_only_sets_flags(router, store):
    send(router, "/add_market poland remote_only")
    market = [m for m in store.read_snapshot().markets if m.name == "Poland (remote only)"][0]
    assert market.remote and market.remote_only
    assert market.location == "Poland"


def test_add_market_region_has_no_country(router, store):
    reply = send(router, "/add_market Europe")
    market = [m for m in store.read_snapshot().markets if m.name == "Europe"][0]
    assert market.countries == () and market.languages == ("en",)
    assert "Warning:" in reply and "no country" in reply


def test_add_market_usage(router):
    assert "Usage: /add_market" in send(router, "/add_market")
    assert "Usage: /add_market" in send(router, "/add_market remote")


def test_exclude_prefix_shows_backtest_line(router, store):
    reply = send(router, "/exclude <i>x</i> prefix")
    assert "+ rule block &lt;i&gt;x&lt;/i&gt;" in reply
    assert "not meaningful" in reply
    assert "<i>" not in reply
    rule = [r for r in store.read_snapshot().rules if r.source == "user"][0]
    assert (rule.kind, rule.match_mode, rule.reason) == ("block", "prefix", USER_RULE_REASON)
    assert send(router, "/exclude <i>x</i> prefix").startswith("Not applied")  # duplicate


def test_exclude_exact_and_usage(router, store):
    send(router, "/exclude intern")
    rule = [r for r in store.read_snapshot().rules if r.source == "user"][0]
    assert rule.match_mode == "exact"
    send(router, "/exclude prefix")  # a lone "prefix" is the term, not the mode flag
    terms = {(r.term, r.match_mode) for r in store.read_snapshot().rules if r.source == "user"}
    assert ("prefix", "exact") in terms
    assert "Usage: /exclude" in send(router, "/exclude")


def test_exclude_backtest_reports_counts_with_enough_jobs(router, db):
    for n in range(205):
        db.save_job(_job(n, "Intern Helper" if n < 3 else f"Title {n}"))
    reply = send(router, "/exclude intern")
    assert "backtest: 3 of 205 stored jobs would now be dropped" in reply
    assert "  - Intern Helper" in reply


def test_threshold_comma_decimal_and_invariant_error(router, store):
    reply = send(router, "/threshold wow 8,0")
    assert "wow_threshold" in reply and store.read_snapshot().fields.wow_threshold == 8.0
    before = version(store)
    reply = send(router, "/threshold digest 9.5")
    assert reply.startswith("Not applied")
    assert version(store) == before
    send(router, "/threshold digest 7")
    assert store.read_snapshot().fields.digest_threshold == 7.0


@pytest.mark.parametrize(
    "args", ["", "wow", "wow abc", "other 7", "wow 7 8", "wow nan", "digest inf", "wow 11", "wow -1", "digest 3"]
)
def test_threshold_usage_or_rejection(router, store, args):
    before = version(store)
    reply = send(router, f"/threshold {args}".strip())
    assert reply.startswith(("Usage: /threshold", "Not applied"))
    assert version(store) == before


def test_pause_resume_triggers_cycle(router, store, control):
    reply = send(router, "/pause")
    assert "paused: False -&gt; True" in reply
    assert store.read_snapshot().fields.paused
    assert "already paused" in send(router, "/pause")
    reply = send(router, "/resume")
    assert "paused: True -&gt; False" in reply and "A cycle starts now." in reply
    assert control.trigger_calls == 1
    assert control.notify_flags == [True]  # /resume asks for a run report
    assert send(router, "/resume") == "Scraping is not paused. Send /run_now to start a cycle."
    assert control.trigger_calls == 1


def test_resume_keeps_the_reply_when_the_trigger_raises(router, store, control):
    send(router, "/pause")

    def boom(notify=False):
        raise RuntimeError("scheduler gone")

    control.trigger = boom  # type: ignore[assignment]
    router._ctx.trigger_cycle = boom
    reply = send(router, "/resume")
    assert "paused: True -&gt; False" in reply
    assert "Scraping resumes at the next scheduled cycle." in reply
    assert not store.read_snapshot().fields.paused


@pytest.mark.parametrize(
    "result, expected",
    [
        (TRIGGER_BUSY, "A cycle is already running; the next one will scrape."),
        (TRIGGER_NOT_READY, "Scraping resumes at the next scheduled cycle."),
    ],
)
def test_resume_trigger_variants(router, store, control, result, expected):
    send(router, "/pause")
    control.trigger_result = result
    assert expected in send(router, "/resume")


def test_resume_without_queries_is_refused_and_does_not_trigger(router, store, control):
    send(router, "/remove_role <role>")
    send(router, "/pause")
    before = version(store)
    reply = send(router, "/resume")
    assert reply.startswith("Not applied: cannot resume")
    assert version(store) == before
    assert control.trigger_calls == 0


def test_run_now_replies(router, control):
    assert send(router, "/run_now") == "Cycle started."
    assert control.notify_flags == [True]  # /run_now asks for a run report
    control.trigger_result = TRIGGER_BUSY
    assert send(router, "/run_now") == "A cycle is already running; try again when it has finished."
    control.trigger_result = TRIGGER_NOT_READY
    assert send(router, "/run_now") == "The scheduler is not ready yet."


def test_run_now_on_a_paused_profile_says_it_will_not_scrape(router, control, store):
    apply_changes(store, [SetField("paused", True)], label="pause")

    text = send(router, "/run_now")

    assert text != "Cycle started."
    assert "paused" in text and "/resume" in text
    assert control.trigger_calls == 1  # evaluation and digest still run


def test_run_now_refused_while_running(router, control):
    control.trigger_result = TRIGGER_BUSY
    assert "already running" in send(router, "/run_now")
    assert control.trigger_calls == 1


# --- safety commands ----------------------------------------------------------------------


def test_undo_toggles_and_nothing_to_undo(router, store):
    send(router, "/add_role Second")
    names = lambda: [r.name for r in store.read_snapshot().roles]  # noqa: E731
    assert "Second" in names()
    reply = send(router, "/undo")
    assert "- role Second" in reply and "Second" not in names()
    reply = send(router, "/undo")  # toggles back
    assert "+ role Second" in reply and "Second" in names()


def test_undo_with_nothing_to_undo(router, db, tmp_path, control, telegram):
    fresh = Database(tmp_path / "fresh.db")
    fresh.init_db()
    ctx = BotContext(fresh.db_path, tmp_path / ".env", Settings(_env_file=None), control.trigger, control.is_running, None)
    other = CommandRouter(ctx, client=TelegramBotClient("<TOKEN>", transport=telegram.transport))
    reply = send(other, "/undo")
    assert reply.startswith("Nothing to undo")


def test_history_and_restore(router, store):
    send(router, "/add_role Second")
    listing = send(router, "/history")
    lines = listing.splitlines()
    assert lines[0].startswith("v3  ") and "add role Second" in lines[0]
    assert any(line.startswith("v2  ") and "fixture" in line for line in lines)
    assert any(line.startswith("v1  ") and "init starter profile" in line for line in lines)
    assert re.match(r"v3  \d{4}-\d{2}-\d{2} \d{2}:\d{2}  add role Second", lines[0])
    reply = send(router, "/restore 2")
    assert "- role Second" in reply
    assert "Second" not in [r.name for r in store.read_snapshot().roles]
    assert "not among the last 10" in send(router, "/restore 999")
    assert "Usage: /restore" in send(router, "/restore")
    assert "Usage: /restore" in send(router, "/restore x")
    for bad in ("0", "-3"):
        before = version(store)
        reply = send(router, f"/restore {bad}")
        assert reply.startswith(("Usage: /restore", "Not applied")), bad
        assert version(store) == before


def test_requeue_refused_while_cycle_running(router, db, control):
    control.running = True
    reply = send(router, "/requeue")
    assert "A cycle is running" in reply


def _prefiltered(db: Database, n: int, title: str) -> int:
    job_id, _ = db.save_job(_job(n, title))
    with db._get_connection() as conn:
        conn.execute(
            "UPDATE jobs SET pre_filtered = 1, pre_filter_reason = 'EXCLUDED_USER_TERM: x', "
            "notification_status = 'dismissed' WHERE id = ?",
            (job_id,),
        )
    return job_id


def test_requeue_and_thresholds_report_counts(router, db):
    _prefiltered(db, 1, "Alpha Role")
    _prefiltered(db, 2, "Beta Role")
    reply = send(router, "/requeue")
    assert "checked: 2, requeued: 2, still dropped: 0, too old: 0" in reply
    assert "  - Alpha Role" in reply or "  - Beta Role" in reply
    reply = send(router, "/requeue thresholds")
    assert "checked: 0, changed: 0, skipped (missing stored fields): 0, to digest: 0, to dismissed: 0, too old: 0" in reply
    assert "Usage: /requeue" in send(router, "/requeue bogus")


def test_requeue_thresholds_reports_a_non_zero_rebucket(router, db, store):
    job_id, _ = db.save_job(_job(1, "<b>Scored</b>", eval_status="evaluated", eval_score=5.0,
                                 notification_status="dismissed"))
    with db._get_connection() as conn:
        conn.execute(
            "UPDATE jobs SET eval_raw_score = 7.0, eval_is_core_role = 1, eval_bonus_flag = 0, "
            "eval_hard_reject = 0 WHERE id = ?",
            (job_id,),
        )
    reply = send(router, "/requeue thresholds")
    assert "checked: 1, changed: 1, skipped (missing stored fields): 0, to digest: 1, to dismissed: 0" in reply
    assert "  - &lt;b&gt;Scored&lt;/b&gt;" in reply and "<b>" not in reply
    assert db.get_job_by_id(job_id).notification_status == "pending"


def test_requeue_samples_are_escaped_and_capped(router, db):
    for n in range(8):
        _prefiltered(db, n, f"<b>Role {n}</b>")
    reply = send(router, "/requeue")
    assert "requeued: 8" in reply
    assert "&lt;b&gt;Role" in reply and "<b>Role" not in reply
    assert reply.count("  - ") == 5


# --- read commands ------------------------------------------------------------------------


def test_dropped_lists_last_20_escaped(router, db):
    assert "Nothing has been pre-filtered" in send(router, "/dropped")
    for n in range(25):
        _prefiltered(db, n, f"<i>Title {n}</i>")
    reply = send(router, "/dropped")
    lines = reply.splitlines()
    assert len(lines) == 20
    assert lines[0] == "&lt;i&gt;Title 24&lt;/i&gt; - Example Co (linkedin): EXCLUDED_USER_TERM: x"
    assert "<i>" not in reply


def test_preview_paginates(router, store):
    names = [AddRole(f"Role {n:02d}") for n in range(39)]
    apply_changes(store, [SetField("max_calls_per_cycle", 1000)] + names, label="many")
    first = send(router, "/preview")
    assert "page 1/3" in first and "send /preview 2" in first
    last = send(router, "/preview 3")
    assert "page 3/3" in last and "send /preview" not in last
    assert "calls per cycle" in last.lower()
    assert "Usage: /preview" in send(router, "/preview x")
    # an out-of-range page clamps to the last page
    assert "page 3/3" in send(router, "/preview 99")


def test_preview_lists_query_boards(router):
    reply = send(router, "/preview")
    assert "role=&lt;ROLE&gt;;term=&lt;ROLE&gt;;market=&lt;MARKET&gt;: linkedin, google" in reply
    assert "LinkedIn calls" in reply and "cap" in reply


def test_profile_overview(router):
    reply = send(router, "/profile")
    assert "Profile v2" in reply
    assert "running" in reply
    assert "WOW 8.5" in reply and "digest 6.5" in reply
    assert "Output language: en" in reply
    assert "Roles: 1" in reply and "markets: 1" in reply
    assert "Calls per cycle: 6 of 100" in reply


def test_profile_without_profile(tmp_path, control, telegram):
    fresh = Database(tmp_path / "fresh.db")
    fresh.init_db()
    ctx = BotContext(fresh.db_path, tmp_path / ".env", Settings(_env_file=None), control.trigger, control.is_running, None)
    other = CommandRouter(ctx, client=TelegramBotClient("<TOKEN>", transport=telegram.transport))
    for command in ("/profile", "/roles", "/markets", "/exclusions", "/preview"):
        assert "No profile yet" in send(other, command)


def test_roles_markets_exclusions_format(router, store):
    send(router, "/exclude intern")
    apply_changes(
        store,
        [AddRole("Other", (RoleTerm("anderes", "de"), RoleTerm("plain", None)))],
        label="x",
    )
    roles = send(router, "/roles").splitlines()
    assert roles[0] == "&lt;ROLE&gt; [enabled]: &lt;SYN&gt;"
    assert roles[1] == "Other [enabled]: anderes [de], plain"
    markets = send(router, "/markets")
    assert markets == "&lt;MARKET&gt;: &lt;LOCATION&gt;; poland; not remote; pl, en; enabled"
    exclusions = send(router, "/exclusions")
    assert "block:" in exclusions and "  intern (exact)" in exclusions


def test_exclusions_group_by_kind_and_count_seed_rules(router, store):
    from src.profile.legacy_seed import seed_rules

    apply_changes(
        store,
        [
            AddRule(Rule("soft", "junior", "prefix", "SOFT", "llm")),
            AddRule(Rule("block", "intern", "exact", USER_RULE_REASON, "user")),
            *[AddRule(r) for r in seed_rules()],
        ],
        label="rules",
    )
    reply = send(router, "/exclusions")
    assert reply.index("block:") < reply.index("soft:")
    assert "  junior (prefix)" in reply
    seed_count = len(seed_rules())
    assert f"{seed_count} built-in legacy patterns" in reply
    assert "(pattern)" not in reply


def test_user_terms_escaped_everywhere(router, store):
    role = ("<b>& R&D " + "x" * 70)[:70]
    reply_add = send(router, f"/add_role {role}")
    assert "&lt;b&gt;&amp; R&amp;D" in reply_add
    reply_rule = send(router, "/exclude <i>&amp;x</i>")
    assert "+ rule block &lt;i&gt;&amp;amp;x&lt;/i&gt;" in reply_rule
    for command in ("/roles", "/exclusions", "/profile", "/history", "/markets", "/preview", "/dropped"):
        text = send(router, command)
        assert "<i>" not in text and "<b>&" not in text, command
    assert "&lt;b&gt;&amp; R&amp;D" in send(router, "/roles")
    assert "&lt;i&gt;&amp;amp;x&lt;/i&gt;" in send(router, "/exclusions")
    assert "&lt;b&gt;" in send(router, "/profile")


def test_exclusions_header_is_never_orphaned_at_a_page_end(router, store):
    block = [AddRule(Rule("block", f"term{n:02d}", "exact", USER_RULE_REASON, "user")) for n in range(28)]
    soft = [AddRule(Rule("soft", f"soft{n:02d}", "exact", "SOFT", "user")) for n in range(3)]
    apply_changes(store, block + soft, label="rules")
    first = send(router, "/exclusions").splitlines()
    assert first[0] == "<b>block:</b>"
    body = [line for line in first if not line.startswith("page ")]
    assert not body[-1].startswith("<b>")
    second = send(router, "/exclusions 2").splitlines()
    assert second[0] == "<b>soft:</b>"
    assert second.count("<b>soft:</b>") == 1


def test_long_emoji_term_in_storage_is_chunked_through_roles(router, store, db):
    snapshot = store.read_snapshot()
    huge = RoleTerm("<b>& R&D \U0001F600" * 400, None)  # about 5,000 characters, bypassing ops limits
    role = replace(snapshot.roles[0], terms=snapshot.roles[0].terms + (huge,))
    store.write_snapshot(replace(snapshot, roles=(role,)), expected_version=snapshot.version, label="huge")
    messages = router.handle(parse_update(message_update(1, "/roles")))
    assert len(messages) >= 2
    for message in messages:
        assert utf16_len(message.text) <= 3800
        assert "<b>" not in message.text
        assert not re.search(r"&[#\w]*$", message.text)


def test_long_output_is_chunked_within_limits(router, store):
    terms = tuple(RoleTerm(f"<b>& {n} R&D \U0001F600 " + "y" * 40, None) for n in range(70))
    apply_changes(store, [SetField("max_calls_per_cycle", 1000), AddRole("Long", terms)], label="long")
    messages = router.handle(parse_update(message_update(1, "/roles")))
    assert len(messages) >= 2
    for message in messages:
        assert utf16_len(message.text) <= 3800
        assert "<b>" not in message.text
        assert not re.search(r"&\w*$", message.text)  # no entity cut at a chunk end


# --- routing -----------------------------------------------------------------------------


def test_help_lists_every_command(router):
    for command in ("/help", "/start"):
        reply = send(router, command)
        for name, description in COMMANDS:
            assert f"/{name} - " in reply
    names = [name for name, _ in COMMANDS]
    assert names[:7] == ["help", "profile", "roles", "markets", "exclusions", "preview", "dropped"]
    assert {"setup", "stats", "test", "undo", "history", "restore", "requeue", "run_now"} <= set(names)
    assert len(names) == len(set(names))
    for name, description in COMMANDS:
        assert name == name.lower() and 1 <= len(name) <= 32
        assert 3 <= len(description) <= 256


def test_unknown_command_and_plain_text_hint(router):
    assert send(router, "/nonsense") == "Unknown command. Send /help."
    assert send(router, "hello there") == "Send /help to see what I can do."
    assert "Unknown command" not in send(router, "/setup")  # the wizard is plugged in by the router itself


def test_plain_text_without_llm_keeps_hint(router):
    # no chat LLM behind the router: free text, bare words and uploads keep today's hint, nothing is queued
    from tests.fake_telegram import document_update

    assert router.conversation is not None
    assert send(router, "please add a role") == "Send /help to see what I can do."
    messages = router.handle(parse_update(document_update(5, "f1", "notes.txt")))
    assert [m.text for m in messages] == ["Send /help to see what I can do."]
    assert router.conversation.process_next(10**12) is False


def test_ok_and_cancel_are_routed_to_the_conversation(router):
    assert send(router, "/ok") == "There is no pending proposal. Send /help for the commands."
    assert send(router, "/cancel") == "There is nothing to cancel."
    assert send(router, "yes") == "There is no pending proposal. Send /help for the commands."
    assert "no longer active" in "\n".join(
        m.text for m in router.handle(parse_update(callback_update(7, "pc:abc:1:a")))
    )
    assert "no longer active" in "\n".join(
        m.text for m in router.handle(parse_update(callback_update(8, "cs:abc:yes")))
    )
    names = [name for name, _ in COMMANDS]
    assert names[names.index("setup") + 1:names.index("setup") + 3] == ["ok", "cancel"]


def test_args_too_long(router, store):
    before = version(store)
    assert send(router, "/add_role " + "x" * (MAX_ARGS_CHARS + 1)) == "That is too long."
    assert version(store) == before


def test_command_with_bot_suffix_and_case(router):
    assert "WOW" in send(router, "/PROFILE@some_bot")


def test_register_handlers(router):
    router.register("/hello", lambda update: [OutgoingMessage("hi " + update.args)])
    router.register("setup", lambda update: [OutgoingMessage("wizard")])
    assert send(router, "/hello you") == "hi you"
    assert send(router, "/setup") == "wizard"
    seen: List[str] = []
    router.register_callback("xx:", lambda update: seen.append(update.text) or [OutgoingMessage("step")])
    messages = router.handle(parse_update(callback_update(2, "xx:abc")))
    assert [m.text for m in messages] == ["step"] and seen == ["xx:abc"]
    messages = router.handle(parse_update(callback_update(3, "other:1")))
    assert [m.text for m in messages] == ["This button is no longer active."]
    router.set_text_handler(lambda update: [OutgoingMessage("text " + update.text)] if update.text == "ok" else None)
    assert send(router, "ok") == "text ok"
    assert send(router, "nope") == "Send /help to see what I can do."


def test_errors_are_translated(router, store, monkeypatch):
    from src.profile.errors import ProfileCompileError, ProfileValidationError, VersionConflict
    import src.bot.commands as commands

    def conflict(*a, **kw):
        raise VersionConflict("x")

    monkeypatch.setattr(commands, "apply_changes", conflict)
    assert send(router, "/pause") == "The profile changed at the same time; send the command again."

    def invalid(*a, **kw):
        raise ProfileValidationError(["<one>", "two"])

    monkeypatch.setattr(commands, "apply_changes", invalid)
    reply = send(router, "/pause")
    assert reply.startswith("Not applied: &lt;one&gt;") and "two" in reply

    def broken(*a, **kw):
        raise ProfileCompileError("bad <regex>")

    monkeypatch.setattr(commands, "apply_changes", broken)
    assert send(router, "/pause") == "The stored profile cannot be compiled: bad &lt;regex&gt;"


def test_mutating_set_matches_spec(router):
    assert MUTATING_COMMANDS == frozenset(
        {"/add_role", "/remove_role", "/add_market", "/exclude", "/threshold", "/pause", "/resume",
         "/run_now", "/undo", "/restore", "/requeue", "/setup", "/test", "/ok", "/cancel"}
    )
    for command in MUTATING_COMMANDS:
        assert router.is_mutating(parse_update(message_update(1, f"{command} x")))
        assert router.is_mutating(parse_update(message_update(1, f"{command.upper()}@bot x")))
    for command in ("/help", "/start", "/profile", "/roles", "/markets", "/exclusions", "/preview",
                    "/dropped", "/history", "/stats"):
        assert not router.is_mutating(parse_update(message_update(1, command)))
    assert not router.is_mutating(parse_update(message_update(1, "plain text")))
    assert not router.is_mutating(parse_update(callback_update(1, "wz:1")))


def test_market_from_match_is_what_add_market_stores():
    from src.bot.commands import market_from_match
    from src.bot.countries import resolve_country

    match = resolve_country("Germany").match
    assert market_from_match(match, False, False) == Market("Germany", "Germany", ("germany",), False, False, ("de", "en"))
    remote_only = market_from_match(match, False, True)
    assert remote_only.name == "Germany (remote only)" and remote_only.remote and remote_only.remote_only
    region = market_from_match(resolve_country("Europe").match, True, False)
    assert region.countries == () and region.location == "Europe" and region.remote
