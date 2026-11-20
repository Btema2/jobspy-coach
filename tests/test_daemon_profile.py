"""The daemon scrapes and evaluates from the compiled profile (one snapshot per cycle)."""
import json
import sqlite3
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import pytest

from src.config import Settings
from src.evaluator.chain import ProviderChainEvaluator
from src.evaluator.circuit_breaker import CircuitBreaker
from src.evaluator.providers import CompletionResult
from src.evaluator.schemas import JobEvaluationResult
from src.profile.compiler import compile_profile
from src.profile.importer import build_imported_snapshot, parse_queries_json
from src.profile.errors import ProfileCompileError
from src.profile.ops import AddPinnedQuery, ReplaceProfile, SetField, apply_changes
from src.profile.snapshot import PinnedQuery, ProfileFields, ProfileSnapshot
from src.scheduler.daemon import STOPPED_PROFILE_BUSY, DaemonService, EvalSummary
from src.scraper import query_catalog
from src.scraper.jobspy_scraper import JobSpyScraper
from src.scraper.outcome import STATUS_EMPTY, ScrapeOutcome
from src.scraper.query_catalog import SearchQuery
from src.storage.models import JobPosting

FIXTURES = Path(__file__).parent / "fixtures"
QUERIES_FIXTURE = FIXTURES / "legacy_queries.json"
PERSONA_FIXTURE = FIXTURES / "legacy_persona.md"


@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path, monkeypatch):
    """Nothing here may read the real queries.json / .candidate_profile.md from the repo."""
    monkeypatch.chdir(tmp_path)


def _legacy_parse(path: Path) -> List[SearchQuery]:
    """The former queries.json loader mapping (proves import equivalence)."""
    return [
        SearchQuery(
            query_id=item["query_id"],
            search_term=item["search_term"],
            location=item["location"],
            site_names=item["site_names"],
            country_code=item.get("country_code"),
            is_remote=item.get("is_remote", False),
            google_search_term=item.get("google_search_term"),
        )
        for item in json.loads(path.read_text(encoding="utf-8"))
    ]


def _settings(**overrides) -> Settings:
    base = dict(EVAL_MAX_WORKERS=2, EVAL_MAX_ATTEMPTS=2, EVAL_BATCH_LIMIT=50)
    base.update(overrides)
    return Settings(_env_file=None, **base)


class FakeScraper:
    """No begin_cycle / scrape_board_outcome: the daemon runs every listed board."""

    def __init__(self, on_call=None):
        self.calls: List[Tuple[str, str]] = []
        self.on_call = on_call

    def scrape_board(self, site, query):
        self.calls.append((query.query_id, site))
        if self.on_call is not None and len(self.calls) == 1:
            self.on_call()
        return []


class RuleScraper:
    """Scraper double with the real ``scrape_board_outcome`` keyword; records the rules it was given."""

    def __init__(self, on_call=None):
        self.calls: List[Tuple[str, str]] = []
        self.rules: List[object] = []
        self.on_call = on_call

    def scrape_board_outcome(self, site, query, *, track_cycle=True, rules=None):
        self.calls.append((query.query_id, site))
        self.rules.append(rules)
        if self.on_call is not None and len(self.calls) == 1:
            self.on_call()
        return ScrapeOutcome(postings=[], rows=0, status=STATUS_EMPTY)


class RecordingChain:
    """Test double with the per-call policy signature; scores a job by its title (default 7.0)."""

    def __init__(self, scores: Optional[Dict[str, float]] = None, is_core_role: bool = True, on_call=None):
        self.prompts: List[Optional[str]] = []
        self.policies: List[object] = []
        self.scores = scores or {}
        self.is_core_role = is_core_role
        self.on_call = on_call
        self.calls = 0

    def evaluate_job(self, job, policy=None):
        self.calls += 1
        if self.on_call is not None and self.calls == 1:
            self.on_call()
        self.policies.append(policy)
        self.prompts.append(policy.system_prompt if policy is not None else None)
        return JobEvaluationResult(
            score=self.scores.get(job.title, 7.0),
            summary="A sufficiently long summary.",
            reasoning="Because.",
            eval_model="m/x",
            is_core_role=self.is_core_role,
        )


class ScriptedProvider:
    """Provider double answering with a JSON payload built from the job title; records every system prompt."""

    name = "scripted"
    model = "vendor/scripted"
    max_tokens = 100

    def __init__(self, payload: Callable[[str], dict]):
        self.payload = payload
        self.systems: List[str] = []
        self.breaker = CircuitBreaker(failure_threshold=3, cooldown_seconds=600)

    def is_available(self) -> bool:
        return True

    def complete(self, system, user, max_tokens=None):
        self.systems.append(system)
        title = user.split("Title:", 1)[1].splitlines()[0].strip() if "Title:" in user else ""
        return CompletionResult(
            content=json.dumps(self.payload(title)), model_used=self.model, latency=0.0, finish_reason="stop"
        )


def _answer(score: float, **extra) -> dict:
    return {"score": score, "summary": "A sufficiently long summary.", "reasoning": "r", **extra}


class LegacyChain:
    """Test double with the pre-profile one-argument signature."""

    def __init__(self):
        self.jobs: List[str] = []

    def evaluate_job(self, job):
        self.jobs.append(job.title)
        return JobEvaluationResult(
            score=7.0, summary="A sufficiently long summary.", reasoning="Because.", eval_model="m/x"
        )


def _make(tmp_path, scraper=None, chain=None) -> DaemonService:
    daemon = DaemonService(
        db_path=tmp_path / "p.db",
        scraper=scraper if scraper is not None else FakeScraper(),
        evaluator=chain if chain is not None else RecordingChain(),
        notifier=MagicMock(),
    )
    daemon.settings = _settings()
    return daemon


def _import_fixture(daemon: DaemonService, *, paused: bool = False) -> None:
    snapshot, _ = build_imported_snapshot(
        parse_queries_json(QUERIES_FIXTURE),
        PERSONA_FIXTURE.read_text(encoding="utf-8"),
        wow_threshold=8.5,
        digest_threshold=6.5,
    )
    ops = [ReplaceProfile(snapshot)]
    apply_changes(daemon.profile_store, ops, label="test import", expected_version=0)
    if paused:
        apply_changes(daemon.profile_store, [SetField("paused", True)], label="pause")


def _seed_job(daemon: DaemonService, n: int = 1) -> None:
    daemon.db.save_job(
        JobPosting(
            dedup_hash=f"hash_{n}",
            site="indeed",
            job_url=f"https://example.test/{n}",
            title=f"Career Coach {n}",
            company="Example Co",
            location="Remote",
            description="A coaching role description that is long enough to be meaningful.",
        )
    )


def _alerts(daemon: DaemonService) -> List[str]:
    return [c.args[0] for c in daemon.notifier.send_message.call_args_list]


def _pinned(query_id: str) -> PinnedQuery:
    return PinnedQuery(query_id, "<ROLE>", "<MARKET>", ("linkedin",))


def _set_pinned_enabled(daemon: DaemonService, enabled: bool) -> None:
    """Raw SQL: ops refuses an unpaused profile without enabled queries, a hand edit does not."""
    with daemon.db._get_connection() as conn:
        conn.execute("UPDATE pinned_queries SET enabled = ?", (int(enabled),))
        conn.commit()


def _start_with_empty_compiled_list(daemon: DaemonService) -> None:
    snapshot = ProfileSnapshot(ProfileFields(), pinned_queries=(_pinned("Q-1"),))
    apply_changes(daemon.profile_store, [ReplaceProfile(snapshot)], label="p", expected_version=0)
    _set_pinned_enabled(daemon, False)


# --- scraping from the compiled profile ---------------------------------------------------------


def test_cycle_scrapes_compiled_queries(tmp_path):
    scraper = FakeScraper()
    daemon = _make(tmp_path, scraper)
    _import_fixture(daemon)

    daemon.run_pipeline_cycle()

    expected = [(q.query_id, s) for q in _legacy_parse(QUERIES_FIXTURE) for s in q.site_names]
    assert scraper.calls == expected
    assert daemon.notifier.send_message.call_count == 0


def test_cycle_uses_snapshot_taken_at_start(tmp_path):
    def commit_change():
        apply_changes(daemon.profile_store, [AddPinnedQuery(_pinned("Q-LATE"))], label="mid-cycle")

    scraper = FakeScraper(on_call=commit_change)
    daemon = _make(tmp_path, scraper)
    _import_fixture(daemon)

    daemon.run_pipeline_cycle()
    first_cycle = list(scraper.calls)
    assert all(qid != "Q-LATE" for qid, _ in first_cycle)

    scraper.calls.clear()
    daemon.run_pipeline_cycle()
    assert ("Q-LATE", "linkedin") in scraper.calls


def test_empty_compiled_list_skips_scraping_and_alerts_once(tmp_path):
    scraper = FakeScraper()
    daemon = _make(tmp_path, scraper)
    _start_with_empty_compiled_list(daemon)

    daemon.run_pipeline_cycle()
    daemon.run_pipeline_cycle()

    assert scraper.calls == []
    assert daemon.notifier.send_message.call_count == 1


def test_empty_list_alert_rearms_after_queries_return(tmp_path):
    scraper = FakeScraper()
    daemon = _make(tmp_path, scraper)
    _start_with_empty_compiled_list(daemon)
    daemon.run_pipeline_cycle()
    _set_pinned_enabled(daemon, True)
    daemon.run_pipeline_cycle()
    assert scraper.calls == [("Q-1", "linkedin")]
    _set_pinned_enabled(daemon, False)

    daemon.run_pipeline_cycle()

    assert daemon.notifier.send_message.call_count == 2


def test_no_profile_skips_scraping_and_alerts_once(tmp_path):
    scraper = FakeScraper()
    daemon = _make(tmp_path, scraper)

    daemon.run_pipeline_cycle()
    daemon.run_pipeline_cycle()

    assert scraper.calls == []
    assert _alerts(daemon) == ["no profile: run jobspy-coach init"]


def test_paused_profile_skips_scraping_but_evaluates(tmp_path):
    scraper = FakeScraper()
    daemon = _make(tmp_path, scraper)
    _import_fixture(daemon, paused=True)
    daemon.evaluate_unprocessed = MagicMock(return_value=EvalSummary(0, 0, 0, 0, None))

    daemon.run_pipeline_cycle()

    assert scraper.calls == []
    daemon.evaluate_unprocessed.assert_called_once()
    assert daemon.evaluate_unprocessed.call_args.kwargs["profile"].paused is True
    assert daemon.notifier.send_message.call_count == 0


def test_dry_run_cycle_sends_no_alert_and_does_not_evaluate(tmp_path):
    daemon = _make(tmp_path)
    daemon.evaluate_unprocessed = MagicMock(return_value=EvalSummary(0, 0, 0, 0, None))

    assert daemon.run_pipeline_cycle(dry_run=True) == (0, 0)

    assert daemon.notifier.send_message.call_count == 0
    daemon.evaluate_unprocessed.assert_not_called()


# --- compile failures ---------------------------------------------------------------------------


def _break_pattern_rules(daemon: DaemonService, term: str) -> None:
    with daemon.db._get_connection() as conn:
        conn.execute("UPDATE rules SET term = ? WHERE match_mode = 'pattern'", (term,))
        conn.commit()


def test_compile_failure_uses_last_good_and_alerts_once_per_error(tmp_path):
    scraper = FakeScraper()
    chain = RecordingChain()
    daemon = _make(tmp_path, scraper, chain)
    _import_fixture(daemon)
    good_prompt = compile_profile(daemon.profile_store.read_snapshot()).system_prompt
    daemon.run_pipeline_cycle()
    assert scraper.calls and daemon.notifier.send_message.call_count == 0

    scraper.calls.clear()
    _seed_job(daemon, 1)
    _break_pattern_rules(daemon, "(")
    daemon.run_pipeline_cycle()
    _seed_job(daemon, 2)
    daemon.run_pipeline_cycle()

    assert scraper.calls == []
    assert daemon.notifier.send_message.call_count == 1
    assert chain.prompts == [good_prompt, good_prompt]

    _break_pattern_rules(daemon, "[")
    daemon.run_pipeline_cycle()

    assert daemon.notifier.send_message.call_count == 2
    assert scraper.calls == []


def test_compile_failure_without_memory_rebuilds_from_versions(tmp_path):
    first = _make(tmp_path)
    _import_fixture(first)
    apply_changes(
        first.profile_store, [SetField("persona", "<NEWEST PERSONA>")], label="persona change"
    )
    expected_prompt = compile_profile(first.profile_store.read_snapshot()).system_prompt
    _break_pattern_rules(first, "(")

    scraper = FakeScraper()
    chain = RecordingChain()
    daemon = DaemonService(
        db_path=tmp_path / "p.db", scraper=scraper, evaluator=chain, notifier=MagicMock()
    )
    daemon.settings = _settings()
    _seed_job(daemon)

    daemon.run_pipeline_cycle()
    daemon.run_pipeline_cycle()

    assert scraper.calls == []
    assert daemon.notifier.send_message.call_count == 1
    assert chain.prompts == [expected_prompt]
    assert "<NEWEST PERSONA>" in expected_prompt


def test_unreadable_row_is_compile_error(tmp_path):
    scraper = FakeScraper()
    chain = RecordingChain()
    daemon = _make(tmp_path, scraper, chain)
    _import_fixture(daemon)
    good_prompt = compile_profile(daemon.profile_store.read_snapshot()).system_prompt
    with daemon.db._get_connection() as conn:
        conn.execute("INSERT INTO markets (name, location, countries, remote, remote_only, languages, enabled)"
                     " VALUES ('<M>', '<L>', 'not json', 0, 0, '[]', 1)")
        conn.commit()
    _seed_job(daemon)

    daemon.run_pipeline_cycle()

    assert scraper.calls == []
    assert daemon.notifier.send_message.call_count == 1
    assert chain.prompts == [good_prompt]


def test_compile_failure_with_no_fallback_evaluates_without_profile(tmp_path):
    chain = LegacyChain()
    daemon = _make(tmp_path, FakeScraper(), chain)
    _import_fixture(daemon)
    with daemon.db._get_connection() as conn:
        conn.execute("DELETE FROM profile_versions")
        conn.commit()
    _break_pattern_rules(daemon, "(")
    _seed_job(daemon)

    daemon.run_pipeline_cycle()

    assert daemon.notifier.send_message.call_count == 1
    assert chain.jobs == ["Career Coach 1"]


def test_alert_send_failure_is_logged_not_raised(tmp_path, caplog):
    daemon = _make(tmp_path)
    daemon.notifier.send_message.side_effect = RuntimeError("boom")

    daemon.run_pipeline_cycle()  # no profile -> alert attempt fails

    assert any("alert" in r.getMessage().lower() for r in caplog.records)


def test_alert_not_recorded_when_send_returns_false_so_it_retries(tmp_path):
    daemon = _make(tmp_path)
    daemon.notifier.send_message.return_value = False

    daemon._alert_once("k", "first")
    daemon._alert_once("k", "second")
    assert daemon.notifier.send_message.call_count == 2  # not recorded: retried

    daemon.notifier.send_message.return_value = True
    daemon._alert_once("k", "third")
    daemon._alert_once("k", "fourth")
    assert daemon.notifier.send_message.call_count == 3  # recorded after a success


def test_alert_not_recorded_when_send_raises(tmp_path):
    daemon = _make(tmp_path)
    daemon.notifier.send_message.side_effect = RuntimeError("boom")

    daemon._alert_once("k", "x")
    daemon._alert_once("k", "x")

    assert daemon.notifier.send_message.call_count == 2


def test_compile_error_log_is_redacted_and_truncated(tmp_path, caplog):
    import logging

    daemon = _make(tmp_path)
    _import_fixture(daemon)
    secret = "http://user:hunter2@proxy.example:8080"
    long_error = secret + " " + "x" * 1000

    def boom(*args, **kwargs):
        raise ProfileCompileError(long_error)

    daemon.profile_store.read_snapshot = boom
    with caplog.at_level(logging.ERROR):
        daemon.current_profile()

    text = " ".join(r.getMessage() for r in caplog.records)
    assert "hunter2" not in text
    assert "profile cannot be compiled" in text
    compile_records = [r.getMessage() for r in caplog.records if "cannot be compiled" in r.getMessage()]
    assert all(len(m) <= len("profile cannot be compiled: ") + 300 for m in compile_records)
    alerts = _alerts(daemon)
    assert alerts and all("hunter2" not in a and len(a) <= 400 for a in alerts)


def test_alert_text_is_html_escaped(tmp_path):
    daemon = _make(tmp_path)

    daemon._alert_once("k", "bad <term> & more")

    assert _alerts(daemon) == ["bad &lt;term&gt; &amp; more"]


# --- evaluation ---------------------------------------------------------------------------------


def test_evaluation_passes_compiled_system_prompt(tmp_path):
    chain = RecordingChain()
    daemon = _make(tmp_path, FakeScraper(), chain)
    _import_fixture(daemon)
    _seed_job(daemon)
    prompt = compile_profile(daemon.profile_store.read_snapshot()).system_prompt

    daemon.evaluate_unprocessed()

    assert chain.prompts == [prompt]


def test_evaluation_without_profile_calls_legacy_signature(tmp_path):
    chain = LegacyChain()
    daemon = _make(tmp_path, FakeScraper(), chain)
    _seed_job(daemon)

    evaluated, _ = daemon.evaluate_unprocessed().as_tuple()

    assert evaluated == 1
    assert chain.jobs == ["Career Coach 1"]


def test_explicit_queries_bypass_profile(tmp_path):
    scraper = FakeScraper()
    daemon = _make(tmp_path, scraper)  # no profile at all
    query = SearchQuery("Q-X", "<ROLE>", "<MARKET>", ["linkedin", "google"])

    daemon.run_pipeline_cycle(queries=[query])

    assert scraper.calls == [("Q-X", "linkedin"), ("Q-X", "google")]
    assert daemon.notifier.send_message.call_count == 0


def test_chain_is_built_without_a_bonus_flag(tmp_path, monkeypatch):
    builder = MagicMock(return_value=RecordingChain())
    monkeypatch.setattr("src.scheduler.daemon.build_chain_from_settings", builder)
    daemon = DaemonService(db_path=tmp_path / "p.db", scraper=FakeScraper(), notifier=MagicMock())
    daemon.settings = _settings()
    snapshot = ProfileSnapshot(
        ProfileFields(persona="<PERSONA>", bonus_criteria="<CREDENTIAL>"), pinned_queries=(_pinned("Q-1"),)
    )
    apply_changes(daemon.profile_store, [ReplaceProfile(snapshot)], label="p", expected_version=0)
    _seed_job(daemon)

    daemon.evaluate_unprocessed()

    assert builder.call_count == 1
    assert "bonus_enabled" not in builder.call_args.kwargs


def test_chain_is_built_without_a_bonus_flag_and_without_a_profile(tmp_path, monkeypatch):
    builder = MagicMock(return_value=LegacyChain())
    monkeypatch.setattr("src.scheduler.daemon.build_chain_from_settings", builder)
    daemon = DaemonService(db_path=tmp_path / "p.db", scraper=FakeScraper(), notifier=MagicMock())
    daemon.settings = _settings()
    _seed_job(daemon)

    daemon.evaluate_unprocessed()

    assert "bonus_enabled" not in builder.call_args.kwargs


# --- profile thresholds, bonus, cap and rules per cycle -------------------------------------------


def _profile(daemon: DaemonService, **fields) -> None:
    fields.setdefault("persona", "<PERSONA>")
    snapshot = ProfileSnapshot(ProfileFields(**fields), pinned_queries=(_pinned("Q-1"),))
    apply_changes(daemon.profile_store, [ReplaceProfile(snapshot)], label="p", expected_version=0)


def _real_chain_daemon(tmp_path, provider: ScriptedProvider, **settings) -> DaemonService:
    chain = ProviderChainEvaluator([provider])
    daemon = _make(tmp_path, FakeScraper(), chain)
    daemon.settings = _settings(**settings)
    daemon.notifier.send_wow_alert.return_value = True
    return daemon


def _save(daemon: DaemonService, n: int, title: str) -> int:
    return daemon.db.save_job(
        JobPosting(
            dedup_hash=f"h{n}", site="indeed", job_url=f"https://example.test/{n}", title=title,
            company="Example Co", location="Remote",
            description="A coaching role description that is long enough to be meaningful.",
        )
    )[0]


def test_profile_thresholds_drive_wow_and_dismissal(tmp_path):
    chain = RecordingChain({"Core A": 8.2, "Core B": 6.2, "Core C": 5.5})
    daemon = _make(tmp_path, FakeScraper(), chain)
    daemon.settings = _settings(WOW_SCORE_THRESHOLD=8.5, DIGEST_SCORE_THRESHOLD=6.5)
    _profile(daemon, wow_threshold=8.0, digest_threshold=6.0, core_role_definition="<CORE>")
    ids = [_save(daemon, n, t) for n, t in enumerate(["Core A", "Core B", "Core C"], 1)]

    evaluated, wow = daemon.evaluate_unprocessed().as_tuple()

    assert (evaluated, wow) == (3, 1)
    a, b, c = (daemon.db.get_job_by_id(i) for i in ids)
    assert a.notification_status == "instant_sent" and a.eval_is_wow
    assert b.notification_status == "pending"  # 6.2 is above the profile digest threshold, below .env's 6.5
    assert c.notification_status == "dismissed"
    assert daemon.notifier.send_wow_alert.call_count == 1


def test_non_core_never_wow_with_lowered_threshold(tmp_path):
    provider = ScriptedProvider(lambda title: _answer(8.9, is_core_role=False))
    daemon = _real_chain_daemon(tmp_path, provider)
    _profile(daemon, wow_threshold=7.0, digest_threshold=6.0, core_role_definition="<CORE>")
    job_id = _save(daemon, 1, "Trainer")

    evaluated, wow = daemon.evaluate_unprocessed().as_tuple()

    assert (evaluated, wow) == (1, 0)
    job = daemon.db.get_job_by_id(job_id)
    assert job.eval_score == 6.9 and not job.eval_is_wow and job.notification_status == "pending"
    daemon.notifier.send_wow_alert.assert_not_called()


def test_profile_without_bonus_criteria_gets_no_bonus(tmp_path):
    provider = ScriptedProvider(lambda title: _answer(6.5, is_core_role=True, matches_bonus_criteria=True))
    daemon = _real_chain_daemon(tmp_path, provider)
    _profile(daemon, bonus_criteria="")
    job_id = _save(daemon, 1, "Coach")

    daemon.evaluate_unprocessed()

    assert daemon.db.get_job_by_id(job_id).eval_score == 6.5


def test_bonus_follows_profile_change_without_restart(tmp_path):
    provider = ScriptedProvider(lambda title: _answer(6.5, is_core_role=True, matches_bonus_criteria=True))
    daemon = _real_chain_daemon(tmp_path, provider)
    _profile(daemon, bonus_criteria="<CREDENTIAL>", bonus_points=1.0)
    chain = daemon.evaluator
    first = _save(daemon, 1, "Coach")
    daemon.run_pipeline_cycle(queries=[])  # a cycle with a profile: evaluation only
    assert daemon.db.get_job_by_id(first).eval_score == 7.5

    apply_changes(daemon.profile_store, [SetField("bonus_criteria", "")], label="drop bonus")
    second = _save(daemon, 2, "Coach")
    daemon.run_pipeline_cycle(queries=[])

    assert daemon.db.get_job_by_id(second).eval_score == 6.5
    assert daemon.evaluator is chain


def test_evaluation_keeps_cycle_snapshot_when_persona_changes_mid_batch(tmp_path):
    def change():
        apply_changes(daemon.profile_store, [SetField("persona", "<NEW PERSONA>")], label="mid-batch")

    chain = RecordingChain(on_call=change)
    daemon = _make(tmp_path, FakeScraper(), chain)
    _profile(daemon, persona="<OLD PERSONA>")
    for n in range(1, 4):
        _save(daemon, n, f"Coach {n}")

    daemon.evaluate_unprocessed()

    assert len(chain.prompts) == 3
    assert all("<OLD PERSONA>" in p and "<NEW PERSONA>" not in p for p in chain.prompts)

    _save(daemon, 4, "Coach 4")
    daemon.evaluate_unprocessed()
    assert "<NEW PERSONA>" in chain.prompts[-1]


def test_none_profile_not_reresolved_mid_cycle(tmp_path):
    def create_profile():
        _profile(daemon, persona="<LATE PERSONA>")

    chain = LegacyChain()
    daemon = _make(tmp_path, RuleScraper(on_call=create_profile), chain)
    _save(daemon, 1, "Coach")
    daemon.current_profile = MagicMock(wraps=daemon.current_profile)
    daemon.evaluate_unprocessed = MagicMock(wraps=daemon.evaluate_unprocessed)
    query = SearchQuery("Q-X", "<ROLE>", "<MARKET>", ["linkedin"])

    daemon.run_pipeline_cycle(queries=[query])

    assert daemon.current_profile.call_count == 1
    assert daemon.evaluate_unprocessed.call_args.kwargs["profile"] is None
    assert chain.jobs == ["Coach"]  # no profile was in force for this cycle's evaluation


def test_evaluate_unprocessed_resolves_the_profile_only_for_the_unset_default(tmp_path):
    chain = RecordingChain()
    daemon = _make(tmp_path, FakeScraper(), chain)
    _profile(daemon, persona="<PERSONA>")
    _save(daemon, 1, "Coach")

    daemon.evaluate_unprocessed(profile=None)

    assert chain.prompts == [None]  # an explicit None is "no profile", not "look it up"


def test_compile_alert_rearms_after_recovery(tmp_path):
    daemon = _make(tmp_path)
    _import_fixture(daemon)
    with daemon.db._get_connection() as conn:
        originals = conn.execute("SELECT id, term FROM rules WHERE match_mode = 'pattern'").fetchall()

    def restore():
        with daemon.db._get_connection() as conn:
            for row in originals:
                conn.execute("UPDATE rules SET term = ? WHERE id = ?", (row["term"], row["id"]))
            conn.commit()

    _break_pattern_rules(daemon, "(")
    daemon.run_pipeline_cycle()
    daemon.run_pipeline_cycle()
    assert daemon.notifier.send_message.call_count == 1

    restore()
    daemon.run_pipeline_cycle()  # good cycle re-arms
    _break_pattern_rules(daemon, "(")
    daemon.run_pipeline_cycle()

    assert daemon.notifier.send_message.call_count == 2


def test_no_profile_alert_rearms(tmp_path):
    daemon = _make(tmp_path)
    daemon.run_pipeline_cycle()
    daemon.run_pipeline_cycle()
    assert daemon.notifier.send_message.call_count == 1

    _import_fixture(daemon)
    daemon.run_pipeline_cycle()
    with daemon.db._get_connection() as conn:
        conn.execute("DELETE FROM profile")
        conn.commit()
    daemon.run_pipeline_cycle()

    assert daemon.notifier.send_message.call_count == 2


def test_db_locked_skips_scraping_without_alert(tmp_path, monkeypatch):
    scraper = FakeScraper()
    chain = RecordingChain()
    daemon = _make(tmp_path, scraper, chain)
    _import_fixture(daemon)
    good_prompt = compile_profile(daemon.profile_store.read_snapshot()).system_prompt
    daemon.run_pipeline_cycle()  # remembers the last good profile
    scraper.calls.clear()

    def locked():
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(daemon.profile_store, "read_snapshot", locked)
    _seed_job(daemon)
    daemon.run_pipeline_cycle()

    assert scraper.calls == []
    assert daemon.notifier.send_message.call_count == 0
    assert chain.prompts == [good_prompt]


def test_unexpected_exception_from_the_store_is_not_swallowed(tmp_path, monkeypatch):
    daemon = _make(tmp_path)

    def boom():
        raise KeyError("bug")

    monkeypatch.setattr(daemon.profile_store, "read_snapshot", boom)
    with pytest.raises(KeyError):
        daemon.current_profile()


def test_cycle_rules_reach_scraper_and_safety_net(tmp_path):
    from src.profile.ops import AddRule
    from src.profile.snapshot import Rule

    scraper = RuleScraper()
    chain = RecordingChain()
    daemon = _make(tmp_path, scraper, chain)
    _profile(daemon)
    apply_changes(daemon.profile_store, [AddRule(Rule("block", "<TERM>", "exact", "USER_BLOCK", "user"))], label="rule")
    expected = compile_profile(daemon.profile_store.read_snapshot()).rules
    _save(daemon, 1, "Coach")

    daemon.run_pipeline_cycle()
    daemon.run_pipeline_cycle(queries=[SearchQuery("Q-X", "<ROLE>", "<MARKET>", ["linkedin"])])

    assert len(scraper.rules) == 2 and all(r == expected for r in scraper.rules)
    assert any(r.kind == "block" for r in expected)
    assert chain.policies[0].rules == expected


def test_no_profile_cycle_passes_no_rules(tmp_path):
    scraper = RuleScraper()
    daemon = _make(tmp_path, scraper)

    daemon.run_pipeline_cycle(queries=[SearchQuery("Q-X", "<ROLE>", "<MARKET>", ["linkedin"])])

    assert scraper.rules == [None]


def test_digest_uses_profile_threshold(tmp_path):
    chain = RecordingChain({"Coach": 6.2})
    daemon = _make(tmp_path, FakeScraper(), chain)
    daemon.settings = _settings(DIGEST_SCORE_THRESHOLD=6.5)
    _profile(daemon, digest_threshold=6.0, wow_threshold=8.0)
    _save(daemon, 1, "Coach")
    daemon.evaluate_unprocessed()

    assert daemon.run_daily_digest_cycle(dry_run=True) == 1


def test_digest_without_profile_uses_env_threshold(tmp_path):
    chain = RecordingChain({"Coach": 6.7})
    daemon = _make(tmp_path, FakeScraper(), chain)
    daemon.settings = _settings(DIGEST_SCORE_THRESHOLD=6.8)
    _save(daemon, 1, "Coach")
    daemon.evaluate_unprocessed()  # no profile: dismissed below the .env threshold

    assert daemon.run_daily_digest_cycle(dry_run=True) == 0


def test_alert_and_digest_pass_the_profile_language(tmp_path, monkeypatch):
    monkeypatch.setattr("src.scheduler.daemon.check_jobspy_update", lambda: None)
    chain = RecordingChain({"Core A": 8.6, "Core B": 7.0})
    daemon = _make(tmp_path, FakeScraper(), chain)
    daemon.settings = _settings()
    daemon.notifier.send_wow_alert.return_value = True
    daemon.notifier.send_daily_digest.return_value = 1
    _profile(daemon, wow_threshold=8.5, digest_threshold=6.5, core_role_definition="<CORE>", output_language="en")
    _save(daemon, 1, "Core A")
    _save(daemon, 2, "Core B")

    daemon.evaluate_unprocessed()
    daemon.run_daily_digest_cycle()

    assert daemon.notifier.send_wow_alert.call_args.kwargs == {"lang": "en"}
    assert daemon.notifier.send_daily_digest.call_args.kwargs["lang"] == "en"


def test_alert_and_digest_language_is_none_without_a_language(tmp_path, monkeypatch):
    monkeypatch.setattr("src.scheduler.daemon.check_jobspy_update", lambda: None)
    chain = RecordingChain({"Core A": 8.6, "Core B": 7.0})
    daemon = _make(tmp_path, FakeScraper(), chain)
    daemon.settings = _settings()
    daemon.notifier.send_wow_alert.return_value = True
    daemon.notifier.send_daily_digest.return_value = 1
    _profile(daemon, wow_threshold=8.5, digest_threshold=6.5, core_role_definition="<CORE>")
    _save(daemon, 1, "Core A")
    _save(daemon, 2, "Core B")

    daemon.evaluate_unprocessed()
    daemon.run_daily_digest_cycle()

    assert daemon.notifier.send_wow_alert.call_args.kwargs == {"lang": None}
    assert daemon.notifier.send_daily_digest.call_args.kwargs["lang"] is None


def test_digest_update_notice_uses_the_profile_language(tmp_path, monkeypatch):
    from src.scraper.version_checker import VersionCheckResult

    monkeypatch.setattr(
        "src.scheduler.daemon.check_jobspy_update", lambda: VersionCheckResult(True, "1.0.0", "2.0.0")
    )
    daemon = _make(tmp_path, FakeScraper(), RecordingChain({"Core B": 7.0}))
    daemon.settings = _settings()
    daemon.notifier.send_daily_digest.return_value = 1
    _profile(daemon, wow_threshold=8.5, digest_threshold=6.5, core_role_definition="<CORE>", output_language="en")
    _save(daemon, 1, "Core B")
    daemon.evaluate_unprocessed()

    daemon.run_daily_digest_cycle()

    notice = daemon.notifier.send_daily_digest.call_args.kwargs["update_notice"]
    assert "update available" in notice and "v2.0.0" in notice
    assert "Доступне" not in notice


# --- removed catalog ----------------------------------------------------------------------------


def test_query_catalog_has_no_default_catalog():
    for name in ("DEFAULT_QUERY_CATALOG", "get_active_queries", "load_custom_queries"):
        assert not hasattr(query_catalog, name)


def test_scrape_all_queries_requires_queries(monkeypatch):
    monkeypatch.setattr("jobspy.scrape_jobs", MagicMock(side_effect=AssertionError("no network in tests")))
    scraper = JobSpyScraper(jitter_min=0.0, jitter_max=0.0)
    with pytest.raises(TypeError):
        scraper.scrape_all_queries()  # type: ignore[call-arg]


def test_daemon_has_profile_store(tmp_path):
    daemon = _make(tmp_path)
    assert daemon.profile_store.has_profile() is False


def _store_raises(daemon: DaemonService, monkeypatch, error: Exception) -> None:
    def fail():
        raise error

    monkeypatch.setattr(daemon.profile_store, "read_snapshot", fail)


def test_db_busy_without_a_last_good_profile_skips_evaluation(tmp_path, monkeypatch):
    chain = RecordingChain()
    daemon = _make(tmp_path, FakeScraper(), chain)
    _import_fixture(daemon)  # stored, but this process never read it successfully
    _seed_job(daemon)
    _store_raises(daemon, monkeypatch, sqlite3.OperationalError("database is locked"))

    assert daemon.run_pipeline_cycle() == (0, 0)
    summary = daemon.evaluate_unprocessed()  # the UNSET default resolves the profile itself
    assert summary.as_tuple() == (0, 0)
    assert (summary.not_scored_in_batch, summary.beyond_limit) == (1, 0)
    assert summary.stopped_reason == STOPPED_PROFILE_BUSY

    assert chain.calls == 0
    assert [j.eval_status for j in daemon.db.get_unprocessed_jobs()] == ["unprocessed"]
    assert daemon.notifier.send_message.call_count == 0


@pytest.mark.parametrize("message", ["database is locked", "database table is locked", "database is busy"])
def test_busy_and_locked_errors_are_db_busy(tmp_path, monkeypatch, message):
    daemon = _make(tmp_path)
    _store_raises(daemon, monkeypatch, sqlite3.OperationalError(message))
    assert daemon.current_profile() == (None, "db_busy")
    assert daemon.notifier.send_message.call_count == 0


def test_permanent_database_errors_alert_once_per_distinct_error(tmp_path, monkeypatch):
    scraper = FakeScraper()
    daemon = _make(tmp_path, scraper)
    _store_raises(daemon, monkeypatch, sqlite3.OperationalError("no such table: profile"))

    daemon.run_pipeline_cycle()
    daemon.run_pipeline_cycle()

    assert daemon.current_profile()[1] == "compile_error"
    assert scraper.calls == []
    assert len(_alerts(daemon)) == 1 and "no such table" in _alerts(daemon)[0]
    _store_raises(daemon, monkeypatch, sqlite3.OperationalError("disk I/O error"))
    daemon.run_pipeline_cycle()
    assert len(_alerts(daemon)) == 2


def test_legacy_profile_behaves_as_today_at_8_5_and_6_5(tmp_path):
    """An imported legacy profile: core 8.5 alerts, core 6.4 is dismissed, non-core 9.0 lands at 8.4 in the digest."""
    answers = {
        "Core WOW": _answer(8.5, is_core_role=True),
        "Core Low": _answer(6.4, is_core_role=True),
        "Trainer": _answer(9.0, is_core_role=False),
    }
    daemon = _real_chain_daemon(tmp_path, ScriptedProvider(lambda title: answers[title]))
    _import_fixture(daemon)
    ids = {title: _save(daemon, n, title) for n, title in enumerate(answers, 1)}

    evaluated, wow = daemon.evaluate_unprocessed().as_tuple()

    assert (evaluated, wow) == (3, 1)
    wow_job, low, trainer = (daemon.db.get_job_by_id(ids[t]) for t in ("Core WOW", "Core Low", "Trainer"))
    assert (wow_job.eval_score, wow_job.notification_status, wow_job.eval_is_wow) == (8.5, "instant_sent", True)
    assert (low.eval_score, low.notification_status) == (6.4, "dismissed")
    assert (trainer.eval_score, trainer.notification_status, trainer.eval_is_wow) == (8.4, "pending", False)
    assert daemon.notifier.send_wow_alert.call_count == 1


class OldSignatureScraper:
    """Scraper double whose scrape_board_outcome cannot take the cycle's rules."""

    def __init__(self):
        self.calls = 0

    def scrape_board_outcome(self, site, query):
        self.calls += 1
        return ScrapeOutcome(postings=[], rows=0, status=STATUS_EMPTY)


def test_rules_not_forwarded_to_an_old_signature_is_logged_once(tmp_path, caplog):
    scraper = OldSignatureScraper()
    daemon = _make(tmp_path, scraper)
    _import_fixture(daemon)

    with caplog.at_level("WARNING", logger="src.scheduler.daemon"):
        daemon.run_pipeline_cycle()
        daemon.run_pipeline_cycle()

    assert scraper.calls >= 2
    warnings = [r for r in caplog.records if "rules" in r.getMessage() and r.levelname == "WARNING"]
    assert len(warnings) == 1


def test_no_profile_policy_has_one_source(tmp_path):
    from src.evaluator.scoring import no_profile_policy

    daemon = _make(tmp_path)
    daemon.settings = _settings(WOW_SCORE_THRESHOLD=9.0, DIGEST_SCORE_THRESHOLD=7.0)
    assert daemon._policy(None) == no_profile_policy("", 9.0, 7.0)
