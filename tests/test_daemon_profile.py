"""The daemon scrapes and evaluates from the compiled profile (one snapshot per cycle)."""
import json
from pathlib import Path
from typing import List, Optional, Tuple
from unittest.mock import MagicMock

import pytest

from src.config import Settings
from src.evaluator.schemas import JobEvaluationResult
from src.profile.compiler import compile_profile
from src.profile.importer import build_imported_snapshot, parse_queries_json
from src.profile.ops import AddPinnedQuery, ReplaceProfile, SetField, apply_changes
from src.profile.snapshot import PinnedQuery, ProfileFields, ProfileSnapshot
from src.scheduler.daemon import DaemonService
from src.scraper import query_catalog
from src.scraper.jobspy_scraper import JobSpyScraper
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


class RecordingChain:
    def __init__(self):
        self.prompts: List[Optional[str]] = []

    def evaluate_job(self, job, system_prompt=None):
        self.prompts.append(system_prompt)
        return JobEvaluationResult(
            score=7.0, summary_uk="A sufficiently long summary.", reasoning="Because.", eval_model="m/x"
        )


class LegacyChain:
    """Test double with the pre-profile one-argument signature."""

    def __init__(self):
        self.jobs: List[str] = []

    def evaluate_job(self, job):
        self.jobs.append(job.title)
        return JobEvaluationResult(
            score=7.0, summary_uk="A sufficiently long summary.", reasoning="Because.", eval_model="m/x"
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
    daemon.evaluate_unprocessed = MagicMock(return_value=(0, 0))

    daemon.run_pipeline_cycle()

    assert scraper.calls == []
    daemon.evaluate_unprocessed.assert_called_once()
    assert daemon.evaluate_unprocessed.call_args.kwargs["profile"].paused is True
    assert daemon.notifier.send_message.call_count == 0


def test_dry_run_cycle_sends_no_alert_and_does_not_evaluate(tmp_path):
    daemon = _make(tmp_path)
    daemon.evaluate_unprocessed = MagicMock(return_value=(0, 0))

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


def test_unreadable_snapshot_is_a_compile_error_with_fallback(tmp_path):
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

    evaluated, _ = daemon.evaluate_unprocessed()

    assert evaluated == 1
    assert chain.jobs == ["Career Coach 1"]


def test_explicit_queries_bypass_profile(tmp_path):
    scraper = FakeScraper()
    daemon = _make(tmp_path, scraper)  # no profile at all
    query = SearchQuery("Q-X", "<ROLE>", "<MARKET>", ["linkedin", "google"])

    daemon.run_pipeline_cycle(queries=[query])

    assert scraper.calls == [("Q-X", "linkedin"), ("Q-X", "google")]
    assert daemon.notifier.send_message.call_count == 0


def test_chain_built_with_profile_bonus_flag(tmp_path, monkeypatch):
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

    assert builder.call_args.kwargs["bonus_enabled"] is True


def test_chain_built_without_profile_keeps_old_call(tmp_path, monkeypatch):
    builder = MagicMock(return_value=LegacyChain())
    monkeypatch.setattr("src.scheduler.daemon.build_chain_from_settings", builder)
    daemon = DaemonService(db_path=tmp_path / "p.db", scraper=FakeScraper(), notifier=MagicMock())
    daemon.settings = _settings()
    _seed_job(daemon)

    daemon.evaluate_unprocessed()

    assert "bonus_enabled" not in builder.call_args.kwargs


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
