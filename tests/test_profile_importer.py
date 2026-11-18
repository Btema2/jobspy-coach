import json
import os
import shutil
import stat
import subprocess
import threading
from dataclasses import replace
from pathlib import Path
from typing import List

import pytest

from src.profile.compiler import compile_profile
from src.profile.errors import (
    ExportRefused,
    LegacyImportError,
    ProfileValidationError,
)
from src.profile.importer import (
    DEFAULT_EXPORT_PATH,
    LegacyFiles,
    auto_import,
    build_imported_snapshot,
    export_profile,
    extract_bonus_criteria,
    find_legacy_files,
    load_profile_yaml,
    parse_queries_json,
)
from src.profile.legacy_seed import (
    LEGACY_ADJACENT_ROLE_DEFINITION,
    LEGACY_CORE_ROLE_DEFINITION,
    LEGACY_DEALBREAKERS,
    seed_rules,
)
from src.profile.ops import ReplaceProfile, apply_changes
from src.profile.snapshot import PinnedQuery, ProfileFields, ProfileSnapshot, snapshot_to_dict
from src.profile.store import ProfileStore
from src.scraper.query_catalog import SearchQuery, query_key
from src.storage.db import Database

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).parent / "fixtures"
QUERIES_FIXTURE = FIXTURES / "legacy_queries.json"
PERSONA_FIXTURE = FIXTURES / "legacy_persona.md"


@pytest.fixture
def db(tmp_path: Path) -> Database:
    database = Database(tmp_path / "data" / "profile.db")
    database.init_db()
    return database


@pytest.fixture
def store(db: Database) -> ProfileStore:
    return ProfileStore(db)


@pytest.fixture
def legacy_dir(tmp_path: Path) -> Path:
    base = tmp_path / "legacy"
    base.mkdir()
    shutil.copy(QUERIES_FIXTURE, base / "queries.json")
    shutil.copy(PERSONA_FIXTURE, base / ".candidate_profile.md")
    return base


def _legacy_parse(path: Path) -> List[SearchQuery]:
    """Copy of the former queries.json loader mapping that import replaces (kept to prove equivalence)."""
    data = json.loads(path.read_text(encoding="utf-8"))
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
        for item in data
    ]


def _import(store: ProfileStore, base: Path, wow=8.5, digest=6.5):
    return auto_import(store, base, wow_threshold=wow, digest_threshold=digest)


# --- parse_queries_json ------------------------------------------------------------------------


def test_parse_queries_json_maps_fields():
    queries = parse_queries_json(QUERIES_FIXTURE)

    assert [q.query_id for q in queries] == ["Q-FIX-A", "Q-FIX-B", "Q-FIX-C", "Q-FIX-D"]
    a, b, c, d = queries
    assert a == PinnedQuery("Q-FIX-A", "<ROLE> OR <ROLE B>", "<MARKET>", ("linkedin", "indeed"),
                            country="poland", is_remote=False)
    assert b.country is None and b.is_remote is True
    assert c.google_search_term == "<ROLE B> jobs near <MARKET> in the last 3 days"
    assert d.boards == ("linkedin", "indeed", "glassdoor", "google") and d.country == "usa"
    assert all(q.enabled for q in queries)


def test_example_queries_json_parses():
    queries = parse_queries_json(ROOT / "queries.example.json")

    assert len(queries) >= 1
    assert all(q.boards for q in queries)


def _write(tmp_path: Path, content) -> Path:
    path = tmp_path / "queries.json"
    path.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")
    return path


_GOOD = {"query_id": "Q1", "search_term": "<T>", "location": "<L>", "site_names": ["linkedin"]}


@pytest.mark.parametrize(
    "content,index",
    [
        ("{not json", None),
        ({"query_id": "Q1"}, None),
        ([_GOOD, {k: v for k, v in _GOOD.items() if k != "search_term"}], 1),
        ([_GOOD, {**_GOOD}], 1),
        ([{**_GOOD, "site_names": []}], 0),
        ([_GOOD, {**_GOOD, "query_id": "Q2", "site_names": "linkedin"}], 1),
        ([_GOOD, {**_GOOD, "query_id": "Q2", "is_remote": "yes"}], 1),
        ([_GOOD, {**_GOOD, "query_id": "Q2", "country_code": 5}], 1),
        ([_GOOD, "not an object"], 1),
        ([{**_GOOD, "query_id": ""}], 0),
    ],
)
def test_parse_queries_json_errors_name_file_and_entry(tmp_path: Path, content, index):
    path = _write(tmp_path, content)

    with pytest.raises(LegacyImportError) as exc:
        parse_queries_json(path)

    assert "queries.json" in str(exc.value)
    if index is not None:
        assert f"entries[{index}]" in str(exc.value)


def test_parse_queries_json_accepts_empty_list(tmp_path: Path):
    assert parse_queries_json(_write(tmp_path, [])) == ()


def test_parse_queries_json_unreadable_file(tmp_path: Path):
    with pytest.raises(LegacyImportError) as exc:
        parse_queries_json(tmp_path / "missing.json")
    assert "missing.json" in str(exc.value)


# --- find_legacy_files -------------------------------------------------------------------------


def test_find_legacy_files_search_order(tmp_path: Path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "queries.json").write_text("[]")
    (tmp_path / "config" / "candidate_profile.md").write_text("<PERSONA>")
    assert find_legacy_files(tmp_path) == LegacyFiles(
        tmp_path / "config" / "queries.json", tmp_path / "config" / "candidate_profile.md"
    )

    (tmp_path / "queries.json").write_text("[]")
    (tmp_path / ".candidate_profile.md").write_text("<PERSONA>")
    assert find_legacy_files(tmp_path) == LegacyFiles(
        tmp_path / "queries.json", tmp_path / ".candidate_profile.md"
    )


def test_find_legacy_files_empty_persona_counts_as_absent(tmp_path: Path):
    (tmp_path / ".candidate_profile.md").write_text("  \n \n")
    assert find_legacy_files(tmp_path) == LegacyFiles(None, None)

    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "candidate_profile.md").write_text("<PERSONA>")
    assert find_legacy_files(tmp_path).persona_path == tmp_path / "config" / "candidate_profile.md"


# --- extract_bonus_criteria --------------------------------------------------------------------


def test_bonus_criteria_from_heading_section():
    persona = PERSONA_FIXTURE.read_text(encoding="utf-8")

    bonus = extract_bonus_criteria(persona)

    assert "Holds an accredited <CREDENTIAL> certification." in bonus
    assert "Has experience with <SPECIALTY> clients." in bonus
    assert "## Languages" not in bonus and "<LANGUAGE>" not in bonus
    assert bonus.startswith("## Bonus criteria")


def test_bonus_criteria_section_ends_at_same_or_higher_level_heading_only():
    persona = "# Top\n\n## Bonus\n- a\n### Detail\n- b\n# Next\n- c\n"

    assert extract_bonus_criteria(persona) == "## Bonus\n- a\n### Detail\n- b"


def test_bonus_criteria_from_lines_without_heading():
    persona = "# Profile\n\n- Background: <X>\n- Bonus: holds <CREDENTIAL>\n- Languages: <L>\n- A BONUS if <Y>\n"

    assert extract_bonus_criteria(persona) == "- Bonus: holds <CREDENTIAL>\n- A BONUS if <Y>"


def test_no_bonus_word_leaves_bonus_criteria_empty():
    assert extract_bonus_criteria("# Profile\n\n- Background: <X>\n") == ""
    assert extract_bonus_criteria("") == ""


@pytest.mark.parametrize(
    "persona",
    [
        "no mention",
        "## Bonus\n- x",
        "line with Bonus in it",
        "# Bonus Criteria",
        "x\n\n### the BONUS list\n",
    ],
)
def test_bonus_criteria_non_empty_exactly_when_bonus_is_mentioned(persona):
    assert bool(extract_bonus_criteria(persona)) == ("bonus" in persona.lower())


# --- build_imported_snapshot -------------------------------------------------------------------


def test_import_seeds_rules_and_definitions():
    persona = PERSONA_FIXTURE.read_text(encoding="utf-8")
    queries = parse_queries_json(QUERIES_FIXTURE)

    snap, notes = build_imported_snapshot(queries, persona, wow_threshold=8.5, digest_threshold=6.5)

    assert notes == []
    assert snap.rules == seed_rules()
    f = snap.fields
    assert f.persona == persona.strip()
    assert f.core_role_definition == LEGACY_CORE_ROLE_DEFINITION
    assert f.adjacent_role_definition == LEGACY_ADJACENT_ROLE_DEFINITION
    assert f.dealbreakers == LEGACY_DEALBREAKERS
    assert f.bonus_criteria == extract_bonus_criteria(persona.strip())
    assert f.output_language is None and f.paused is False
    assert snap.pinned_queries == queries and snap.roles == () and snap.markets == ()


def test_persona_only_import_is_paused_with_no_queries(store: ProfileStore, tmp_path: Path):
    base = tmp_path / "only-persona"
    base.mkdir()
    (base / ".candidate_profile.md").write_text("<PERSONA>\n", encoding="utf-8")

    result = _import(store, base)

    assert result is not None and result.version == 1
    snap = store.read_snapshot()
    assert snap is not None
    assert snap.fields.paused is True and snap.pinned_queries == ()
    assert snap.fields.persona == "<PERSONA>"


def test_queries_only_import_is_unpaused_with_generic_persona(store: ProfileStore, tmp_path: Path):
    base = tmp_path / "only-queries"
    base.mkdir()
    shutil.copy(QUERIES_FIXTURE, base / "queries.json")

    _import(store, base)

    snap = store.read_snapshot()
    assert snap is not None and snap.fields.paused is False and snap.fields.persona == ""
    assert len(snap.pinned_queries) == 4


def test_import_thresholds_from_settings_and_invalid_fallback():
    queries = (PinnedQuery("Q1", "<ROLE>", "<MARKET>", ("linkedin",)),)
    snap, notes = build_imported_snapshot(queries, "", wow_threshold=8.0, digest_threshold=6.0)
    assert (snap.fields.wow_threshold, snap.fields.digest_threshold, notes) == (8.0, 6.0, [])

    snap, notes = build_imported_snapshot(queries, "", wow_threshold=6.0, digest_threshold=7.0)
    assert (snap.fields.wow_threshold, snap.fields.digest_threshold) == (8.5, 6.5)
    assert len(notes) == 1 and "8.5" in notes[0] and "6.5" in notes[0]


def test_import_threshold_note_reaches_the_apply_result(store: ProfileStore, legacy_dir: Path):
    result = _import(store, legacy_dir, wow=6.0, digest=7.0)

    assert result is not None and any("6.5" in w for w in result.warnings)


def test_import_raises_call_cap_to_fit():
    queries = tuple(
        PinnedQuery(f"Q-{i}", "<T>", "<L>", ("linkedin", "google")) for i in range(60)
    )

    snap, _ = build_imported_snapshot(queries, "", wow_threshold=8.5, digest_threshold=6.5)

    assert snap.fields.max_calls_per_cycle >= 120
    assert compile_profile(snap).estimate.total_calls <= snap.fields.max_calls_per_cycle


def test_small_import_keeps_default_call_cap():
    queries = parse_queries_json(QUERIES_FIXTURE)
    snap, _ = build_imported_snapshot(queries, "", wow_threshold=8.5, digest_threshold=6.5)
    assert snap.fields.max_calls_per_cycle == 100


def test_persona_over_8000_stops_with_message_and_writes_nothing(store: ProfileStore, tmp_path: Path):
    base = tmp_path / "big"
    base.mkdir()
    (base / ".candidate_profile.md").write_text("x" * 8001, encoding="utf-8")
    shutil.copy(QUERIES_FIXTURE, base / "queries.json")

    with pytest.raises(LegacyImportError) as exc:
        _import(store, base)

    assert "8000" in str(exc.value) and ".candidate_profile.md" in str(exc.value)
    assert "8001" in str(exc.value)
    assert store.has_profile() is False


def test_build_snapshot_persona_limit_message_uses_given_source():
    with pytest.raises(LegacyImportError) as exc:
        build_imported_snapshot(
            (), "y" * 8001, wow_threshold=8.5, digest_threshold=6.5, persona_source="some/file.md"
        )
    assert str(exc.value) == (
        "some/file.md: persona is 8001 characters; the limit is 8000. Shorten it and run again."
    )


def test_persona_of_exactly_8000_characters_is_imported():
    snap, _ = build_imported_snapshot((), "y" * 8000, wow_threshold=8.5, digest_threshold=6.5)
    assert len(snap.fields.persona) == 8000


# --- auto_import -------------------------------------------------------------------------------


@pytest.mark.parametrize("path", [QUERIES_FIXTURE, ROOT / "queries.example.json"], ids=["fixture", "example"])
def test_imported_queries_equal_todays_queries(store: ProfileStore, tmp_path: Path, path: Path):
    base = tmp_path / "base"
    base.mkdir()
    shutil.copy(path, base / "queries.json")

    _import(store, base)

    snap = store.read_snapshot()
    assert snap is not None
    compiled = compile_profile(snap)
    legacy = _legacy_parse(base / "queries.json")
    assert list(compiled.queries) == legacy
    assert [query_key(q, b) for q in compiled.queries for b in q.site_names] == [
        query_key(q, b) for q in legacy for b in q.site_names
    ]


def test_auto_import_is_idempotent(store: ProfileStore, legacy_dir: Path):
    files = [legacy_dir / "queries.json", legacy_dir / ".candidate_profile.md"]
    before = [(p.read_bytes(), p.stat().st_mtime_ns) for p in files]

    first = _import(store, legacy_dir)
    second = _import(store, legacy_dir)

    assert first is not None and first.version == 1
    assert second is None
    assert len(store.list_versions()) == 1
    assert store.list_versions()[0].label == "import legacy files"
    assert [(p.read_bytes(), p.stat().st_mtime_ns) for p in files] == before


def test_auto_import_race_imports_once(tmp_path: Path, legacy_dir: Path):
    path = tmp_path / "shared" / "race.db"
    path.parent.mkdir()
    Database(path).init_db()
    barrier = threading.Barrier(2)
    results: list = []
    errors: list = []

    def worker():
        try:
            local = ProfileStore(Database(path))
            barrier.wait(timeout=10)
            results.append(_import(local, legacy_dir))
        except BaseException as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert errors == []
    assert sorted(r is None for r in results) == [False, True]
    store = ProfileStore(Database(path))
    assert [v.version for v in store.list_versions()] == [1]
    snap = store.read_snapshot()
    assert snap is not None and len(snap.pinned_queries) == 4


def test_auto_import_without_legacy_files_does_nothing(store: ProfileStore, tmp_path: Path):
    base = tmp_path / "empty"
    base.mkdir()

    assert _import(store, base) is None
    assert store.has_profile() is False


def test_auto_import_does_nothing_when_a_profile_exists(store: ProfileStore, legacy_dir: Path):
    apply_changes(store, [ReplaceProfile(ProfileSnapshot(ProfileFields(paused=True)))], label="x",
                  expected_version=0)

    assert _import(store, legacy_dir) is None
    assert len(store.list_versions()) == 1


def test_broken_queries_json_stops_import_and_writes_nothing(store: ProfileStore, legacy_dir: Path):
    (legacy_dir / "queries.json").write_text('[{"query_id": "Q1"}]', encoding="utf-8")

    with pytest.raises(LegacyImportError) as exc:
        _import(store, legacy_dir)

    assert "queries.json" in str(exc.value) and "entries[0]" in str(exc.value)
    assert store.has_profile() is False


def test_import_with_unknown_country_stops_with_legacy_error(store: ProfileStore, tmp_path: Path):
    base = tmp_path / "badcountry"
    base.mkdir()
    entry = {"query_id": "Q1", "search_term": "<T>", "location": "<L>", "site_names": ["linkedin"],
             "country_code": "polnad"}
    (base / "queries.json").write_text(json.dumps([entry]), encoding="utf-8")

    with pytest.raises(LegacyImportError) as exc:
        _import(store, base)

    assert "polnad" in str(exc.value)
    assert store.has_profile() is False


# --- export / import YAML ----------------------------------------------------------------------


def _full_snapshot(store: ProfileStore, legacy_dir: Path) -> ProfileSnapshot:
    _import(store, legacy_dir)
    snap = store.read_snapshot()
    assert snap is not None
    return snap


def test_export_yaml_roundtrip(store: ProfileStore, legacy_dir: Path, tmp_path: Path):
    snap = _full_snapshot(store, legacy_dir)

    out = export_profile(store, tmp_path / "out" / "profile-export.yaml")

    loaded = load_profile_yaml(out)
    assert replace(loaded, version=0) == replace(snap, version=0)
    text = out.read_text(encoding="utf-8")
    assert text.startswith("snapshot_schema: 1")  # key order preserved, not sorted


def test_export_default_path():
    assert DEFAULT_EXPORT_PATH == Path("local/profile-export.yaml")


def test_export_keeps_unicode_readable(store: ProfileStore, tmp_path: Path):
    snap = ProfileSnapshot(ProfileFields(persona="<PERSONA> ąę", paused=True))
    apply_changes(store, [ReplaceProfile(snap)], label="x", expected_version=0)

    out = export_profile(store, tmp_path / "p.yaml")

    assert "ąę" in out.read_text(encoding="utf-8")
    assert load_profile_yaml(out).fields.persona == "<PERSONA> ąę"


def test_export_without_a_profile_is_an_error(store: ProfileStore, tmp_path: Path):
    with pytest.raises(ProfileValidationError):
        export_profile(store, tmp_path / "p.yaml")


needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


@needs_git
def test_export_refuses_unignored_path_in_git_repo(store: ProfileStore, legacy_dir: Path, tmp_path: Path):
    _full_snapshot(store, legacy_dir)
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    target = repo / "p.yaml"

    with pytest.raises(ExportRefused):
        export_profile(store, target)
    assert not target.exists()

    assert export_profile(store, target, force=True) == target
    assert target.exists()


@needs_git
def test_export_allowed_in_ignored_path(store: ProfileStore, legacy_dir: Path, tmp_path: Path):
    _full_snapshot(store, legacy_dir)
    repo = tmp_path / "repo2"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / ".gitignore").write_text("local/\n", encoding="utf-8")

    out = export_profile(store, repo / "local" / "profile-export.yaml")

    assert out.exists()


@needs_git
def test_export_outside_git_repo_allowed(store: ProfileStore, legacy_dir: Path, tmp_path: Path):
    _full_snapshot(store, legacy_dir)

    out = export_profile(store, tmp_path / "plain" / "nested" / "p.yaml")

    assert out.exists()


def test_export_file_mode_600(store: ProfileStore, legacy_dir: Path, tmp_path: Path):
    _full_snapshot(store, legacy_dir)
    target = tmp_path / "wide.yaml"
    target.write_text("old", encoding="utf-8")
    os.chmod(target, 0o644)

    export_profile(store, target)

    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    fresh = export_profile(store, tmp_path / "fresh.yaml")
    assert stat.S_IMODE(fresh.stat().st_mode) == 0o600


def test_gitignore_has_profile_export_pattern():
    lines = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "profile-export*.yaml" in lines


def test_yaml_import_validates_through_ops(store: ProfileStore, legacy_dir: Path, tmp_path: Path):
    snap = _full_snapshot(store, legacy_dir)
    data = snapshot_to_dict(snap)
    data["profile"]["wow_threshold"] = 11
    yaml_path = tmp_path / "bad.yaml"
    import yaml

    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    loaded = load_profile_yaml(yaml_path)
    with pytest.raises(ProfileValidationError):
        apply_changes(store, [ReplaceProfile(loaded)], label="import yaml", expected_version=1)

    assert [v.version for v in store.list_versions()] == [1]
    current = store.read_snapshot()
    assert current is not None and current.fields.wow_threshold == 8.5


@pytest.mark.parametrize(
    "text",
    ["{unclosed", "- a\n- b\n", "", "snapshot_schema: 1\nprofile: {persnoa: x}\n"],
)
def test_load_profile_yaml_errors_are_validation_errors(tmp_path: Path, text):
    path = tmp_path / "bad.yaml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ProfileValidationError):
        load_profile_yaml(path)


def test_load_profile_yaml_missing_file(tmp_path: Path):
    with pytest.raises(ProfileValidationError):
        load_profile_yaml(tmp_path / "nope.yaml")


# --- review fixes ------------------------------------------------------------------------------


def test_yaml_nan_and_inf_are_validation_errors_and_nothing_is_written(
    store: ProfileStore, legacy_dir: Path, tmp_path: Path
):
    import yaml

    snap = _full_snapshot(store, legacy_dir)
    for literal in (".nan", ".inf"):
        data = snapshot_to_dict(snap)
        text = yaml.safe_dump(data, sort_keys=False).replace(
            "wow_threshold: 8.5", f"wow_threshold: {literal}"
        )
        path = tmp_path / f"bad{literal}.yaml"
        path.write_text(text, encoding="utf-8")

        with pytest.raises(ProfileValidationError):
            apply_changes(store, [ReplaceProfile(load_profile_yaml(path))], label="y", expected_version=1)

    assert [v.version for v in store.list_versions()] == [1]


def test_empty_country_code_is_treated_as_absent(tmp_path: Path):
    path = _write(tmp_path, [{**_GOOD, "country_code": ""}])

    (query,) = parse_queries_json(path)

    assert query.country is None


def test_export_refuses_symlink_target(store: ProfileStore, legacy_dir: Path, tmp_path: Path):
    _full_snapshot(store, legacy_dir)
    victim = tmp_path / "victim.txt"
    victim.write_text("keep", encoding="utf-8")
    link = tmp_path / "link.yaml"
    link.symlink_to(victim)

    with pytest.raises(ExportRefused):
        export_profile(store, link, force=True)

    assert victim.read_text(encoding="utf-8") == "keep"


def test_export_never_leaves_a_wide_mode_file_even_while_writing(
    store: ProfileStore, legacy_dir: Path, tmp_path: Path, monkeypatch
):
    _full_snapshot(store, legacy_dir)
    target = tmp_path / "wide.yaml"
    target.write_text("old", encoding="utf-8")
    os.chmod(target, 0o644)
    seen = []
    real_fdopen = os.fdopen

    def spying_fdopen(fd, *args, **kwargs):
        seen.append(stat.S_IMODE(os.fstat(fd).st_mode))
        return real_fdopen(fd, *args, **kwargs)

    monkeypatch.setattr(os, "fdopen", spying_fdopen)

    export_profile(store, target)

    assert seen == [0o600]


@needs_git
def test_export_refuses_tracked_path_without_force(store: ProfileStore, legacy_dir: Path, tmp_path: Path):
    _full_snapshot(store, legacy_dir)
    repo = tmp_path / "repo3"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    target = repo / "tracked.yaml"
    target.write_text("old", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "tracked.yaml"], check=True)

    with pytest.raises(ExportRefused):
        export_profile(store, target)

    assert target.read_text(encoding="utf-8") == "old"


# --- final review fixes ------------------------------------------------------------------------


def _write_queries(path: Path, entries: list) -> Path:
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path


def _entry(query_id: str, **kw) -> dict:
    base = {"query_id": query_id, "search_term": "<ROLE>", "location": "<MARKET>", "site_names": ["linkedin"]}
    base.update(kw)
    return base


@pytest.mark.parametrize(
    "legacy, canonical",
    [("us", "usa"), ("United States", "usa"), ("united kingdom", "uk"), ("czechia", "czech republic"),
     ("turkey", "türkiye"), ("UAE", "united arab emirates")],
)
def test_legacy_queries_json_with_country_aliases_imports(store: ProfileStore, tmp_path: Path, legacy, canonical):
    base = tmp_path / "alias"
    base.mkdir()
    _write_queries(base / "queries.json", [_entry("Q1", country_code=legacy, site_names=["linkedin", "indeed"])])

    result = _import(store, base)

    assert result is not None
    snap = store.read_snapshot()
    assert snap.pinned_queries[0].country == canonical


def test_legacy_alias_country_counts_for_glassdoor_support(store: ProfileStore, tmp_path: Path):
    base = tmp_path / "gd"
    base.mkdir()
    _write_queries(base / "queries.json", [_entry("Q1", country_code="united kingdom", site_names=["glassdoor"])])

    result = _import(store, base)

    assert result is not None and not any("glassdoor" in w for w in result.warnings)


def test_import_preserves_query_ids_exactly(store: ProfileStore, tmp_path: Path):
    base = tmp_path / "ids"
    base.mkdir()
    ids = [" Q-A ", "Q-b", "q-B"]  # padding kept; case-only twin gets a deterministic distinct id
    _write_queries(base / "queries.json", [_entry(i) for i in ids])

    result = _import(store, base)

    assert result is not None
    got = [q.query_id for q in store.read_snapshot().pinned_queries]
    assert got[0] == " Q-A " and got[1] == "Q-b"
    assert got[2] != "q-B" and got[2].casefold() != "q-b" and got[2].startswith("q-B")
    assert len({g.casefold() for g in got}) == 3
    assert any("q-B" in w for w in result.warnings)  # the renamed id is reported


def test_case_twin_disambiguation_is_deterministic_and_avoids_existing_ids():
    queries = [PinnedQuery(i, "<ROLE>", "<MARKET>", ("linkedin",)) for i in ("A", "a", "a~2", "a")]
    snap1, notes1 = build_imported_snapshot(queries, "", wow_threshold=8.5, digest_threshold=6.5)
    snap2, _ = build_imported_snapshot(queries, "", wow_threshold=8.5, digest_threshold=6.5)

    ids = [q.query_id for q in snap1.pinned_queries]
    assert ids == [q.query_id for q in snap2.pinned_queries]
    assert ids[0] == "A" and ids[2] == "a~2"  # non-colliding / first ids unchanged
    assert len({i.casefold() for i in ids}) == 4
    assert len(notes1) == 2


def test_persona_without_queries_prints_a_paused_note(store: ProfileStore, tmp_path: Path):
    base = tmp_path / "note"
    base.mkdir()
    (base / ".candidate_profile.md").write_text("<PERSONA>\n", encoding="utf-8")

    result = _import(store, base)

    assert any("paused" in w and "no search queries" in w for w in result.warnings)


def test_estimate_is_computed_after_normalising_boards():
    entries = [PinnedQuery(f"Q{i}", "<ROLE>", "<MARKET>", ("LinkedIn ", "INDEED"), country="Poland")
               for i in range(60)]  # 120 calls once normalised

    snap, notes = build_imported_snapshot(entries, "", wow_threshold=8.5, digest_threshold=6.5)

    assert snap.fields.max_calls_per_cycle == 120
    assert any("max_calls_per_cycle raised to 120" in n for n in notes)
