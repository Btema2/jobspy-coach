"""Import of the legacy ``queries.json`` / ``.candidate_profile.md`` files and YAML export / import.

The legacy files are only ever read, never modified. Import is idempotent: it runs only while no profile
exists, and a concurrent import that loses the race is a silent no-op.
"""
import json
import os
import re
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import yaml

from src.profile.compiler import compile_profile
from src.profile.errors import (
    ExportRefused,
    LegacyImportError,
    ProfileCompileError,
    ProfileValidationError,
    VersionConflict,
)
from src.profile.legacy_seed import (
    LEGACY_ADJACENT_ROLE_DEFINITION,
    LEGACY_CORE_ROLE_DEFINITION,
    LEGACY_DEALBREAKERS,
    seed_rules,
)
from src.profile.ops import (
    MAX_TEXT_CHARS,
    ApplyResult,
    ReplaceProfile,
    apply_changes,
    threshold_problems,
)
from src.profile.snapshot import (
    PinnedQuery,
    ProfileFields,
    ProfileSnapshot,
    snapshot_from_dict,
    snapshot_to_dict,
)
from src.profile.store import ProfileStore

DEFAULT_EXPORT_PATH = Path("local/profile-export.yaml")
FALLBACK_WOW_THRESHOLD = 8.5
FALLBACK_DIGEST_THRESHOLD = 6.5
DEFAULT_MAX_CALLS = 100

_QUERIES_CANDIDATES = ("queries.json", "config/queries.json")
_PERSONA_CANDIDATES = (".candidate_profile.md", "config/candidate_profile.md")
_HEADING_RE = re.compile(r"^ {0,3}(#{1,6})\s+(.*)$")


@dataclass(frozen=True)
class LegacyFiles:
    queries_path: Optional[Path]
    persona_path: Optional[Path]


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise LegacyImportError(f"{path}: cannot be read ({type(exc).__name__})") from exc


def find_legacy_files(base_dir: Path) -> LegacyFiles:
    """Locate the legacy files in today's search order; an empty persona file counts as absent."""
    queries = next((base_dir / c for c in _QUERIES_CANDIDATES if (base_dir / c).is_file()), None)
    persona: Optional[Path] = None
    for candidate in _PERSONA_CANDIDATES:
        path = base_dir / candidate
        if path.is_file() and _read_text(path).strip():
            persona = path
            break
    return LegacyFiles(queries, persona)


# --- queries.json ------------------------------------------------------------------------------


def _entry_error(path: Path, index: int, problem: str) -> LegacyImportError:
    return LegacyImportError(f"{path}: entries[{index}]: {problem}")


def _required_str(path: Path, i: int, item: dict, key: str, *, non_empty: bool) -> str:
    value = item.get(key)
    if not isinstance(value, str):
        raise _entry_error(path, i, f"'{key}' is missing or not text")
    if non_empty and not value.strip():
        raise _entry_error(path, i, f"'{key}' must not be empty")
    return value


def parse_queries_json(path: Path) -> Tuple[PinnedQuery, ...]:
    """Strictly parse a legacy ``queries.json`` into pinned queries (1:1); any problem is an error."""
    try:
        data = json.loads(_read_text(path))
    except json.JSONDecodeError as exc:
        raise LegacyImportError(f"{path}: not valid JSON ({exc.msg} at line {exc.lineno})") from exc
    if not isinstance(data, list):
        raise LegacyImportError(f"{path}: expected a JSON list of query objects")
    queries: List[PinnedQuery] = []
    seen: set = set()
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise _entry_error(path, i, "expected an object")
        query_id = _required_str(path, i, item, "query_id", non_empty=True)
        term = _required_str(path, i, item, "search_term", non_empty=True)
        location = _required_str(path, i, item, "location", non_empty=False)
        sites = item.get("site_names")
        if not isinstance(sites, list) or not sites or not all(isinstance(s, str) for s in sites):
            raise _entry_error(path, i, "'site_names' must be a non-empty list of text")
        country = item.get("country_code")
        if country is not None and not isinstance(country, str):
            raise _entry_error(path, i, "'country_code' must be text or null")
        is_remote = item.get("is_remote", False)
        if not isinstance(is_remote, bool):
            raise _entry_error(path, i, "'is_remote' must be true or false")
        google = item.get("google_search_term")
        if google is not None and not isinstance(google, str):
            raise _entry_error(path, i, "'google_search_term' must be text or null")
        if query_id in seen:
            raise _entry_error(path, i, f"duplicate query_id '{query_id}'")
        seen.add(query_id)
        queries.append(PinnedQuery(query_id, term, location, tuple(sites), country, is_remote, google))
    return tuple(queries)


# --- persona -----------------------------------------------------------------------------------


def extract_bonus_criteria(persona: str) -> str:
    """The persona's bonus criteria; non-empty exactly when ``"bonus"`` occurs in it (any case).

    A markdown heading containing "bonus" yields that section (up to the next heading of the same or a
    higher level); without such a heading, every line containing "bonus" is returned.
    """
    if "bonus" not in persona.lower():
        return ""
    lines = persona.splitlines()
    sections: List[str] = []
    i = 0
    while i < len(lines):
        m = _HEADING_RE.match(lines[i])
        if m and "bonus" in m.group(2).lower():
            level = len(m.group(1))
            j = i + 1
            while j < len(lines):
                nxt = _HEADING_RE.match(lines[j])
                if nxt and len(nxt.group(1)) <= level:
                    break
                j += 1
            sections.append("\n".join(lines[i:j]).strip())
            i = j
        else:
            i += 1
    if sections:
        return "\n\n".join(sections)
    return "\n".join(line.rstrip() for line in lines if "bonus" in line.lower())


def build_imported_snapshot(
    queries: Sequence[PinnedQuery],
    persona: str,
    *,
    wow_threshold: float,
    digest_threshold: float,
    persona_source: Optional[str] = None,
) -> Tuple[ProfileSnapshot, List[str]]:
    """The profile an existing install migrates to, and notes about adjustments made on the way."""
    notes: List[str] = []
    persona = persona.strip()
    if len(persona) > MAX_TEXT_CHARS:
        prefix = f"{persona_source}: persona" if persona_source else "persona"
        raise LegacyImportError(
            f"{prefix} is {len(persona)} characters; the limit is {MAX_TEXT_CHARS}. "
            "Shorten it and run again."
        )
    if threshold_problems(wow_threshold, digest_threshold):
        notes.append(
            f"thresholds wow={wow_threshold:g} / digest={digest_threshold:g} are not valid "
            f"(need 3.0 < digest <= wow - 0.1 and wow <= 10); using {FALLBACK_WOW_THRESHOLD:g} and "
            f"{FALLBACK_DIGEST_THRESHOLD:g}"
        )
        wow_threshold, digest_threshold = FALLBACK_WOW_THRESHOLD, FALLBACK_DIGEST_THRESHOLD
    fields = ProfileFields(
        persona=persona,
        core_role_definition=LEGACY_CORE_ROLE_DEFINITION,
        adjacent_role_definition=LEGACY_ADJACENT_ROLE_DEFINITION,
        dealbreakers=LEGACY_DEALBREAKERS,
        bonus_criteria=extract_bonus_criteria(persona),
        wow_threshold=wow_threshold,
        digest_threshold=digest_threshold,
        max_calls_per_cycle=DEFAULT_MAX_CALLS,
        paused=not queries,
        output_language=None,
    )
    snapshot = ProfileSnapshot(fields, pinned_queries=tuple(queries), rules=seed_rules())
    try:
        calls = compile_profile(snapshot).estimate.total_calls
    except ProfileCompileError:
        calls = 0  # reported with the file's own wording by validation when the import is applied
    if calls > DEFAULT_MAX_CALLS:
        snapshot = replace(snapshot, fields=replace(fields, max_calls_per_cycle=calls))
        notes.append(f"max_calls_per_cycle raised to {calls} to fit the imported queries")
    return snapshot, notes


def auto_import(
    store: ProfileStore, base_dir: Path, *, wow_threshold: float, digest_threshold: float
) -> Optional[ApplyResult]:
    """Import the legacy files when no profile exists yet. Returns None when there is nothing to do."""
    if store.has_profile():
        return None
    files = find_legacy_files(base_dir)
    if files.queries_path is None and files.persona_path is None:
        return None
    queries = parse_queries_json(files.queries_path) if files.queries_path else ()
    persona = _read_text(files.persona_path) if files.persona_path else ""
    snapshot, notes = build_imported_snapshot(
        queries,
        persona,
        wow_threshold=wow_threshold,
        digest_threshold=digest_threshold,
        persona_source=str(files.persona_path) if files.persona_path else None,
    )
    try:
        result = apply_changes(
            store, [ReplaceProfile(snapshot)], label="import legacy files", expected_version=0
        )
    except VersionConflict:
        return None  # another process imported first
    except ProfileValidationError as exc:
        raise LegacyImportError(
            "the legacy files do not make a valid profile: " + "; ".join(exc.messages)
        ) from exc
    result.warnings = notes + result.warnings
    return result


# --- YAML export / import ----------------------------------------------------------------------


def _existing_ancestor(path: Path) -> Path:
    for candidate in [path, *path.parents]:
        if candidate.is_dir():
            return candidate
    return Path(".")


def _is_unignored_in_git(path: Path) -> bool:
    """True only when git says the path is inside a work tree and not ignored (exit code 1)."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(_existing_ancestor(path.parent)), "check-ignore", "-q", str(path)],
            capture_output=True,
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return False  # no git available: not a checkout we can reason about
    return proc.returncode == 1


def export_profile(store: ProfileStore, path: Path, *, force: bool = False) -> Path:
    """Write the stored profile as YAML (mode 0600); refuses a path that git would not ignore."""
    snapshot = store.read_snapshot()
    if snapshot is None:
        raise ProfileValidationError("there is no profile to export")
    target = Path(path).absolute()
    if not force and _is_unignored_in_git(target):
        raise ExportRefused(
            f"{path} is inside a git work tree and not git-ignored; the export contains private "
            "profile data. Use a git-ignored path (for example under local/) or --force."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(snapshot_to_dict(snapshot), sort_keys=False, allow_unicode=True)
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.chmod(target, 0o600)
    return Path(path)


def load_profile_yaml(path: Path) -> ProfileSnapshot:
    """Parse an exported profile file; the result still has to go through ``apply_changes``."""
    try:
        data: Any = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        raise ProfileValidationError(f"{path}: cannot be read ({type(exc).__name__})") from exc
    except yaml.YAMLError as exc:
        raise ProfileValidationError(f"{path}: not valid YAML ({type(exc).__name__})") from exc
    return snapshot_from_dict(data)
