"""Plain, immutable profile snapshot and its (versioned) dict shape.

A snapshot holds no database ids: it is the whole profile state that is read, validated, compiled and
written as one unit. ``snapshot_to_dict`` / ``snapshot_from_dict`` define the stored (``profile_versions``)
and exported shape; the dict carries a ``snapshot_schema`` stamp so older stored snapshots can be upgraded.
"""
from dataclasses import dataclass, fields as dc_fields
from typing import Any, Dict, List, Optional, Tuple

from src.profile.errors import ProfileValidationError

SNAPSHOT_SCHEMA = 1


@dataclass(frozen=True)
class ProfileFields:
    persona: str = ""
    core_role_definition: str = ""
    adjacent_role_definition: str = ""
    dealbreakers: str = ""
    bonus_criteria: str = ""
    bonus_points: float = 1.5
    output_language: Optional[str] = None
    target_level: Optional[str] = None
    wow_threshold: float = 8.5
    digest_threshold: float = 6.5
    max_calls_per_cycle: int = 100
    paused: bool = False


@dataclass(frozen=True)
class RoleTerm:
    term: str
    language: Optional[str] = None


@dataclass(frozen=True)
class Role:
    name: str
    enabled: bool = True
    terms: Tuple[RoleTerm, ...] = ()


@dataclass(frozen=True)
class Market:
    name: str
    location: str
    countries: Tuple[str, ...] = ()
    remote: bool = False
    remote_only: bool = False
    languages: Tuple[str, ...] = ()
    enabled: bool = True


@dataclass(frozen=True)
class PinnedQuery:
    query_id: str
    term: str
    location: str
    boards: Tuple[str, ...]
    country: Optional[str] = None
    is_remote: bool = False
    google_search_term: Optional[str] = None
    enabled: bool = True


@dataclass(frozen=True)
class Rule:
    kind: str
    term: str
    match_mode: str
    reason: str
    source: str


@dataclass(frozen=True)
class ProfileSnapshot:
    fields: ProfileFields
    roles: Tuple[Role, ...] = ()
    markets: Tuple[Market, ...] = ()
    pinned_queries: Tuple[PinnedQuery, ...] = ()
    rules: Tuple[Rule, ...] = ()
    version: int = 0


@dataclass(frozen=True)
class VersionInfo:
    version: int
    label: str
    created_at: str


# --- dict shape -------------------------------------------------------------------------------

_TOP_KEYS = {"snapshot_schema", "profile", "roles", "markets", "pinned_queries", "rules"}


def snapshot_to_dict(s: ProfileSnapshot) -> Dict[str, Any]:
    """JSON/YAML-safe dict of the snapshot (``version`` is stored beside it, not inside)."""
    return {
        "snapshot_schema": SNAPSHOT_SCHEMA,
        "profile": {f.name: getattr(s.fields, f.name) for f in dc_fields(ProfileFields)},
        "roles": [
            {
                "name": r.name,
                "enabled": r.enabled,
                "terms": [{"term": t.term, "language": t.language} for t in r.terms],
            }
            for r in s.roles
        ],
        "markets": [
            {
                "name": m.name,
                "location": m.location,
                "countries": list(m.countries),
                "remote": m.remote,
                "remote_only": m.remote_only,
                "languages": list(m.languages),
                "enabled": m.enabled,
            }
            for m in s.markets
        ],
        "pinned_queries": [
            {
                "query_id": p.query_id,
                "term": p.term,
                "location": p.location,
                "boards": list(p.boards),
                "country": p.country,
                "is_remote": p.is_remote,
                "google_search_term": p.google_search_term,
                "enabled": p.enabled,
            }
            for p in s.pinned_queries
        ],
        "rules": [
            {"kind": r.kind, "term": r.term, "match_mode": r.match_mode, "reason": r.reason, "source": r.source}
            for r in s.rules
        ],
    }


def upgrade_snapshot_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Bring a stored dict up to ``SNAPSHOT_SCHEMA``. Stamp 1 passes through; others are refused."""
    if not isinstance(d, dict):
        raise ProfileValidationError("snapshot must be a mapping")
    stamp = d.get("snapshot_schema")
    if isinstance(stamp, bool) or not isinstance(stamp, int):
        raise ProfileValidationError("snapshot_schema is missing or not an integer")
    if stamp > SNAPSHOT_SCHEMA:
        raise ProfileValidationError(
            f"snapshot_schema {stamp} is newer than this version supports ({SNAPSHOT_SCHEMA})"
        )
    if stamp != 1:
        raise ProfileValidationError(f"unsupported snapshot_schema {stamp}")
    return d


class _Problems:
    def __init__(self) -> None:
        self.messages: List[str] = []

    def add(self, msg: str) -> None:
        self.messages.append(msg)


def _check_keys(d: Any, where: str, allowed: set, required: set, p: _Problems) -> bool:
    if not isinstance(d, dict):
        p.add(f"{where}: expected a mapping")
        return False
    ok = True
    for key in sorted(map(str, d.keys())):
        if key not in allowed:
            p.add(f"{where}: unknown key '{key}'")
            ok = False
    for key in sorted(required):
        if key not in d:
            p.add(f"{where}: missing required key '{key}'")
            ok = False
    return ok


def _str(d: dict, key: str, where: str, default: str, p: _Problems) -> str:
    val = d.get(key, default)
    if not isinstance(val, str):
        p.add(f"{where}.{key}: expected text")
        return default
    return val


def _opt_str(d: dict, key: str, where: str, p: _Problems) -> Optional[str]:
    val = d.get(key)
    if val is not None and not isinstance(val, str):
        p.add(f"{where}.{key}: expected text or null")
        return None
    return val


def _bool(d: dict, key: str, where: str, default: bool, p: _Problems) -> bool:
    val = d.get(key, default)
    if not isinstance(val, bool):
        p.add(f"{where}.{key}: expected true or false")
        return default
    return val


def _float(d: dict, key: str, where: str, default: float, p: _Problems) -> float:
    val = d.get(key, default)
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        p.add(f"{where}.{key}: expected a number")
        return default
    return float(val)


def _int(d: dict, key: str, where: str, default: int, p: _Problems) -> int:
    val = d.get(key, default)
    if isinstance(val, bool) or not isinstance(val, int):
        p.add(f"{where}.{key}: expected an integer")
        return default
    return val


def _str_tuple(d: dict, key: str, where: str, p: _Problems) -> Tuple[str, ...]:
    val = d.get(key, [])
    if not isinstance(val, (list, tuple)) or not all(isinstance(x, str) for x in val):
        p.add(f"{where}.{key}: expected a list of text")
        return ()
    return tuple(val)


def _list(d: dict, key: str, p: _Problems) -> list:
    val = d.get(key, [])
    if not isinstance(val, (list, tuple)):
        p.add(f"{key}: expected a list")
        return []
    return list(val)


def _parse_fields(raw: Any, p: _Problems) -> ProfileFields:
    names = {f.name for f in dc_fields(ProfileFields)}
    if raw is None:
        return ProfileFields()
    if not _check_keys(raw, "profile", names, set(), p):
        return ProfileFields()
    defaults = ProfileFields()
    w = "profile"
    return ProfileFields(
        persona=_str(raw, "persona", w, defaults.persona, p),
        core_role_definition=_str(raw, "core_role_definition", w, "", p),
        adjacent_role_definition=_str(raw, "adjacent_role_definition", w, "", p),
        dealbreakers=_str(raw, "dealbreakers", w, "", p),
        bonus_criteria=_str(raw, "bonus_criteria", w, "", p),
        bonus_points=_float(raw, "bonus_points", w, defaults.bonus_points, p),
        output_language=_opt_str(raw, "output_language", w, p),
        target_level=_opt_str(raw, "target_level", w, p),
        wow_threshold=_float(raw, "wow_threshold", w, defaults.wow_threshold, p),
        digest_threshold=_float(raw, "digest_threshold", w, defaults.digest_threshold, p),
        max_calls_per_cycle=_int(raw, "max_calls_per_cycle", w, defaults.max_calls_per_cycle, p),
        paused=_bool(raw, "paused", w, False, p),
    )


def _parse_role(raw: Any, i: int, p: _Problems) -> Optional[Role]:
    w = f"roles[{i}]"
    if not _check_keys(raw, w, {"name", "enabled", "terms"}, {"name"}, p):
        return None
    terms: List[RoleTerm] = []
    for j, t in enumerate(_list(raw, "terms", p)):
        tw = f"{w}.terms[{j}]"
        if _check_keys(t, tw, {"term", "language"}, {"term"}, p):
            terms.append(RoleTerm(_str(t, "term", tw, "", p), _opt_str(t, "language", tw, p)))
    return Role(_str(raw, "name", w, "", p), _bool(raw, "enabled", w, True, p), tuple(terms))


def _parse_market(raw: Any, i: int, p: _Problems) -> Optional[Market]:
    w = f"markets[{i}]"
    allowed = {"name", "location", "countries", "remote", "remote_only", "languages", "enabled"}
    if not _check_keys(raw, w, allowed, {"name", "location"}, p):
        return None
    return Market(
        name=_str(raw, "name", w, "", p),
        location=_str(raw, "location", w, "", p),
        countries=_str_tuple(raw, "countries", w, p),
        remote=_bool(raw, "remote", w, False, p),
        remote_only=_bool(raw, "remote_only", w, False, p),
        languages=_str_tuple(raw, "languages", w, p),
        enabled=_bool(raw, "enabled", w, True, p),
    )


def _parse_pinned(raw: Any, i: int, p: _Problems) -> Optional[PinnedQuery]:
    w = f"pinned_queries[{i}]"
    allowed = {"query_id", "term", "location", "boards", "country", "is_remote", "google_search_term", "enabled"}
    if not _check_keys(raw, w, allowed, {"query_id", "term", "location", "boards"}, p):
        return None
    return PinnedQuery(
        query_id=_str(raw, "query_id", w, "", p),
        term=_str(raw, "term", w, "", p),
        location=_str(raw, "location", w, "", p),
        boards=_str_tuple(raw, "boards", w, p),
        country=_opt_str(raw, "country", w, p),
        is_remote=_bool(raw, "is_remote", w, False, p),
        google_search_term=_opt_str(raw, "google_search_term", w, p),
        enabled=_bool(raw, "enabled", w, True, p),
    )


def _parse_rule(raw: Any, i: int, p: _Problems) -> Optional[Rule]:
    w = f"rules[{i}]"
    keys = {"kind", "term", "match_mode", "reason", "source"}
    if not _check_keys(raw, w, keys, keys, p):
        return None
    return Rule(*(_str(raw, k, w, "", p) for k in ("kind", "term", "match_mode", "reason", "source")))


def snapshot_from_dict(d: Dict[str, Any], version: int = 0) -> ProfileSnapshot:
    """Strictly parse a snapshot dict; unknown keys and wrong types raise ``ProfileValidationError``."""
    d = upgrade_snapshot_dict(d)
    p = _Problems()
    _check_keys(d, "snapshot", _TOP_KEYS, set(), p)
    fields = _parse_fields(d.get("profile"), p)
    roles = [r for i, raw in enumerate(_list(d, "roles", p)) if (r := _parse_role(raw, i, p))]
    markets = [m for i, raw in enumerate(_list(d, "markets", p)) if (m := _parse_market(raw, i, p))]
    pinned = [q for i, raw in enumerate(_list(d, "pinned_queries", p)) if (q := _parse_pinned(raw, i, p))]
    rules = [r for i, raw in enumerate(_list(d, "rules", p)) if (r := _parse_rule(raw, i, p))]
    if p.messages:
        raise ProfileValidationError(p.messages)
    return ProfileSnapshot(fields, tuple(roles), tuple(markets), tuple(pinned), tuple(rules), version)
