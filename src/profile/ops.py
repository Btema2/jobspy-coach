"""Profile operations: a closed set of pure edits, validation, diff and version handling.

Every change is "read snapshot -> pure operations -> normalise -> validate (including a compile) ->
write the whole snapshot in one transaction guarded by an optimistic version check". Nothing is written
when validation fails. ``validate_snapshot`` is pure and strict (it does not normalise); ``apply_changes``
normalises text that has one obvious canonical form (case and surrounding spaces of countries, languages
and board names; python-jobspy country aliases such as ``us`` map to their canonical value; ``query_id`` is
never changed, because it is part of the stored scrape keys) before validating, so a value such as ``"USA "`` is stored as ``"usa"`` while a typo such
as ``"polnad"`` is rejected instead of silently losing the country-bound boards.
"""
import math
import re
from dataclasses import dataclass, field, fields as dc_fields, replace
from typing import Any, Callable, List, Optional, Sequence, Tuple, TypeVar, Union

from src.filters.seniority import SENIORITY_LEVELS
from src.profile.compiler import (
    KNOWN_COUNTRIES,
    RULE_KINDS,
    RULE_MODES,
    SUPPORTED_BOARDS,
    canonical_country,
    compile_profile,
)
from src.profile.errors import (
    NothingToUndo,
    ProfileCompileError,
    ProfileValidationError,
    VersionConflict,
)
from src.profile.legacy_seed import SEED_RULES
from src.profile.snapshot import (
    Market,
    PinnedQuery,
    ProfileFields,
    ProfileSnapshot,
    Role,
    RoleTerm,
    Rule,
    VersionInfo,
)
from src.profile.store import KEEP_VERSIONS, ProfileStore

MAX_TEXT_CHARS = 8000
MAX_NAME_CHARS = 80
MIN_ROLE_TERM_CHARS = 2
MIN_RULE_TERM_CHARS = 3
MAX_RULE_TERM_CHARS = 60
MAX_RULE_REASON_CHARS = 60
MAX_RULES = 500
MAX_BONUS_POINTS = 5.0
MIN_DIGEST_EXCLUSIVE = 3.0
MAX_WOW = 10.0
FALLBACK_WOW_THRESHOLD = 8.5
FALLBACK_DIGEST_THRESHOLD = 6.5

RULE_SOURCES = frozenset({"user", "llm", "import", "seed"})
_TEXT_FIELDS = (
    "persona", "core_role_definition", "adjacent_role_definition", "dealbreakers", "bonus_criteria",
    "output_language", "target_level",
)
_LONG_TEXT_FIELDS = frozenset(
    {"persona", "core_role_definition", "adjacent_role_definition", "dealbreakers", "bonus_criteria"}
)
_LANGUAGE_RE = re.compile(r"^[a-z]{2,3}$")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_ID_SEPARATOR_RE = re.compile(r"[;=]")  # keep parse_expanded_query_id unambiguous
_SEED_BY_KEY = {(s.kind, s.pattern, s.reason) for s in SEED_RULES}


# --- operations --------------------------------------------------------------------------------


@dataclass(frozen=True)
class SetField:
    name: str
    value: Any


@dataclass(frozen=True)
class AddRole:
    name: str
    terms: Tuple[RoleTerm, ...] = ()


@dataclass(frozen=True)
class RemoveRole:
    name: str


@dataclass(frozen=True)
class AddRoleTerm:
    role: str
    term: str
    language: Optional[str] = None


@dataclass(frozen=True)
class RemoveRoleTerm:
    """Remove a term; ``language=None`` matches the term under any language tag."""

    role: str
    term: str
    language: Optional[str] = None


@dataclass(frozen=True)
class AddMarket:
    market: Market


@dataclass(frozen=True)
class RemoveMarket:
    name: str


@dataclass(frozen=True)
class AddPinnedQuery:
    query: PinnedQuery


@dataclass(frozen=True)
class RemovePinnedQuery:
    query_id: str


@dataclass(frozen=True)
class SetEnabled:
    item: str  # "role", "market" or "pinned_query"
    name: str
    enabled: bool


@dataclass(frozen=True)
class AddRule:
    rule: Rule


@dataclass(frozen=True)
class RemoveRule:
    kind: str
    term: str


@dataclass(frozen=True)
class ReplaceProfile:
    snapshot: ProfileSnapshot


Operation = Union[
    SetField, AddRole, RemoveRole, AddRoleTerm, RemoveRoleTerm, AddMarket, RemoveMarket,
    AddPinnedQuery, RemovePinnedQuery, SetEnabled, AddRule, RemoveRule, ReplaceProfile,
]


@dataclass
class ApplyResult:
    version: int
    diff: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    old: Optional[ProfileSnapshot] = None  # the profile this change was applied to (None: there was none)
    new: Optional[ProfileSnapshot] = None  # the stored result, carrying its new version


# --- apply_operation ---------------------------------------------------------------------------

_T = TypeVar("_T")


def _same(a: str, b: str) -> bool:
    return a.strip().casefold() == b.strip().casefold()


def _index_of(items: Sequence[_T], key: Callable[[_T], str], name: str, what: str) -> int:
    """Index of the item whose key equals ``name`` (exact match preferred, then case-insensitive)."""
    for i, item in enumerate(items):
        if key(item) == name:
            return i
    for i, item in enumerate(items):
        if _same(key(item), name):
            return i
    raise ProfileValidationError(f"no {what} named '{name}'")


def _set_field(fields: ProfileFields, name: str, value: Any) -> ProfileFields:
    names = [f.name for f in dc_fields(ProfileFields)]
    if name not in names:
        raise ProfileValidationError(f"unknown profile field '{name}'")
    default = getattr(ProfileFields(), name)
    if name in _TEXT_FIELDS:
        ok = isinstance(value, str) or (value is None and default is None)
    elif isinstance(default, bool):
        ok = isinstance(value, bool)
    elif isinstance(default, int):
        ok = isinstance(value, int) and not isinstance(value, bool)
    else:
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        value = float(value) if ok else value
    if not ok:
        raise ProfileValidationError(f"profile field '{name}' got a value of the wrong type")
    return replace(fields, **{name: value})


def _replace_at(items: Tuple[_T, ...], index: int, new: _T) -> Tuple[_T, ...]:
    return items[:index] + (new,) + items[index + 1:]


def _drop_at(items: Tuple[_T, ...], index: int) -> Tuple[_T, ...]:
    return items[:index] + items[index + 1:]


def apply_operation(snapshot: ProfileSnapshot, op: Operation) -> ProfileSnapshot:
    """Pure: the snapshot with ``op`` applied. An unknown target raises ``ProfileValidationError``."""
    if isinstance(op, SetField):
        return replace(snapshot, fields=_set_field(snapshot.fields, op.name, op.value))
    if isinstance(op, AddRole):
        return replace(snapshot, roles=snapshot.roles + (Role(op.name, True, tuple(op.terms)),))
    if isinstance(op, RemoveRole):
        i = _index_of(snapshot.roles, lambda r: r.name, op.name, "role")
        return replace(snapshot, roles=_drop_at(snapshot.roles, i))
    if isinstance(op, AddRoleTerm):
        i = _index_of(snapshot.roles, lambda r: r.name, op.role, "role")
        role = snapshot.roles[i]
        role = replace(role, terms=role.terms + (RoleTerm(op.term, op.language),))
        return replace(snapshot, roles=_replace_at(snapshot.roles, i, role))
    if isinstance(op, RemoveRoleTerm):
        i = _index_of(snapshot.roles, lambda r: r.name, op.role, "role")
        role = snapshot.roles[i]
        kept = tuple(
            t for t in role.terms
            if not (_same(t.term, op.term) and (op.language is None or t.language == op.language))
        )
        if len(kept) == len(role.terms):
            raise ProfileValidationError(f"role '{role.name}' has no term '{op.term}'")
        return replace(snapshot, roles=_replace_at(snapshot.roles, i, replace(role, terms=kept)))
    if isinstance(op, AddMarket):
        return replace(snapshot, markets=snapshot.markets + (op.market,))
    if isinstance(op, RemoveMarket):
        i = _index_of(snapshot.markets, lambda m: m.name, op.name, "market")
        return replace(snapshot, markets=_drop_at(snapshot.markets, i))
    if isinstance(op, AddPinnedQuery):
        return replace(snapshot, pinned_queries=snapshot.pinned_queries + (op.query,))
    if isinstance(op, RemovePinnedQuery):
        i = _index_of(snapshot.pinned_queries, lambda q: q.query_id, op.query_id, "pinned query")
        return replace(snapshot, pinned_queries=_drop_at(snapshot.pinned_queries, i))
    if isinstance(op, SetEnabled):
        if op.item == "role":
            i = _index_of(snapshot.roles, lambda r: r.name, op.name, "role")
            return replace(
                snapshot, roles=_replace_at(snapshot.roles, i, replace(snapshot.roles[i], enabled=op.enabled))
            )
        if op.item == "market":
            i = _index_of(snapshot.markets, lambda m: m.name, op.name, "market")
            return replace(
                snapshot,
                markets=_replace_at(snapshot.markets, i, replace(snapshot.markets[i], enabled=op.enabled)),
            )
        if op.item == "pinned_query":
            i = _index_of(snapshot.pinned_queries, lambda q: q.query_id, op.name, "pinned query")
            return replace(
                snapshot,
                pinned_queries=_replace_at(
                    snapshot.pinned_queries, i, replace(snapshot.pinned_queries[i], enabled=op.enabled)
                ),
            )
        raise ProfileValidationError(f"cannot enable or disable '{op.item}'")
    if isinstance(op, AddRule):
        return replace(snapshot, rules=snapshot.rules + (op.rule,))
    if isinstance(op, RemoveRule):
        for i, rule in enumerate(snapshot.rules):
            if rule.kind == op.kind and _same(rule.term, op.term):
                return replace(snapshot, rules=_drop_at(snapshot.rules, i))
        raise ProfileValidationError(f"no {op.kind} rule '{op.term}'")
    if isinstance(op, ReplaceProfile):
        return replace(op.snapshot, version=snapshot.version)
    raise ProfileValidationError(f"unknown operation {type(op).__name__}")


# --- normalisation -----------------------------------------------------------------------------


def _norm_token(value: str) -> str:
    return value.strip().lower()


def normalise_snapshot(snap: ProfileSnapshot) -> ProfileSnapshot:
    """Canonical form of values that have one obvious spelling (see the module docstring)."""
    roles = tuple(
        replace(
            r,
            name=r.name.strip(),
            terms=tuple(
                RoleTerm(t.term.strip(), _norm_token(t.language) if t.language is not None else None)
                for t in r.terms
            ),
        )
        for r in snap.roles
    )
    markets = tuple(
        replace(
            m,
            name=m.name.strip(),
            location=m.location.strip(),
            countries=tuple(canonical_country(c) for c in m.countries),
            languages=tuple(_norm_token(x) for x in m.languages),
        )
        for m in snap.markets
    )
    pinned = tuple(
        replace(
            q,
            boards=tuple(_norm_token(b) for b in q.boards),
            country=canonical_country(q.country) if q.country is not None else None,
        )
        for q in snap.pinned_queries
    )
    rules = tuple(
        replace(r, term=r.term.strip()) if r.match_mode != "pattern" else r for r in snap.rules
    )
    level = snap.fields.target_level
    fields = snap.fields
    if level is not None:
        fields = replace(fields, target_level=level.strip().lower() or None)
    return replace(snap, fields=fields, roles=roles, markets=markets, pinned_queries=pinned, rules=rules)


# --- validation --------------------------------------------------------------------------------


def threshold_problems(wow: float, digest: float) -> List[str]:
    """Violations of ``3.0 < digest <= wow - 0.1`` and ``wow <= 10`` (NaN and infinity are rejected)."""
    if not (math.isfinite(wow) and math.isfinite(digest)):
        return ["wow_threshold and digest_threshold must be finite numbers"]
    problems: List[str] = []
    if wow > MAX_WOW:
        problems.append(f"wow_threshold {wow:g} is above the maximum of {MAX_WOW:g}")
    if digest <= MIN_DIGEST_EXCLUSIVE:
        problems.append(f"digest_threshold {digest:g} must be above {MIN_DIGEST_EXCLUSIVE:g}")
    if digest > round(wow - 0.1, 6):
        problems.append(
            f"digest_threshold {digest:g} must be at most wow_threshold - 0.1 ({round(wow - 0.1, 6):g})"
        )
    return problems


def _name_problems(kind: str, names: List[str], problems: List[str]) -> None:
    seen: set = set()
    for name in names:
        if _ID_SEPARATOR_RE.search(name):
            problems.append(f"{kind} name '{name[:20]}' must not contain ';' or '='; rename the item")
        if not 1 <= len(name.strip()) <= MAX_NAME_CHARS:
            problems.append(f"{kind} name '{name[:20]}' must be 1-{MAX_NAME_CHARS} characters")
        key = name.strip().casefold()
        if key in seen:
            problems.append(f"duplicate {kind} name '{name}'")
        seen.add(key)


def _validate_fields(f: ProfileFields, problems: List[str]) -> None:
    if f.target_level is not None and f.target_level not in SENIORITY_LEVELS:
        problems.append(
            f"target_level '{f.target_level[:20]}' must be one of {', '.join(SENIORITY_LEVELS)} or unset"
        )
    if f.output_language is not None and not _LANGUAGE_RE.fullmatch(f.output_language):
        problems.append(f"output_language '{f.output_language[:20]}' must be a language code such as 'en' or unset")
    for name in _TEXT_FIELDS:
        value = getattr(f, name)
        if value is not None and len(value) > MAX_TEXT_CHARS:
            problems.append(f"{name} is {len(value)} characters; the limit is {MAX_TEXT_CHARS}")
    problems.extend(threshold_problems(f.wow_threshold, f.digest_threshold))
    if not (math.isfinite(f.bonus_points) and 0 <= f.bonus_points <= MAX_BONUS_POINTS):
        problems.append(f"bonus_points {f.bonus_points:g} must be between 0 and {MAX_BONUS_POINTS:g}")
    if f.max_calls_per_cycle < 1:
        problems.append(f"max_calls_per_cycle {f.max_calls_per_cycle} must be at least 1")


def _validate_roles(snap: ProfileSnapshot, problems: List[str]) -> None:
    _name_problems("role", [r.name for r in snap.roles], problems)
    for role in snap.roles:
        seen: set = set()
        for t in role.terms:
            if not MIN_ROLE_TERM_CHARS <= len(t.term.strip()) <= MAX_NAME_CHARS:
                problems.append(
                    f"role '{role.name}': term '{t.term[:20]}' must be "
                    f"{MIN_ROLE_TERM_CHARS}-{MAX_NAME_CHARS} characters"
                )
            if t.language is not None and not _LANGUAGE_RE.match(t.language):
                problems.append(
                    f"role '{role.name}': term '{t.term}' has language tag '{t.language}', "
                    "expected a 2-3 letter lowercase code"
                )
            if _ID_SEPARATOR_RE.search(t.term):
                problems.append(f"role '{role.name}': term '{t.term[:20]}' must not contain ';' or '='; rename the item")
            key = (t.term.strip().casefold(), t.language)
            if key in seen:
                problems.append(f"role '{role.name}': duplicate term '{t.term}'")
            seen.add(key)


def _validate_markets(snap: ProfileSnapshot, problems: List[str]) -> None:
    _name_problems("market", [m.name for m in snap.markets], problems)
    for m in snap.markets:
        for country in m.countries:
            if country not in KNOWN_COUNTRIES:
                problems.append(f"market '{m.name}': unknown country '{country}'")
        for lang in m.languages:
            if not _LANGUAGE_RE.match(lang):
                problems.append(
                    f"market '{m.name}': language '{lang}' must be a 2-3 letter lowercase code"
                )
        if m.remote_only and not m.remote:
            problems.append(f"market '{m.name}': remote_only requires remote")
        if not m.location.strip():
            problems.append(f"market '{m.name}': location must not be empty")


def _validate_pinned(snap: ProfileSnapshot, problems: List[str]) -> None:
    seen: set = set()
    for q in snap.pinned_queries:
        label = f"pinned query '{q.query_id}'"
        if not q.query_id.strip():
            problems.append("pinned query: query_id must not be empty")
        elif q.query_id.casefold() in seen:
            problems.append(f"{label}: duplicate query_id")
        seen.add(q.query_id.casefold())
        if not q.term.strip():
            problems.append(f"{label}: term must not be empty")
        if not q.boards:
            problems.append(f"{label}: at least one board is required")
        for board in q.boards:
            if board not in SUPPORTED_BOARDS:
                problems.append(f"{label}: unknown board '{board}'")
        if q.country is not None and q.country not in KNOWN_COUNTRIES:
            problems.append(f"{label}: unknown country '{q.country}'")


def _validate_rules(snap: ProfileSnapshot, problems: List[str]) -> None:
    if len(snap.rules) > MAX_RULES:
        problems.append(f"{len(snap.rules)} rules; at most {MAX_RULES} are allowed")
    seen: set = set()
    for r in snap.rules:
        label = f"rule {r.kind} '{r.term[:30]}'"
        if r.kind not in RULE_KINDS:
            problems.append(f"{label}: unknown kind '{r.kind}'")
        if r.match_mode not in RULE_MODES:
            problems.append(f"{label}: unknown match mode '{r.match_mode}'")
        if r.source not in RULE_SOURCES:
            problems.append(f"{label}: unknown source '{r.source}'")
        if len(r.reason) > MAX_RULE_REASON_CHARS:
            problems.append(f"{label}: reason is {len(r.reason)} characters; the limit is {MAX_RULE_REASON_CHARS}")
        if r.match_mode == "pattern":
            if r.source != "seed" or (r.kind, r.term, r.reason) not in _SEED_BY_KEY:
                problems.append(f"{label}: pattern rules are only allowed as an unchanged legacy seed entry")
        elif r.match_mode in ("exact", "prefix"):
            if _CONTROL_RE.search(r.term):
                problems.append(f"{label}: term must not contain control characters or line breaks")
            if not MIN_RULE_TERM_CHARS <= len(r.term.strip()) <= MAX_RULE_TERM_CHARS or r.term != r.term.strip():
                problems.append(
                    f"{label}: term must be {MIN_RULE_TERM_CHARS}-{MAX_RULE_TERM_CHARS} characters "
                    "without leading or trailing spaces"
                )
        key = (r.kind, r.term.casefold())
        if key in seen:
            problems.append(f"{label}: duplicate rule")
        seen.add(key)


def validate_snapshot(new: ProfileSnapshot, old: Optional[ProfileSnapshot]) -> List[str]:
    """Pure validation; an empty list means valid. ``old`` is the snapshot being replaced (or None)."""
    problems: List[str] = []
    _validate_fields(new.fields, problems)
    _validate_roles(new, problems)
    _validate_markets(new, problems)
    _validate_pinned(new, problems)
    _validate_rules(new, problems)

    try:
        compiled = compile_profile(new)
    except ProfileCompileError as exc:
        problems.append(str(exc))
        return problems
    if new.fields.max_calls_per_cycle >= 1 and compiled.estimate.total_calls > new.fields.max_calls_per_cycle:
        problems.append(
            f"the profile needs {compiled.estimate.total_calls} calls per cycle, more than "
            f"max_calls_per_cycle {new.fields.max_calls_per_cycle}"
        )
    resuming = new.fields.paused is False and (old is None or old.fields.paused)
    if resuming and not compiled.queries:
        problems.append("cannot resume: the profile has no enabled search queries")
    return problems


# --- diff --------------------------------------------------------------------------------------


def _one_line(value: Any, limit: int = 80) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _show(value: Any) -> str:
    if value is None:
        return "(unset)"
    if isinstance(value, (tuple, list)):
        return ",".join(_one_line(v) for v in value) or "(none)"
    return _one_line(value)


def _term_line(sign: str, term: RoleTerm, role: str) -> str:
    lang = f" [{_one_line(term.language)}]" if term.language else ""
    return f"{sign} term {_one_line(term.term)}{lang} (role {_one_line(role)})"


def _index(items: Sequence[_T], key: Callable[[_T], str]) -> dict:
    return {key(i).strip().casefold(): i for i in items}


def _enabled_line(kind: str, name: str, old_enabled: bool, new_enabled: bool) -> List[str]:
    if old_enabled == new_enabled:
        return []
    return [f"~ {kind} {_one_line(name)} {'enabled' if new_enabled else 'disabled'}"]


def diff_snapshots(old: Optional[ProfileSnapshot], new: ProfileSnapshot) -> List[str]:
    """One code-rendered line per change; no line contains a newline."""
    lines: List[str] = []
    old_fields = old.fields if old is not None else ProfileFields()
    for f in dc_fields(ProfileFields):
        a, b = getattr(old_fields, f.name), getattr(new.fields, f.name)
        if a == b:
            continue
        if f.name in _LONG_TEXT_FIELDS:
            lines.append(f"{f.name}: {len(a or '')} -> {len(b or '')} chars")
        else:
            lines.append(f"{f.name}: {_show(a)} -> {_show(b)}")

    old_roles = _index(old.roles if old else (), lambda r: r.name)
    new_roles = _index(new.roles, lambda r: r.name)
    for key, role in old_roles.items():
        if key not in new_roles:
            lines.append(f"- role {_one_line(role.name)}")
    term_lines: List[str] = []
    for key, role in new_roles.items():
        before = old_roles.get(key)
        if before is None:
            lines.append(f"+ role {_one_line(role.name)}")
        else:
            lines.extend(_enabled_line("role", role.name, before.enabled, role.enabled))
        before_terms = {(t.term.strip().casefold(), t.language): t for t in (before.terms if before else ())}
        now_terms = {(t.term.strip().casefold(), t.language): t for t in role.terms}
        term_lines.extend(_term_line("-", t, role.name) for k, t in before_terms.items() if k not in now_terms)
        term_lines.extend(_term_line("+", t, role.name) for k, t in now_terms.items() if k not in before_terms)
    lines.extend(term_lines)

    old_markets = _index(old.markets if old else (), lambda m: m.name)
    new_markets = _index(new.markets, lambda m: m.name)
    for key, market in old_markets.items():
        if key not in new_markets:
            lines.append(f"- market {_one_line(market.name)}")
    for key, market in new_markets.items():
        before = old_markets.get(key)
        if before is None:
            lines.append(f"+ market {_one_line(market.name)}")
            continue
        lines.extend(_enabled_line("market", market.name, before.enabled, market.enabled))
        for name in ("location", "countries", "remote", "remote_only", "languages"):
            a, b = getattr(before, name), getattr(market, name)
            if a != b:
                lines.append(f"~ market {_one_line(market.name)} {name}: {_show(a)} -> {_show(b)}")

    old_pinned = {q.query_id: q for q in (old.pinned_queries if old else ())}
    new_pinned = {q.query_id: q for q in new.pinned_queries}
    for qid in old_pinned:
        if qid not in new_pinned:
            lines.append(f"- pinned {_one_line(qid)}")
    for qid, query in new_pinned.items():
        before = old_pinned.get(qid)
        if before is None:
            lines.append(f"+ pinned {_one_line(qid)}")
            continue
        lines.extend(_enabled_line("pinned", qid, before.enabled, query.enabled))
        for name in ("term", "location", "boards", "country", "is_remote", "google_search_term"):
            a, b = getattr(before, name), getattr(query, name)
            if a != b:
                lines.append(f"~ pinned {_one_line(qid)} {name}: {_show(a)} -> {_show(b)}")

    old_rules = {(r.kind, r.term.casefold()): r for r in (old.rules if old else ())}
    new_rules = {(r.kind, r.term.casefold()): r for r in new.rules}
    for key, rule in old_rules.items():
        if key not in new_rules:
            lines.append(f"- rule {rule.kind} {_one_line(rule.term)}")
    for key, rule in new_rules.items():
        if key not in old_rules:
            lines.append(f"+ rule {rule.kind} {_one_line(rule.term)}")
    return lines


# --- apply / versions --------------------------------------------------------------------------


def apply_changes(
    store: ProfileStore,
    ops: Sequence[Operation],
    *,
    label: str,
    expected_version: Optional[int] = None,
) -> ApplyResult:
    """Apply ``ops`` to the stored profile as one validated, versioned change."""
    current = store.read_snapshot()
    base = current if current is not None else ProfileSnapshot(ProfileFields())
    if expected_version is not None and base.version != expected_version:
        raise VersionConflict(
            f"profile is at version {base.version}, change was based on version {expected_version}"
        )
    new = base
    for op in ops:
        new = apply_operation(new, op)
    new = normalise_snapshot(new)
    problems = validate_snapshot(new, current)
    if problems:
        raise ProfileValidationError(problems)
    version = store.write_snapshot(new, expected_version=base.version, label=label)
    warnings = list(compile_profile(new).warnings)
    return ApplyResult(version, diff_snapshots(current, new), warnings, current, replace(new, version=version))


def history(store: ProfileStore) -> List[VersionInfo]:
    return store.list_versions()


def _restore_version(store: ProfileStore, current_version: int, target: int, label: str) -> ApplyResult:
    snapshot = store.load_version(target)
    return apply_changes(
        store, [ReplaceProfile(snapshot)], label=label, expected_version=current_version
    )


def undo(store: ProfileStore) -> ApplyResult:
    """Restore the newest stored version below the current one, as a new version."""
    current = store.read_snapshot()
    if current is None:
        raise NothingToUndo("there is no profile yet")
    earlier = [v.version for v in store.list_versions() if v.version < current.version]
    if not earlier:
        raise NothingToUndo("there is no earlier version to return to")
    target = max(earlier)
    return _restore_version(store, current.version, target, f"undo (back to v{target})")


def restore(store: ProfileStore, version: int) -> ApplyResult:
    """Restore one of the stored versions as a new version."""
    current = store.read_snapshot()
    try:
        if current is None:
            raise KeyError(version)
        snapshot = store.load_version(version)
    except KeyError:
        raise ProfileValidationError(
            f"version {version} is not among the last {KEEP_VERSIONS}"
        ) from None
    return apply_changes(
        store, [ReplaceProfile(snapshot)], label=f"restore v{version}", expected_version=current.version
    )


def resolve_thresholds(wow: float, digest: float) -> Tuple[float, float, Optional[str]]:
    """The thresholds to store: the given ones when valid, else the fallbacks and a note saying so."""
    if not threshold_problems(wow, digest):
        return wow, digest, None
    note = (
        f"thresholds wow={wow:g} / digest={digest:g} are not valid "
        f"(need 3.0 < digest <= wow - 0.1 and wow <= 10); using {FALLBACK_WOW_THRESHOLD:g} and "
        f"{FALLBACK_DIGEST_THRESHOLD:g}"
    )
    return FALLBACK_WOW_THRESHOLD, FALLBACK_DIGEST_THRESHOLD, note


def create_starter_profile(
    store: ProfileStore, *, wow_threshold: float, digest_threshold: float
) -> ApplyResult:
    """An empty, paused profile (used by ``init`` when no legacy files exist).

    Invalid ``.env`` thresholds fall back to the defaults with a note in ``warnings`` (like the import).
    """
    wow, digest, note = resolve_thresholds(wow_threshold, digest_threshold)
    snapshot = ProfileSnapshot(ProfileFields(paused=True, wow_threshold=wow, digest_threshold=digest))
    result = apply_changes(
        store, [ReplaceProfile(snapshot)], label="init starter profile", expected_version=0
    )
    if note:
        result.warnings = [note] + result.warnings
    return result
