"""``/stats``: scrape statistics of the last 7 days, one row per role (or pinned / legacy query).

Rows are built from ``scrape_runs`` (calls and outcomes per ``query_key``) and ``job_hits`` (which query keys
surfaced which evaluated jobs). Failed calls are reported apart from empty results, and ``relevant`` counts
distinct evaluated jobs at or above the profile's current digest threshold.
"""
from dataclasses import dataclass
from typing import AbstractSet, Dict, Iterable, List, Sequence, Set, Tuple

from src.bot.render import chunk_html, esc, paginate
from src.bot.updates import OutgoingMessage
from src.config import Settings
from src.profile.compiler import parse_expanded_query_id
from src.profile.store import ProfileStore
from src.storage.db import Database, QueryKeyRuns

STATS_DAYS = 7
LABEL_CLIP = 80
_USAGE = "Usage: /stats [page]"


@dataclass(frozen=True)
class RoleStats:
    label: str
    calls: int
    ok: int
    empty: int
    failed: int
    new_jobs: int
    relevant: int


def group_label(query_key: str, pinned_ids: AbstractSet[str], role_names: AbstractSet[str]) -> str:
    """The row a query key belongs to. Pinned ids are recognised first: one may look like an expanded id."""
    query_id = query_key.rsplit("|", 1)[0]
    if query_id in pinned_ids:
        return f"pinned {query_id}"
    parsed = parse_expanded_query_id(query_id)
    if parsed is not None:
        role = parsed["role"]
        if role.casefold() in {name.casefold() for name in role_names}:
            return role
        return f"{role} (removed)"
    return f"query {query_id}"


def build_role_stats(
    runs: Iterable[QueryKeyRuns],
    hits: Iterable[Tuple[str, int]],
    pinned_ids: AbstractSet[str],
    role_names: AbstractSet[str],
) -> List[RoleStats]:
    totals: Dict[str, List[int]] = {}
    for run in runs:
        label = group_label(run.query_key, pinned_ids, role_names)
        row = totals.setdefault(label, [0, 0, 0, 0, 0])
        for index, value in enumerate((run.calls, run.ok, run.empty, run.failed, run.new_jobs)):
            row[index] += value
    relevant: Dict[str, Set[int]] = {}
    for query_key, job_id in hits:
        label = group_label(query_key, pinned_ids, role_names)
        if label in totals:
            relevant.setdefault(label, set()).add(job_id)
    rows = [
        RoleStats(label, *counts, len(relevant.get(label, ()))) for label, counts in totals.items()
    ]
    return sorted(rows, key=lambda r: (-r.relevant, -r.calls, r.label))


def _clip(text: str) -> str:
    return text if len(text) <= LABEL_CLIP else text[: LABEL_CLIP - 3] + "..."


def render_stats(rows: Sequence[RoleStats], digest_threshold: float, page: int) -> List[OutgoingMessage]:
    if not rows:
        return [OutgoingMessage(f"No scrape runs in the last {STATS_DAYS} days.")]
    lines = [
        f"{esc(_clip(r.label))}: calls {r.calls} (ok {r.ok}, empty {r.empty}, failed {r.failed}), "
        f"new {r.new_jobs}, relevant {r.relevant}"
        for r in rows
    ]
    shown, page, pages = paginate(lines, page)
    out = [
        f"<b>Stats for the last {STATS_DAYS} days</b>",
        esc(f"relevant = evaluated, score >= {digest_threshold:g}"),
    ]
    out.extend(shown)
    if pages > 1:
        out.append(f"page {page}/{pages}" + (f", send /stats {page + 1}" if page < pages else ""))
    return [OutgoingMessage(piece) for piece in chunk_html("\n".join(out))]


def _page(args: str):
    token = args.strip()
    if not token:
        return 1
    return int(token) if token.isascii() and token.isdigit() and len(token) < 9 else None


def stats_command(db: Database, store: ProfileStore, settings: Settings, args: str) -> List[OutgoingMessage]:
    """The reply of ``/stats [page]``; the digest threshold is the stored profile's, else the ``.env`` one."""
    page = _page(args)
    if page is None:
        return [OutgoingMessage(_USAGE)]
    snapshot = store.read_snapshot()
    if snapshot is None:
        digest_threshold = settings.digest_score_threshold
        pinned: Set[str] = set()
        role_names: Set[str] = set()
    else:
        digest_threshold = snapshot.fields.digest_threshold
        pinned = {q.query_id for q in snapshot.pinned_queries}
        role_names = {r.name for r in snapshot.roles}
    rows = build_role_stats(
        db.get_query_key_runs(STATS_DAYS), db.get_relevant_hits(STATS_DAYS, digest_threshold), pinned, role_names
    )
    return render_stats(rows, digest_threshold, page)
