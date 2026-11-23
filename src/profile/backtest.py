"""Dry run of a rule change against the stored job titles (pure apart from the title read)."""
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from src.filters.rules import CompiledRule, legacy_rules, match_title
from src.profile.compiler import compile_profile
from src.profile.snapshot import ProfileSnapshot
from src.storage.db import Database

MIN_BACKTEST_JOBS = 200  # below this a count says nothing about the rule change


@dataclass(frozen=True)
class BacktestResult:
    total_jobs: int
    meaningful: bool
    newly_dropped: int
    newly_passing: int
    dropped_samples: Tuple[str, ...]
    passing_samples: Tuple[str, ...]


def _add_sample(samples: List[str], title: str, limit: int) -> None:
    if len(samples) < limit and title not in samples:
        samples.append(title)


def backtest_rules(
    titles: Sequence[str],
    old_rules: Optional[Sequence[CompiledRule]],
    new_rules: Sequence[CompiledRule],
    sample: int = 5,
) -> BacktestResult:
    """Titles the new rules would drop that the old ones pass, and the reverse (``None`` rules: legacy seed)."""
    total = len(titles)
    if total < MIN_BACKTEST_JOBS:
        return BacktestResult(total, False, 0, 0, (), ())
    dropped = passing = 0
    dropped_samples: List[str] = []
    passing_samples: List[str] = []
    for title in titles:
        was_dropped = match_title(title, old_rules) is not None
        now_dropped = match_title(title, new_rules) is not None
        if now_dropped and not was_dropped:
            dropped += 1
            _add_sample(dropped_samples, title, sample)
        elif was_dropped and not now_dropped:
            passing += 1
            _add_sample(passing_samples, title, sample)
    return BacktestResult(total, True, dropped, passing, tuple(dropped_samples), tuple(passing_samples))


def backtest_snapshots(
    db: Database, old: Optional[ProfileSnapshot], new: ProfileSnapshot
) -> Optional[BacktestResult]:
    """Backtest of ``new`` against ``old`` (no old profile: the legacy seed ruleset); None if no rule changed."""
    new_rules = compile_profile(new).rules
    old_rules = compile_profile(old).rules if old is not None else tuple(legacy_rules())
    if [r.key for r in old_rules] == [r.key for r in new_rules]:
        return None
    return backtest_rules(db.get_titles_for_backtest(), old_rules, new_rules)


def render_backtest(result: BacktestResult) -> List[str]:
    if not result.meaningful:
        return [f"backtest not meaningful: only {result.total_jobs} stored jobs (needs {MIN_BACKTEST_JOBS})"]
    lines = [
        f"backtest: {result.newly_dropped} of {result.total_jobs} stored jobs would now be dropped, "
        f"{result.newly_passing} would now pass"
    ]
    lines.extend(f"  - {title}" for title in result.dropped_samples)
    lines.extend(f"  + {title}" for title in result.passing_samples)
    return lines
