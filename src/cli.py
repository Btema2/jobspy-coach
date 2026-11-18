import importlib.metadata
from pathlib import Path
from typing import Optional
import typer
from typer.core import TyperGroup
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from src.config import get_settings
from src.evaluator.errors import ConfigurationError
from src.logging_config import configure_logging
from src.notifier.telegram_notifier import TelegramNotifier
from src.profile.backtest import backtest_snapshots, render_backtest
from src.profile.compiler import compile_profile
from src.profile.errors import (
    LegacyImportError,
    ProfileCompileError,
    ProfileError,
    ProfileValidationError,
    VersionConflict,
)
from src.profile.importer import DEFAULT_EXPORT_PATH, auto_import, export_profile, load_profile_yaml
from src.profile.ops import ReplaceProfile, apply_changes, create_starter_profile
from src.profile.requeue import rebucket_evaluated, requeue_prefiltered
from src.profile.store import ProfileStore
from src.scheduler.daemon import DaemonService
from src.scraper.version_checker import check_jobspy_update
from src.storage.db import Database

_SUBCOMMAND_ARGS = "jobspy_coach.subcommand_args"


class _RecordingGroup(TyperGroup):
    """Remembers the arguments of the subcommand: Click only parses them after the group callback ran."""

    def resolve_command(self, ctx, args):
        ctx.meta[_SUBCOMMAND_ARGS] = list(args)
        return super().resolve_command(ctx, args)


app = typer.Typer(cls=_RecordingGroup, help="JobSpy Coach: Autonomous Coaching Job Monitor Daemon")
console = Console()

configure_logging()

profile_app = typer.Typer(help="Inspect, export and import the stored search profile")
app.add_typer(profile_app, name="profile")


def _fail(message: str, *, hint: str = "") -> "typer.Exit":
    """Print an error without Rich markup (messages contain brackets and backticks) and return Exit(1)."""
    console.print(message, style="bold red", markup=False, highlight=False)
    if hint:
        console.print(hint, markup=False, highlight=False)
    return typer.Exit(code=1)


def _fail_profile_error(error: ProfileError, *, hint: str = "") -> "typer.Exit":
    messages = error.messages if isinstance(error, ProfileValidationError) else [str(error)]
    for message in messages:
        console.print(f"- {message}", style="bold red", markup=False, highlight=False)
    if hint:
        console.print(hint, markup=False, highlight=False)
    return typer.Exit(code=1)


def _profile_store() -> ProfileStore:
    db = Database(get_settings().db_path)
    db.init_db()
    return ProfileStore(db)


_LEGACY_HINT = (
    "Fix the file and entry named above (or move the file away), then run the command again. "
    "Nothing was imported."
)


def _help_requested(ctx: typer.Context) -> bool:
    """True when the remaining arguments ask for help (`<command> --help`): no database work for that."""
    return ctx.resilient_parsing or "--help" in ctx.meta.get(_SUBCOMMAND_ARGS, [])


@app.callback()
def main(ctx: typer.Context) -> None:
    """Open the database and import the legacy queries.json / .candidate_profile.md once, before any command."""
    if _help_requested(ctx):
        return
    settings = get_settings()
    store = _profile_store()
    try:
        result = auto_import(
            store,
            Path.cwd(),
            wow_threshold=settings.wow_score_threshold,
            digest_threshold=settings.digest_score_threshold,
        )
    except LegacyImportError as exc:
        raise _fail_profile_error(exc, hint=_LEGACY_HINT) from None
    except ProfileValidationError as exc:
        raise _fail_profile_error(exc, hint=_LEGACY_HINT) from None
    if result is not None:
        console.print(f"Imported legacy profile (version {result.version})", style="bold green", markup=False)
        for note in result.warnings:
            console.print(f"- {note}", style="yellow", markup=False, highlight=False)


def _exit_on_missing_provider(error: ConfigurationError) -> "typer.Exit":
    console.print(f"[bold red]Configuration error:[/bold red] {error}")
    return typer.Exit(code=1)


@app.command()
def run():
    """Start the 24/7 background daemon scheduler."""
    if not _profile_store().has_profile():
        raise _fail("no profile, run `jobspy-coach init`")
    console.print("[bold green]Starting JobSpy Coach Daemon...[/bold green]")
    daemon = DaemonService()
    try:
        daemon.start()
    except ConfigurationError as e:
        raise _exit_on_missing_provider(e) from None


@app.command()
def init() -> None:
    """Create an empty, paused starter profile (does nothing when a profile exists). Never prompts."""
    settings = get_settings()
    store = _profile_store()
    existing = store.read_snapshot()
    if existing is not None:
        console.print(f"Profile already exists (version {existing.version}); nothing to do.", markup=False)
        return
    try:
        result = create_starter_profile(
            store,
            wow_threshold=settings.wow_score_threshold,
            digest_threshold=settings.digest_score_threshold,
        )
    except VersionConflict:  # a concurrent `init` (or import) won the race
        current = store.read_snapshot()
        console.print(
            f"Profile already exists (version {current.version if current else '?'}); nothing to do.",
            markup=False,
        )
        return
    except ProfileError as exc:
        raise _fail_profile_error(exc) from None
    console.print(f"Created an empty starter profile (version {result.version}); it is paused.", markup=False)
    for note in result.warnings:
        console.print(f"- {note}", style="yellow", markup=False, highlight=False)
    console.print(
        "Fill it in: run `jobspy-coach profile export`, edit the YAML file, then run "
        "`jobspy-coach profile import <file>`.",
        markup=False,
    )


@profile_app.command("preview")
def profile_preview() -> None:
    """Show the search queries the stored profile compiles to, with the cost estimate."""
    store = _profile_store()
    snapshot = store.read_snapshot()
    if snapshot is None:
        raise _fail("no profile, run `jobspy-coach init`")
    try:
        compiled = compile_profile(snapshot, scrape_interval_hours=get_settings().scrape_interval_hours)
    except ProfileCompileError as exc:
        raise _fail(f"profile cannot be compiled: {exc}") from None

    table = Table(title=f"Profile version {compiled.version}: search queries")
    table.add_column("Query id", style="cyan", no_wrap=True)
    table.add_column("Term")
    table.add_column("Location")
    table.add_column("Country")
    table.add_column("Boards")
    table.add_column("Remote")
    for q in compiled.queries:
        table.add_row(
            escape(q.query_id),
            escape(q.search_term),
            escape(q.location),
            escape(q.country_code or "-"),
            escape(", ".join(q.site_names)),
            "yes" if q.is_remote else "no",
        )
    console.print(table)
    estimate = compiled.estimate
    minutes = estimate.duration_seconds / 60
    console.print(
        f"Calls per cycle: {estimate.total_calls} (LinkedIn: {estimate.linkedin_calls}), "
        f"cap: {snapshot.fields.max_calls_per_cycle}, estimated duration: ~{minutes:.0f} min",
        markup=False,
    )
    console.print(f"Paused: {'yes' if compiled.paused else 'no'}", markup=False)
    for warning in compiled.warnings:
        console.print(f"Warning: {warning}", style="yellow", markup=False, highlight=False)


@profile_app.command("export")
def profile_export(
    path: Path = typer.Option(DEFAULT_EXPORT_PATH, "--path", help="Where to write the YAML export"),
    force: bool = typer.Option(False, "--force", help="Write even when the path is not git-ignored"),
) -> None:
    """Write the stored profile as YAML (private data: only to a git-ignored path)."""
    store = _profile_store()
    if not store.has_profile():
        raise _fail("no profile, run `jobspy-coach init`")
    try:
        written = export_profile(store, path, force=force)
    except ProfileError as exc:  # includes ExportRefused
        raise _fail(str(exc)) from None
    except OSError as exc:
        raise _fail(f"cannot write {path} ({type(exc).__name__}: {exc.strerror or 'error'})") from None
    console.print(f"Exported the profile to {written} (private data, file mode 0600).", markup=False)


@profile_app.command("import")
def profile_import(file: Path = typer.Argument(..., help="YAML file written by `profile export`")) -> None:
    """Validate a YAML profile file and apply it as a new profile version."""
    store = _profile_store()
    try:
        old = store.read_snapshot()
        result = apply_changes(store, [ReplaceProfile(load_profile_yaml(file))], label=f"import {file.name}")
    except ProfileError as exc:
        raise _fail_profile_error(exc, hint="Nothing was changed.") from None
    for line in result.diff:
        console.print(line, markup=False, highlight=False)
    try:  # the import is already applied: a backtest problem must not turn it into a failure
        new = store.read_snapshot()
        backtest = backtest_snapshots(Database(get_settings().db_path), old, new) if new is not None else None
    except ProfileError:
        backtest = None
    if backtest is not None:
        for line in render_backtest(backtest):
            console.print(line, markup=False, highlight=False)
    for warning in result.warnings:
        console.print(f"Warning: {warning}", style="yellow", markup=False, highlight=False)
    console.print(f"Profile is now at version {result.version}.", markup=False)


@app.command()
def requeue(
    thresholds: bool = typer.Option(
        False, "--thresholds", help="Re-bucket evaluated rows under the profile's thresholds (no LLM call)"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Report what would change without writing"),
) -> None:
    """Revive pre-filtered rows that the current rules pass, or (--thresholds) re-bucket evaluated rows.

    Never run while the daemon is running. Re-bucketed rows never send instant alerts.
    """
    settings = get_settings()
    db = Database(settings.db_path)
    db.init_db()
    snapshot = ProfileStore(db).read_snapshot()
    if snapshot is None:
        raise _fail("no profile, run `jobspy-coach init`")
    try:
        compiled = compile_profile(snapshot, scrape_interval_hours=settings.scrape_interval_hours)
    except ProfileCompileError as exc:
        raise _fail(f"profile cannot be compiled: {exc}") from None
    prefix = "dry run: " if dry_run else ""
    if thresholds:
        rebucket = rebucket_evaluated(db, compiled.scoring, dry_run=dry_run)
        console.print(
            f"{prefix}checked: {rebucket.checked}, changed: {rebucket.changed}, "
            f"skipped (missing stored fields): {rebucket.skipped_missing}, to digest: {rebucket.to_digest}, "
            f"to dismissed: {rebucket.to_dismissed}",
            markup=False,
            soft_wrap=True,
        )
        samples = rebucket.samples
    else:
        revived = requeue_prefiltered(db, compiled.rules, dry_run=dry_run)
        console.print(
            f"{prefix}checked: {revived.checked}, requeued: {revived.requeued}, "
            f"still dropped: {revived.still_dropped}, too old: {revived.too_old}",
            markup=False,
            soft_wrap=True,
        )
        samples = revived.samples
    for title in samples:
        console.print(f"  - {title}", markup=False, highlight=False, soft_wrap=True)


@app.command()
def scrape(
    dry_run: bool = typer.Option(False, "--dry-run", help="Run scraper without saving to database"),
):
    """Run a single one-shot scraping cycle."""
    console.print(f"[bold blue]Executing one-shot scrape (dry_run={dry_run})...[/bold blue]")
    daemon = DaemonService()
    try:
        new_jobs, wow_sent = daemon.run_pipeline_cycle(dry_run=dry_run)
    except ConfigurationError as e:
        raise _exit_on_missing_provider(e) from None
    console.print(f"[bold green]Done! New jobs: {new_jobs}, Instant WOW alerts: {wow_sent}[/bold green]")


@app.command()
def evaluate(
    limit: Optional[int] = typer.Option(
        None, "--limit", help="Max jobs to evaluate (default: EVAL_BATCH_LIMIT from settings)"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Evaluate without saving or notifying"),
):
    """Evaluate queued unprocessed vacancies independently from scraping."""
    if limit is None:
        limit = get_settings().eval_batch_limit
    console.print(f"[bold blue]Evaluating up to {limit} unprocessed vacancies (dry_run={dry_run})...[/bold blue]")
    daemon = DaemonService()
    try:
        evaluated, wow_sent = daemon.evaluate_unprocessed(limit=limit, dry_run=dry_run)
    except ConfigurationError as e:
        raise _exit_on_missing_provider(e) from None
    console.print(f"[bold green]Evaluation completed. Evaluated: {evaluated}, Instant WOW alerts: {wow_sent}[/bold green]")


@app.command()
def digest(
    dry_run: bool = typer.Option(False, "--dry-run", help="Render digest without sending"),
):
    """Dispatch the daily coaching vacancy digest."""
    console.print(f"[bold blue]Running daily digest (dry_run={dry_run})...[/bold blue]")
    daemon = DaemonService()
    count = daemon.run_daily_digest_cycle(dry_run=dry_run)
    console.print(f"[bold green]Digest completed. Processed {count} jobs.[/bold green]")


@app.command(name="test-telegram")
def test_telegram():
    """Send an authentication and connectivity test message to Telegram."""
    settings = get_settings()
    msg = "👋 <b>JobSpy Coach Daemon</b> | Підключення успішне! Тестове сповіщення доставлено."
    with TelegramNotifier(token=settings.telegram_bot_token, chat_id=settings.telegram_chat_id) as notifier:
        success = notifier.send_message(msg)
    if success:
        console.print("[bold green]Telegram test message sent successfully![/bold green]")
    else:
        console.print(
            "[bold red]Failed to send Telegram test message. Check TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.[/bold red]"
        )
        raise typer.Exit(code=1)


@app.command()
def rehash(
    dry_run: bool = typer.Option(False, "--dry-run", help="Report what would change without writing"),
):
    """Recompute dedup hashes with the current normalization and merge duplicate postings.

    Run once after upgrading (and never while the daemon is running). Keeps the row that already sent an
    alert, else an evaluated row, else the oldest; rows that already sent an alert are never deleted.
    """
    db = Database(get_settings().db_path)
    db.init_db()
    result = db.rehash_all(dry_run=dry_run)
    verb = "Would change" if dry_run else "Done"
    console.print(
        f"[bold green]{verb}: scanned {result['total']}, rehashed {result['rehashed']}, "
        f"merged (deleted) {result['merged']}, kept alert-sent twins {result['kept_twins']}.[/bold green]"
    )


@app.command("cleanup-empty")
def cleanup_empty(
    site: str = typer.Option("linkedin", "--site", help="Board whose empty-description rows are removed"),
):
    """Delete stub rows with an empty description (e.g. LinkedIn rows from a failed detail fetch).

    Their dedup hash would otherwise keep suppressing the real posting. Never run while the daemon is running.
    """
    db = Database(get_settings().db_path)
    db.init_db()
    removed = db.delete_empty_description_jobs(site)
    console.print(f"[bold green]Removed {removed} empty-description {site} row(s).[/bold green]")


@app.command()
def stats():
    """Display database statistics and lifecycle state counts."""
    settings = get_settings()
    db = Database(settings.db_path)
    db.init_db()
    digest_threshold = settings.digest_score_threshold  # .env only without a (compilable) profile
    try:
        snapshot = ProfileStore(db).read_snapshot()
        if snapshot is not None:
            digest_threshold = compile_profile(
                snapshot, scrape_interval_hours=settings.scrape_interval_hours
            ).scoring.digest_threshold
    except ProfileError:
        pass
    data = db.get_stats(pending_threshold=digest_threshold)

    table = Table(title="JobSpy Coach: Repository Statistics")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    for k, v in data.items():
        table.add_row(k.replace("_", " ").title(), str(v))

    try:
        version_info = check_jobspy_update()
        if version_info:
            version_str = (
                f"{version_info.installed_version} (Update: v{version_info.latest_version})"
                if version_info.has_update
                else f"{version_info.installed_version} (Latest)"
            )
        else:
            version_str = importlib.metadata.version("python-jobspy")
    except Exception:
        version_str = "Unknown"

    table.add_row("Jobspy Version", version_str)

    console.print(table)


if __name__ == "__main__":
    app()
