import importlib.metadata
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table

from src.config import get_settings
from src.evaluator.errors import ConfigurationError
from src.logging_config import configure_logging
from src.notifier.telegram_notifier import TelegramNotifier
from src.scheduler.daemon import DaemonService
from src.scraper.version_checker import check_jobspy_update
from src.storage.db import Database

app = typer.Typer(help="JobSpy Coach: Autonomous Coaching Job Monitor Daemon")
console = Console()

configure_logging()


def _exit_on_missing_provider(error: ConfigurationError) -> "typer.Exit":
    console.print(f"[bold red]Configuration error:[/bold red] {error}")
    return typer.Exit(code=1)


@app.command()
def run():
    """Start the 24/7 background daemon scheduler."""
    console.print("[bold green]Starting JobSpy Coach Daemon...[/bold green]")
    daemon = DaemonService()
    try:
        daemon.start()
    except ConfigurationError as e:
        raise _exit_on_missing_provider(e) from None


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
    data = db.get_stats(pending_threshold=settings.digest_score_threshold)

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
