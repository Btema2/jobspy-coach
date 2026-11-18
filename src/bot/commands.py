"""Command router of the bot: read, change and safety commands, all through ``src.profile.ops``.

Every change applies immediately (validated, versioned, snapshotted) and is answered with the code-rendered
diff. Every user-supplied or stored term passes through ``esc``; long output is paginated and chunked, so a
reply never exceeds Telegram's limit. ``/setup``, ``/stats`` and ``/test`` are plugged in by the wizard, stats
and probe modules (see ``_register_features``) through ``register`` / ``register_callback`` /
``set_text_handler``.
"""
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.bot.client import TelegramBotClient
from src.bot.countries import resolve_country
from src.bot.render import PAGE_SIZE, chunk_html, esc, paginate, render_apply
from src.bot.updates import OutgoingMessage, Update
from src.config import Settings
from src.profile.backtest import backtest_snapshots, render_backtest
from src.profile.compiler import compile_profile
from src.profile.errors import (
    NothingToUndo,
    ProfileCompileError,
    ProfileValidationError,
    VersionConflict,
)
from src.profile.ops import (
    AddMarket,
    AddRole,
    AddRoleTerm,
    AddRule,
    ApplyResult,
    Operation,
    RemoveMarket,
    RemoveRole,
    SetField,
    apply_changes,
    history,
    restore,
    undo,
)
from src.profile.requeue import rebucket_evaluated, requeue_prefiltered
from src.profile.snapshot import Market, ProfileSnapshot, Role, RoleTerm, Rule
from src.profile.store import ProfileStore
from src.storage.db import Database

logger = logging.getLogger(__name__)

TRIGGER_STARTED = "started"
TRIGGER_BUSY = "running"
TRIGGER_NOT_READY = "not_ready"

MAX_ARGS_CHARS = 500
USER_RULE_REASON = "EXCLUDED_USER_TERM"
DROPPED_LIMIT = 20
NAME_PREVIEW_CHARS = 30
LISTED_ROLE_NAMES = 5
MAX_ERROR_LINES = 8
FIELD_CLIP = 100

# (name, English description) in the order of the spec's command table; also the setMyCommands menu.
COMMANDS: Tuple[Tuple[str, str], ...] = (
    ("help", "List the commands"),
    ("profile", "Show the profile summary"),
    ("roles", "List the roles and their synonyms"),
    ("markets", "List the markets"),
    ("exclusions", "List the exclusion rules"),
    ("preview", "Show the searches the profile runs"),
    ("dropped", "Show the last pre-filtered vacancies"),
    ("add_role", "Add a role: /add_role name = synonym, synonym"),
    ("remove_role", "Remove a role: /remove_role name"),
    ("add_market", "Add a market: /add_market country [remote] [remote_only]"),
    ("exclude", "Exclude a title term: /exclude term [prefix]"),
    ("threshold", "Set a threshold: /threshold wow|digest number"),
    ("pause", "Pause scraping"),
    ("resume", "Resume scraping and start a cycle"),
    ("run_now", "Start a scrape cycle now"),
    ("setup", "Guided setup: role, synonyms, market"),
    ("undo", "Undo the last change (repeat to toggle)"),
    ("history", "List the stored profile versions"),
    ("restore", "Restore a version: /restore number"),
    ("requeue", "Re-check dropped vacancies: /requeue [thresholds]"),
    ("stats", "Show 7-day search statistics per role"),
    ("test", "Try a role search: /test role"),
)

MUTATING_COMMANDS = frozenset({
    "/add_role", "/remove_role", "/add_market", "/exclude", "/threshold", "/pause", "/resume", "/run_now",
    "/undo", "/restore", "/requeue", "/setup", "/test",
})

_KIND_ORDER = ("block", "soft", "seniority", "rescue")  # RULE_KINDS is a set: keep the listing stable
_NO_PROFILE = "No profile yet. Send /setup or run jobspy-coach init."
_Handler = Callable[[Update], List[OutgoingMessage]]


@dataclass
class BotContext:
    db_path: Path
    env_path: Path
    settings: Settings
    trigger_cycle: Callable[[], str]
    is_cycle_running: Callable[[], bool]
    scraper: Any  # has scrape_board_outcome
    clock: Callable[[], float] = time.time


def _reply(text: str) -> List[OutgoingMessage]:
    return [OutgoingMessage(text)]


def _clip(text: str, limit: int = FIELD_CLIP) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _words(text: str) -> str:
    return " ".join(text.split())


def _page_arg(args: str) -> Optional[int]:
    """The page number of a list command (1 without argument); None when the argument is not a number."""
    if not args.strip():
        return 1
    token = args.strip()
    return int(token) if token.isascii() and token.isdigit() and len(token) < 9 else None


def _paged(command: str, lines: Sequence[str], page: int, tail: Sequence[str] = ()) -> str:
    shown, page, pages = paginate(lines, page)
    out = list(shown)
    while out and not out[-1]:  # padding that keeps a group header off the end of a page
        out.pop()
    out.extend(tail if page == pages else ())
    if pages > 1:
        footer = f"page {page}/{pages}"
        out.append(footer + (f", send {command} {page + 1}" if page < pages else ""))
    return "\n".join(out)


def _not_applied(messages: Sequence[str]) -> str:
    shown = [esc(m) for m in messages[:MAX_ERROR_LINES]]
    lines = ["Not applied: " + shown[0]] + ["  " + m for m in shown[1:]]
    if len(messages) > MAX_ERROR_LINES:
        lines.append(f"  ... and {len(messages) - MAX_ERROR_LINES} more")
    return "\n".join(lines)


class CommandRouter:
    def __init__(self, ctx: BotContext, *, client: TelegramBotClient):
        self._ctx = ctx
        self.client = client
        self._db = Database(ctx.db_path)
        self._store = ProfileStore(self._db)
        self._handlers: Dict[str, _Handler] = {}
        self._callbacks: List[Tuple[str, _Handler]] = []
        self._text_handler: Optional[Callable[[Update], Optional[List[OutgoingMessage]]]] = None
        for name, handler in (
            ("help", self._help), ("start", self._help), ("profile", self._profile), ("roles", self._roles),
            ("markets", self._markets), ("exclusions", self._exclusions), ("preview", self._preview),
            ("dropped", self._dropped), ("add_role", self._add_role), ("remove_role", self._remove_role),
            ("add_market", self._add_market), ("exclude", self._exclude), ("threshold", self._threshold),
            ("pause", self._pause), ("resume", self._resume), ("run_now", self._run_now),
            ("undo", self._undo), ("history", self._history), ("restore", self._restore),
            ("requeue", self._requeue),
        ):
            self.register(name, handler)
        self._register_features()

    def _register_features(self) -> None:
        """Plug in /setup, /stats and /test (their modules import this one, hence the local imports)."""
        from src.bot.probe import ProbeRunner
        from src.bot.state import BotStateStore
        from src.bot.stats import stats_command
        from src.bot.wizard import SetupWizard

        wizard = SetupWizard(BotStateStore(self._db), self._store, clock=self._ctx.clock)
        self.register("setup", wizard.start)
        self.register_callback("wz:", wizard.on_callback)
        self.set_text_handler(wizard.on_text)
        self.register("stats", lambda update: stats_command(self._db, self._store, self._ctx.settings, update.args))
        self.register("test", ProbeRunner(self._ctx, self.client, clock=self._ctx.clock).request)

    # --- registration -------------------------------------------------------------------

    def register(self, command: str, handler: _Handler) -> None:
        self._handlers["/" + command.lstrip("/").lower()] = handler

    def register_callback(self, prefix: str, handler: _Handler) -> None:
        self._callbacks.append((prefix, handler))

    def set_text_handler(self, handler: Callable[[Update], Optional[List[OutgoingMessage]]]) -> None:
        self._text_handler = handler

    # --- dispatch -----------------------------------------------------------------------

    def is_mutating(self, update: Update) -> bool:
        return update.kind == "message" and update.command in MUTATING_COMMANDS

    def handle(self, update: Update) -> List[OutgoingMessage]:
        try:
            messages = self._dispatch(update)
        except ProfileValidationError as exc:
            messages = _reply(_not_applied(exc.messages))
        except VersionConflict:
            messages = _reply("The profile changed at the same time; send the command again.")
        except ProfileCompileError as exc:
            messages = _reply("The stored profile cannot be compiled: " + esc(str(exc)))
        return self._chunked(messages)

    @staticmethod
    def _chunked(messages: Sequence[OutgoingMessage]) -> List[OutgoingMessage]:
        out: List[OutgoingMessage] = []
        for message in messages:
            pieces = chunk_html(message.text)
            if len(pieces) <= 1:
                out.append(message)
                continue
            for index, piece in enumerate(pieces):
                out.append(OutgoingMessage(
                    piece,
                    message.reply_markup if index == len(pieces) - 1 else None,
                    message.edit_message_id if index == 0 else None,
                ))
        return out

    def _dispatch(self, update: Update) -> List[OutgoingMessage]:
        if update.kind == "callback":
            for prefix, handler in self._callbacks:
                if update.text.startswith(prefix):
                    return handler(update)
            return _reply("This button is no longer active.")
        if update.kind != "message":
            return []
        if update.command is not None:
            if len(update.args) > MAX_ARGS_CHARS:
                return _reply("That is too long.")
            handler = self._handlers.get(update.command)
            if handler is None:
                return _reply("Unknown command. Send /help.")
            return handler(update)
        if self._text_handler is not None:
            result = self._text_handler(update)
            if result is not None:
                return result
        return _reply("Send /help to see what I can do.")

    # --- helpers ------------------------------------------------------------------------

    def _snapshot(self) -> Optional[ProfileSnapshot]:
        return self._store.read_snapshot()

    def _compiled(self, snapshot: ProfileSnapshot):
        return compile_profile(snapshot, scrape_interval_hours=self._ctx.settings.scrape_interval_hours)

    def _apply(self, ops: Sequence[Operation], label: str) -> ApplyResult:
        return apply_changes(self._store, list(ops), label=_clip(label, 80))

    def _applied(self, result: ApplyResult, extra: Sequence[str] = ()) -> List[OutgoingMessage]:
        return _reply(render_apply(result, extra))

    # --- read commands ------------------------------------------------------------------

    def _help(self, update: Update) -> List[OutgoingMessage]:
        return _reply("\n".join(f"/{name} - {esc(description)}" for name, description in COMMANDS))

    def _profile(self, update: Update) -> List[OutgoingMessage]:
        snapshot = self._snapshot()
        if snapshot is None:
            return _reply(_NO_PROFILE)
        compiled = self._compiled(snapshot)
        f = snapshot.fields
        user_rules = sum(1 for r in snapshot.rules if r.source != "seed")
        names = [esc(_clip(r.name, NAME_PREVIEW_CHARS)) for r in snapshot.roles[:LISTED_ROLE_NAMES]]
        more = len(snapshot.roles) - LISTED_ROLE_NAMES
        role_line = f"Roles: {len(snapshot.roles)}"
        if names:
            role_line += " (" + ", ".join(names) + (f", +{more} more" if more > 0 else "") + ")"
        lines = [
            f"<b>Profile v{snapshot.version}</b> ({'paused' if f.paused else 'running'})",
            f"Thresholds: WOW {f.wow_threshold:g}, digest {f.digest_threshold:g}",
            f"Output language: {esc(f.output_language or '(unset)')}",
            f"Target level: {esc(f.target_level or '(unset)')}",
            f"{role_line}, markets: {len(snapshot.markets)}, pinned queries: {len(snapshot.pinned_queries)}, "
            f"user rules: {user_rules}",
            f"Persona: {len(f.persona)} characters",
            f"Calls per cycle: {compiled.estimate.total_calls} of {f.max_calls_per_cycle} max",
        ]
        return _reply("\n".join(lines))

    def _roles(self, update: Update) -> List[OutgoingMessage]:
        page = _page_arg(update.args)
        if page is None:
            return _reply("Usage: /roles [page]")
        snapshot = self._snapshot()
        if snapshot is None:
            return _reply(_NO_PROFILE)
        if not snapshot.roles:
            return _reply("No roles yet. Add one with /add_role.")
        return _reply(_paged("/roles", [self._role_line(r) for r in snapshot.roles], page))

    @staticmethod
    def _role_line(role: Role) -> str:
        terms = ", ".join(esc(t.term) + (f" [{esc(t.language)}]" if t.language else "") for t in role.terms)
        state = "enabled" if role.enabled else "disabled"
        return f"{esc(role.name)} [{state}]: {terms or '(no synonyms)'}"

    def _markets(self, update: Update) -> List[OutgoingMessage]:
        page = _page_arg(update.args)
        if page is None:
            return _reply("Usage: /markets [page]")
        snapshot = self._snapshot()
        if snapshot is None:
            return _reply(_NO_PROFILE)
        if not snapshot.markets:
            return _reply("No markets yet. Add one with /add_market.")
        lines = []
        for m in snapshot.markets:
            remote = "remote only" if m.remote_only else "remote" if m.remote else "not remote"
            lines.append(
                f"{esc(m.name)}: {esc(m.location)}; "
                f"{esc(', '.join(m.countries)) if m.countries else 'no country'}; {remote}; "
                f"{esc(', '.join(m.languages)) if m.languages else 'no languages'}; "
                f"{'enabled' if m.enabled else 'disabled'}"
            )
        return _reply(_paged("/markets", lines, page))

    def _exclusions(self, update: Update) -> List[OutgoingMessage]:
        page = _page_arg(update.args)
        if page is None:
            return _reply("Usage: /exclusions [page]")
        snapshot = self._snapshot()
        if snapshot is None:
            return _reply(_NO_PROFILE)
        lines: List[str] = []
        kinds = list(_KIND_ORDER) + sorted({r.kind for r in snapshot.rules} - set(_KIND_ORDER))
        for kind in kinds:
            rules = [r for r in snapshot.rules if r.kind == kind and r.source != "seed"]
            if rules:
                if len(lines) % PAGE_SIZE == PAGE_SIZE - 1:
                    lines.append("")  # a header alone at the end of a page moves to the next one
                lines.append(f"<b>{esc(kind)}:</b>")
                lines.extend(f"  {esc(r.term)} ({esc(r.match_mode)})" for r in rules)
        seeds = sum(1 for r in snapshot.rules if r.source == "seed")
        tail = [f"{seeds} built-in legacy patterns"] if seeds else []
        if not lines and not tail:
            return _reply("No exclusion rules. Add one with /exclude.")
        return _reply(_paged("/exclusions", lines, page, tail))

    def _preview(self, update: Update) -> List[OutgoingMessage]:
        page = _page_arg(update.args)
        if page is None:
            return _reply("Usage: /preview [page]")
        snapshot = self._snapshot()
        if snapshot is None:
            return _reply(_NO_PROFILE)
        compiled = self._compiled(snapshot)
        lines = [f"{esc(q.query_id)}: {esc(', '.join(q.site_names))}" for q in compiled.queries]
        estimate = compiled.estimate
        tail = [
            f"Calls per cycle: {estimate.total_calls} (LinkedIn calls: {estimate.linkedin_calls}), "
            f"about {estimate.duration_seconds / 60:.0f} min, cap {snapshot.fields.max_calls_per_cycle}"
        ]
        tail.extend("Warning: " + esc(w) for w in compiled.warnings)
        if not lines:
            return _reply("No searches yet: add a role and a market.\n" + "\n".join(tail))
        return _reply(_paged("/preview", lines, page, tail))

    def _dropped(self, update: Update) -> List[OutgoingMessage]:
        jobs = self._db.get_recent_prefiltered(DROPPED_LIMIT)
        if not jobs:
            return _reply("Nothing has been pre-filtered yet.")
        return _reply("\n".join(
            f"{esc(_clip(j.title))} - {esc(_clip(j.company, 60))} ({esc(j.site)}): {esc(_clip(j.reason or '-', 160))}"
            for j in jobs
        ))

    # --- change commands ----------------------------------------------------------------

    def _add_role(self, update: Update) -> List[OutgoingMessage]:
        name_part, _, synonym_part = update.args.partition("=")
        name = _words(name_part)
        if not name:
            return _reply("Usage: /add_role name [= synonym, synonym]")
        synonyms: List[str] = []
        for raw in synonym_part.split(","):
            synonym = _words(raw)
            if synonym and synonym.casefold() not in {s.casefold() for s in synonyms}:
                synonyms.append(synonym)
        snapshot = self._snapshot()
        existing = next(
            (r for r in (snapshot.roles if snapshot else ()) if r.name.strip().casefold() == name.casefold()),
            None,
        )
        if existing is None:
            ops: List[Operation] = [AddRole(name, tuple(RoleTerm(s, None) for s in synonyms))]
        else:
            present = {existing.name.casefold()} | {t.term.strip().casefold() for t in existing.terms}
            new_terms = [s for s in synonyms if s.casefold() not in present]
            if not new_terms:
                return _reply(f"Role {esc(existing.name)} already exists.")
            ops = [AddRoleTerm(existing.name, s) for s in new_terms]
        return self._applied(self._apply(ops, f"add role {name}"))

    def _remove_role(self, update: Update) -> List[OutgoingMessage]:
        name = _words(update.args)
        if not name:
            return _reply("Usage: /remove_role name")
        return self._applied(self._apply([RemoveRole(name)], f"remove role {name}"))

    def _add_market(self, update: Update) -> List[OutgoingMessage]:
        tokens = update.args.split()
        remote = remote_only = False
        while tokens and tokens[-1].lower() in ("remote", "remote_only"):
            if tokens.pop().lower() == "remote":
                remote = True
            else:
                remote_only = True
        text = " ".join(tokens)
        if not text:
            return _reply("Usage: /add_market country [remote] [remote_only]")
        resolution = resolve_country(text)
        if resolution.match is None:
            if resolution.options:
                return _reply("Which one did you mean? " + ", ".join(esc(o.name) for o in resolution.options))
            return _reply(f"Unknown country '{esc(_clip(text, 60))}'. Try the English name, e.g. Germany.")
        match = resolution.match
        name = match.name + (" (remote only)" if remote_only else "")
        market = Market(
            name=name,
            location=match.name,
            countries=(match.country,) if match.country else (),
            remote=remote or remote_only,
            remote_only=remote_only,
            languages=match.languages,
        )
        snapshot = self._snapshot()
        ops: List[Operation] = []
        if snapshot is not None and any(m.name.strip().casefold() == name.casefold() for m in snapshot.markets):
            ops.append(RemoveMarket(name))
        ops.append(AddMarket(market))
        return self._applied(self._apply(ops, f"add market {name}"))

    def _exclude(self, update: Update) -> List[OutgoingMessage]:
        tokens = update.args.split()
        mode = "exact"
        if len(tokens) > 1 and tokens[-1].lower() == "prefix":
            tokens.pop()
            mode = "prefix"
        term = " ".join(tokens)
        if not term:
            return _reply("Usage: /exclude term [prefix]")
        result = self._apply(
            [AddRule(Rule("block", term, mode, USER_RULE_REASON, "user"))], f"exclude {term}"
        )
        extra: List[str] = []
        if result.new is not None:
            backtest = backtest_snapshots(self._db, result.old, result.new)
            if backtest is not None:
                extra = render_backtest(backtest)
        return self._applied(result, extra)

    def _threshold(self, update: Update) -> List[OutgoingMessage]:
        usage = _reply("Usage: /threshold wow|digest number")
        parts = update.args.split()
        if len(parts) != 2 or parts[0].lower() not in ("wow", "digest"):
            return usage
        try:
            value = float(parts[1].replace(",", "."))
        except ValueError:
            return usage
        which = parts[0].lower()
        return self._applied(self._apply([SetField(f"{which}_threshold", value)], f"{which} threshold {value:g}"))

    def _pause(self, update: Update) -> List[OutgoingMessage]:
        snapshot = self._snapshot()
        if snapshot is not None and snapshot.fields.paused:
            return _reply("Scraping is already paused.")
        return self._applied(self._apply([SetField("paused", True)], "pause"))

    def _resume(self, update: Update) -> List[OutgoingMessage]:
        snapshot = self._snapshot()
        if snapshot is not None and not snapshot.fields.paused:
            return _reply("Scraping is not paused. Send /run_now to start a cycle.")
        result = self._apply([SetField("paused", False)], "resume")
        try:
            outcome = self._ctx.trigger_cycle()
        except Exception as exc:  # the profile is already changed: the reply must still be sent
            logger.warning("trigger_cycle failed: %s", type(exc).__name__)
            outcome = TRIGGER_NOT_READY
        if outcome == TRIGGER_STARTED:
            note = "A cycle starts now."
        elif outcome == TRIGGER_BUSY:
            note = "A cycle is already running; the next one will scrape."
        else:
            note = "Scraping resumes at the next scheduled cycle."
        return self._applied(result, [note])

    def _run_now(self, update: Update) -> List[OutgoingMessage]:
        outcome = self._ctx.trigger_cycle()
        if outcome == TRIGGER_STARTED:
            return _reply("Cycle started.")
        if outcome == TRIGGER_BUSY:
            return _reply("A cycle is already running; try again when it has finished.")
        return _reply("The scheduler is not ready yet.")

    # --- safety commands ----------------------------------------------------------------

    def _undo(self, update: Update) -> List[OutgoingMessage]:
        try:
            result = undo(self._store)
        except NothingToUndo as exc:
            return _reply("Nothing to undo: " + esc(str(exc)))
        return self._applied(result)

    def _history(self, update: Update) -> List[OutgoingMessage]:
        versions = history(self._store)
        if not versions:
            return _reply("No versions stored yet.")
        return _reply("\n".join(
            f"v{v.version}  {self._stamp(v.created_at)}  {esc(_clip(v.label, 80))}" for v in versions
        ))

    @staticmethod
    def _stamp(created_at: str) -> str:
        try:
            return datetime.fromisoformat(created_at).strftime("%Y-%m-%d %H:%M")
        except ValueError:
            return esc(created_at[:16])

    def _restore(self, update: Update) -> List[OutgoingMessage]:
        token = update.args.strip()
        if not (token.isascii() and token.isdigit()) or len(token) > 9:
            return _reply("Usage: /restore number (see /history)")
        return self._applied(restore(self._store, int(token)))

    def _requeue(self, update: Update) -> List[OutgoingMessage]:
        argument = update.args.strip().lower()
        if argument not in ("", "thresholds"):
            return _reply("Usage: /requeue [thresholds]")
        if self._ctx.is_cycle_running():
            return _reply("A cycle is running; send /requeue when it has finished.")
        snapshot = self._snapshot()
        if snapshot is None:
            return _reply(_NO_PROFILE)
        compiled = self._compiled(snapshot)
        if argument == "thresholds":
            rebucket = rebucket_evaluated(self._db, compiled.scoring)
            head = (
                f"checked: {rebucket.checked}, changed: {rebucket.changed}, "
                f"skipped (missing stored fields): {rebucket.skipped_missing}, to digest: {rebucket.to_digest}, "
                f"to dismissed: {rebucket.to_dismissed}, too old: {rebucket.too_old}"
            )
            samples = rebucket.samples
        else:
            revived = requeue_prefiltered(self._db, compiled.rules)
            head = (
                f"checked: {revived.checked}, requeued: {revived.requeued}, "
                f"still dropped: {revived.still_dropped}, too old: {revived.too_old}"
            )
            samples = revived.samples
        lines = [esc(head)] + [f"  - {esc(_clip(title))}" for title in samples]
        return _reply("\n".join(lines))
