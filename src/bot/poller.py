"""Long-poll loop of the bot: one daemon thread, one update at a time, auth on every update.

Delivery is at-most-once: the update offset is persisted BEFORE anything of an update executes, so a crash or
restart can lose a command (the user resends) but never replay a non-idempotent one such as ``/undo``.
Nothing here logs message text, callback data, a token or a request URL; rejected updates are logged by id,
kind and chat type only.
"""
import logging
import threading
import time
from pathlib import Path
from typing import Callable, List, Protocol, Sequence, Tuple

from src.bot.auth import AuthDecision, Authorizer
from src.bot.client import (
    TelegramBotClient,
    TelegramConflict,
    TelegramRetryAfter,
    TelegramUnauthorized,
)
from src.bot.render import chunk_html
from src.bot.state import BotStateStore
from src.bot.updates import OutgoingMessage, Update, parse_update
from src.envfile import merge_env_file

logger = logging.getLogger(__name__)

LONG_POLL_SECONDS = 25
STALE_SECONDS = 900
MAX_BACKOFF_SECONDS = 60
CONFLICT_BACKOFF_SECONDS = 30
ERROR_REPLY = "Something went wrong; nothing was changed unless a diff was shown."


class UnreadableUpdate(Exception):
    """The last update of a batch had no readable id: back off instead of re-fetching it in a tight loop."""


class Router(Protocol):
    def is_mutating(self, update: Update) -> bool: ...

    def handle(self, update: Update) -> List[OutgoingMessage]: ...


class BotPoller:
    def __init__(
        self,
        client: TelegramBotClient,
        state: BotStateStore,
        authorizer: Authorizer,
        router: Router,
        *,
        stop_event: threading.Event,
        env_path: Path,
        on_paired: Callable[[int], None] = lambda chat_id: None,
        commands: Sequence[Tuple[str, str]] = (),
        clock: Callable[[], float] = time.time,
    ):
        self._client = client
        self._state = state
        self._authorizer = authorizer
        self._router = router
        self._stop = stop_event
        self._env_path = env_path
        self._on_paired = on_paired
        self._commands = list(commands)
        self._clock = clock

    # --- lifecycle -------------------------------------------------------------------------

    def start(self) -> threading.Thread:
        thread = threading.Thread(target=self.run_forever, name="bot-poller", daemon=True)
        thread.start()
        return thread

    def run_forever(self) -> None:
        if self._commands:  # an empty list would wipe commands registered elsewhere
            try:
                self._client.set_my_commands(self._commands)
            except Exception as exc:  # the menu is a convenience: never stop polling for it
                logger.warning("setMyCommands failed: %s", type(exc).__name__)
        failures = 0
        outage_logged = False
        conflict_logged = False
        while not self._stop.is_set():
            try:
                self.poll_once()
            except TelegramUnauthorized:
                logger.error("bot token rejected; bot polling stopped")
                return
            except TelegramRetryAfter as exc:
                self._stop.wait(exc.retry_after)
            except TelegramConflict:
                if not conflict_logged:
                    conflict_logged = True
                    logger.warning(
                        "another consumer is polling this bot (409); backing off %d s", CONFLICT_BACKOFF_SECONDS
                    )
                self._stop.wait(CONFLICT_BACKOFF_SECONDS)
            except Exception as exc:  # network errors, API errors, a database hiccup: back off and retry
                # One WARNING per outage; repeats stay at DEBUG so a long outage does not flood the log.
                logger.log(
                    logging.DEBUG if outage_logged else logging.WARNING,
                    "bot polling failed: %s",
                    type(exc).__name__,
                )
                outage_logged = True
                self._stop.wait(min(MAX_BACKOFF_SECONDS, 2 ** min(failures, 6)))
                failures += 1
            else:
                failures = 0
                outage_logged = False
                conflict_logged = False

    def poll_once(self) -> int:
        updates = self._client.get_updates(self._state.get_offset(), LONG_POLL_SECONDS)
        handled = 0
        unreadable_last = False
        for raw in updates:
            if self._stop.is_set():
                break  # not acknowledged: Telegram re-delivers it after the restart
            unreadable_last = parse_update(raw).update_id < 0
            self.process(raw)
            handled += 1
        if unreadable_last:
            # Nothing later in the batch moved the offset past it, so Telegram would hand it out again at once.
            raise UnreadableUpdate("update without a readable id")
        return handled

    # --- one update ------------------------------------------------------------------------

    def process(self, update_raw: dict) -> None:
        update = parse_update(update_raw)
        if update.update_id >= 0:
            self._state.set_offset(update.update_id + 1)  # before anything executes: at-most-once
        decision = self._authorizer.decide(update)
        if decision is AuthDecision.REJECT:
            logger.info("rejected update %d (%s, %s)", update.update_id, update.kind, update.chat_type)
            return
        if decision is AuthDecision.PAIRING_ATTEMPT:
            self._pair(update)
            return
        self._handle_accepted(update)

    def _pair(self, update: Update) -> None:
        outcome = self._authorizer.try_pair(update)
        if outcome.paired:
            try:
                merge_env_file(self._env_path, {"TELEGRAM_CHAT_ID": str(update.chat_id)})
            except OSError as exc:  # pairing stays; only the notifier's .env copy is missing
                logger.warning("could not write TELEGRAM_CHAT_ID to .env: %s", type(exc).__name__)
            self._on_paired(update.chat_id)
        if outcome.reply:
            self._send(update.chat_id, [OutgoingMessage(outcome.reply)])

    def _handle_accepted(self, update: Update) -> None:
        if update.kind == "callback" and update.callback_id:
            try:  # a stale button ("query is too old") must not drop the press
                self._client.answer_callback_query(update.callback_id)
            except Exception as exc:
                logger.debug("answerCallbackQuery failed: %s", type(exc).__name__)
        try:
            if (
                self._router.is_mutating(update)
                and update.date
                and self._clock() - update.date > STALE_SECONDS
            ):
                messages = [
                    OutgoingMessage(
                        f"Ignored {update.command}: it was sent more than 15 minutes ago. "
                        "Send it again if you still want it."
                    )
                ]
            else:
                messages = self._router.handle(update)
        except Exception as exc:  # never include the message: it can hold user text
            logger.error("handler failed on update %d: %s", update.update_id, type(exc).__name__)
            messages = [OutgoingMessage(ERROR_REPLY)]
        self._send(update.chat_id, messages)

    def _send(self, chat_id: int, messages: Sequence[OutgoingMessage]) -> None:
        for message in messages:
            pieces = chunk_html(message.text)
            if not pieces:
                continue
            last = len(pieces) - 1
            for index, piece in enumerate(pieces):
                markup = message.reply_markup if index == last else None
                if index == 0 and message.edit_message_id is not None:
                    self._client.edit_message_text(chat_id, message.edit_message_id, piece, markup)
                else:
                    self._client.send_message(chat_id, piece, markup)
