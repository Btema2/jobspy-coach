"""Free text in the paired chat: debounce, consent, clarification, one pending proposal, confirm / cancel.

Threads (AGENTS.md 3.C.4, amended by Branch 5): the BotPoller thread buffers text parts (memory only), takes
the consent button, handles ``/ok``, ``/cancel`` and the ``pc:`` buttons, and is the only thread that writes
profile tables (``take_pending`` + ``apply_changes``) or the ``llm_consent`` table. One ``bot-conversation``
worker thread flushes the debounce buffer, downloads uploads, masks, gates on consent, calls the translator,
and writes only ``pending_change`` / ``conversation`` rows; it has its own ``Database`` instance. A
``threading.Timer`` only sends the progress note. Memory state is guarded by one ``threading.Condition``.

Retention: raw text lives in memory and, while a clarifying question is open, in ``conversation.context``
(masked, at most 24 hours); it is purged on apply, cancel, supersede and expiry. ``pending_change`` holds the
validated operation list only. An interview (``/setup``, state ``awaiting_interview``) exists only after
consent; its row is created by the worker (``begin``) and cleared on the poller only together with
``_drop_memory()`` and a nonce check; the poller learns "in an interview" from the ``_interview`` mirror.
Its 24 hours count from the first stored answer (``context.created_at``; ``updated_at`` while that is null).
Nothing here logs user text, masked text, questions or replies: only lengths,
request ids and exception type names.
"""
import logging
import secrets
import threading
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.bot.chat_llm import ChatLLMError, ChatLLMLimited, ChatLLMNotConfigured, describe_failure
from src.bot.client import TelegramApiError, TelegramBotClient
from src.bot.commands import BotContext, not_applied, trigger_note
from src.bot.conversation_store import ConversationState, ConversationStore, PendingChange
from src.bot.interview import InterviewLLM, InterviewState, new_state, next_step
from src.bot.privacy import CONSENT_NOTICE, CONSENT_VERSION, mask_personal_data
from src.bot.render import chunk_html, esc, render_apply
from src.bot.translator import (
    MAX_INPUT_CHARS,
    MODES,
    Clarification,
    Proposal,
    TranslationProblem,
    Translator,
    operations_from_json,
    render_proposal,
    to_operations,
)
from src.bot.state import BotStateStore
from src.bot.updates import DocumentRef, OutgoingMessage, Update
from src.bot.wizard import STALE as WIZARD_STALE
from src.bot.wizard import SetupWizard
from src.evaluator.errors import StopRequested
from src.profile.backtest import backtest_snapshots, render_backtest
from src.profile.compiler import compile_profile
from src.profile.errors import ProfileCompileError, ProfileValidationError, VersionConflict
from src.profile.ops import apply_changes
from src.profile.store import ProfileStore
from src.storage.db import Database

logger = logging.getLogger(__name__)

DEBOUNCE_SECONDS = 3.0
MAX_PARTS = 3
MAX_MERGED_CHARS = 12000
MAX_UPLOAD_BYTES = 20480
PENDING_TTL_SECONDS = 86400
CONTEXT_TTL_SECONDS = 86400
SWEEP_SECONDS = 60.0
PROGRESS_NOTE_SECONDS = 20.0
MAX_CLARIFICATIONS = 2
MAX_QUEUED_REQUESTS = 5  # requests waiting behind the one in flight (an upload or a debounced paste each)
BARE_CONFIRM_MAX_AGE_SECONDS = 900
_IDLE_WAIT_SECONDS = 0.5  # the longest the worker sleeps: it must notice the stop event within a second
_UPLOAD_SUFFIXES = (".txt", ".md")
_BOM = b"\xef\xbb\xbf"

BARE_CONFIRM = frozenset({"ok", "okay", "yes", "y", "yep", "sure", "confirm", "apply"})
BARE_CANCEL = frozenset({"no", "n", "nope", "cancel"})

TOO_LONG_REPLY = "That is too long; send it as a .txt or .md file (up to 20 KB)."
ONLY_TEXT_FILES_REPLY = "Only .txt and .md files are read."
FILE_TOO_LARGE_REPLY = "That file is larger than 20 KB; send a shorter one."
NOT_UTF8_REPLY = "The file must be UTF-8 text."
EMPTY_FILE_REPLY = "The file is empty."
DOWNLOAD_FAILED_REPLY = "I could not download that file; send it again or paste the text."
SWALLOWED_REPLY = "Ignored: still dropping your too-long paste; resend in a few seconds."
FINISH_SETUP_REPLY = "Finish or cancel the /setup step first."
BUSY_REPLY = "Still working on your earlier requests; send this again in a moment."
NOT_CONFIGURED_REPLY = (
    "Free-text requests need an LLM key (NVIDIA_API_KEY or OPENROUTER_API_KEY in .env); "
    "the commands keep working: /help"
)
NOT_UNDERSTOOD_REPLY = "Couldn't understand that, try a command (/help)."
ERROR_REPLY = "Something went wrong; nothing was changed."
WORKING_REPLY = "Working on it..."
PROGRESS_REPLY = "Still working; this can take a few minutes."
RETRY_BUTTON = "Retry"
DELIVERY_FAILED_REPLY = "Could not deliver the proposal; send your request again."
NO_CHANGE_REPLY = "That does not look like a profile change. Send /help for the commands."
QUESTION_PREFIX = "Question: "
NO_LONGER_ACTIVE_REPLY = "This proposal is no longer active."
PROPOSAL_UNREADABLE_REPLY = "That proposal could not be read; send your request again."
EXPIRED_REPLY = "Your proposal expired after 24 hours and was discarded; send your request again."
CONFLICT_REPLY = "The profile changed since this proposal; send your request again."
SUPERSEDED_REPLY = "Your previous proposal was replaced by this one."
NO_PENDING_REPLY = "There is no pending proposal. Send /help for the commands."
BARE_STALE_REPLY = 'That "ok" is more than 15 minutes old; send /ok to apply.'
CANCELLED_REPLY = "Cancelled. Nothing was changed."
NOTHING_TO_CANCEL_REPLY = "There is nothing to cancel."
CONSENT_DECLINED_REPLY = "Nothing was sent. Use the commands (/help) or /setup guided."
CONSENT_STALE_REPLY = "This choice is no longer active; send your request again."
COMPILE_ERROR_REPLY = "The stored profile cannot be compiled; nothing was changed."
PAUSED_NOTE = "The profile is paused; send /resume to start searching."
SETUP_NO_LLM_REPLY = "No LLM is configured; using step-by-step questions."
PROPOSAL_DISCARDED_REPLY = "Your pending proposal was discarded."
CLARIFICATION_DISCARDED_REPLY = "Your open question was dropped."
DRAFT_BUTTON = "Draft now"
GUIDED_BUTTON = "Step-by-step questions instead"
CANCEL_BUTTON = "Cancel"
INTERVIEW_STALE_REPLY = "This choice is no longer active; send your answer again."
CONSENT_DECLINED_SETUP_REPLY = "Nothing was sent. Continuing with step-by-step questions."
INTERVIEW_INTRO = (
    "Let's set up your search. I will ask 6 to 8 short questions; it takes a few minutes. "
    "You can send a CV (PDF), a .txt / .md file or pasted text at any point: one answer can cover several "
    "questions. Your answers are kept at most 24 hours. Nothing changes until you confirm the draft I show you.\n"
    "Draft now: draft the profile from what you have said so far. "
    "Step-by-step questions instead: switch to fixed questions without the language model. "
    "Cancel: stop the interview."
)

STATE_IDLE = "idle"
STATE_CLARIFICATION = "awaiting_clarification"
STATE_CONFIRM = "awaiting_confirm"
STATE_INTERVIEW = "awaiting_interview"
_INTERVIEW_ACTIONS = ("draft", "cancel", "retry", "guided")


def _reply(text: str) -> List[OutgoingMessage]:
    return [OutgoingMessage(text)]


def _consent_markup(nonce: str) -> Dict[str, Any]:
    return {
        "inline_keyboard": [[
            {"text": "Yes, continue", "callback_data": f"cs:{nonce}:yes"},
            {"text": "No", "callback_data": f"cs:{nonce}:no"},
        ]]
    }


def _bare_word(text: str) -> Optional[str]:
    """``confirm`` or ``cancel`` when ``text`` is just such a word (case and trailing punctuation ignored)."""
    word = text.strip().lower().rstrip(".!?,;: ").strip()
    if word in BARE_CONFIRM:
        return "confirm"
    if word in BARE_CANCEL:
        return "cancel"
    return None


@dataclass
class _Buffer:
    """One request while it waits for its debounce window to close (or an upload / released consent text)."""

    chat_id: int
    parts: List[str]
    chars: int
    last: float
    closed: bool = False  # no more parts are merged into it
    immediate: bool = False  # due at once (uploads, text released by the consent button)
    doc: Optional[DocumentRef] = None
    caption: str = ""
    # Interview inputs: ``interview`` marks them (a free-text request has False); ``action`` is a worker action
    # (``begin``, ``draft``, ``retry``; None for an answer), ``mode`` the interview mode for ``begin``, ``nonce``
    # the interview the input belongs to and ``slot`` the question current when it arrived (None: unknown,
    # e.g. after a restart).
    interview: bool = False
    action: Optional[str] = None
    mode: Optional[str] = None
    nonce: Optional[str] = None
    slot: Optional[str] = None
    ask_first: bool = True  # ``begin``: post the first question (False when started from a CV)

    def due(self, now: float) -> bool:
        return self.immediate or now - self.last >= DEBOUNCE_SECONDS

    def due_at(self) -> float:
        return float("-inf") if self.immediate else self.last + DEBOUNCE_SECONDS


@dataclass
class _Held:
    """Masked request text waiting in memory (never in the database) for the consent answer."""

    nonce: str
    chat_id: int
    parts: List[str]
    created: float
    message_id: Optional[int] = None  # the consent notice, so its buttons can be removed when it is replaced
    setup: bool = False  # ``/setup`` waits for consent: ``yes`` starts the interview, ``no`` the wizard


@dataclass
class _Retry:
    """Masked request text kept in memory (never the database) so the Retry button can run it again."""

    nonce: str
    chat_id: int
    parts: List[str]
    created: float
    message_id: Optional[int] = None  # the failure reply carrying the button, so it can be removed when replaced


@dataclass
class _Work:
    buffer: _Buffer
    generation: int  # a /cancel after the request was taken makes its result stale
    done: bool = False  # the model call ended: a progress note that has not been sent yet must stay unsent


class ConversationService:
    def __init__(
        self,
        ctx: BotContext,
        client: TelegramBotClient,
        wizard: SetupWizard,
        *,
        translator: Optional[Translator] = None,
        interview_llm: Optional[InterviewLLM] = None,
        clock: Callable[[], float] = time.time,
    ):
        self._ctx = ctx
        self._client = client
        self._wizard = wizard
        self._clock = clock
        # The poller thread's instances ...
        self._db = Database(ctx.db_path)
        self._conv = ConversationStore(self._db)
        self._bot_state = BotStateStore(self._db)
        self._profiles = ProfileStore(self._db)
        # ... and the worker's own (pending_change / conversation writes only).
        self._wdb = Database(ctx.db_path)
        self._wconv = ConversationStore(self._wdb)
        self._wprofiles = ProfileStore(self._wdb)
        if translator is None and ctx.chat_llm is not None:
            translator = Translator(ctx.chat_llm)
        self._translator = translator
        if interview_llm is None and ctx.chat_llm is not None and translator is not None:
            interview_llm = InterviewLLM(ctx.chat_llm, translator)
        self._interview_llm = interview_llm
        self._stop = ctx.stop_event if ctx.stop_event is not None else threading.Event()
        self._note_lock = threading.Lock()  # a progress note is sent (or skipped) as a whole; see _progress_note
        self._cond = threading.Condition()  # memory state only: nothing slow (database, network) runs under it
        self._state_lock = threading.RLock()  # pending row + conversation state transitions (poller and worker)
        self._context_chats: set = set()
        self._buffers: List[_Buffer] = []
        # chat id, time of the last part of a dropped paste, and whether the user was told about it
        self._swallow: Optional[List[Any]] = None
        self._held: Optional[_Held] = None
        self._retry: Optional[_Retry] = None
        # chat id -> (interview nonce, current slot): set by the worker before it posts each question, removed
        # when the interview ends; the poller reads only this to know "in an interview" (guarded by ``_cond``).
        self._interview: Dict[int, Tuple[str, Optional[str]]] = {}
        self._generation = 0
        self._inflight = False
        self._next_sweep = 0.0

    # --- poller thread: entry points --------------------------------------------------------

    def on_text(self, update: Update) -> Optional[List[OutgoingMessage]]:
        """The router's text handler: wizard first, then bare words, then the language model."""
        if update.document is not None:
            # An upload never answers a wizard step: its caption is not a step answer.
            wizard_reply = self._wizard.on_text(replace(update, text="", document=None))
            if wizard_reply is not None:
                return wizard_reply if wizard_reply[0].text == WIZARD_STALE else _reply(FINISH_SETUP_REPLY)
        else:
            wizard_reply = self._wizard.on_text(update)
            if wizard_reply is not None:
                return wizard_reply
        if update.chat_id is None or (update.document is None and not update.text.strip()):
            return None
        tag = self._interview_tag(update.chat_id)
        if tag is not None:  # an interview input: never the bare-word router, never the free-text translator
            return self._add_interview_input(update, tag)
        if update.document is None:
            kind = _bare_word(update.text)
            if kind is not None:
                routed = self._route_bare(update, kind)
                if routed is not None:
                    return routed
        if self._translator is None:
            return None  # no language model behind this router: the router's own hint answers
        llm = self._ctx.chat_llm
        if llm is not None and not llm.available():
            return _reply(NOT_CONFIGURED_REPLY)
        if update.document is not None:
            return self._add_upload(update)
        return self._add_part(update.chat_id, update.text, self._clock())

    def start_setup(self, update: Update) -> List[OutgoingMessage]:
        """``/setup``: the interview when a language model is configured, else the wizard.

        ``/setup guided`` always runs the wizard. Either way a pending proposal and an open question are
        discarded (with notices), a request still in flight loses its result and any earlier wizard or interview
        is replaced. Consent comes first: without it only the notice is shown and nothing is stored (the hold
        lives in memory); with it the worker starts the interview (``begin``).
        """
        chat_id = update.chat_id
        guided = update.args.strip().lower() == "guided"
        llm = self._ctx.chat_llm
        ready = self._translator is not None and (llm is None or llm.available())
        notes = self._reset_for_setup(chat_id) if chat_id is not None else []
        if guided or not ready or chat_id is None:
            extra = [] if guided or chat_id is None else _reply(SETUP_NO_LLM_REPLY)
            return notes + extra + self._wizard.start(update)
        self._bot_state.set_wizard(None)
        if not self._conv.has_consent(chat_id, CONSENT_VERSION):
            self._hold_setup(chat_id, notes)
            return []
        self._queue_begin(chat_id)
        return notes

    def _starter_mode(self) -> str:
        """``onboarding`` when the profile has no enabled role and no market, else ``update``."""
        snapshot = self._profiles.read_snapshot()
        if snapshot is None or (not any(role.enabled for role in snapshot.roles) and not snapshot.markets):
            return "onboarding"
        return "update"

    def _queue_begin(self, chat_id: int, *, ask_first: bool = True) -> None:
        mode = self._starter_mode()
        with self._cond:
            self._buffers.insert(
                0,
                _Buffer(chat_id, [], 0, self._clock(), closed=True, immediate=True, interview=True,
                        action="begin", mode=mode, ask_first=ask_first),
            )
            self._cond.notify_all()

    def _hold_setup(self, chat_id: int, notes: List[OutgoingMessage]) -> None:
        """``/setup`` without consent: the notice (after the discard notes); nothing is stored before the answer."""
        held = _Held(secrets.token_hex(4), chat_id, [], self._clock(), setup=True)
        with self._cond:
            replaced, self._held = self._held, held
        if replaced is not None and replaced.message_id is not None:
            self._clear_buttons(replaced.chat_id, replaced.message_id)
        if notes:
            self._send(chat_id, notes)
        message_id = self._send(chat_id, [OutgoingMessage(CONSENT_NOTICE, _consent_markup(held.nonce))])
        with self._cond:
            current = self._held is held
            if current:
                held.message_id = message_id
        if not current:  # answered or cancelled while the notice was being sent
            self._clear_buttons(chat_id, message_id)

    def _reset_for_setup(self, chat_id: int) -> List[OutgoingMessage]:
        self._drop_memory()
        with self._state_lock:
            pending = self._conv.delete_pending(chat_id)
            state = self._conv.get_state(chat_id)
            if state.state != STATE_IDLE or state.context is not None:
                self._conv.set_state(chat_id, STATE_IDLE, None, self._clock())
        notes: List[OutgoingMessage] = []
        if pending is not None:
            self._clear_buttons(chat_id, pending.message_id)
            notes += _reply(PROPOSAL_DISCARDED_REPLY)
        if state.state == STATE_CLARIFICATION:
            notes += _reply(CLARIFICATION_DISCARDED_REPLY)
        return notes

    def _row_interview_nonce(self, chat_id: int) -> Optional[str]:
        """The nonce of the stored interview (None: the chat is not in one)."""
        state = self._conv.get_state(chat_id)
        if state.state != STATE_INTERVIEW:
            return None
        nonce = state.context.get("nonce") if isinstance(state.context, dict) else None
        return nonce if isinstance(nonce, str) else ""

    def _interview_tag(self, chat_id: int) -> Optional[Tuple[Optional[str], Optional[str]]]:
        """``(nonce, slot)`` when the chat is in an interview: the mirror, else one read of the row (after a
        restart the slot is unknown); None otherwise."""
        with self._cond:
            entry = self._interview.get(chat_id)
        if entry is not None:
            return entry
        nonce = self._row_interview_nonce(chat_id)
        return None if nonce is None else (nonce or None, None)

    def _add_interview_input(
        self, update: Update, tag: Tuple[Optional[str], Optional[str]]
    ) -> List[OutgoingMessage]:
        """Queue an interview answer (or upload) for the worker, in arrival order and without debounce."""
        nonce, slot = tag
        document = update.document
        if document is not None:
            if not document.file_name.lower().endswith(_UPLOAD_SUFFIXES):
                return _reply(ONLY_TEXT_FILES_REPLY)
            if document.file_size is not None and document.file_size > MAX_UPLOAD_BYTES:
                return _reply(FILE_TOO_LARGE_REPLY)
        text = update.text if document is None else ""
        with self._cond:
            if len(self._buffers) >= MAX_QUEUED_REQUESTS:
                return _reply(BUSY_REPLY)
            self._buffers.append(
                _Buffer(update.chat_id, [text] if text else [], len(text), self._clock(), closed=True,
                        immediate=True, doc=document, caption=update.text.strip() if document else "",
                        interview=True, nonce=nonce, slot=slot)
            )
            self._cond.notify_all()
        return []

    def _on_interview_button(self, update: Update) -> List[OutgoingMessage]:
        """``iv:<nonce>:draft|cancel|retry|guided`` (poller thread)."""
        chat_id = update.chat_id
        parts = update.text.split(":")
        if chat_id is None or len(parts) != 3 or parts[2] not in _INTERVIEW_ACTIONS:
            return _reply(INTERVIEW_STALE_REPLY)
        nonce, action = parts[1], parts[2]
        tag = self._interview_tag(chat_id)
        if tag is None or tag[0] != nonce:
            return _reply(INTERVIEW_STALE_REPLY)
        if action in ("cancel", "guided"):
            self._clear_buttons(chat_id, update.message_id)
            self._drop_memory()  # bumps the generation: a worker step of this interview loses its result
            with self._state_lock:
                if self._row_interview_nonce(chat_id) == nonce:
                    self._conv.set_state(chat_id, STATE_IDLE, None, self._clock())
            if action == "guided":
                return self._wizard.start(update)
            return _reply(CANCELLED_REPLY)
        with self._cond:
            if len(self._buffers) >= MAX_QUEUED_REQUESTS:
                return _reply(BUSY_REPLY)
            self._buffers.append(
                _Buffer(chat_id, [], 0, self._clock(), closed=True, immediate=True, interview=True,
                        action=action, nonce=nonce, slot=tag[1])
            )
            self._cond.notify_all()
        return []

    def on_ok(self, update: Update) -> List[OutgoingMessage]:
        return self._confirm(update.chat_id, None, None)

    def on_cancel(self, update: Update) -> List[OutgoingMessage]:
        """Discard everything in flight: buffer, held text, running request, clarification and proposal."""
        chat_id = update.chat_id
        anything = self._drop_memory()
        with self._state_lock:
            pending = self._conv.delete_pending(chat_id)
            state = self._conv.get_state(chat_id)
            if state.state != STATE_IDLE or state.context is not None:
                self._conv.set_state(chat_id, STATE_IDLE, None, self._clock())
                anything = True
        if pending is not None:
            self._clear_buttons(chat_id, pending.message_id)
        return _reply(CANCELLED_REPLY if anything or pending is not None else NOTHING_TO_CANCEL_REPLY)

    def reset_memory(self) -> None:
        """Forget buffers, held text and any running result (unpair and a new pairing; the tables are cleared
        separately by ``ConversationStore.clear_all``)."""
        self._drop_memory()
        self._context_chats.clear()

    def _drop_memory(self) -> bool:
        with self._cond:
            self._generation += 1  # a request already taken by the worker loses its result
            held, retry = self._held, self._retry
            anything = bool(self._buffers) or held is not None or retry is not None or self._inflight
            self._buffers.clear()
            self._held = None
            self._retry = None
            self._swallow = None
            self._interview.clear()  # the single paired chat: its interview mirror goes with the generation
        if held is not None and held.message_id is not None:
            self._clear_buttons(held.chat_id, held.message_id)
        if retry is not None and retry.message_id is not None:
            self._clear_buttons(retry.chat_id, retry.message_id)
        return anything

    def on_callback(self, update: Update) -> List[OutgoingMessage]:
        if update.text.startswith("pc:"):
            return self._on_proposal_button(update)
        if update.text.startswith("cs:"):
            return self._on_consent_button(update)
        if update.text.startswith("rt:"):
            return self._on_retry_button(update)
        if update.text.startswith("iv:"):
            return self._on_interview_button(update)
        return _reply(NO_LONGER_ACTIVE_REPLY)

    # --- poller thread: buffering ------------------------------------------------------------

    def _route_bare(self, update: Update, kind: str) -> Optional[List[OutgoingMessage]]:
        """A bare yes / no: the answer to a question, a confirmation, or a hint (never an LLM call by itself)."""
        chat_id = update.chat_id
        if self._conv.get_state(chat_id).state == STATE_CLARIFICATION:
            return None  # the answer to the open question: it goes to the translator with the history
        pending = self._conv.get_pending(chat_id)
        if pending is None:
            return _reply(NO_PENDING_REPLY)
        if kind == "confirm":
            if update.date and self._clock() - update.date > BARE_CONFIRM_MAX_AGE_SECONDS:
                return _reply(BARE_STALE_REPLY)
            return self._confirm(chat_id, pending.change_id, None)
        return self._cancel_proposal(chat_id, pending.change_id, None)

    def _open_buffer(self, chat_id: int, now: float) -> Optional[_Buffer]:
        """The newest text buffer of this chat that is still inside its debounce window."""
        for buffer in reversed(self._buffers):
            if buffer.chat_id == chat_id and not buffer.closed:
                return buffer if now - buffer.last < DEBOUNCE_SECONDS else None
        return None

    def _add_part(self, chat_id: int, text: str, now: float) -> List[OutgoingMessage]:
        with self._cond:
            if self._swallow is not None:
                swallow_chat, swallow_last, told = self._swallow
                if swallow_chat == chat_id and now - swallow_last < DEBOUNCE_SECONDS:
                    # The tail of a paste that was already refused: the user is told once, then it is silent.
                    self._swallow = [chat_id, now, True]
                    return [] if told else _reply(SWALLOWED_REPLY)
                self._swallow = None
            buffer = self._open_buffer(chat_id, now)
            if buffer is None:
                if len(self._buffers) >= MAX_QUEUED_REQUESTS:
                    return _reply(BUSY_REPLY)
                buffer = _Buffer(chat_id, [], 0, now)
                self._buffers.append(buffer)
            if len(buffer.parts) == MAX_PARTS or buffer.chars + len(text) > MAX_MERGED_CHARS:
                # A truncated paste is worse than none: the whole merged request is dropped.
                self._buffers.remove(buffer)
                self._swallow = [chat_id, now, False]
                return _reply(TOO_LONG_REPLY)
            buffer.parts.append(text)
            buffer.chars += len(text)
            buffer.last = now
            self._cond.notify_all()
        return []

    def _add_upload(self, update: Update) -> List[OutgoingMessage]:
        document = update.document
        if not document.file_name.lower().endswith(_UPLOAD_SUFFIXES):
            return _reply(ONLY_TEXT_FILES_REPLY)
        if document.file_size is not None and document.file_size > MAX_UPLOAD_BYTES:
            return _reply(FILE_TOO_LARGE_REPLY)
        with self._cond:
            if len(self._buffers) >= MAX_QUEUED_REQUESTS:
                return _reply(BUSY_REPLY)
            self._buffers.append(
                _Buffer(update.chat_id, [], 0, self._clock(), closed=True, immediate=True,
                        doc=document, caption=update.text.strip())
            )
            self._cond.notify_all()
        return []

    # --- poller thread: consent button -------------------------------------------------------

    def _on_consent_button(self, update: Update) -> List[OutgoingMessage]:
        parts = update.text.split(":")
        if len(parts) != 3 or parts[2] not in ("yes", "no"):
            return _reply(CONSENT_STALE_REPLY)
        with self._cond:
            held = self._held
            if held is None or held.nonce != parts[1] or held.chat_id != update.chat_id:
                held = None
            else:
                self._held = None
        if held is None:
            return _reply(CONSENT_STALE_REPLY)
        self._clear_buttons(update.chat_id, update.message_id)
        if parts[2] == "no":
            if held.setup:  # /setup without the language model: the step-by-step wizard takes over
                return _reply(CONSENT_DECLINED_SETUP_REPLY) + self._wizard.start(update)
            return _reply(CONSENT_DECLINED_REPLY)
        self._conv.record_consent(update.chat_id, CONSENT_VERSION)  # the poller thread writes llm_consent
        if held.setup:
            self._queue_begin(held.chat_id)
            return []
        with self._cond:
            self._buffers.insert(
                0,
                _Buffer(held.chat_id, held.parts, sum(len(p) for p in held.parts), self._clock(),
                        closed=True, immediate=True),
            )
            self._cond.notify_all()
        return []

    def _on_retry_button(self, update: Update) -> List[OutgoingMessage]:
        """``rt:<nonce>:go``: run the failed request again with the same (already masked) parts."""
        parts = update.text.split(":")
        if len(parts) != 3 or parts[2] != "go":
            return _reply(CONSENT_STALE_REPLY)
        with self._cond:
            retry = self._retry
            if retry is None or retry.nonce != parts[1] or retry.chat_id != update.chat_id \
                    or self._clock() - retry.created > CONTEXT_TTL_SECONDS:
                retry = None
            else:
                self._retry = None
        if retry is None:
            return _reply(CONSENT_STALE_REPLY)
        self._clear_buttons(update.chat_id, update.message_id)
        with self._cond:
            self._buffers.insert(
                0,
                _Buffer(retry.chat_id, retry.parts, sum(len(p) for p in retry.parts), self._clock(),
                        closed=True, immediate=True),
            )
            self._cond.notify_all()
        return []

    # --- poller thread: confirm and cancel -----------------------------------------------------

    def _on_proposal_button(self, update: Update) -> List[OutgoingMessage]:
        parts = update.text.split(":")
        if len(parts) != 4 or not (parts[2].isascii() and parts[2].isdigit()) or parts[3] not in ("a", "c"):
            return _reply(NO_LONGER_ACTIVE_REPLY)
        pending = self._conv.get_pending(update.chat_id)
        if pending is None or pending.change_id != parts[1] or pending.base_version != int(parts[2]):
            return _reply(NO_LONGER_ACTIVE_REPLY)
        if parts[3] == "a":
            return self._confirm(update.chat_id, pending.change_id, update.message_id)
        return self._cancel_proposal(update.chat_id, pending.change_id, update.message_id)

    def _cancel_proposal(
        self, chat_id: int, change_id: str, button_message_id: Optional[int]
    ) -> List[OutgoingMessage]:
        taken = self._conv.take_pending(chat_id, change_id)
        if taken is None:
            return _reply(NO_LONGER_ACTIVE_REPLY)
        self._clear_buttons(chat_id, button_message_id if button_message_id is not None else taken.message_id)
        self._idle_unless_pending(chat_id, only_from=STATE_CONFIRM)
        return _reply(CANCELLED_REPLY)

    def _idle_unless_pending(self, chat_id: int, only_from: Optional[str] = None) -> None:
        """Back to idle, but never over a newer proposal the worker stored meanwhile."""
        with self._state_lock:
            if self._conv.get_pending(chat_id) is not None:
                return
            if only_from is None or self._conv.get_state(chat_id).state == only_from:
                self._conv.set_state(chat_id, STATE_IDLE, None, self._clock())

    def _confirm(
        self, chat_id: int, change_id: Optional[str], button_message_id: Optional[int]
    ) -> List[OutgoingMessage]:
        """Apply the pending proposal once (``/ok``, a bare word or a button); every profile write is here."""
        if change_id is None:
            pending = self._conv.get_pending(chat_id)
            if pending is None:
                return _reply(NO_PENDING_REPLY)
            change_id = pending.change_id
        taken = self._conv.take_pending(chat_id, change_id)  # atomic: a second press finds nothing
        if taken is None:
            return _reply(NO_LONGER_ACTIVE_REPLY)
        self._clear_buttons(chat_id, button_message_id if button_message_id is not None else taken.message_id)
        self._idle_unless_pending(chat_id)
        return self._apply(taken)

    def _apply(self, taken: PendingChange) -> List[OutgoingMessage]:
        if self._clock() - taken.created_at > PENDING_TTL_SECONDS:
            return _reply(EXPIRED_REPLY)
        try:
            operations = operations_from_json(taken.operations)  # stored JSON is re-validated, never trusted
        except ValueError:
            return _reply(PROPOSAL_UNREADABLE_REPLY)
        snapshot = self._profiles.read_snapshot()
        if (snapshot.version if snapshot is not None else 0) != taken.base_version:
            return _reply(CONFLICT_REPLY)
        label = "llm onboarding" if taken.mode == "onboarding" else f"chat change ({len(operations)} operations)"
        try:
            profile_ops = to_operations(operations, snapshot)
            result = apply_changes(self._profiles, profile_ops, label=label, expected_version=taken.base_version)
        except VersionConflict:
            return _reply(CONFLICT_REPLY)
        except (TranslationProblem, ProfileValidationError) as exc:
            return _reply(not_applied(exc.messages))
        except ProfileCompileError:
            return _reply(COMPILE_ERROR_REPLY)
        extra: List[str] = []
        if result.old is not None and result.old.fields.paused and result.new is not None \
                and not result.new.fields.paused:
            extra.append(self._trigger())
        if taken.mode == "onboarding" and result.new is not None and result.new.fields.paused:
            extra.append(PAUSED_NOTE)
        return _reply(render_apply(result, extra))

    def _trigger(self) -> str:
        try:
            outcome = self._ctx.trigger_cycle()
        except Exception as exc:  # the profile is already changed: the reply must still be sent
            logger.warning("trigger_cycle failed: %s", type(exc).__name__)
            outcome = ""
        return trigger_note(outcome)

    # --- worker thread ---------------------------------------------------------------------------

    def start(self) -> threading.Thread:
        thread = threading.Thread(target=self.run_forever, name="bot-conversation", daemon=True)
        thread.start()
        return thread

    def run_forever(self) -> None:
        while not self._stop.is_set():
            try:
                with self._cond:
                    wait = self._wait_seconds(self._clock())
                    if wait > 0:
                        self._cond.wait(wait)
                if self._stop.is_set():
                    break
                now = self._clock()
                self.process_next(now)
                if now >= self._next_sweep:
                    self._next_sweep = now + SWEEP_SECONDS
                    self.sweep(now)
            except Exception as exc:  # never the message: it can hold user text
                logger.error("conversation worker failed: %s", type(exc).__name__)
                self._stop.wait(0.2)

    def _wait_seconds(self, now: float) -> float:
        if self._next_sweep <= now:
            return 0.0
        wait = min(_IDLE_WAIT_SECONDS, self._next_sweep - now)
        for buffer in self._buffers:
            wait = min(wait, buffer.due_at() - now)
        return max(0.0, wait)

    def process_next(self, now: float) -> bool:
        """Flush one due request through the pipeline; False when nothing is due."""
        with self._cond:
            for index, buffer in enumerate(self._buffers):
                if buffer.due(now):
                    del self._buffers[index]
                    work = _Work(buffer, self._generation)
                    self._inflight = True
                    break
            else:
                return False
        try:
            self._process(work)
        finally:
            with self._cond:
                self._inflight = False
        return True

    def sweep(self, now: float) -> None:
        """Expire proposals older than 24 hours (with a notice) and purge stale conversation contexts."""
        for stale in self._wconv.expired_pending(now - PENDING_TTL_SECONDS):
            taken = self._wconv.take_pending(stale.chat_id, stale.change_id)
            if taken is None:  # confirmed or cancelled in the meantime
                continue
            self._clear_buttons(taken.chat_id, taken.message_id)
            if self._wconv.get_state(taken.chat_id).state == STATE_CONFIRM:
                self._wconv.set_state(taken.chat_id, STATE_IDLE, None, now)
            self._send(taken.chat_id, _reply(EXPIRED_REPLY))
        self._wconv.purge_contexts(now - CONTEXT_TTL_SECONDS)
        self._purge_old_contexts(now)
        with self._cond:
            held = self._held
            if held is not None and now - held.created > CONTEXT_TTL_SECONDS:
                self._held = None  # held text expires silently (its buttons just go away)
            else:
                held = None
            retry = self._retry
            if retry is not None and now - retry.created > CONTEXT_TTL_SECONDS:
                self._retry = None
            else:
                retry = None
        if held is not None and held.message_id is not None:
            self._clear_buttons(held.chat_id, held.message_id)
        if retry is not None and retry.message_id is not None:
            self._clear_buttons(retry.chat_id, retry.message_id)

    @staticmethod
    def _context_expired(state: ConversationState, now: float) -> bool:
        """The 24 hours count from when the first round's text was stored, not from the last round."""
        created = state.context.get("created_at") if isinstance(state.context, dict) else None
        start = float(created) if isinstance(created, (int, float)) and not isinstance(created, bool) else state.updated_at
        return now - start > CONTEXT_TTL_SECONDS

    def _purge_old_contexts(self, now: float) -> None:
        chats = set(self._context_chats)
        pairing = BotStateStore(self._wdb).get_pairing()
        if pairing is not None:
            chats.add(pairing.chat_id)
        for chat_id in chats:
            state = self._wconv.get_state(chat_id)
            if state.state not in (STATE_CLARIFICATION, STATE_INTERVIEW) or not self._context_expired(state, now):
                continue
            with self._state_lock:
                current = self._wconv.get_state(chat_id)
                if current.state != state.state or current.context != state.context:
                    continue  # rewritten meanwhile: the next sweep looks again
                self._wconv.set_state(chat_id, self._rest_state(chat_id), None, now)
            if state.state == STATE_INTERVIEW:
                with self._cond:
                    self._interview.pop(chat_id, None)

    # --- worker thread: one request --------------------------------------------------------------

    def _stale(self, work: _Work) -> bool:
        with self._cond:
            return work.generation != self._generation

    def _process(self, work: _Work) -> None:
        request_id = secrets.token_hex(4)
        try:
            self._run(work, request_id)
        except Exception as exc:  # the worker must outlive any one request; never log the message
            logger.error("conversation request %s failed: %s", request_id, type(exc).__name__)
            try:
                self._finish(work, _reply(ERROR_REPLY), purge=True)
            except Exception as exc2:
                logger.error("conversation request %s: reply failed: %s", request_id, type(exc2).__name__)

    def _run(self, work: _Work, request_id: str) -> None:
        buffer = work.buffer
        chat_id = buffer.chat_id
        if buffer.action == "begin":
            self._begin(work)
            return
        if buffer.interview:
            self._run_interview(work, request_id)
            return
        parts = self._collect_parts(work)
        if parts is None:
            return
        parts = [mask_personal_data(part)[0] for part in parts]
        now = self._clock()
        mode, history = self._mode_and_history(self._wconv.get_state(chat_id), now)
        logger.info("conversation request %s: %d parts, %d chars", request_id, len(parts), sum(map(len, parts)))
        if not self._wconv.has_consent(chat_id, CONSENT_VERSION):
            self._hold_for_consent(work, parts, now)
            return
        self._translate(work, parts, mode, history, request_id, now)

    def _collect_parts(self, work: _Work) -> Optional[List[str]]:
        buffer = work.buffer
        if buffer.doc is None:
            return list(buffer.parts)
        path = self._client.get_file_path(buffer.doc.file_id)
        if path is None:
            self._finish(work, _reply(DOWNLOAD_FAILED_REPLY), purge=False)
            return None
        try:
            data = self._client.download_file(path, MAX_UPLOAD_BYTES)
        except TelegramApiError as exc:
            reply = FILE_TOO_LARGE_REPLY if exc.status == 413 else DOWNLOAD_FAILED_REPLY
            self._finish(work, _reply(reply), purge=False)
            return None
        if data.startswith(_BOM):
            data = data[len(_BOM):]
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            self._finish(work, _reply(NOT_UTF8_REPLY), purge=False)
            return None
        if not text.strip():
            self._finish(work, _reply(EMPTY_FILE_REPLY), purge=False)
            return None
        parts = ([buffer.caption] if buffer.caption else []) + [text]
        if len("\n\n".join(parts)) > MAX_INPUT_CHARS:  # the translator would cut the end silently
            self._finish(work, _reply(TOO_LONG_REPLY), purge=False)
            return None
        return parts

    @staticmethod
    def _mode_and_history(state: ConversationState, now: float) -> Tuple[str, List[Tuple[str, str]]]:
        context = state.context
        if state.state == STATE_CLARIFICATION and isinstance(context, dict) \
                and not ConversationService._context_expired(state, now):
            mode = context.get("mode") if context.get("mode") in MODES else "chat"
            history: List[Tuple[str, str]] = []
            rounds = context.get("rounds")
            for item in rounds if isinstance(rounds, list) else []:
                if isinstance(item, list) and len(item) == 2 and all(isinstance(x, str) for x in item):
                    history.append((item[0], item[1]))
            return mode, history[:MAX_CLARIFICATIONS]
        return "chat", []

    def _hold_for_consent(self, work: _Work, parts: List[str], now: float) -> None:
        nonce = secrets.token_hex(4)
        held = _Held(nonce, work.buffer.chat_id, parts, now)  # memory only, never the database
        with self._cond:
            if work.generation != self._generation:
                return
            replaced, self._held = self._held, held
        if replaced is not None and replaced.message_id is not None:
            self._clear_buttons(replaced.chat_id, replaced.message_id)
        message_id = self._send(work.buffer.chat_id, [OutgoingMessage(CONSENT_NOTICE, _consent_markup(nonce))])
        with self._cond:
            current = self._held is held
            if current:
                held.message_id = message_id
        if not current:  # answered, replaced or cancelled while the notice was being sent
            self._clear_buttons(work.buffer.chat_id, message_id)

    def _translate(
        self,
        work: _Work,
        parts: List[str],
        mode: str,
        history: List[Tuple[str, str]],
        request_id: str,
        now: float,
    ) -> None:
        chat_id = work.buffer.chat_id
        snapshot = self._wprofiles.read_snapshot()
        if self._stale(work):
            return
        self._send(chat_id, _reply(WORKING_REPLY))
        timer = threading.Timer(PROGRESS_NOTE_SECONDS, self._progress_note, args=(work,))
        timer.daemon = True
        timer.start()
        failure: Optional[ChatLLMError] = None
        try:
            outcome = self._translator.translate(
                parts, snapshot, mode=mode, history=history, request_id=request_id
            )
        except StopRequested:
            return
        except ChatLLMError as exc:
            failure = exc
        except (TranslationProblem, ProfileValidationError) as exc:
            self._finish(work, _reply(not_applied(exc.messages)), purge=True)
            return
        except Exception as exc:  # ValueError, TypeError, httpx errors: the type name only, never the text
            logger.error("conversation request %s failed: %s", request_id, type(exc).__name__)
            self._finish(work, _reply(ERROR_REPLY), purge=True)
            return
        finally:
            timer.cancel()
            with self._note_lock:  # waits for a note that is already being sent, so it precedes the answer
                work.done = True
        if failure is not None:
            self._reply_failure(work, parts, failure, request_id)
            return
        if isinstance(outcome, Proposal):
            self._store_proposal(work, outcome, mode)
        elif isinstance(outcome, Clarification):
            self._ask(work, outcome, parts, history, mode, now)
        else:
            self._finish(work, _reply(NO_CHANGE_REPLY), purge=True)

    def _reply_failure(self, work: _Work, parts: List[str], failure: ChatLLMError, request_id: str) -> None:
        """Say why the model request failed (code-written text) and offer Retry; the conversation state is kept.

        Not configured is the exception: there is nothing to retry until a key is set.
        """
        chat_id = work.buffer.chat_id
        if isinstance(failure, ChatLLMNotConfigured):
            self._finish(work, _reply(NOT_CONFIGURED_REPLY), purge=False)
            return
        daily = bool(getattr(failure, "daily", False)) if isinstance(failure, ChatLLMLimited) else False
        text = describe_failure(failure.failure, failure.retry_in, daily=daily)
        nonce = secrets.token_hex(4)
        retry = _Retry(nonce, chat_id, list(parts), self._clock())  # memory only, never the database
        with self._cond:
            if work.generation != self._generation:
                return  # /cancel arrived meanwhile: the result is discarded
            replaced, self._retry = self._retry, retry
        if replaced is not None and replaced.message_id is not None:
            self._clear_buttons(replaced.chat_id, replaced.message_id)
        markup = {"inline_keyboard": [[{"text": RETRY_BUTTON, "callback_data": f"rt:{nonce}:go"}]]}
        message_id = self._send(chat_id, [OutgoingMessage(text, markup)])
        with self._cond:
            current = self._retry is retry
            if current:
                retry.message_id = message_id
        if not current:  # replaced, used or cancelled while the reply was being sent
            self._clear_buttons(chat_id, message_id)

    def _progress_note(self, work: _Work) -> None:
        # A separate lock, not ``_cond``: the send is network I/O and nothing slow may run under ``_cond``.
        with self._note_lock:
            if not work.done and not self._stale(work):
                self._send(work.buffer.chat_id, _reply(PROGRESS_REPLY))

    def _rest_state(self, chat_id: int) -> str:
        """The state a conversation returns to: waiting for the old proposal's answer, or idle."""
        return STATE_CONFIRM if self._wconv.get_pending(chat_id) is not None else STATE_IDLE

    def _finish(self, work: _Work, messages: Sequence[OutgoingMessage], *, purge: bool) -> None:
        """Send the final reply of a request; ``purge`` also ends any clarification (raw context is dropped).

        Writes run outside the memory lock; the generation is re-checked after them, so a ``/cancel`` that
        lands in between wins (the state is reset again and nothing is sent).
        """
        chat_id = work.buffer.chat_id
        if self._stale(work):
            return  # /cancel arrived meanwhile: the result is discarded
        if purge:
            self._reset_state(chat_id)
            if self._stale(work):
                self._reset_state(chat_id)
                return
        self._send(chat_id, messages)

    def _reset_state(self, chat_id: int) -> None:
        with self._state_lock:
            if self._wconv.get_state(chat_id).state == STATE_INTERVIEW:
                return  # never over an interview: only its own steps end it
            self._wconv.set_state(chat_id, self._rest_state(chat_id), None, self._clock())

    def _ask(
        self,
        work: _Work,
        clarification: Clarification,
        parts: List[str],
        history: List[Tuple[str, str]],
        mode: str,
        now: float,
    ) -> None:
        if len(history) >= MAX_CLARIFICATIONS:
            self._finish(work, _reply(NOT_UNDERSTOOD_REPLY), purge=True)
            return
        chat_id = work.buffer.chat_id
        rounds = [[earlier, question] for earlier, question in history]
        rounds.append(["\n\n".join(parts)[:MAX_INPUT_CHARS], clarification.question])
        state = self._wconv.get_state(chat_id)
        created = now
        if history and isinstance(state.context, dict) and isinstance(state.context.get("created_at"), (int, float)):
            created = float(state.context["created_at"])
        context: Dict[str, Any] = {"mode": mode, "rounds": rounds, "created_at": created}
        if self._stale(work):
            return
        with self._state_lock:
            self._wconv.set_state(chat_id, STATE_CLARIFICATION, context, now)
        self._context_chats.add(chat_id)
        if self._stale(work):  # /cancel landed during the write: drop the context again
            self._reset_state(chat_id)
            return
        self._send(chat_id, _reply(QUESTION_PREFIX + esc(clarification.question)))

    # --- worker thread: the interview ----------------------------------------------------------------

    def _begin(self, work: _Work) -> None:
        """Start an interview: write the row (``created_at`` null), set the mirror, post the intro with its
        buttons and the first question. Sends happen after the lock is released; a ``/cancel`` or ``/setup``
        that lands meanwhile (a new generation) wins and the row is reset."""
        buffer = work.buffer
        chat_id = buffer.chat_id
        nonce = secrets.token_hex(4)
        state = new_state(nonce, buffer.mode if buffer.mode in ("onboarding", "update") else "update")
        step = next_step(state) if buffer.ask_first else None
        now = self._clock()
        with self._state_lock:
            if self._stale(work):
                return
            self._wconv.set_state(chat_id, STATE_INTERVIEW, state.to_json(), now)
            if self._stale(work):  # dropped during the write: undo it unless something newer replaced it
                current = self._wconv.get_state(chat_id)
                if current.state == STATE_INTERVIEW and isinstance(current.context, dict) \
                        and current.context.get("nonce") == nonce:
                    self._wconv.set_state(chat_id, STATE_IDLE, None, now)
                return
        self._context_chats.add(chat_id)
        if not self._set_mirror(work, nonce, step.slot if step is not None else None):
            return
        markup = {
            "inline_keyboard": [
                [{"text": DRAFT_BUTTON, "callback_data": f"iv:{nonce}:draft"}],
                [{"text": GUIDED_BUTTON, "callback_data": f"iv:{nonce}:guided"}],
                [{"text": CANCEL_BUTTON, "callback_data": f"iv:{nonce}:cancel"}],
            ]
        }
        self._send(chat_id, [OutgoingMessage(INTERVIEW_INTRO, markup)])
        if step is not None and step.kind != "draft":
            self._send(chat_id, _reply(esc(step.text)))

    def _set_mirror(self, work: _Work, nonce: str, slot: Optional[str]) -> bool:
        """Record the current question in the mirror (before it is posted); False when the work is stale."""
        with self._cond:
            if work.generation != self._generation:
                return False
            self._interview[work.buffer.chat_id] = (nonce, slot)
            return True

    def _rebuild_mirror(self, work: _Work) -> Optional[InterviewState]:
        """After a restart the mirror is empty while the row says ``awaiting_interview``: rebuild it from the row."""
        chat_id = work.buffer.chat_id
        state = self._wconv.get_state(chat_id)
        interview = InterviewState.from_json(state.context) if state.state == STATE_INTERVIEW else None
        if interview is None:
            return None
        with self._cond:
            if work.generation == self._generation and chat_id not in self._interview:
                self._interview[chat_id] = (interview.nonce, interview.current)
        return interview

    def _run_interview(self, work: _Work, request_id: str) -> None:
        """PLACEHOLDER (Task 6b replaces it): answers, ``Draft now`` and ``Retry`` of an interview.

        For now an interview input only rebuilds the mirror after a restart; it is never sent to the free-text
        translator, and nothing is stored or replied."""
        self._rebuild_mirror(work)
        logger.debug("interview input %s not handled yet", request_id)

    def _proposal_extras(self, proposal: Proposal) -> Tuple[List[str], List[str]]:
        """Compile warnings and a rule backtest for the proposal (best effort: never blocks the proposal)."""
        try:
            compiled = compile_profile(proposal.proposed, scrape_interval_hours=self._ctx.settings.scrape_interval_hours)
            result = backtest_snapshots(self._wdb, proposal.base, proposal.proposed)
            return list(compiled.warnings), render_backtest(result) if result is not None else []
        except Exception as exc:
            logger.debug("proposal extras skipped: %s", type(exc).__name__)
            return [], []

    def _store_proposal(self, work: _Work, proposal: Proposal, mode: str) -> None:
        chat_id = work.buffer.chat_id
        change_id = secrets.token_hex(4)
        warnings, backtest_lines = self._proposal_extras(proposal)
        messages = render_proposal(proposal, change_id, backtest_lines=backtest_lines, warnings=warnings)
        pending = PendingChange(
            chat_id=chat_id,
            change_id=change_id,
            base_version=proposal.base.version if proposal.base is not None else 0,
            mode=mode,
            operations=proposal.operations_json,
            message_id=None,
            created_at=self._clock(),
        )
        if self._stale(work):
            return
        with self._state_lock:
            superseded = self._wconv.put_pending(pending)
            self._wconv.set_state(chat_id, STATE_CONFIRM, None, self._clock())
        if superseded is not None:
            self._clear_buttons(chat_id, superseded.message_id)
        if work.buffer.interview:  # an interview draft: the interview ends with its proposal
            with self._cond:
                self._interview.pop(chat_id, None)
        if self._stale(work):  # /cancel landed after the row was stored: no proposal may follow "Cancelled"
            with self._state_lock:
                self._wconv.take_pending(chat_id, change_id)
            self._reset_state(chat_id)
            return
        if superseded is not None:
            self._send(chat_id, _reply(SUPERSEDED_REPLY))
        last_id, complete = self._send_checked(chat_id, messages)
        if not complete:  # the user may not have seen all of it: a later bare "ok" must not apply it
            with self._state_lock:
                self._wconv.take_pending(chat_id, change_id)
            self._reset_state(chat_id)
            self._clear_buttons(chat_id, last_id)
            self._send(chat_id, _reply(DELIVERY_FAILED_REPLY))
            return
        self._wconv.set_pending_message(chat_id, change_id, last_id)
        current = self._wconv.get_pending(chat_id)
        if current is None or current.change_id != change_id:  # confirmed or cancelled while it was being sent
            self._clear_buttons(chat_id, last_id)

    # --- sending -----------------------------------------------------------------------------------

    def _send(self, chat_id: int, messages: Sequence[OutgoingMessage]) -> Optional[int]:
        """Send messages in chunks (buttons on the last chunk only); the id of the last message sent."""
        return self._send_checked(chat_id, messages)[0]

    def _send_checked(self, chat_id: int, messages: Sequence[OutgoingMessage]) -> Tuple[Optional[int], bool]:
        """Like ``_send``, and whether every chunk was delivered."""
        last_id: Optional[int] = None
        complete = True
        for message in messages:
            pieces = chunk_html(message.text)
            for index, piece in enumerate(pieces):
                markup = message.reply_markup if index == len(pieces) - 1 else None
                sent_id = self._client.send_message(chat_id, piece, markup)
                if sent_id is None:
                    complete = False
                else:
                    last_id = sent_id
        return last_id, complete

    def _clear_buttons(self, chat_id: int, message_id: Optional[int]) -> None:
        if message_id is None:
            return
        try:
            self._client.edit_message_reply_markup(chat_id, message_id)
        except Exception as exc:
            logger.debug("removing buttons failed: %s", type(exc).__name__)
