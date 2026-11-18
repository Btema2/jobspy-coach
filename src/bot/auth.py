"""Pairing and per-update authorisation (security-critical: every update passes ``Authorizer.decide``).

The paired user is identified by user id AND chat id AND a private chat; nothing else is ever accepted. Codes
are single-use, expire after 15 minutes and are stored only as a hash; five wrong codes lock pairing in memory
until the process restarts. Nothing here logs a code, a code attempt or message text.
"""
import enum
import hashlib
import hmac
import logging
import re
import secrets
import time
from dataclasses import dataclass
from typing import Callable, Optional

from src.bot.state import BotStateStore
from src.bot.updates import Update

logger = logging.getLogger(__name__)

CODE_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"  # Crockford base32: 8 characters are 40 bits
CODE_LENGTH = 8
PAIRING_TTL_SECONDS = 900
MAX_PAIRING_ATTEMPTS = 5

_POSITIVE_INT_RE = re.compile(r"[0-9]+")


def generate_pairing_code() -> str:
    return "".join(secrets.choice(CODE_ALPHABET) for _ in range(CODE_LENGTH))


def normalise_code(text: str) -> str:
    """Upper-case, drop spaces and dashes, map the look-alikes O -> 0 and I, L -> 1."""
    cleaned = text.upper().replace(" ", "").replace("-", "")
    return cleaned.replace("O", "0").replace("I", "1").replace("L", "1")


def format_code(code: str) -> str:
    return f"{code[:4]}-{code[4:]}"


def hash_code(code: str) -> str:
    return hashlib.sha256(normalise_code(code).encode("utf-8")).hexdigest()


def issue_pairing_code(state: BotStateStore, now: float) -> str:
    """Store the hash of a fresh code (valid for 15 minutes) and return the plain code, once."""
    code = generate_pairing_code()
    state.set_pairing_code(hash_code(code), now + PAIRING_TTL_SECONDS)
    return code


def adopt_chat_id(state: BotStateStore, raw: Optional[str]) -> bool:
    """Adopt ``TELEGRAM_CHAT_ID`` as the paired chat when unpaired and it is a positive integer.

    The local ``.env`` is the trust anchor; a private chat id equals the user id, so both are that number.
    """
    if state.get_pairing() is not None or raw is None:
        return False
    text = raw.strip()
    if not _POSITIVE_INT_RE.fullmatch(text) or int(text) <= 0:
        return False
    state.set_pairing(int(text), int(text))
    return True


class AuthDecision(enum.Enum):
    ACCEPT = "accept"
    PAIRING_ATTEMPT = "pairing_attempt"
    REJECT = "reject"


@dataclass(frozen=True)
class PairOutcome:
    paired: bool
    reply: Optional[str]


class Authorizer:
    def __init__(self, state: BotStateStore, *, clock: Callable[[], float] = time.time):
        self._state = state
        self._clock = clock
        self._locked = False
        state.reset_attempts()  # "locked until restart": a new process starts with a clean counter

    def decide(self, update: Update) -> AuthDecision:
        if update.kind == "other" or update.user_id is None or update.chat_id is None:
            return AuthDecision.REJECT
        if update.forwarded:  # a forwarded "/undo" carries the forward time: never a command, never pairing
            return AuthDecision.REJECT
        pairing = self._state.get_pairing()
        if pairing is not None:
            ok = (
                update.chat_type == "private"
                and update.user_id == pairing.user_id
                and update.chat_id == pairing.chat_id
            )
            return AuthDecision.ACCEPT if ok else AuthDecision.REJECT
        if (
            update.kind == "message"
            and update.chat_type == "private"
            and update.command == "/start"
            and update.args
            and not self._locked
        ):
            return AuthDecision.PAIRING_ATTEMPT
        return AuthDecision.REJECT

    def try_pair(self, update: Update) -> PairOutcome:
        code = self._state.get_pairing_code()
        if code is None:
            return PairOutcome(False, None)
        if self._clock() > code.expires_at:
            self._state.clear_pairing_code()
            return PairOutcome(False, None)
        if update.user_id is None or update.chat_id is None:
            return PairOutcome(False, None)
        if hmac.compare_digest(hash_code(update.args), code.code_hash):
            self._state.set_pairing(update.user_id, update.chat_id)
            return PairOutcome(True, "Paired. Send /help to see the commands.")
        attempts = self._state.record_failed_attempt()
        if attempts >= MAX_PAIRING_ATTEMPTS:
            self._locked = True
            logger.warning(
                "pairing locked after %d wrong codes; restart the daemon to try again", MAX_PAIRING_ATTEMPTS
            )
            return PairOutcome(False, None)
        return PairOutcome(False, "Pairing code not accepted.")
