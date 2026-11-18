"""Pairing and per-update authorisation."""
import logging

import pytest

from src.bot.auth import (
    CODE_ALPHABET,
    MAX_PAIRING_ATTEMPTS,
    PAIRING_TTL_SECONDS,
    AuthDecision,
    Authorizer,
    adopt_chat_id,
    format_code,
    generate_pairing_code,
    hash_code,
    issue_pairing_code,
    normalise_code,
)
from src.bot.state import BotStateStore, Pairing
from src.bot.updates import parse_update
from src.storage.db import Database
from tests.fake_telegram import callback_update, message_update

NOW = 1_700_000_000.0


@pytest.fixture
def state(tmp_path):
    db = Database(tmp_path / "bot.db")
    db.init_db()
    return BotStateStore(db)


class Clock:
    def __init__(self, value=NOW):
        self.value = value

    def __call__(self):
        return self.value


def _msg(text, **kw):
    return parse_update(message_update(1, text, **kw))


def test_code_is_8_chars_from_32_symbol_alphabet():
    assert len(set(CODE_ALPHABET)) == 32
    codes = {generate_pairing_code() for _ in range(1000)}
    assert len(codes) > 990
    assert all(len(c) == 8 and set(c) <= set(CODE_ALPHABET) for c in codes)


def test_normalise_and_format_code():
    assert normalise_code("abcd-efgh") == "ABCDEFGH"
    assert normalise_code(" o0 il-1l ") == "001111"
    assert format_code("ABCDEFGH") == "ABCD-EFGH"
    assert hash_code("abcd-efgh") == hash_code("ABCDEFGH")
    assert len(hash_code("ABCDEFGH")) == 64


def test_pairing_happy_path_single_use(state):
    clock = Clock()
    auth = Authorizer(state, clock=clock)
    code = issue_pairing_code(state, NOW)
    assert state.get_pairing_code().expires_at == NOW + PAIRING_TTL_SECONDS
    assert code not in str(state.get_pairing_code())

    update = _msg(f"/start {format_code(code)}")
    assert auth.decide(update) is AuthDecision.PAIRING_ATTEMPT
    outcome = auth.try_pair(update)

    assert outcome.paired is True
    assert outcome.reply == "Paired. Send /help to see the commands."
    assert state.get_pairing() == Pairing(111, 111)
    assert state.get_pairing_code() is None
    # Single use: the same code from another private user is rejected, nothing changes.
    other = _msg(f"/start {code}", user_id=222, chat_id=222)
    assert auth.decide(other) is AuthDecision.REJECT
    assert state.get_pairing() == Pairing(111, 111)


def test_wrong_code_counts_and_locks_after_5(state, caplog):
    auth = Authorizer(state, clock=Clock())
    code = issue_pairing_code(state, NOW)
    wrong = _msg("/start ZZZZ-ZZZZ")

    with caplog.at_level(logging.WARNING, logger="src.bot.auth"):
        replies = []
        for _ in range(MAX_PAIRING_ATTEMPTS):
            assert auth.decide(wrong) is AuthDecision.PAIRING_ATTEMPT
            replies.append(auth.try_pair(wrong).reply)

    assert replies[:4] == ["Pairing code not accepted."] * 4
    assert replies[4] is None
    warnings = [r for r in caplog.records if "pairing locked" in r.getMessage()]
    assert len(warnings) == 1
    assert "restart the daemon" in warnings[0].getMessage()
    # Locked: even the right code is rejected until a restart.
    right = _msg(f"/start {code}")
    assert auth.decide(right) is AuthDecision.REJECT
    assert state.get_pairing() is None
    restarted = Authorizer(state, clock=Clock())
    assert restarted.decide(right) is AuthDecision.PAIRING_ATTEMPT
    assert restarted.try_pair(right).paired is True


def test_expired_code_rejected(state):
    clock = Clock()
    auth = Authorizer(state, clock=clock)
    code = issue_pairing_code(state, NOW)
    clock.value = NOW + PAIRING_TTL_SECONDS + 1

    outcome = auth.try_pair(_msg(f"/start {code}"))

    assert (outcome.paired, outcome.reply) == (False, None)
    assert state.get_pairing() is None
    assert state.get_pairing_code() is None  # an expired code is cleared


def test_no_code_issued_means_silent_rejection(state):
    auth = Authorizer(state, clock=Clock())
    update = _msg("/start ABCD-EFGH")
    assert auth.decide(update) is AuthDecision.PAIRING_ATTEMPT
    outcome = auth.try_pair(update)
    assert (outcome.paired, outcome.reply) == (False, None)


def test_group_chat_and_strangers_rejected(state):
    auth = Authorizer(state, clock=Clock())
    code = issue_pairing_code(state, NOW)

    group = _msg(f"/start {code}", chat_id=-100, chat_type="group")
    assert auth.decide(group) is AuthDecision.REJECT
    assert auth.decide(_msg("/start")) is AuthDecision.REJECT  # no code argument
    assert auth.decide(_msg("hello ABCD")) is AuthDecision.REJECT  # not /start
    assert auth.decide(parse_update(callback_update(2, "x"))) is AuthDecision.REJECT  # unpaired callback

    assert auth.try_pair(_msg(f"/start {code}")).paired is True
    assert auth.decide(_msg("/undo")) is AuthDecision.ACCEPT
    assert auth.decide(_msg("/undo", user_id=222, chat_id=222)) is AuthDecision.REJECT
    assert auth.decide(_msg("/undo", user_id=111, chat_id=-100, chat_type="group")) is AuthDecision.REJECT
    assert auth.decide(_msg("/undo", user_id=111, chat_id=111, chat_type="channel")) is AuthDecision.REJECT
    assert auth.decide(_msg("/undo", user_id=111, chat_id=999)) is AuthDecision.REJECT
    raw = message_update(3, "/undo")
    del raw["message"]["from"]
    assert auth.decide(parse_update(raw)) is AuthDecision.REJECT
    assert auth.decide(parse_update({"update_id": 4, "edited_message": {}})) is AuthDecision.REJECT
    assert auth.decide(_msg(f"/start {code}")) is AuthDecision.ACCEPT  # paired user: just a command


def test_callback_auth_checks_from_id_and_chat_type(state):
    state.set_pairing(111, 111)
    auth = Authorizer(state, clock=Clock())

    assert auth.decide(parse_update(callback_update(1, "x"))) is AuthDecision.ACCEPT
    assert auth.decide(parse_update(callback_update(2, "x", user_id=222, chat_id=111))) is AuthDecision.REJECT
    assert auth.decide(parse_update(callback_update(3, "x", chat_type="supergroup"))) is AuthDecision.REJECT
    assert auth.decide(parse_update(callback_update(4, "x", chat_id=-5, chat_type="group"))) is AuthDecision.REJECT


def test_adopt_chat_id_only_positive_int(state):
    for raw in ("-100123", "@chan", "", "  ", None, "0", "12abc", "your_telegram_user_or_chat_id_here", "٤٢"):
        assert adopt_chat_id(state, raw) is False
    assert state.get_pairing() is None

    assert adopt_chat_id(state, "4242") is True
    assert state.get_pairing() == Pairing(4242, 4242)
    assert adopt_chat_id(state, "999") is False  # already paired
    assert state.get_pairing() == Pairing(4242, 4242)


def test_code_comparison_uses_compare_digest_on_equal_length_digests(state, monkeypatch):
    import hmac

    import src.bot.auth as auth_module

    calls = []
    real = hmac.compare_digest

    def spy(a, b):
        calls.append((a, b))
        return real(a, b)

    monkeypatch.setattr(auth_module.hmac, "compare_digest", spy)
    auth = Authorizer(state, clock=Clock())
    issue_pairing_code(state, NOW)

    for guess in ("x", "ABCD-EFGH-IJKL-MNOP" * 20, "\u00e9\u00e9", "0"):
        assert auth.try_pair(_msg(f"/start {guess}", user_id=7, chat_id=7)).paired is False
        auth = Authorizer(state, clock=Clock())  # a restart clears the lock; the code stays valid

    assert len(calls) == 4
    assert all(len(a) == 64 and len(b) == 64 for a, b in calls)  # hex digests, whatever the guess length
    assert state.get_pairing() is None


def test_forwarded_messages_are_rejected_even_from_the_paired_user(state):
    state.set_pairing(111, 111)
    auth = Authorizer(state, clock=Clock())
    raw = message_update(1, "/undo", user_id=111, chat_id=111)
    raw["message"]["forward_origin"] = {"type": "user"}

    assert auth.decide(parse_update(raw)) is AuthDecision.REJECT
    assert parse_update(raw).command is None
    assert auth.decide(_msg("/undo")) is AuthDecision.ACCEPT
