"""Parsing of document messages (uploads) and their authorisation."""
import pytest

from src.bot.auth import AuthDecision, Authorizer
from src.bot.state import BotStateStore
from src.bot.updates import DocumentRef, parse_update
from src.storage.db import Database
from tests.fake_telegram import document_update, message_update


def test_document_message_carries_ref_and_caption_as_text():
    update = parse_update(document_update(3, "f1", "notes.md", caption="  <CAPTION> ", file_size=42))
    assert update.kind == "message"
    assert update.document == DocumentRef("f1", "notes.md", 42, "text/plain")
    assert update.text == "  <CAPTION> "
    assert update.command is None and update.args == ""


def test_caption_that_looks_like_a_command_is_not_a_command():
    update = parse_update(document_update(3, "f1", "notes.txt", caption="/undo"))
    assert update.command is None and update.text == "/undo"


def test_document_without_caption_or_optional_fields():
    update = parse_update(document_update(3, "f1", "n.txt", mime_type=None))
    assert update.document == DocumentRef("f1", "n.txt", None, None)
    assert update.text == ""


@pytest.mark.parametrize(
    "document",
    [{"file_name": "a.txt"}, {"file_id": 5}, "text", None, []],
)
def test_malformed_document_is_ignored_not_raised(document):
    raw = document_update(3, "f1", "a.txt")
    raw["message"]["document"] = document
    update = parse_update(raw)
    assert update.document is None


def test_document_ref_is_frozen_and_bounded():
    raw = document_update(3, "f1", "x" * 1000 + ".txt", file_size=True)
    update = parse_update(raw)
    assert len(update.document.file_name) <= 255
    assert update.document.file_size is None  # a bool is not a size
    with pytest.raises(Exception):
        update.document.file_id = "other"


def test_plain_message_has_no_document():
    assert parse_update(message_update(1, "hello")).document is None


@pytest.fixture
def auth(tmp_path):
    db = Database(tmp_path / "bot.db")
    db.init_db()
    state = BotStateStore(db)
    state.set_pairing(111, 111)
    return Authorizer(state)


def test_document_passes_authorizer_like_any_update(auth):
    assert auth.decide(parse_update(document_update(1, "f", "a.txt"))) is AuthDecision.ACCEPT
    assert auth.decide(parse_update(document_update(1, "f", "a.txt", chat_type="group"))) is AuthDecision.REJECT
    assert auth.decide(parse_update(document_update(1, "f", "a.txt", user_id=222))) is AuthDecision.REJECT
    forwarded = document_update(1, "f", "a.txt")
    forwarded["message"]["forward_origin"] = {"type": "user"}
    assert auth.decide(parse_update(forwarded)) is AuthDecision.REJECT
