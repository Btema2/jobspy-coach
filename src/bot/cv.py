"""CV reading for the onboarding interview: PDFs in a child process, text files directly, both masked.

The PDF is parsed only by ``src.bot.pdf_text`` in a separate interpreter (its own memory and CPU limits, killed
after ``PDF_TIMEOUT_SECONDS``). Nothing here logs text, bytes or a file name; no function takes a file name.
The returned text has e-mail addresses, phone numbers, links and handles masked and the contact header removed.
"""
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass

from src.bot.pdf_text import MAX_CHILD_CHARS, MAX_CV_CHARS, MAX_CV_PAGES  # noqa: F401 - re-exported constants
from src.bot.privacy import mask_all, strip_contact_header

logger = logging.getLogger(__name__)

MAX_CV_BYTES = 5 * 1024 * 1024
PDF_TIMEOUT_SECONDS = 15.0
PDF_MARKER_WINDOW = 1024
MIN_TEXT_CHARS = 200
PREVIEW_CHARS = 300

PASSWORD_REPLY = "This PDF is password-protected. Send an unprotected copy or paste the text."
NO_TEXT_REPLY = "This PDF has no readable text (it looks scanned). Paste the text or send a .txt file."
UNREADABLE_REPLY = "This PDF could not be read. Paste the text or send a .txt file."
TOO_LARGE_REPLY = "That PDF is larger than 5 MB. Send a smaller one or paste the text."
TOO_MANY_PAGES_REPLY = "That PDF has more than 10 pages. Send a shorter one or paste the text."

# Resolved through the installed (editable) package, never the working directory (``-I``). Tests patch it.
_CHILD_ARGV = [sys.executable, "-I", "-m", "src.bot.pdf_text"]

_ERROR_REPLIES = {
    "encrypted": PASSWORD_REPLY,
    "too_many_pages": TOO_MANY_PAGES_REPLY,
    "unreadable": UNREADABLE_REPLY,
}


class CvError(Exception):
    """A CV that cannot be used; ``reply`` is the exact text for the user."""

    def __init__(self, reply: str) -> None:
        super().__init__(reply)
        self.reply = reply


@dataclass(frozen=True)
class CvText:
    pages: int  # 0 for a text file
    text: str  # masked, header removed


def _minimal_env() -> dict:
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}
    if "LANG" in os.environ:
        env["LANG"] = os.environ["LANG"]
    return env


def _run_child(data: bytes, timeout: float) -> dict:
    try:
        proc = subprocess.run(
            list(_CHILD_ARGV),
            input=data,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=_minimal_env(),
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        logger.warning("CV extraction timed out")  # killed by subprocess.run
        raise CvError(UNREADABLE_REPLY) from None
    except Exception as exc:  # noqa: BLE001 - spawn failures: the type name only
        logger.warning("CV extraction failed to start (%s)", type(exc).__name__)
        raise CvError(UNREADABLE_REPLY) from None
    if proc.returncode != 0:
        logger.warning("CV extraction child exited abnormally")
        raise CvError(UNREADABLE_REPLY)
    # stdout is read whole: accepted risk, the child's output is at most ~240 KB by construction
    # (MAX_CHILD_CHARS characters, JSON-escaped at up to 12 bytes each).
    try:
        result = json.loads(proc.stdout.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        logger.warning("CV extraction returned unreadable output")
        raise CvError(UNREADABLE_REPLY) from None
    if not isinstance(result, dict):
        raise CvError(UNREADABLE_REPLY)
    return result


def _finish(text: str, pages: int) -> CvText:
    """Header removed and everything masked on the full text; only then cut (no half e-mail at the cut)."""
    cleaned = mask_all(strip_contact_header(text))[:MAX_CV_CHARS].strip()
    if not cleaned:
        raise CvError(NO_TEXT_REPLY)  # everything was header
    return CvText(pages=pages, text=cleaned)


def read_cv(data: bytes, *, timeout: float = PDF_TIMEOUT_SECONDS) -> CvText:
    """Masked text of a PDF given as bytes; raises ``CvError`` with the user reply on any refusal."""
    if len(data) > MAX_CV_BYTES:
        raise CvError(TOO_LARGE_REPLY)
    if data.find(b"%PDF-", 0, PDF_MARKER_WINDOW + 5) < 0:
        raise CvError(UNREADABLE_REPLY)
    result = _run_child(data, timeout)
    if result.get("ok") is not True:
        error = result.get("error")
        raise CvError(_ERROR_REPLIES.get(error if isinstance(error, str) else "", UNREADABLE_REPLY))
    pages, text = result.get("pages"), result.get("text")
    if not isinstance(pages, int) or isinstance(pages, bool) or not isinstance(text, str):
        raise CvError(UNREADABLE_REPLY)
    if pages > MAX_CV_PAGES:
        raise CvError(TOO_MANY_PAGES_REPLY)
    text = text[:MAX_CHILD_CHARS]
    if sum(1 for ch in text if not ch.isspace()) < MIN_TEXT_CHARS:
        raise CvError(NO_TEXT_REPLY)
    return _finish(text, pages)


def read_text_file(text: str) -> CvText:
    """A ``.txt`` / ``.md`` CV: the same masking and header removal, ``pages == 0``; ``CvError`` when empty."""
    return _finish(text[:MAX_CHILD_CHARS], 0)


def build_preview(cv: CvText) -> str:
    """``Read <N> pages, <M> characters. First lines: <first ~300 masked characters>`` (no page count for text)."""
    snippet = " ".join(cv.text.split())[:PREVIEW_CHARS]
    count = "" if not cv.pages else ("1 page, " if cv.pages == 1 else f"{cv.pages} pages, ")
    return f"Read {count}{len(cv.text)} characters. First lines: {snippet}"
