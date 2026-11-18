"""Rendering helpers for bot output: escaping, UTF-16-aware chunking, pagination."""
import html
import math
import re
from typing import List, Sequence, Tuple

from src.profile.ops import ApplyResult

CHUNK_LIMIT = 3800
PAGE_SIZE = 30

# A character Telegram must never see cut in two: an entity (&amp;) or a tag (<b>, <a href="...">).
_ATOMIC_RE = re.compile(r"&#?\w+;|<[^<>]*>")


def esc(value: object) -> str:
    return html.escape(str(value), quote=False)


def utf16_len(text: str) -> int:
    """Length in UTF-16 code units, the unit Telegram counts message length in."""
    return len(text.encode("utf-16-le")) // 2


def _fit_prefix(text: str, limit: int) -> int:
    """Index of the longest prefix of ``text`` whose UTF-16 length is at most ``limit``."""
    units = 0
    for index, char in enumerate(text):
        units += 2 if ord(char) > 0xFFFF else 1
        if units > limit:
            return index
    return len(text)


def _hard_split(line: str, limit: int) -> List[str]:
    pieces: List[str] = []
    rest = line
    while utf16_len(rest) > limit:
        cut = _fit_prefix(rest, limit)
        # Back off out of a split entity/tag; the last '&' or '<' before the cut is the only candidate.
        start = max(rest.rfind("&", 0, cut), rest.rfind("<", 0, cut))
        if start > 0:
            match = _ATOMIC_RE.match(rest, start)
            if match and match.end() > cut:
                cut = start
        cut = max(cut, 1)
        pieces.append(rest[:cut])
        rest = rest[cut:]
    if rest:
        pieces.append(rest)
    return pieces


def chunk_html(text: str, limit: int = CHUNK_LIMIT) -> List[str]:
    """Split on newlines into chunks of at most ``limit`` UTF-16 units; overlong lines are hard-split."""
    if not text:
        return []
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        joined = "\n".join(current)
        if joined.strip():
            chunks.append(joined)
        current, current_len = [], 0

    for line in text.split("\n"):
        for piece in _hard_split(line, limit) if utf16_len(line) > limit else [line]:
            size = utf16_len(piece)
            if current and current_len + 1 + size > limit:
                flush()
            current_len += size + (1 if current else 0)
            current.append(piece)
    flush()
    return chunks


def paginate(lines: Sequence[str], page: int, page_size: int = PAGE_SIZE) -> Tuple[List[str], int, int]:
    pages = max(1, math.ceil(len(lines) / page_size))
    page = min(max(page, 1), pages)
    start = (page - 1) * page_size
    return list(lines[start:start + page_size]), page, pages


def render_apply(result: ApplyResult, extra: Sequence[str] = ()) -> str:
    lines = [f"<b>Profile v{result.version}</b>"]
    lines.extend(esc(line) for line in result.diff)
    lines.extend("Warning: " + esc(w) for w in result.warnings)
    lines.extend(esc(line) for line in extra)
    return "\n".join(lines)
