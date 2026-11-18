"""Safe, line-preserving edits of the local ``.env`` file (it holds secrets).

``merge_env_file`` rewrites only the lines of the keys it is given, keeps every other byte, writes through a
temp file created with mode 600 and swaps it in atomically, so a crash never leaves a half-written file and
the secrets are never world-readable, not even briefly.
"""
import os
import re
import secrets
import unicodedata
from pathlib import Path
from typing import List, Mapping, Optional

from dotenv import dotenv_values

_KEY_RE = re.compile(r"[A-Z][A-Z0-9_]*")
# Characters that need quoting or escaping in a dotenv file, or that dotenv would interpret: whitespace, quotes,
# comments (#), escapes (\), and variable interpolation ($). A pasted secret containing one is a paste error.
_FORBIDDEN_CHARS = set("\"'`#\\$")


def validate_env_value(value: str) -> None:
    """Raise ``ValueError`` (never echoing the value) unless it can be written unquoted. Empty is allowed."""
    for char in value:
        if char.isspace() or char in _FORBIDDEN_CHARS or unicodedata.category(char).startswith("C"):
            raise ValueError("value contains whitespace, a quote, '#', '$', a backslash or a control character")


def read_env_value(path: Path, key: str) -> Optional[str]:
    """The value of ``key`` in the dotenv file ``path``; None when the file or the key is missing."""
    return dotenv_values(path).get(key)


def _line_ending(line: bytes) -> bytes:
    if line.endswith(b"\r\n"):
        return b"\r\n"
    if line.endswith(b"\n"):
        return b"\n"
    if line.endswith(b"\r"):
        return b"\r"
    return b""


def _merge(data: bytes, updates: Mapping[str, str]) -> bytes:
    lines: List[bytes] = data.splitlines(keepends=True)
    patterns = {key: re.compile(rb"^\s*(?:export\s+)?" + re.escape(key.encode()) + rb"\s*=") for key in updates}
    done = set()
    out: List[bytes] = []
    for line in lines:
        matched = next((key for key, pattern in patterns.items() if pattern.match(line)), None)
        if matched is None:
            out.append(line)
        elif matched not in done:
            done.add(matched)
            out.append(f"{matched}={updates[matched]}".encode() + _line_ending(line))
        # else: a duplicate of a key that was already replaced is dropped
    missing = [key for key in updates if key not in done]
    if missing:
        newline = b"\r\n" if b"\r\n" in data else b"\n"
        if out and not _line_ending(out[-1]):
            out[-1] += newline
        out.extend(f"{key}={updates[key]}".encode() + newline for key in missing)
    return b"".join(out)


def merge_env_file(path: Path, updates: Mapping[str, str]) -> None:
    """Set ``updates`` in the dotenv file ``path`` (created when missing), mode 600, atomically."""
    for key, value in updates.items():
        if not _KEY_RE.fullmatch(key):
            raise ValueError("invalid environment variable name")
        validate_env_value(value)
    target = Path(path).resolve()  # a symlinked .env stays a symlink; its target is updated
    try:
        existing = target.read_bytes()
    except FileNotFoundError:
        existing = b""
    merged = _merge(existing, updates)

    temp = target.with_name(f".{target.name}.{secrets.token_hex(6)}.tmp")
    fd = os.open(temp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(merged)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, target)
    except BaseException:
        try:
            os.unlink(temp)
        except OSError:
            pass
        raise
    os.chmod(target, 0o600)
