"""Child-process PDF text extractor: ``python -I -m src.bot.pdf_text`` (bytes on stdin, one JSON line on stdout).

A PDF is hostile input, so the parent (``src/bot/cv.py``) never parses one itself. This process sets its own
memory and CPU limits before importing the PDF library, refuses long files before reading a page, and always
answers with exit code 0 and exactly one JSON line: ``{"ok": true, "pages": N, "text": "..."}`` or
``{"ok": false, "error": "encrypted" | "too_many_pages" | "unreadable"}``. It never prints text to stderr and
never receives or logs a file name.
"""
import io
import json
import sys

MAX_CV_PAGES = 10
MAX_CV_CHARS = 20000
# The child sends a little more than the cap so the parent can mask before cutting (no half e-mail at the cut).
TEXT_SLACK_CHARS = 256
MAX_CHILD_CHARS = MAX_CV_CHARS + TEXT_SLACK_CHARS
ADDRESS_SPACE_BYTES = 512 * 1024 * 1024
CPU_SECONDS = 10


def _apply_limits() -> None:
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (ADDRESS_SPACE_BYTES, ADDRESS_SPACE_BYTES))
        resource.setrlimit(resource.RLIMIT_CPU, (CPU_SECONDS, CPU_SECONDS))
    except (ImportError, ValueError, OSError):
        pass  # not Linux: the parent's wall-clock kill is the only limit then


def _extract(data: bytes) -> dict:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted:
        try:
            opened = reader.decrypt("")
        except Exception:  # noqa: BLE001 - e.g. a missing crypto library: same answer as a wrong password
            return {"ok": False, "error": "encrypted"}
        if not opened:
            return {"ok": False, "error": "encrypted"}
    pages = len(reader.pages)
    if pages > MAX_CV_PAGES:
        return {"ok": False, "error": "too_many_pages"}
    chunks = []
    total = 0
    for page in reader.pages:
        piece = page.extract_text() or ""
        chunks.append(piece)
        total += len(piece) + 1
        if total >= MAX_CHILD_CHARS:
            break
    text = "\n".join(chunks)[:MAX_CHILD_CHARS]
    return {"ok": True, "pages": pages, "text": text}


def main() -> int:
    _apply_limits()
    try:
        data = sys.stdin.buffer.read()
        result = _extract(data)
    except BaseException:  # noqa: BLE001 - MemoryError, parse errors, recursion: all "unreadable"
        result = {"ok": False, "error": "unreadable"}
    sys.stdout.write(json.dumps(result) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
