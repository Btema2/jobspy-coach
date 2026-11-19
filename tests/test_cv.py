"""CV reading: PDFs are hostile input, parsed only in a child process with limits; text is masked."""
import inspect
import io
import json
import subprocess
import sys
import time
import zlib

import pytest

from src.bot import cv as cv_module
from src.bot.cv import (
    MAX_CV_BYTES,
    MAX_CV_CHARS,
    MAX_CV_PAGES,
    PDF_TIMEOUT_SECONDS,
    CvError,
    CvText,
    build_preview,
    read_cv,
    read_text_file,
)

PASSWORD_REPLY = "This PDF is password-protected. Send an unprotected copy or paste the text."
NO_TEXT_REPLY = "This PDF has no readable text (it looks scanned). Paste the text or send a .txt file."
UNREADABLE_REPLY = "This PDF could not be read. Paste the text or send a .txt file."

CV_LINES = [
    "Sample Person",
    "sample.person@example.com",
    "+1 415 555 0101",
    "https://www.example.com/in/sample",
    "Experience",
    "Led a small support team and wrote onboarding material for new colleagues over several years.",
    "Built a weekly review routine and trained twelve people in structured feedback conversations.",
    "Education",
    "Bachelor degree in applied psychology with a thesis on adult learning and motivation.",
]


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_pdf(pages=1, lines=None, junk: bytes = b"", content_override=None) -> bytes:
    """A minimal hand-written PDF: ``pages`` identical pages with Helvetica text lines."""
    lines = CV_LINES if lines is None else lines
    objects = {}
    kids = []
    next_id = 4
    for _ in range(pages):
        page_id, content_id = next_id, next_id + 1
        next_id += 2
        kids.append(f"{page_id} 0 R")
        if content_override is not None:
            stream = content_override
            filt = b"/Filter /FlateDecode "
        else:
            ops = ["BT /F1 11 Tf 72 740 Td 14 TL"]
            for line in lines:
                ops.append(f"({_escape(line)}) Tj T*")
            ops.append("ET")
            stream = "\n".join(ops).encode("latin-1")
            filt = b""
        objects[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content_id} 0 R "
            f"/Resources << /Font << /F1 3 0 R >> >> >>"
        ).encode()
        objects[content_id] = (
            b"<< " + filt + b"/Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"
        )
    objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objects[2] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {pages} >>".encode()
    objects[3] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    out = bytearray(junk + b"%PDF-1.4\n")
    offsets = {}
    for num in sorted(objects):
        offsets[num] = len(out)
        out += f"{num} 0 obj\n".encode() + objects[num] + b"\nendobj\n"
    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode() + b"0000000000 65535 f \n"
    for num in sorted(objects):
        out += f"{offsets[num] - len(junk):010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_at - len(junk)}\n%%EOF\n".encode()
    )
    return bytes(out)


def encrypted_pdf(user_password: str, owner_password: str) -> bytes:
    from pypdf import PdfWriter

    writer = PdfWriter(clone_from=io.BytesIO(build_pdf()))
    writer.encrypt(user_password=user_password, owner_password=owner_password, algorithm="RC4-128")
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def reply_of(exc_info) -> str:
    return exc_info.value.reply


# --- the happy path -----------------------------------------------------------------------------------

def test_normal_pdf_gives_masked_text_without_header():
    cv = read_cv(build_pdf())
    assert isinstance(cv, CvText)
    assert cv.pages == 1
    assert "onboarding material" in cv.text
    assert "sample.person@example.com" not in cv.text
    assert "555" not in cv.text
    assert "example.com" not in cv.text
    assert "Sample Person" not in cv.text  # header block is dropped
    assert "Experience" in cv.text


def test_read_cv_takes_no_file_name():
    params = inspect.signature(read_cv).parameters
    assert list(params) == ["data", "timeout"]
    assert not any("name" in p or "path" in p for p in params)


def test_constants_match_the_spec():
    assert MAX_CV_BYTES == 5 * 1024 * 1024
    assert MAX_CV_PAGES == 10
    assert MAX_CV_CHARS == 20000
    assert PDF_TIMEOUT_SECONDS == 15.0


# --- %PDF- marker and size --------------------------------------------------------------------------

def test_junk_before_marker_within_1024_bytes_is_accepted():
    cv = read_cv(build_pdf(junk=b"\x00junk\n" * 100))  # 600 bytes
    assert cv.pages == 1


def test_junk_beyond_1024_bytes_is_refused_before_spawning(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("no child may be started")

    monkeypatch.setattr(cv_module.subprocess, "run", boom)
    with pytest.raises(CvError) as exc:
        read_cv(b"x" * 1100 + build_pdf())
    assert reply_of(exc) == UNREADABLE_REPLY


def test_oversize_is_refused_before_spawning(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("no child may be started")

    monkeypatch.setattr(cv_module.subprocess, "run", boom)
    with pytest.raises(CvError) as exc:
        read_cv(b"%PDF-" + b"0" * MAX_CV_BYTES)
    assert "5 MB" in reply_of(exc)


def test_non_pdf_bytes_are_unreadable_without_spawning(monkeypatch):
    monkeypatch.setattr(cv_module.subprocess, "run", lambda *a, **k: pytest.fail("spawned"))
    with pytest.raises(CvError) as exc:
        read_cv(b"This is just a text file pretending to be a PDF." * 50)
    assert reply_of(exc) == UNREADABLE_REPLY


def test_marker_present_but_garbage_body_is_unreadable():
    with pytest.raises(CvError) as exc:
        read_cv(b"%PDF-1.7\nthis is not a real pdf body at all\n" * 20)
    assert reply_of(exc) == UNREADABLE_REPLY


# --- page, encryption and text rules ----------------------------------------------------------------

def test_ten_pages_are_accepted_and_eleven_refused():
    assert read_cv(build_pdf(pages=10)).pages == 10
    with pytest.raises(CvError) as exc:
        read_cv(build_pdf(pages=11))
    assert "10 pages" in reply_of(exc)


def test_user_password_gives_the_password_reply():
    with pytest.raises(CvError) as exc:
        read_cv(encrypted_pdf("secret", "owner"))
    assert reply_of(exc) == PASSWORD_REPLY


def test_owner_password_only_opens_with_empty_password():
    cv = read_cv(encrypted_pdf("", "owner"))
    assert cv.pages == 1
    assert "onboarding material" in cv.text


def test_blank_page_gives_the_no_text_reply():
    with pytest.raises(CvError) as exc:
        read_cv(build_pdf(lines=[]))
    assert reply_of(exc) == NO_TEXT_REPLY


def test_fewer_than_200_non_whitespace_characters_is_no_text():
    with pytest.raises(CvError) as exc:
        read_cv(build_pdf(lines=["Experience", "Short text only. " * 3]))
    assert reply_of(exc) == NO_TEXT_REPLY


def test_text_is_capped_at_max_chars():
    line = "Experience with many words in one long line of text. " * 4
    cv = read_cv(build_pdf(pages=10, lines=["Experience"] + [line] * 40))
    assert len(cv.text) <= MAX_CV_CHARS


# --- hostile input ----------------------------------------------------------------------------------

def test_hanging_child_is_killed_at_the_timeout(monkeypatch):
    monkeypatch.setattr(cv_module, "_CHILD_ARGV", [sys.executable, "-I", "-c", "import time; time.sleep(60)"])
    start = time.monotonic()
    with pytest.raises(CvError) as exc:
        read_cv(build_pdf(), timeout=1.0)
    assert time.monotonic() - start < 10
    assert reply_of(exc) == UNREADABLE_REPLY


def test_decompression_bomb_is_unreadable_and_parent_survives():
    # ~700 MB of zeros in a Flate stream (about 0.7 MB compressed) exceeds the child's 512 MB address space.
    comp = zlib.compressobj(9)
    chunk = b"\x00" * (1024 * 1024)
    parts = [comp.compress(chunk) for _ in range(700)]
    parts.append(comp.flush())
    bomb = build_pdf(content_override=b"".join(parts))
    assert len(bomb) < MAX_CV_BYTES
    with pytest.raises(CvError) as exc:
        read_cv(bomb)
    assert reply_of(exc) == UNREADABLE_REPLY
    assert read_cv(build_pdf()).pages == 1  # the parent is still fine


def test_non_zero_exit_and_bad_json_are_unreadable(monkeypatch):
    for code in ("raise SystemExit(3)", "print('not json')", "print('[1, 2]')", "print('{\"ok\": true}')"):
        monkeypatch.setattr(cv_module, "_CHILD_ARGV", [sys.executable, "-I", "-c", code])
        with pytest.raises(CvError) as exc:
            read_cv(build_pdf())
        assert reply_of(exc) == UNREADABLE_REPLY


def test_child_stderr_never_reaches_the_parent(monkeypatch, capfd):
    code = "import sys; sys.stderr.write('SECRET-STDERR'); print('{\"ok\": false, \"error\": \"unreadable\"}')"
    monkeypatch.setattr(cv_module, "_CHILD_ARGV", [sys.executable, "-I", "-c", code])
    with pytest.raises(CvError):
        read_cv(build_pdf())
    captured = capfd.readouterr()
    assert "SECRET-STDERR" not in captured.out + captured.err


def test_child_is_started_without_preexec_with_minimal_env_and_devnull(monkeypatch):
    seen = {}
    real_run = subprocess.run

    def spy(argv, **kwargs):
        seen["argv"], seen["kwargs"] = argv, kwargs
        return real_run(argv, **kwargs)

    monkeypatch.setattr(cv_module.subprocess, "run", spy)
    read_cv(build_pdf())
    kwargs = seen["kwargs"]
    assert seen["argv"][1:] == ["-I", "-m", "src.bot.pdf_text"]
    assert kwargs["stderr"] == subprocess.DEVNULL
    assert "preexec_fn" not in kwargs
    assert set(kwargs["env"]) <= {"PATH", "LANG"}
    assert kwargs["timeout"] == PDF_TIMEOUT_SECONDS
    assert kwargs["input"].startswith(b"%PDF-")


def test_logs_carry_no_text(monkeypatch, caplog):
    monkeypatch.setattr(cv_module, "_CHILD_ARGV", [sys.executable, "-I", "-c", "import time; time.sleep(60)"])
    with caplog.at_level("DEBUG"):
        with pytest.raises(CvError):
            read_cv(build_pdf(), timeout=0.5)
    assert "Sample Person" not in caplog.text and "example.com" not in caplog.text


# --- the child's own protocol -----------------------------------------------------------------------

def test_child_prints_one_json_line_and_exits_zero_even_on_failure():
    for data, expected in ((build_pdf(), True), (b"%PDF-garbage", False), (build_pdf(pages=11), False)):
        proc = subprocess.run(
            [sys.executable, "-I", "-m", "src.bot.pdf_text"], input=data, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, timeout=30,
        )
        assert proc.returncode == 0
        lines = proc.stdout.decode().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["ok"] is expected


def test_child_error_codes():
    def run(data):
        proc = subprocess.run(
            [sys.executable, "-I", "-m", "src.bot.pdf_text"], input=data, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, timeout=30,
        )
        return json.loads(proc.stdout.decode())

    assert run(build_pdf(pages=11))["error"] == "too_many_pages"
    assert run(encrypted_pdf("secret", "owner"))["error"] == "encrypted"
    assert run(b"%PDF-garbage")["error"] == "unreadable"


# --- text files and preview -------------------------------------------------------------------------

def test_read_text_file_masks_like_a_pdf():
    text = "\n".join(CV_LINES)
    from_pdf = read_cv(build_pdf())
    from_txt = read_text_file(text)
    assert from_txt.pages == 0
    assert from_txt.text == from_pdf.text
    assert "sample.person@example.com" not in from_txt.text


def test_read_text_file_caps_length():
    cv = read_text_file("Experience\n" + "word " * 10000)
    assert len(cv.text) <= MAX_CV_CHARS


def test_preview_has_counts_and_only_masked_text():
    cv = read_cv(build_pdf())
    preview = build_preview(cv)
    assert preview.startswith("Read 1 pages, %d characters. First lines: " % len(cv.text))
    for forbidden in ("@", "555", "example.com", "http", "Sample Person"):
        assert forbidden not in preview
    assert "\n" not in preview
    assert len(preview.split("First lines: ", 1)[1]) <= 303


def test_preview_for_text_file_omits_pages():
    preview = build_preview(read_text_file("\n".join(CV_LINES)))
    assert preview.startswith("Read ") and "pages" not in preview
    assert "characters. First lines: " in preview
