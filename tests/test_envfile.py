"""``merge_env_file``: line-preserving, atomic, mode 600, validated values."""
import os
import stat
from pathlib import Path

import pytest

from src.envfile import merge_env_file, read_env_value, validate_env_value

SECRET = "123456:<TOKEN>"


def _mode(path: Path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


def test_merge_env_preserves_other_lines_and_mode(tmp_path):
    path = tmp_path / ".env"
    before = (
        b"# a comment\n\nexport A=1\nTIMEZONE=UTC  # trailing\r\nTELEGRAM_BOT_TOKEN=old\n"
        b"\n# end\n"
    )
    path.write_bytes(before)
    os.chmod(path, 0o644)

    merge_env_file(path, {"TELEGRAM_BOT_TOKEN": SECRET})

    after = path.read_bytes().split(b"\n")
    assert after == before.replace(b"TELEGRAM_BOT_TOKEN=old", b"TELEGRAM_BOT_TOKEN=" + SECRET.encode()).split(b"\n")
    assert b"export A=1\n" in path.read_bytes()
    assert b"TIMEZONE=UTC  # trailing\r\n" in path.read_bytes()
    assert _mode(path) == 0o600
    assert read_env_value(path, "TELEGRAM_BOT_TOKEN") == SECRET


def test_merge_replaces_export_line_and_keeps_its_line_ending(tmp_path):
    path = tmp_path / ".env"
    path.write_bytes(b"export KEY = old\r\nOTHER=1\r\n")

    merge_env_file(path, {"KEY": "new"})

    assert path.read_bytes() == b"KEY=new\r\nOTHER=1\r\n"


def test_merge_replaces_duplicates_and_appends_missing(tmp_path):
    path = tmp_path / ".env"
    path.write_bytes(b"KEY=1\nOTHER=x\nKEY=2\n#KEY=3\nLAST=y")

    merge_env_file(path, {"KEY": "9", "NEW_ONE": "v"})

    assert path.read_bytes() == b"KEY=9\nOTHER=x\n#KEY=3\nLAST=y\nNEW_ONE=v\n"


def test_merge_creates_missing_file_0600(tmp_path):
    path = tmp_path / ".env"

    merge_env_file(path, {"A": "1", "B": ""})

    assert path.read_text() == "A=1\nB=\n"
    assert _mode(path) == 0o600
    assert read_env_value(tmp_path / "missing.env", "A") is None


def test_merge_leaves_no_temp_files(tmp_path):
    path = tmp_path / ".env"
    merge_env_file(path, {"A": "1"})
    merge_env_file(path, {"A": "2"})

    assert sorted(p.name for p in tmp_path.iterdir()) == [".env"]


@pytest.mark.parametrize(
    "value",
    ["a b", "a\nb", "a\rb", 'a"b', "a'b", "a#b", "a\\b", "a\tb", "a\x00b", "a\x7fb", "a$b", "a b"],
)
def test_merge_rejects_newline_quote_hash_space(tmp_path, value):
    path = tmp_path / ".env"
    path.write_bytes(b"KEEP=1\n")

    with pytest.raises(ValueError):
        merge_env_file(path, {"KEY": value})

    assert path.read_bytes() == b"KEEP=1\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == [".env"]


def test_validate_env_value_allows_empty_and_token_characters():
    for value in ("", SECRET, "nvapi-abc_DEF.123=", "a/b:c,d/e"):
        validate_env_value(value)


@pytest.mark.parametrize("key", ["lower", "1A", "A B", "A=B", "", "A\n"])
def test_merge_rejects_bad_keys(tmp_path, key):
    path = tmp_path / ".env"
    with pytest.raises(ValueError):
        merge_env_file(path, {key: "v"})
    assert not path.exists()


def test_merge_follows_symlink(tmp_path):
    target = tmp_path / "real.env"
    target.write_text("A=1\n")
    link = tmp_path / ".env"
    link.symlink_to(target)

    merge_env_file(link, {"A": "2"})

    assert link.is_symlink()
    assert target.read_text() == "A=2\n"
    assert _mode(target) == 0o600
