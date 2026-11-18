import html.parser
import re
import string

import pytest

from src.i18n import (
    DEFAULT_LANGUAGE,
    LEGACY_LANGUAGE,
    REQUIRED_PLACEHOLDERS,
    TABLES,
    t,
    table_language,
)

ALLOWED_TAGS = {"b", "i", "u", "s", "code", "pre", "a"}


def _fields(value: str) -> set:
    return {name for _, name, _, _ in string.Formatter().parse(value) if name}


def test_tables_have_identical_keys():
    assert set(TABLES) == {"en", "uk"}
    keys = set(REQUIRED_PLACEHOLDERS)
    for lang, table in TABLES.items():
        assert set(table) == keys, lang


def test_required_placeholders_present():
    for lang, table in TABLES.items():
        for key, value in table.items():
            assert _fields(value) == set(REQUIRED_PLACEHOLDERS[key]), (lang, key)


class _TagChecker(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack = []
        self.bad = []

    def handle_starttag(self, tag, attrs):
        if tag not in ALLOWED_TAGS:
            self.bad.append(tag)
        self.stack.append(tag)

    def handle_endtag(self, tag):
        if not self.stack or self.stack[-1] != tag:
            self.bad.append("/" + tag)
        else:
            self.stack.pop()


def test_tables_have_valid_tags():
    for lang, table in TABLES.items():
        for key, value in table.items():
            rendered = value.format(**{name: "x" for name in _fields(value)})
            checker = _TagChecker()
            checker.feed(rendered)
            checker.close()
            assert not checker.bad and not checker.stack, (lang, key)


def test_table_language_fallbacks():
    assert DEFAULT_LANGUAGE == "en" and LEGACY_LANGUAGE == "uk"
    assert table_language(None) == "uk"
    assert table_language("uk") == "uk"
    assert table_language("en") == "en"
    assert table_language("de") == "en"


def test_english_has_no_cyrillic():
    for key, value in TABLES["en"].items():
        assert not re.search("[Ѐ-ӿ]", value), key


def test_english_is_domain_neutral():
    for key, value in TABLES["en"].items():
        assert "coach" not in value.replace("JobSpy Coach", "").lower(), key


def test_t_formats_and_falls_back():
    assert t("digest_part_header", "en", part="2") == TABLES["en"]["digest_part_header"].format(part="2")
    assert t("digest_part_header", None, part="2") == TABLES["uk"]["digest_part_header"].format(part="2")
    assert t("digest_part_header", "fr", part="2") == t("digest_part_header", "en", part="2")
    with pytest.raises(KeyError):
        t("no_such_key", "en")
