import re

import pytest

from src.bot.render import CHUNK_LIMIT, PAGE_SIZE, chunk_html, esc, paginate, render_apply, utf16_len
from src.profile.ops import ApplyResult


def test_esc_escapes_html():
    assert esc("<b>&amp; R&D</b> \"q\"") == '&lt;b&gt;&amp;amp; R&amp;D&lt;/b&gt; "q"'
    assert esc(5) == "5"


def test_utf16_len_counts_surrogates():
    assert utf16_len("a") == 1
    assert utf16_len("\U0001F600") == 2
    assert utf16_len("é") == 1


def test_chunk_html_keeps_short_text_whole():
    assert chunk_html("one\ntwo") == ["one\ntwo"]
    assert chunk_html("") == []


def test_chunk_html_utf16_and_entities():
    long_line = esc("&amp;\U0001F600<b>" * 600)
    assert len(long_line) > 5000
    lines = [long_line] + [f"line {i} &lt;x&gt;" for i in range(300)]
    text = "\n".join(lines)

    chunks = chunk_html(text)

    assert len(chunks) > 1
    for chunk in chunks:
        assert utf16_len(chunk) <= CHUNK_LIMIT
        # no chunk ends or starts inside an entity or a tag
        assert not re.search(r"&[#\w]*$", chunk)
        assert chunk.count("<") == chunk.count(">")
        assert "&" not in chunk.replace("&amp;", "").replace("&lt;", "").replace("&gt;", "")
    # hard-split pieces of the long line are contiguous; whole lines survive
    assert "".join(chunks[:2]).startswith(long_line[:100])
    assert "\n".join(chunks).replace("\n", "").endswith("line 299 &lt;x&gt;")
    rejoined = "".join(c.replace("\n", "") for c in chunks)
    assert rejoined == text.replace("\n", "")


_ATOMS = [
    "&amp;",
    "<b>",
    '<a href="https://example.test/?a=1&amp;b=2&amp;c=3">',
    "&#128512;",
]


def _assert_well_cut(chunks, text):
    assert "".join(chunks) == text
    for chunk in chunks:
        assert utf16_len(chunk) <= CHUNK_LIMIT
        assert chunk.count("<") == chunk.count(">"), chunk[-80:]
        assert "&" not in re.sub(r"&#?\w+;", "", chunk), chunk[-80:]


@pytest.mark.parametrize("atom", _ATOMS)
def test_chunk_html_backs_off_when_the_cut_lands_inside_an_atom(atom):
    # Sweep the padding so the 3,800 cut falls at every offset inside the atom (and just outside it).
    for inside in range(len(atom) + 2):
        text = "x" * (CHUNK_LIMIT - inside) + atom + "y" * 40
        chunks = chunk_html(text)
        assert len(chunks) == 2
        _assert_well_cut(chunks, text)


def test_chunk_html_never_splits_inside_a_tag_whose_attribute_has_entities():
    tag = '<a href="https://example.test/?a=1&amp;b=2&amp;c=3&amp;d=4">'
    for inside in range(1, len(tag)):
        text = "x" * (CHUNK_LIMIT - inside) + tag + "label</a>"
        _assert_well_cut(chunk_html(text), text)


def test_chunk_html_counts_surrogate_pairs_as_two_units():
    for pad in (CHUNK_LIMIT - 3, CHUNK_LIMIT - 2, CHUNK_LIMIT - 1):
        text = "x" * pad + "\U0001F600" * 3
        chunks = chunk_html(text)
        _assert_well_cut(chunks, text)
        assert len(chunks) == 2
    # An emoji that would straddle the limit moves whole to the next chunk.
    chunks = chunk_html("x" * (CHUNK_LIMIT - 1) + "\U0001F600")
    assert chunks == ["x" * (CHUNK_LIMIT - 1), "\U0001F600"]


def test_chunk_html_packs_lines_up_to_the_limit():
    line = "a" * 1000
    chunks = chunk_html("\n".join([line] * 5))
    assert [c.count("a") for c in chunks] == [3000, 2000]
    assert all(utf16_len(c) <= CHUNK_LIMIT for c in chunks)


def test_paginate_clamps_page():
    lines = [str(i) for i in range(PAGE_SIZE * 2 + 5)]
    page_lines, page, pages = paginate(lines, 1)
    assert (page, pages, len(page_lines)) == (1, 3, PAGE_SIZE)
    page_lines, page, pages = paginate(lines, 99)
    assert (page, pages, page_lines) == (3, 3, lines[-5:])
    _, page, _ = paginate(lines, -4)
    assert page == 1
    assert paginate([], 1) == ([], 1, 1)


def test_render_apply_escapes_diff_and_warnings():
    result = ApplyResult(version=7, diff=["add role <b>&amp; R&D"], warnings=["w <x>"])
    out = render_apply(result, extra=["note & more"])
    lines = out.split("\n")
    assert lines[0] == "<b>Profile v7</b>"
    assert "add role &lt;b&gt;&amp;amp; R&amp;D" in lines
    assert "Warning: w &lt;x&gt;" in lines
    assert lines[-1] == "note &amp; more"
