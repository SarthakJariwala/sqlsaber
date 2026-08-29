"""Tests for the block vocabulary and the one-formatter invariant."""

from __future__ import annotations

import json
from io import StringIO

import pytest

from sqlsaber.render.blocks import (
    ControlSequenceError,
    ansi,
    code,
    error,
    image,
    json_block,
    key_values,
    md,
    note,
    panel,
    success,
    table,
    warn,
)
from sqlsaber.render.html import html_of
from sqlsaber.render.markdown_text import fence, markdown_source, md_of
from sqlsaber.render.surface import AskConfirm, PromptUnavailable
from sqlsaber.render.terminal import PlainSurface
from sqlsaber.render.tui_blocks import components_for, markdown_component_source


def test_error_note_owns_the_stable_handle() -> None:
    block = error("could not connect")
    assert markdown_source(block) == "**Error:** could not connect"


def test_sql_error_label() -> None:
    block = error("syntax error", label="SQL error")
    assert markdown_source(block) == "**SQL error:** syntax error"


def test_md_rejects_ansi() -> None:
    with pytest.raises(ControlSequenceError):
        md("hello\x1b[31mred")


def test_md_rejects_c0() -> None:
    with pytest.raises(ControlSequenceError):
        note("bell\x07")


def test_md_allows_newline_and_tab() -> None:
    block = md("a\tb\nc")
    assert markdown_source(block) == "a\tb\nc"


def test_image_requires_alt_markdown() -> None:
    block = image(b"\x89PNG", "image/png", filename="plot.png")
    assert block.alt_markdown == "*[image: plot.png]*"
    with pytest.raises(ControlSequenceError):
        image(b"x", "image/png", alt_markdown="\x1b[0m")


def test_ansi_fallback_is_control_free() -> None:
    block = ansi("\x1b[31mchart\x1b[0m")
    assert "\x1b" not in block.fallback_markdown
    assert "chart" in block.fallback_markdown


def test_key_values_are_markdown_lists() -> None:
    block = key_values({"Connected to": "verify-sqlite (sqlite)", "Model": "gpt"})
    source = markdown_source(block)
    assert source == ("- **Connected to**: verify-sqlite (sqlite)\n- **Model**: gpt")


def test_table_github_pipes_and_truncation() -> None:
    rows = [{"a": i, "b": i * 2} for i in range(25)]
    block = table(rows, caption="Results (25 rows):", max_rows=20)
    source = markdown_source(block)
    assert source.startswith("**Results (25 rows):**")
    assert "| a | b |" in source or "| a |" in source
    assert "*... and 5 more rows.*" in source


def test_table_column_cap() -> None:
    row = {f"c{i}": i for i in range(20)}
    block = table([row], max_columns=15)
    assert block.omitted_columns == 5
    assert "Showing first 15 of 20 columns" in markdown_source(block)


def test_empty_table() -> None:
    block = table([], empty_text="0 rows returned")
    assert markdown_source(block) == "*0 rows returned*"


def test_fence_widens_past_backticks() -> None:
    assert fence("``` inner", "sql").startswith("````sql")


def test_json_block_is_fenced_json() -> None:
    block = json_block({"ok": True})
    source = markdown_source(block)
    assert source.startswith("```json")
    assert json.loads(source.split("\n", 1)[1].rsplit("```", 1)[0]) == {"ok": True}


def test_md_of_single_text_block_equals_markdown_source() -> None:
    block = error("boom")
    assert md_of([block]) == markdown_source(block)


def test_md_of_joins_with_blank_lines() -> None:
    text = md_of([note("one"), note("two")])
    assert text == "one\n\ntwo"


def test_panel_title_then_children() -> None:
    block = panel([md("body")], title="Analysis notebook")
    assert md_of([block]) == "**Analysis notebook**\n\nbody"


def test_html_error_class_contract() -> None:
    html = html_of([error("nope")])
    assert 'class="sql-error"' in html
    assert "<strong>Error:</strong>" in html
    assert "nope" in html


def test_html_table_class_contract() -> None:
    html = html_of([table([{"n": 1}], caption="Results (1 rows):")])
    assert 'class="sql-results"' in html
    assert 'class="table-wrapper"' in html
    assert 'class="result-count"' in html


def test_html_code_language_class() -> None:
    html = html_of([code("select 1", "sql")])
    assert 'class="language-sql"' in html


def test_assert_single_formatter_text_blocks() -> None:
    from sqlsaber.theme.styles import get_styles

    styles = get_styles()
    cases = [
        md("**hello**"),
        error("boom"),
        warn("careful", label="Warning"),
        success("ok"),
        code("print(1)", "python"),
        table([{"name": "users", "schema": "public"}], caption="Tables (1 total)"),
        key_values({"limit": 10}),
    ]
    for block in cases:
        source = markdown_source(block)
        assert md_of([block]) == source
        components = components_for([block], styles)
        assert components
        constructed = markdown_component_source(components[0])
        assert constructed == source


@pytest.mark.asyncio
async def test_plain_surface_writes_markdown() -> None:
    buf = StringIO()
    surface = PlainSurface(buf)
    surface.emit(error("could not connect"))
    assert buf.getvalue() == "**Error:** could not connect\n"


@pytest.mark.asyncio
async def test_plain_surface_buffers_stream() -> None:
    buf = StringIO()
    surface = PlainSurface(buf)
    stream = surface.stream()
    stream.append("hel")
    stream.append("lo")
    assert buf.getvalue() == ""
    stream.close()
    assert buf.getvalue() == "hello\n"


@pytest.mark.asyncio
async def test_plain_surface_discard_drops_buffer() -> None:
    buf = StringIO()
    surface = PlainSurface(buf)
    stream = surface.stream()
    stream.append("secret")
    stream.discard()
    assert buf.getvalue() == ""


@pytest.mark.asyncio
async def test_plain_surface_status_is_noop() -> None:
    buf = StringIO()
    surface = PlainSurface(buf)
    surface.status("Crunching data...")
    assert buf.getvalue() == ""


@pytest.mark.asyncio
async def test_plain_surface_ask_yes_short_circuit() -> None:
    buf = StringIO()
    surface = PlainSurface(buf)
    assert await surface.ask(AskConfirm("Reset?", assume_yes=True)) is True


@pytest.mark.asyncio
async def test_plain_surface_ask_raises() -> None:
    buf = StringIO()
    surface = PlainSurface(buf)
    with pytest.raises(PromptUnavailable) as exc:
        await surface.ask(
            AskConfirm("Reset?", unavailable_hint="saber theme reset --yes")
        )
    assert "saber theme reset --yes" in str(exc.value)


@pytest.mark.asyncio
async def test_reset_io_binds_stringio() -> None:
    from sqlsaber.render import cli_out, reset_io

    buf = StringIO()
    reset_io(stdout=buf, stderr=StringIO(), tty=False)
    try:
        cli_out().emit(error("piped"))
        assert "**Error:** piped" in buf.getvalue()
    finally:
        reset_io()
