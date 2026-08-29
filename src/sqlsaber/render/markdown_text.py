"""The single markdown formatter plus the pipe serializer."""

from __future__ import annotations

import json
import string
from collections.abc import Sequence

from .blocks import (
    Ansi,
    Block,
    Cell,
    Code,
    Image,
    KeyValues,
    Md,
    Note,
    Panel,
    Table,
    TextBlock,
    is_text_block,
)


def markdown_source(block: TextBlock) -> str:
    """Render one text-bearing block to markdown source.

    This output is both what a pipe receives and what the saber-tui
    ``Markdown`` component is constructed with.

    Args:
        block: A text-bearing block.

    Returns:
        Markdown source without a trailing extra blank line.
    """
    if isinstance(block, Md):
        return block.text
    if isinstance(block, Note):
        if block.label:
            return f"**{escape_cell(block.label)}:** {block.text}"
        return block.text
    if isinstance(block, Code):
        return fence(block.text, block.language)
    if isinstance(block, Table):
        return _table_markdown(block)
    return _key_values_markdown(block)


def escape_cell(value: Cell) -> str:
    """Sanitize control leftovers, collapse newlines, escape markdown punctuation.

    Args:
        value: A table or key-value cell.

    Returns:
        A single-line markdown-safe string.
    """
    text = stringify_cell(value)
    text = text.replace("\n", " ").replace("\t", " ").replace("\r", " ")
    return "".join(
        "`\\`" if char == "\\" else f"\\{char}" if char in string.punctuation else char
        for char in text
    )


def fence(text: str, language: str) -> str:
    """Wrap text in a fence wider than any backtick run in the body.

    Args:
        text: Source text.
        language: Optional language tag.

    Returns:
        A fenced markdown code block.
    """
    longest = 0
    run = 0
    for char in text:
        if char == "`":
            run += 1
            longest = max(longest, run)
        else:
            run = 0
    ticks = "`" * max(3, longest + 1)
    body = text if text.endswith("\n") or text == "" else f"{text}\n"
    return f"{ticks}{language}\n{body}{ticks}"


def md_of(blocks: Sequence[Block]) -> str:
    """Unstyled markdown for pipes and non-TTY stdout.

    Text blocks delegate to ``markdown_source``. Image and Ansi use their
    fallback markdown. Panel becomes an optional title plus children.
    Blocks are joined with a blank line.

    Args:
        blocks: Blocks to serialize.

    Returns:
        Markdown source. A single text block equals ``markdown_source``.
    """
    parts = [_one(block) for block in blocks]
    parts = [part for part in parts if part]
    return "\n\n".join(parts)


def _one(block: Block) -> str:
    if is_text_block(block):
        return markdown_source(block)
    if isinstance(block, Image):
        return block.alt_markdown
    if isinstance(block, Ansi):
        return block.fallback_markdown
    if isinstance(block, Panel):
        children = md_of(block.blocks)
        if block.title:
            title = f"**{block.title}**"
            return f"{title}\n\n{children}" if children else title
        return children
    return ""


def _table_markdown(block: Table) -> str:
    lines: list[str] = []
    if block.caption:
        lines.append(f"**{block.caption}**")
        lines.append("")
    if not block.rows or not block.columns:
        lines.append(f"*{block.empty_text}*")
        return "\n".join(lines).rstrip()

    total_columns = len(block.columns) + block.omitted_columns
    if block.omitted_columns:
        lines.append(
            f"*Showing first {len(block.columns)} of {total_columns} columns.*"
        )
        lines.append("")

    headers = [escape_cell(col.header) for col in block.columns]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in block.columns) + " |")
    shown = block.rows[: block.max_rows]
    for row in shown:
        cells = [escape_cell(row.get(col.field)) for col in block.columns]
        lines.append("| " + " | ".join(cells) + " |")
    omitted_rows = len(block.rows) - len(shown)
    if omitted_rows > 0:
        lines.append("")
        lines.append(f"*... and {omitted_rows} more rows.*")
    return "\n".join(lines)


def _key_values_markdown(block: KeyValues) -> str:
    lines: list[str] = []
    if block.caption:
        lines.append(f"**{block.caption}**")
        lines.append("")
    for key, value in block.pairs:
        lines.append(f"- **{key}**: {stringify_cell(value)}")
    return "\n".join(lines)


def stringify_cell(value: Cell) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    if isinstance(value, int | float):
        return str(value)
    return json.dumps(value, ensure_ascii=False)
