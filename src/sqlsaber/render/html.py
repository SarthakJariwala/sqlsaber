"""HTML export serializer. No terminal, no Console, no Styles."""

from __future__ import annotations

from collections.abc import Sequence
from html import escape

from .blocks import (
    Ansi,
    Block,
    Code,
    Image,
    KeyValues,
    Md,
    Note,
    Panel,
    Table,
)
from .markdown_text import stringify_cell

HTML_TABLE_MAX_ROWS = 100


def html_of(blocks: Sequence[Block]) -> str:
    """HTML for thread export.

    Preserves the class contract: ``sql-error``, ``sql-results``,
    ``table-wrapper``, ``result-count``, ``language-*``, ``markdown-content``.

    Args:
        blocks: Blocks to serialize.

    Returns:
        Concatenated HTML fragments.
    """
    return "".join(_block_html(block) for block in blocks)


def _block_html(block: Block) -> str:
    if isinstance(block, Note) and block.role == "error":
        label = escape(block.label or "Error")
        return (
            f'<div class="sql-error"><strong>{label}:</strong> '
            f"{escape(block.text)}</div>"
        )
    if isinstance(block, Note):
        body = escape(block.text)
        if block.label:
            body = f"<strong>{escape(block.label)}:</strong> {body}"
        return f'<p class="result-count">{body}</p>'
    if isinstance(block, Md):
        return f'<div class="markdown-content">{escape(block.text)}</div>'
    if isinstance(block, Code):
        class_attr = (
            f' class="language-{escape(block.language)}"' if block.language else ""
        )
        return f"<pre><code{class_attr}>{escape(block.text)}</code></pre>"
    if isinstance(block, Table):
        return _table_html(block)
    if isinstance(block, KeyValues):
        items = "".join(
            f"<li><strong>{escape(key)}</strong>: {escape(stringify_cell(value))}</li>"
            for key, value in block.pairs
        )
        title = (
            f'<p class="result-count">{escape(block.caption)}</p>'
            if block.caption
            else ""
        )
        return f"{title}<ul>{items}</ul>"
    if isinstance(block, Image):
        return f'<p class="result-count">{escape(block.alt_markdown)}</p>'
    if isinstance(block, Ansi):
        return f'<pre class="viz-chart">{escape(block.fallback_markdown)}</pre>'
    if isinstance(block, Panel):
        title = (
            f'<p class="result-count">{escape(block.title)}</p>' if block.title else ""
        )
        return f"{title}{html_of(block.blocks)}"
    return ""


def _table_html(block: Table) -> str:
    if not block.rows or not block.columns:
        return f'<p class="result-count">{escape(block.empty_text)}</p>'
    cap = max(block.max_rows, HTML_TABLE_MAX_ROWS)
    shown = block.rows[:cap]
    header = "".join(f"<th>{escape(col.header)}</th>" for col in block.columns)
    body = []
    for row in shown:
        cells = "".join(
            f"<td>{escape(stringify_cell(row.get(col.field)))}</td>"
            for col in block.columns
        )
        body.append(f"<tr>{cells}</tr>")
    title = (
        f'<p class="result-count">{escape(block.caption)}</p>' if block.caption else ""
    )
    count = ""
    if len(block.rows) > cap:
        count = f'<p class="result-count">Showing {cap} of {len(block.rows)} rows</p>'
    return (
        f'{title}{count}<div class="table-wrapper">'
        f'<table class="sql-results"><thead><tr>{header}</tr></thead>'
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )
