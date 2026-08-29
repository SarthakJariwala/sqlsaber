"""Immutable document vocabulary. Stdlib only; safe on the --help path."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypeGuard

type Role = Literal["primary", "accent", "success", "warning", "error", "info", "muted"]
type Cell = str | int | float | bool | None

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")
_ANSI_RE = re.compile(
    r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))"
)


class ControlSequenceError(ValueError):
    """Raised when markdown text contains terminal control sequences."""


def reject_controls(text: str, *, field: str) -> None:
    """Reject ANSI and C0/C1 controls. Newline and tab remain valid.

    Args:
        text: Candidate markdown or fallback text.
        field: Name used in the error message.

    Raises:
        ControlSequenceError: If a forbidden character or escape is present.
    """
    if _ANSI_RE.search(text) or _CONTROL_RE.search(text):
        raise ControlSequenceError(
            f"{field} must not contain ANSI or terminal control sequences"
        )


def strip_ansi(text: str) -> str:
    """Remove ANSI CSI/OSC sequences. Used to derive Ansi fallbacks."""
    return _ANSI_RE.sub("", text)


@dataclass(frozen=True, slots=True)
class Md:
    """Free-form markdown. ``role`` tints the whole block on styled sinks."""

    text: str
    role: Role | None = None

    def __post_init__(self) -> None:
        reject_controls(self.text, field="Md.text")


@dataclass(frozen=True, slots=True)
class Note:
    """A labelled line such as ``**Error:** could not connect``."""

    text: str
    role: Role = "info"
    label: str | None = None

    def __post_init__(self) -> None:
        reject_controls(self.text, field="Note.text")
        if self.label is not None:
            reject_controls(self.label, field="Note.label")


@dataclass(frozen=True, slots=True)
class Code:
    text: str
    language: str = ""

    def __post_init__(self) -> None:
        reject_controls(self.text, field="Code.text")
        reject_controls(self.language, field="Code.language")


@dataclass(frozen=True, slots=True)
class Column:
    field: str
    header: str
    role: Role | None = None


@dataclass(frozen=True, slots=True)
class Table:
    """Row-oriented tabular data. Serializers apply ``max_rows``."""

    rows: tuple[Mapping[str, Cell], ...]
    columns: tuple[Column, ...]
    caption: str | None = None
    max_rows: int = 20
    empty_text: str = "No results"
    omitted_columns: int = 0

    def __post_init__(self) -> None:
        if self.caption is not None:
            reject_controls(self.caption, field="Table.caption")
        reject_controls(self.empty_text, field="Table.empty_text")


@dataclass(frozen=True, slots=True)
class KeyValues:
    pairs: tuple[tuple[str, Cell], ...]
    caption: str | None = None

    def __post_init__(self) -> None:
        if self.caption is not None:
            reject_controls(self.caption, field="KeyValues.caption")
        for key, _value in self.pairs:
            reject_controls(key, field="KeyValues.key")


@dataclass(frozen=True, slots=True)
class Image:
    data: bytes
    mime_type: str
    alt_markdown: str
    filename: str | None = None
    max_width_cells: int | None = None

    def __post_init__(self) -> None:
        reject_controls(self.alt_markdown, field="Image.alt_markdown")
        reject_controls(self.mime_type, field="Image.mime_type")
        if self.filename is not None:
            reject_controls(self.filename, field="Image.filename")


@dataclass(frozen=True, slots=True)
class Ansi:
    """Pre-rendered ANSI art (plotext). Pipes emit ``fallback_markdown``."""

    text: str
    fallback_markdown: str

    def __post_init__(self) -> None:
        reject_controls(self.fallback_markdown, field="Ansi.fallback_markdown")


@dataclass(frozen=True, slots=True)
class Panel:
    """Grouping with an optional title. Styled sinks tint a background box."""

    blocks: tuple[Block, ...]
    title: str | None = None
    role: Role | None = None

    def __post_init__(self) -> None:
        if self.title is not None:
            reject_controls(self.title, field="Panel.title")


type Block = Md | Note | Code | Table | KeyValues | Image | Ansi | Panel
type TextBlock = Md | Note | Code | Table | KeyValues


def md(text: str, *, role: Role | None = None) -> Md:
    """Build a free-form markdown block.

    Args:
        text: Markdown source.
        role: Optional theme role applied by styled sinks.

    Returns:
        A validated ``Md`` block.
    """
    return Md(text, role=role)


def note(text: str, *, role: Role = "info", label: str | None = None) -> Note:
    """Build a labelled note.

    Args:
        text: Body text after the optional label.
        role: Theme role.
        label: Prefix rendered as ``**label:**``.

    Returns:
        A validated ``Note`` block.
    """
    return Note(text, role=role, label=label)


def error(text: str, *, label: str = "Error") -> Note:
    """Build ``**Error:** text`` (or a custom label), tinted error.

    Args:
        text: Error body.
        label: Prefix. Defaults to ``Error`` so pipes keep the stable handle.

    Returns:
        A ``Note`` with role ``error``.
    """
    return Note(text, role="error", label=label)


def warn(text: str, *, label: str | None = None) -> Note:
    """Build a warning note.

    Args:
        text: Body text.
        label: Optional prefix.

    Returns:
        A ``Note`` with role ``warning``.
    """
    return Note(text, role="warning", label=label)


def success(text: str, *, label: str | None = None) -> Note:
    """Build a success note.

    Args:
        text: Body text.
        label: Optional prefix.

    Returns:
        A ``Note`` with role ``success``.
    """
    return Note(text, role="success", label=label)


def code(text: str, language: str = "") -> Code:
    """Build a fenced code block.

    Args:
        text: Source text.
        language: Fence language tag.

    Returns:
        A validated ``Code`` block.
    """
    return Code(text, language=language)


def table(
    rows: Iterable[Mapping[str, Cell] | Cell],
    *,
    columns: Sequence[Column] | None = None,
    caption: str | None = None,
    max_rows: int = 20,
    max_columns: int = 15,
    empty_text: str = "No results",
) -> Table:
    """Normalize rows, infer columns when unspecified, and cap width.

    Args:
        rows: Mappings or scalars. Scalars become ``{"value": x}``.
        columns: Explicit columns. When omitted, union of keys in order.
        caption: Optional heading rendered above the table.
        max_rows: Row cap applied by every serializer.
        max_columns: Column cap applied when inferring columns.
        empty_text: Placeholder when there are no rows.

    Returns:
        A ``Table`` block.
    """
    normalized: list[dict[str, Cell]] = []
    for row in rows:
        if isinstance(row, Mapping):
            normalized.append({str(key): _as_cell(value) for key, value in row.items()})
        else:
            normalized.append({"value": _as_cell(row)})

    omitted = 0
    if columns is not None:
        cols = tuple(columns)
    else:
        keys: list[str] = []
        seen: set[str] = set()
        for row in normalized:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        if len(keys) > max_columns:
            omitted = len(keys) - max_columns
            keys = keys[:max_columns]
        cols = tuple(Column(field=key, header=key) for key in keys)

    frozen_rows = tuple(normalized)
    return Table(
        rows=frozen_rows,
        columns=cols,
        caption=caption,
        max_rows=max_rows,
        empty_text=empty_text,
        omitted_columns=omitted,
    )


def key_values(
    pairs: Mapping[str, Cell] | Sequence[tuple[str, Cell]],
    *,
    caption: str | None = None,
) -> KeyValues:
    """Build a key/value list (``- **key**: value`` in markdown).

    Args:
        pairs: Mapping or sequence of pairs.
        caption: Optional heading.

    Returns:
        A ``KeyValues`` block.
    """
    if isinstance(pairs, Mapping):
        items = tuple((str(key), _as_cell(value)) for key, value in pairs.items())
    else:
        items = tuple((str(key), _as_cell(value)) for key, value in pairs)
    return KeyValues(pairs=items, caption=caption)


def image(
    data: bytes,
    mime_type: str,
    *,
    filename: str | None = None,
    max_width_cells: int | None = None,
    alt_markdown: str | None = None,
) -> Image:
    """Build an image block with mandatory ANSI-free fallback text.

    Args:
        data: Image bytes.
        mime_type: MIME type such as ``image/png``.
        filename: Optional file name for the fallback text.
        max_width_cells: Optional TTY width cap.
        alt_markdown: Pipe/TTY fallback. Defaults to ``*[image: name]*``.

    Returns:
        An ``Image`` block.
    """
    if alt_markdown is None:
        name = filename or "image"
        alt_markdown = f"*[image: {name}]*"
    return Image(
        data=data,
        mime_type=mime_type,
        alt_markdown=alt_markdown,
        filename=filename,
        max_width_cells=max_width_cells,
    )


def ansi(text: str, *, fallback_markdown: str | None = None) -> Ansi:
    """Build an ANSI art block with mandatory ANSI-free fallback markdown.

    Args:
        text: Pre-rendered ANSI.
        fallback_markdown: Pipe text. Defaults to stripped ANSI, fenced.

    Returns:
        An ``Ansi`` block.
    """
    if fallback_markdown is None:
        stripped = strip_ansi(text).rstrip("\n")
        fallback_markdown = stripped if stripped else "*[ANSI chart]*"
        if _CONTROL_RE.search(fallback_markdown):
            fallback_markdown = (
                _CONTROL_RE.sub("", fallback_markdown) or "*[ANSI chart]*"
            )
    return Ansi(text=text, fallback_markdown=fallback_markdown)


def panel(
    blocks: Sequence[Block], *, title: str | None = None, role: Role | None = None
) -> Panel:
    """Group blocks under an optional title.

    Args:
        blocks: Child blocks.
        title: Optional heading.
        role: Optional tint for styled sinks.

    Returns:
        A ``Panel`` block.
    """
    return Panel(blocks=tuple(blocks), title=title, role=role)


def json_block(payload: object) -> Code:
    """Pretty-printed JSON as a fenced ``json`` code block.

    Args:
        payload: Any JSON-serializable object.

    Returns:
        A ``Code`` block tagged ``json``.
    """
    return Code(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), language="json"
    )


def is_text_block(block: Block) -> TypeGuard[TextBlock]:
    """Return True when ``markdown_source`` can format the block.

    Args:
        block: Any block.

    Returns:
        True for Md, Note, Code, Table, and KeyValues.
    """
    return isinstance(block, Md | Note | Code | Table | KeyValues)


def _as_cell(value: object) -> Cell:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)
