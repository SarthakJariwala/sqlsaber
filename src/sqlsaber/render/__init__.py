"""SQLsaber rendering: blocks out, saber-tui or markdown in."""

from __future__ import annotations

import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, TextIO

from . import blocks
from .blocks import (
    Ansi,
    Block,
    Code,
    Column,
    ControlSequenceError,
    Image,
    KeyValues,
    Md,
    Note,
    Panel,
    Role,
    Table,
    TextBlock,
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

if TYPE_CHECKING:
    from .surface import (
        Ask,
        AskChoice,
        AskConfirm,
        AskPath,
        AskSecret,
        AskText,
        Choice,
        PromptUnavailable,
        Surface,
        TextStream,
    )

__all__ = [
    "Ansi",
    "Ask",
    "AskChoice",
    "AskConfirm",
    "AskPath",
    "AskSecret",
    "AskText",
    "Block",
    "Choice",
    "Code",
    "Column",
    "ControlSequenceError",
    "Image",
    "KeyValues",
    "Md",
    "Note",
    "Panel",
    "PromptUnavailable",
    "Role",
    "Surface",
    "Table",
    "TextBlock",
    "TextStream",
    "ansi",
    "blocks",
    "cli_err",
    "cli_out",
    "code",
    "error",
    "html_of",
    "image",
    "json_block",
    "key_values",
    "md",
    "md_of",
    "note",
    "panel",
    "reset_io",
    "success",
    "table",
    "warn",
]

_out: Surface | None = None
_err: Surface | None = None


def _stream_closed(surface: Surface | None) -> bool:
    stream = getattr(surface, "_stream", None)
    return bool(getattr(stream, "closed", False))


def cli_out() -> Surface:
    """Cached stdout surface. TTY gets TerminalSurface, else PlainSurface.

    Returns:
        The stdout ``Surface``.
    """
    global _out
    if _out is None or _stream_closed(_out):
        _out = _make_surface(sys.stdout, stderr=False)
    return _out


def cli_err() -> Surface:
    """Cached stderr surface.

    Returns:
        The stderr ``Surface``.
    """
    global _err
    if _err is None or _stream_closed(_err):
        _err = _make_surface(sys.stderr, stderr=True)
    return _err


def reset_io(
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    tty: bool | None = None,
) -> None:
    """Rebind ``cli_out`` / ``cli_err``.

    Args:
        stdout: Stream for stdout. Defaults to ``sys.stdout``.
        stderr: Stream for stderr. Defaults to ``sys.stderr``.
        tty: When False, force PlainSurface. When True, force TerminalSurface.
            When None, use ``isatty()``.
    """
    global _out, _err
    out_stream = stdout if stdout is not None else sys.stdout
    err_stream = stderr if stderr is not None else sys.stderr
    _out = _make_surface(out_stream, stderr=False, tty=tty)
    _err = _make_surface(err_stream, stderr=True, tty=tty)


@contextmanager
def form_session(surface: Surface) -> Iterator[Surface]:
    """Keep one TUI alive across sequential ``ask`` calls.

    Args:
        surface: Usually ``cli_out()``.

    Yields:
        The same surface.
    """
    start = getattr(surface, "start_form", None)
    end = getattr(surface, "end_form", None)
    if start is not None:
        start()
    try:
        yield surface
    finally:
        if end is not None:
            end()


def _make_surface(stream: TextIO, *, stderr: bool, tty: bool | None = None) -> Surface:
    del stderr
    is_tty = stream.isatty() if tty is None else tty
    if is_tty:
        from sqlsaber.theme.styles import get_styles

        from .terminal import TerminalSurface

        return TerminalSurface(stream, get_styles())
    from .terminal import PlainSurface

    return PlainSurface(stream)


def html_of(blocks_in: Sequence[Block]) -> str:
    from .html import html_of as _html_of

    return _html_of(blocks_in)


def md_of(blocks_in: Sequence[Block]) -> str:
    from .markdown_text import md_of as _md_of

    return _md_of(blocks_in)


def __getattr__(name: str):
    if name in {
        "Ask",
        "AskChoice",
        "AskConfirm",
        "AskPath",
        "AskSecret",
        "AskText",
        "Choice",
        "PromptUnavailable",
        "Surface",
        "TextStream",
    }:
        from . import surface as surface_mod

        return getattr(surface_mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
