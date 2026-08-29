"""TTY serializer: blocks become saber-tui components.

This module may import ``saber_tui.components.markdown``. It must not be
imported from ``cli/commands.py`` at module load.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from saber_tui.components import Box, DefaultTextStyle, Markdown
from saber_tui.components import Image as TuiImage
from saber_tui.components.image import ImageOptions, ImageTheme
from saber_tui.utils import wrap_text_with_ansi

from sqlsaber.render.blocks import (
    Ansi,
    Block,
    Image,
    Panel,
    is_text_block,
)
from sqlsaber.render.markdown_text import markdown_source
from sqlsaber.theme.styles import Styles

_RESPONSIVE_IMAGE_MAX_CELLS = 2**31 - 1


@runtime_checkable
class Component(Protocol):
    """Structural mirror of ``saber_tui.tui.Component``."""

    def render(self, width: int) -> list[str]: ...

    def invalidate(self) -> None: ...


class AnsiLines:
    """Wrap pre-rendered ANSI text as a component. Promoted from ``_AnsiBlock``."""

    def __init__(self, text: str) -> None:
        self.ansi_text = text.rstrip("\n")
        self._cache_width: int | None = None
        self._cache_lines: list[str] | None = None

    def render(self, width: int) -> list[str]:
        """Render wrapped ANSI lines.

        Args:
            width: Terminal width in cells.

        Returns:
            One string per row.
        """
        if self._cache_width == width and self._cache_lines is not None:
            return list(self._cache_lines)
        if not self.ansi_text:
            lines = [""]
        else:
            lines = []
            for line in self.ansi_text.splitlines():
                lines.extend(wrap_text_with_ansi(line, max(1, width)))
            lines = lines or [""]
        self._cache_width = width
        self._cache_lines = lines
        return list(lines)

    def invalidate(self) -> None:
        """Drop the width cache."""
        self._cache_width = None
        self._cache_lines = None


def components_for(
    blocks: Sequence[Block], styles: Styles, *, tui: Any | None = None
) -> list[Component]:
    """Map blocks to saber-tui components.

    Text blocks become ``Markdown(markdown_source(block), ...)``. The
    component never formats text itself.

    Args:
        blocks: Blocks to serialize.
        styles: Resolved theme.
        tui: Live ``TUI`` required for native images. Without it, images
            degrade to their alt markdown.

    Returns:
        A list of components.
    """
    components: list[Component] = []
    for block in blocks:
        components.extend(_one(block, styles, tui=tui))
    return components


def _one(block: Block, styles: Styles, *, tui: Any | None) -> list[Component]:
    if is_text_block(block):
        role = getattr(block, "role", None)
        color = styles.role(role) if role else None
        default_style = (
            DefaultTextStyle(color=color) if color is not None and role else None
        )
        return [
            Markdown(
                markdown_source(block),
                theme=styles.markdown,
                default_text_style=default_style,
            )
        ]
    if isinstance(block, Image):
        if tui is None:
            return [
                Markdown(
                    block.alt_markdown,
                    theme=styles.markdown,
                    default_text_style=DefaultTextStyle(color=styles.muted_fg),
                )
            ]
        max_width = (
            block.max_width_cells
            if block.max_width_cells is not None
            else _RESPONSIVE_IMAGE_MAX_CELLS
        )
        return [
            TuiImage(
                tui,
                block.data,
                block.mime_type,
                theme=ImageTheme(fallback_color=styles.muted_fg),
                options=ImageOptions(
                    filename=block.filename,
                    max_width_cells=max_width,
                ),
            )
        ]
    if isinstance(block, Ansi):
        return [AnsiLines(block.text)]
    if isinstance(block, Panel):
        box = Box(padding_x=1, padding_y=1, bg_fn=styles.panel_bg)
        children = components_for(block.blocks, styles, tui=tui)
        if block.title:
            children = [
                Markdown(
                    f"**{block.title}**",
                    theme=styles.markdown,
                    default_text_style=DefaultTextStyle(color=styles.role(block.role)),
                ),
                *children,
            ]
        for index, child in enumerate(children):
            if index:
                box.add_child(AnsiLines(""))
            box.add_child(child)
        return [box]
    return []


def markdown_component_source(component: Component) -> str | None:
    """Return the markdown source a Markdown component was constructed with.

    Args:
        component: A saber-tui component.

    Returns:
        The source string, or None when the component is not Markdown.
    """
    if isinstance(component, Markdown):
        return component.text
    return None
