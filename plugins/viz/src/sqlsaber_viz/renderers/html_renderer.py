"""Placeholder HTML renderer for future web UI support."""

from __future__ import annotations

from ..spec import VizSpec


class HtmlRenderer:
    """Render VizSpec to HTML.

    Placeholder implementation; currently returns an empty string.
    """

    def render(self, spec: VizSpec, rows: list[dict]) -> str:
        _ = spec
        _ = rows
        return ""
