"""Renderer protocol for visualization outputs."""

from __future__ import annotations

from typing import Protocol

from ..spec import VizSpec


class RendererProtocol(Protocol):
    def render(self, spec: VizSpec, rows: list[dict]) -> str:
        """Render a visualization spec with data rows."""
        ...
