"""Shared capability interfaces for SQLSaber."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from pydantic_ai.capabilities import AbstractCapability

from sqlsaber.tools.base import Tool

if TYPE_CHECKING:
    from sqlsaber.capabilities.plugins import PluginContext


class SqlSaberCapability(AbstractCapability[Any]):
    """Base class for capabilities that expose SQLSaber CLI renderers."""

    @property
    def display_specs(self) -> Mapping[str, Tool]:
        """Return model tool names mapped to their CLI display implementations."""
        return {}

    def update_context(self, context: PluginContext) -> None:
        """Apply refreshed managed-agent context after a successful rebuild."""

    async def close(self) -> None:
        """Release resources owned by this capability, if any."""
