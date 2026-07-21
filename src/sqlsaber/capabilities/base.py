"""Shared capability interfaces for SQLSaber."""

from collections.abc import Mapping
from typing import Any

from pydantic_ai.capabilities import AbstractCapability

from sqlsaber.tools.base import Tool


class SqlSaberCapability(AbstractCapability[Any]):
    """Base class for capabilities that expose SQLSaber CLI renderers."""

    @property
    def display_specs(self) -> Mapping[str, Tool]:
        """Return model tool names mapped to their CLI display implementations."""
        return {}

    async def close(self) -> None:
        """Release resources owned by this capability, if any."""
