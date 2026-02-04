"""SQLSaber visualization plugin."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlsaber.tools.registry import ToolRegistry


def register_tools(registry: "ToolRegistry | None" = None):
    """Register visualization tools.

    Returns list of tool classes for sqlsaber to register.
    """
    from .tools import VizTool

    return [VizTool]


__all__ = ["register_tools"]
