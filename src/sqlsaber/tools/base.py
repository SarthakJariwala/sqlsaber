"""Base class for SQLSaber tools."""

from abc import ABC, abstractmethod
from typing import ClassVar

from rich.console import Console

from sqlsaber.tools.display import ToolDisplaySpec


class Tool(ABC):
    """Abstract base class for all tools."""

    requires_ctx: ClassVar[bool] = False
    display_spec: ClassVar[ToolDisplaySpec | None] = None

    def __init__(self):
        """Initialize the tool."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the tool name."""
        pass

    @abstractmethod
    async def execute(self, *args, **kwargs) -> str:
        """Execute the tool with given inputs.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            JSON string with the tool's output
        """
        pass

    def render_executing(self, console: Console, args: dict) -> bool:
        """Optionally render execution details. Return True if handled."""
        return False

    def render_result(self, console: Console, result: object) -> bool:
        """Optionally render tool results. Return True if handled."""
        return False

    def render_result_html(self, result: object) -> str | None:
        """Optionally render tool results as HTML."""
        return None
