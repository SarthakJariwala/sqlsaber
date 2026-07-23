"""Base class for SQLSaber tools."""

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Protocol

from rich.console import Console

from sqlsaber.tools.display import ToolDisplaySpec


class ToolResultPanel(Protocol):
    """Native components that can be composed inside a themed tool panel."""

    def append_markdown(self, text: str = "", *, muted: bool = False) -> object: ...

    def append_image(
        self,
        data: bytes,
        mime_type: str,
        *,
        filename: str | None = None,
        max_width_cells: int = 60,
        max_height_cells: int | None = None,
    ) -> object: ...


class ToolResultTUI(ToolResultPanel, Protocol):
    """Native TUI surface available to tool renderers."""

    def append_panel(self) -> ToolResultPanel: ...


class Tool(ABC):
    """Abstract base class for all tools."""

    requires_ctx: ClassVar[bool] = False
    multi_db_only: ClassVar[bool] = False
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
    async def execute(self, *args, **kwargs) -> Any:
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

    def render_executing_tui(self, tui: ToolResultTUI, args: dict) -> bool:
        """Optionally append native saber-tui components for tool execution."""
        del tui, args
        return False

    def render_result(self, console: Console, result: object) -> bool:
        """Optionally render tool results. Return True if handled."""
        return False

    def render_result_event(
        self,
        console: Console,
        result: object,
        *,
        tool_call_id: str | None = None,
        metadata: object = None,
    ) -> bool:
        """Render a live/replayed result with framework event context."""
        del tool_call_id, metadata
        return self.render_result(console, result)

    def render_result_tui(
        self,
        tui: ToolResultTUI,
        result: object,
        *,
        tool_call_id: str | None = None,
        metadata: object = None,
    ) -> bool:
        """Optionally append native saber-tui components for a live result."""
        del tui, result, tool_call_id, metadata
        return False

    def render_result_html(self, result: object) -> str | None:
        """Optionally render tool results as HTML."""
        return None
