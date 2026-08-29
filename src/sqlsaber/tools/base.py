"""Base class for SQLSaber tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar

from sqlsaber.tools.display import ToolDisplaySpec

if TYPE_CHECKING:
    from sqlsaber.artifact_resolution import ResolvedArtifactPublication
    from sqlsaber.render.blocks import Block
    from sqlsaber.tools.renderer import ToolRenderContext


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

    def render_executing(self, args: Mapping[str, Any]) -> Sequence[Block] | None:
        """Optionally render execution details. None means not handled."""
        del args
        return None

    def render_result(
        self,
        result: object,
        *,
        context: ToolRenderContext | None = None,
    ) -> Sequence[Block] | None:
        """Optionally render a tool result. None means not handled."""
        del result, context
        return None

    def set_resolved_artifact_publications(
        self,
        publications: Mapping[str, ResolvedArtifactPublication],
    ) -> None:
        """Accept verified publications for optional read-only replay rendering."""

        del publications
