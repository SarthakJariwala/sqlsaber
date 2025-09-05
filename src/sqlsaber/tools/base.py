"""Base class for SQLSaber tools."""

from abc import ABC, abstractmethod
from typing import Any

from .enums import ToolCategory, WorkflowPosition


class Tool(ABC):
    """Abstract base class for all tools."""

    def __init__(self):
        """Initialize the tool."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the tool name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Return the tool description."""
        pass

    @property
    @abstractmethod
    def input_schema(self) -> dict[str, Any]:
        """Return the tool's input schema."""
        pass

    @abstractmethod
    async def execute(self, **kwargs) -> str:
        """Execute the tool with given inputs.

        Args:
            **kwargs: Tool-specific keyword arguments

        Returns:
            JSON string with the tool's output
        """
        pass

    @property
    def category(self) -> ToolCategory:
        """Return the tool category. Override to customize."""
        return ToolCategory.GENERAL

    def get_usage_instructions(self) -> str | None:
        """Return tool-specific usage instructions for LLM guidance.

        Returns:
            Usage instructions string, or None for no specific guidance
        """
        return None

    def get_priority(self) -> int:
        """Return priority for tool ordering in instructions.

        Returns:
            Priority number (lower = higher priority, default = 100)
        """
        return 100

    def get_workflow_position(self) -> WorkflowPosition:
        """Return the typical workflow position for this tool.

        Returns:
            WorkflowPosition enum value
        """
        return WorkflowPosition.OTHER
