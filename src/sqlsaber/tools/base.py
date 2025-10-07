"""Base class for SQLSaber tools."""

from abc import ABC, abstractmethod
from typing import Any


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
