"""Base class for SQLSaber tools."""

from abc import ABC, abstractmethod


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
