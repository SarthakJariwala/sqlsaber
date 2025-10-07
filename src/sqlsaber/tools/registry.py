"""Tool registry for managing available tools."""

from typing import Type

from .base import Tool


class ToolRegistry:
    """Registry for managing and discovering tools."""

    def __init__(self):
        """Initialize the registry."""
        self._tools: dict[str, Type[Tool]] = {}
        self._instances: dict[str, Tool] = {}

    def register(self, tool_class: Type[Tool]) -> None:
        """Register a tool class.

        Args:
            tool_class: The tool class to register
        """
        # Create a temporary instance to get the name
        temp_instance = tool_class()
        name = temp_instance.name

        if name in self._tools:
            raise ValueError(f"Tool '{name}' is already registered")

        self._tools[name] = tool_class

    def unregister(self, name: str) -> None:
        """Unregister a tool.

        Args:
            name: Name of the tool to unregister
        """
        if name in self._tools:
            del self._tools[name]
        if name in self._instances:
            del self._instances[name]

    def get_tool(self, name: str) -> Tool:
        """Get a tool instance by name.

        Args:
            name: Name of the tool

        Returns:
            Tool instance

        Raises:
            KeyError: If tool is not found
        """
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found in registry")

        # Create instance if not already created (singleton pattern)
        if name not in self._instances:
            self._instances[name] = self._tools[name]()

        return self._instances[name]

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def get_all_tools(self) -> list[Tool]:
        """Get all tool instances."""
        return [self.get_tool(name) for name in self.list_tools()]


# Global registry instance
tool_registry = ToolRegistry()


def register_tool(tool_class: Type[Tool]) -> Type[Tool]:
    """Decorator to register a tool class.

    Usage:
        @register_tool
        class MyTool(Tool):
            ...
    """
    tool_registry.register(tool_class)
    return tool_class
