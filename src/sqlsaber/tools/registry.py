"""Tool registry for managing available tools."""

import inspect
from collections.abc import Iterable
from importlib.metadata import entry_points
from typing import Type

from sqlsaber.config.logging import get_logger

from .base import Tool

logger = get_logger(__name__)

PLUGIN_GROUP = "sqlsaber.tools"


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


def _select_entry_points(group: str) -> Iterable:
    eps = entry_points()
    if hasattr(eps, "select"):
        return eps.select(group=group)
    return eps.get(group, [])


def _call_plugin_factory(factory, registry: ToolRegistry):
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return factory(registry)

    params = list(signature.parameters.values())
    has_varargs = any(param.kind == param.VAR_POSITIONAL for param in params)
    positional_params = [
        param
        for param in params
        if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD)
    ]
    if has_varargs or positional_params:
        return factory(registry)
    return factory()


def _normalize_tool_classes(result) -> list[type[Tool]]:
    if result is None:
        return []

    if isinstance(result, type) and issubclass(result, Tool):
        return [result]

    if isinstance(result, Iterable) and not isinstance(result, (str, bytes)):
        classes: list[type[Tool]] = []
        for item in result:
            if isinstance(item, type) and issubclass(item, Tool):
                classes.append(item)
            else:
                logger.warning("Plugin returned non-Tool entry: %r", item)
        return classes

    logger.warning("Plugin returned unsupported result: %r", result)
    return []


def discover_plugins(registry: ToolRegistry | None = None) -> list[str]:
    """Discover and load tool plugins via entry points.

    Plugins register via pyproject.toml entry points:

        [project.entry-points."sqlsaber.tools"]
        my_tool = "my_package.module:MyToolClass"

    Returns:
        List of successfully loaded plugin names.
    """

    target_registry = registry or tool_registry
    loaded: list[str] = []
    for ep in _select_entry_points(PLUGIN_GROUP):
        try:
            plugin_obj = ep.load()

            if isinstance(plugin_obj, type) and issubclass(plugin_obj, Tool):
                target_registry.register(plugin_obj)
                loaded.append(ep.name)
                logger.debug("Loaded plugin tool: %s", ep.name)
                continue

            if callable(plugin_obj):
                result = _call_plugin_factory(plugin_obj, target_registry)
                tool_classes = _normalize_tool_classes(result)
                if tool_classes:
                    for tool_class in tool_classes:
                        try:
                            target_registry.register(tool_class)
                        except ValueError:
                            logger.debug(
                                "Plugin '%s' tool already registered: %s",
                                ep.name,
                                tool_class,
                            )
                loaded.append(ep.name)
                logger.debug("Loaded plugin tools from: %s", ep.name)
                continue

            logger.warning("Plugin '%s' is not a Tool or factory", ep.name)
        except Exception as exc:
            logger.warning("Failed to load plugin '%s': %s", ep.name, exc)

    return loaded


def register_tool(tool_class: Type[Tool]) -> Type[Tool]:
    """Decorator to register a tool class.

    Usage:
        @register_tool
        class MyTool(Tool):
            ...
    """
    tool_registry.register(tool_class)
    return tool_class
