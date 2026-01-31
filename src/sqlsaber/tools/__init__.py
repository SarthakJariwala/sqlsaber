"""SQLSaber tools module."""

from .base import Tool
from .registry import ToolRegistry, discover_plugins, register_tool, tool_registry

# Import concrete tools to register them
from .sql_tools import ExecuteSQLTool, IntrospectSchemaTool, ListTablesTool, SQLTool

# Discover and load any installed plugins
discover_plugins()

__all__ = [
    "Tool",
    "ToolRegistry",
    "tool_registry",
    "register_tool",
    "discover_plugins",
    "SQLTool",
    "ListTablesTool",
    "IntrospectSchemaTool",
    "ExecuteSQLTool",
]
