"""Visualization capability plugin."""

from collections.abc import Mapping
from typing import Any

from pydantic_ai.toolsets import FunctionToolset
from sqlsaber.capabilities.base import SqlSaberCapability
from sqlsaber.capabilities.plugins import PluginContext
from sqlsaber.tools.base import Tool

from .tools import VizTool


class Visualization(SqlSaberCapability):
    """Generate and render terminal visualizations from SQL result payloads."""

    id = "viz"
    description = "Generate a chart from the result of an SQL query."

    def __init__(self, context: PluginContext) -> None:
        self.tool = VizTool(context.query_result_store)
        self.tool.model_overide = context.tool_overrides.get(self.tool.name)
        self._toolset = FunctionToolset[Any](id=self.id)
        self._toolset.add_function(
            self.tool.execute,
            name=self.tool.name,
            takes_ctx=True,
        )

    @property
    def display_specs(self) -> Mapping[str, Tool]:
        return {self.tool.name: self.tool}

    def get_toolset(self) -> FunctionToolset[Any]:
        return self._toolset


def capability(context: PluginContext) -> Visualization:
    """Create the visualization capability for a managed SQLSaber agent."""
    return Visualization(context)
