"""Sandboxed Python capability plugin."""

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import FunctionToolset
from sqlsaber.capabilities.base import SqlSaberCapability
from sqlsaber.capabilities.plugins import PluginContext
from sqlsaber.tools.base import Tool

from .tools import RunPythonTool, sandbox_providers_available


class Sandbox(SqlSaberCapability):
    """Execute Python against SQL result payloads in a remote sandbox."""

    id = "sandbox"
    description = "Run Python analysis in a configured remote sandbox."

    def __init__(self) -> None:
        self.tool = RunPythonTool()
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


def capability(
    context: PluginContext,
) -> AbstractCapability[Any] | Sequence[AbstractCapability[Any]]:
    """Create sandbox tools only when a provider is configured."""
    del context
    if not sandbox_providers_available():
        return ()
    return Sandbox()
