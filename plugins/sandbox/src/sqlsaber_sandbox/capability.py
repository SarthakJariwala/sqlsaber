"""Sandboxed Python capability plugin."""

from collections.abc import Mapping
from typing import Any, Self, cast

from pydantic_ai.toolsets import FunctionToolset
from sqlsaber.capabilities.base import SqlSaberCapability
from sqlsaber.capabilities.plugins import PluginContext
from sqlsaber.tools.base import Tool

from .config import DEFAULT_SANDBOX_CONFIG, SandboxConfig
from .tools import AnalyzeSandboxTool, prepare_analysis


class Sandbox(SqlSaberCapability):
    """Own goal-based persistent sandbox sessions for one SDK runtime."""

    id = "sandbox"
    description = "Delegate analysis to a persistent Python sandbox subagent."

    def __init__(
        self, context: PluginContext, *, config: SandboxConfig = DEFAULT_SANDBOX_CONFIG
    ) -> None:
        self.tool = AnalyzeSandboxTool(context, config)
        self._toolset = FunctionToolset[Any](id=self.id)
        self._toolset.add_function(
            self.tool.execute_with_attachments
            if context.workspace_input_resolver is not None
            else self.tool.execute,
            name=self.tool.name,
            takes_ctx=True,
            prepare=prepare_analysis,
        )
        self._toolset.add_function(
            self.tool.close_session, name="close_sandbox", takes_ctx=True
        )
        self._toolset.add_function(
            self.tool.publish_artifacts,
            name="publish_sandbox_artifacts",
            takes_ctx=True,
        )

    @property
    def display_specs(self) -> Mapping[str, Tool]:
        return {self.tool.name: self.tool}

    def get_toolset(self) -> FunctionToolset[Any]:
        return self._toolset

    def update_context(self, context: PluginContext) -> None:
        self.tool.context = context

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self.close()

    async def close(self) -> None:
        await self.tool.close()


def capability(
    context: PluginContext,
    *,
    config: SandboxConfig = DEFAULT_SANDBOX_CONFIG,
) -> Sandbox:
    """Construct lazily; provider validation happens on the first analysis."""
    return Sandbox(context, config=config)


def display_tools() -> Mapping[str, Tool]:
    return {
        "analyze_in_sandbox": AnalyzeSandboxTool(
            cast(PluginContext, object()), DEFAULT_SANDBOX_CONFIG
        )
    }
