"""Resolve a tool event to blocks: override, then spec, then JSON fallback."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import Any

from sqlsaber.artifacts import artifact_publication_from_metadata
from sqlsaber.render.blocks import Block, json_block, key_values, md, note
from sqlsaber.tools.base import Tool
from sqlsaber.tools.spec_blocks import (
    blocks_from_spec_executing,
    blocks_from_spec_result,
)

logger = logging.getLogger(__name__)
_DISPLAY_TOOLS_GROUP = "sqlsaber.display_tools"


@dataclass(frozen=True, slots=True)
class ToolRenderContext:
    """Context for result rendering. One value instead of growing kwargs."""

    tool_call_id: str | None = None
    metadata: object = None
    replay_messages: Sequence[Any] | None = None
    unavailable_artifacts: frozenset[str] = field(default_factory=frozenset)


def fallback_blocks(result: object) -> tuple[Block, ...]:
    """JSON (or fenced text) used when no tool override or spec applies.

    Args:
        result: Raw tool output.

    Returns:
        A single code/json block.
    """
    if isinstance(result, str):
        try:
            import json

            parsed = json.loads(result)
        except json.JSONDecodeError:
            return (md(f"```\n{result}\n```"),)
        return fallback_blocks(parsed)
    if isinstance(result, dict | list):
        return (json_block(result),)
    return (md(f"```\n{result}\n```"),)


class ToolRenderer:
    """Resolve a tool event to blocks for every sink."""

    def __init__(self, registry: Mapping[str, Tool] | None = None) -> None:
        self._registry = dict(registry or {})

    def executing(self, tool_name: str, args: Mapping[str, Any]) -> Sequence[Block]:
        """Blocks for a tool-executing event.

        Args:
            tool_name: Tool name.
            args: Tool arguments.

        Returns:
            Blocks from the tool override, the display spec, or a JSON fallback.
        """
        tool = self._registry.get(tool_name)
        if tool is not None:
            override = tool.render_executing(dict(args))
            if override is not None:
                return override
            if tool.display_spec is not None:
                return blocks_from_spec_executing(
                    tool_name, dict(args), tool.display_spec
                )
        return fallback_blocks(dict(args))

    def result(
        self,
        tool_name: str,
        result: object,
        *,
        context: ToolRenderContext | None = None,
    ) -> Sequence[Block]:
        """Blocks for a tool result, including artifact references.

        Args:
            tool_name: Tool name.
            result: Tool return value.
            context: Optional render context.

        Returns:
            Blocks from the tool override, the display spec, or a JSON fallback.
        """
        ctx = context or ToolRenderContext()
        tool = self._registry.get(tool_name)
        if tool is not None:
            setter = getattr(tool, "set_replay_messages", None)
            if ctx.replay_messages is not None and callable(setter):
                setter(ctx.replay_messages)
            override = tool.render_result(result, context=ctx)
            if override is not None:
                return (*override, *self._artifact_blocks(ctx))
            if tool.display_spec is not None:
                return (
                    *blocks_from_spec_result(tool_name, result, tool.display_spec),
                    *self._artifact_blocks(ctx),
                )
        return (*fallback_blocks(result), *self._artifact_blocks(ctx))

    def _artifact_blocks(self, ctx: ToolRenderContext) -> tuple[Block, ...]:
        publication = artifact_publication_from_metadata(ctx.metadata)
        if publication is None:
            return ()
        pairs: list[tuple[str, str]] = []
        for artifact in publication.artifacts:
            status = (
                " (unavailable)" if artifact.id in ctx.unavailable_artifacts else ""
            )
            pairs.append(
                (
                    artifact.name,
                    f"{artifact.kind}, {artifact.size} bytes {artifact.uri}{status}",
                )
            )
        return (
            note(f"Artifacts ({publication.kind})"),
            key_values(pairs),
        )


def core_display_registry() -> dict[str, Tool]:
    """Stateless core plus plugin display tools for transcripts and fallbacks.

    Returns:
        A name-to-tool map used when a session registry is not supplied.
    """
    from sqlsaber.tools.knowledge_tool import SearchKnowledgeTool
    from sqlsaber.tools.sql_tools import (
        ExecuteSQLTool,
        IntrospectSchemaTool,
        ListDatabasesTool,
        ListTablesTool,
    )

    tools = [
        SearchKnowledgeTool(),
        ListTablesTool(),
        IntrospectSchemaTool(),
        ExecuteSQLTool(),
        ListDatabasesTool(),
    ]
    registry: dict[str, Tool] = {tool.name: tool for tool in tools}
    discovered = entry_points()
    display_entries = (
        discovered.select(group=_DISPLAY_TOOLS_GROUP)
        if hasattr(discovered, "select")
        else discovered.get(_DISPLAY_TOOLS_GROUP, [])
    )
    for entry_point in sorted(display_entries, key=lambda item: item.name):
        try:
            provided = entry_point.load()()
            if isinstance(provided, Mapping):
                registry.update(
                    {
                        name: tool
                        for name, tool in provided.items()
                        if isinstance(name, str) and isinstance(tool, Tool)
                    }
                )
        except Exception:
            logger.warning(
                "Failed to load display tools from %s",
                entry_point.name,
                exc_info=True,
            )
    return registry
