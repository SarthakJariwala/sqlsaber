"""Composable knowledge search capability."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self

from pydantic_ai.toolsets import FunctionToolset

from sqlsaber.capabilities._wrapping import wrap_add_db_name, wrap_strip_db_name
from sqlsaber.capabilities.base import SqlSaberCapability
from sqlsaber.database.registry import DatabaseRegistry
from sqlsaber.knowledge.manager import KnowledgeManager
from sqlsaber.tools.base import Tool
from sqlsaber.tools.knowledge_tool import SearchKnowledgeTool


class _KnowledgeToolset(FunctionToolset[Any]):
    def __init__(self, manager: KnowledgeManager, *, owned: bool) -> None:
        super().__init__(id="sqlsaber-knowledge")
        self._manager = manager
        self._owned = owned
        self._closed = False

    async def __aenter__(self) -> Self:
        await super().__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        try:
            if self._owned and not self._closed:
                await self._manager.close()
                self._closed = True
        finally:
            return await super().__aexit__(*args)

    async def close(self) -> None:
        if self._owned and not self._closed:
            await self._manager.close()
            self._closed = True


class Knowledge(SqlSaberCapability):
    """Search SQLSaber knowledge associated with the active database."""

    id = "sqlsaber-knowledge"
    description = "Search saved SQL and domain knowledge for the connected database."

    def __init__(
        self,
        knowledge_manager: KnowledgeManager | None = None,
        *,
        registry: DatabaseRegistry | None = None,
        database_name: str | None = None,
    ) -> None:
        owned = knowledge_manager is None
        manager = knowledge_manager or KnowledgeManager()
        self.knowledge_manager = manager
        self.registry = registry
        self._owned = owned
        self._toolset = _KnowledgeToolset(manager, owned=owned)

        tool = SearchKnowledgeTool()
        if registry is not None and len(registry) > 1:
            tool.set_registry(registry)
            tool.knowledge_manager = manager
        else:
            resolved_name = database_name
            if resolved_name is None and registry is not None:
                resolved_name = registry.primary()
            tool.set_context(resolved_name, manager)
        self._tool = tool

        function = tool.execute
        if registry is not None:
            if len(registry) > 1:
                function = wrap_add_db_name(tool, tuple(registry.names()))
            else:
                function = wrap_strip_db_name(tool)
        self._toolset.add_function(function, name=tool.name, takes_ctx=False)

    @property
    def display_specs(self) -> Mapping[str, Tool]:
        return {self._tool.name: self._tool}

    def get_toolset(self) -> FunctionToolset[Any]:
        return self._toolset

    async def close(self) -> None:
        """Close the knowledge manager when this capability created it."""
        await self._toolset.close()
