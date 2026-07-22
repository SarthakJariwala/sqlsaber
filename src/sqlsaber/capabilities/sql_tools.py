"""Composable SQL querying capability."""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from typing import Any, Self

from pydantic_ai import RunContext
from pydantic_ai.toolsets import FunctionToolset

from sqlsaber.capabilities._wrapping import (
    build_db_catalog,
    wrap_add_db_name,
    wrap_strip_db_name,
)
from sqlsaber.capabilities.base import SqlSaberCapability
from sqlsaber.database.registry import DatabaseRegistry
from sqlsaber.database.resolver import resolve_databases
from sqlsaber.prompts.dangerous_mode import DANGEROUS_MODE
from sqlsaber.prompts.sql_guidance import SQL_GUIDANCE, SQL_GUIDANCE_MULTI
from sqlsaber.query_results import InMemoryQueryResultStore, QueryResultStore
from sqlsaber.tools.base import Tool
from sqlsaber.tools.sql_tools import (
    ExecuteSQLTool,
    IntrospectSchemaTool,
    ListDatabasesTool,
    ListTablesTool,
    SQLTool,
)


class _SqlToolset(FunctionToolset[Any]):
    """Function toolset that owns an optional database registry lifecycle."""

    def __init__(self, registry: DatabaseRegistry, *, owned: bool) -> None:
        super().__init__(id="sqlsaber-sql")
        self._registry = registry
        self._owned = owned
        self._entry_count = 0

    async def __aenter__(self) -> Self:
        await super().__aenter__()
        try:
            if self._owned and self._entry_count == 0:
                for entry in self._registry:
                    await entry.connection.get_pool()
        except BaseException as init_error:
            try:
                await self._registry.close()
            except BaseException as cleanup_error:
                init_error.add_note(f"Database cleanup also failed: {cleanup_error!r}")
            try:
                await super().__aexit__(
                    type(init_error), init_error, init_error.__traceback__
                )
            except BaseException as cleanup_error:
                init_error.add_note(f"Toolset cleanup also failed: {cleanup_error!r}")
            raise
        self._entry_count += 1
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        if self._entry_count <= 0:
            raise RuntimeError("SQL toolset context exited without a matching entry.")

        self._entry_count -= 1
        try:
            if self._owned and self._entry_count == 0:
                await self._registry.close()
        finally:
            super_result = await super().__aexit__(*args)
        return super_result

    async def close(self) -> None:
        if self._owned:
            await self._registry.close()


class SqlTools(SqlSaberCapability):
    """SQL querying tools for one or more databases.

    This capability is dependency-agnostic and can be attached to an agent with any
    ``deps_type``. When constructed from database selectors, use ``async with agent:``
    or call :meth:`close` explicitly so owned connections are released.
    """

    def __init__(
        self,
        database: str | Sequence[str] | None = None,
        *,
        registry: DatabaseRegistry | None = None,
        allow_dangerous: bool = False,
        include_catalog_instructions: bool = True,
        query_result_store: QueryResultStore | None = None,
    ) -> None:
        if registry is not None and database is not None:
            raise ValueError("Pass either `database` or `registry`, not both.")

        if registry is None:
            selectors = (
                list(database)
                if isinstance(database, Sequence) and not isinstance(database, str)
                else database
            )
            registry = DatabaseRegistry.from_resolved(resolve_databases(selectors))
            owned = True
        else:
            owned = False

        self.registry = registry
        self.query_result_store = (
            query_result_store
            if query_result_store is not None
            else InMemoryQueryResultStore()
        )
        self.allow_dangerous = allow_dangerous
        self.include_catalog_instructions = include_catalog_instructions
        self._owned = owned
        self._toolset = _SqlToolset(registry, owned=owned)

        tools: list[Tool] = [
            ListTablesTool(),
            IntrospectSchemaTool(),
            ExecuteSQLTool(query_result_store=self.query_result_store),
        ]
        if len(registry) > 1:
            tools.append(ListDatabasesTool())

        primary = registry.get(registry.primary())
        for tool in tools:
            if isinstance(tool, SQLTool):
                if len(registry) > 1:
                    tool.set_registry(registry)
                else:
                    tool.set_connection(primary.connection, primary.schema_manager)
                tool.allow_dangerous = allow_dangerous

        self._tools = {tool.name: tool for tool in tools}
        self._register_tools()

    @property
    def is_multi_db(self) -> bool:
        return len(self.registry) > 1

    @property
    def display_specs(self) -> Mapping[str, Tool]:
        return self._tools

    def _register_tools(self) -> None:
        names = tuple(self.registry.names())
        for tool in self._tools.values():
            signature = inspect.signature(tool.execute)
            if "db_name" not in signature.parameters:
                function = tool.execute
            elif self.is_multi_db:
                function = wrap_add_db_name(tool, names)
            else:
                function = wrap_strip_db_name(tool)
            self._toolset.add_function(
                function,
                name=tool.name,
                takes_ctx=tool.requires_ctx,
            )

    def get_toolset(self) -> FunctionToolset[Any]:
        return self._toolset

    def instructions_text(self) -> str:
        """Render the database-specific instructions for this capability."""
        if self.is_multi_db:
            prompt = SQL_GUIDANCE_MULTI.format(
                db_catalog=build_db_catalog(self.registry)
            )
        else:
            primary = self.registry.get(self.registry.primary())
            prompt = SQL_GUIDANCE.format(db=primary.display_name)
        if self.allow_dangerous:
            prompt += DANGEROUS_MODE
        return prompt

    def get_instructions(self):
        if not self.include_catalog_instructions:
            return None

        def instructions(ctx: RunContext[Any]) -> str:
            del ctx
            return self.instructions_text()

        return instructions

    async def close(self) -> None:
        """Close database connections owned by this capability."""
        await self._toolset.close()
