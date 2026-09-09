"""Show PluginContext sharing with the installed notebook and viz plugins."""

from __future__ import annotations

import asyncio

from sqlsaber import InMemoryQueryResultStore, SqlTools
from sqlsaber.artifacts import InMemoryArtifactStore
from sqlsaber.capabilities.plugins import PluginContext, discover_capabilities
from sqlsaber.config.settings import Config
from sqlsaber.knowledge.manager import KnowledgeManager


async def main() -> None:
    store = InMemoryQueryResultStore()
    artifacts = InMemoryArtifactStore()
    sql = SqlTools(database="sqlite:///:memory:", query_result_store=store)
    context = PluginContext(
        registry=sql.registry,
        knowledge_manager=KnowledgeManager(),
        allow_dangerous=False,
        tool_overrides={},
        config=Config.in_memory(
            model_name="openai:gpt-5.2",
            api_keys={"openai": "test-key"},
        ),
        main_model_name="openai:gpt-5.2",
        query_result_store=store,
        artifact_store=artifacts,
    )
    discovered = discover_capabilities(context)
    print("sql_store", id(sql.query_result_store))
    for capability in discovered:
        tool = getattr(capability, "tool", None)
        plugin_store = getattr(tool, "query_result_store", None)
        if plugin_store is None:
            plugin_store = getattr(
                getattr(tool, "_context", None), "query_result_store", None
            )
        print(
            type(capability).__name__, id(plugin_store), id(plugin_store) == id(store)
        )
    await sql.close()


if __name__ == "__main__":
    asyncio.run(main())
