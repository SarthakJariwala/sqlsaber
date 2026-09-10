"""Explicit capability lists share one query-result store."""

from __future__ import annotations

import json
import sqlite3

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel

from sqlsaber import InMemoryQueryResultStore, SQLSaber, SQLSaberOptions, SqlTools
from sqlsaber.capabilities.plugins import PluginContext
from sqlsaber.config.settings import Config
from sqlsaber.knowledge.manager import KnowledgeManager
from sqlsaber.query_result_resolution import query_result_references_from_messages
from sqlsaber.query_results import QueryResultContext
from sqlsaber_notebook.capability import capability as notebook_factory
from sqlsaber_viz import Visualization, capability as viz_factory


def _settings() -> Config:
    return Config.in_memory(
        model_name="anthropic:claude-3-5-sonnet",
        api_keys={"anthropic": "test-key"},
    )


def _wide_customers(path) -> None:
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, revenue INTEGER)"
    )
    connection.executemany(
        "INSERT INTO customers (name, revenue) VALUES (?, ?)",
        [(f"customer-{index:03d}-" + ("x" * 24), 1000 + index) for index in range(250)],
    )
    connection.commit()
    connection.close()


@pytest.mark.asyncio
async def test_sqlsaber_factories_share_the_session_store() -> None:
    store = InMemoryQueryResultStore()
    saber = SQLSaber(
        options=SQLSaberOptions(
            database="sqlite:///:memory:",
            settings=_settings(),
            query_result_store=store,
            capabilities=[notebook_factory, viz_factory],
        )
    )
    try:
        sql = next(c for c in saber.agent.capabilities if isinstance(c, SqlTools))
        notebook = next(
            c for c in saber.agent.capabilities if getattr(c, "id", None) == "notebook"
        )
        viz = next(
            c for c in saber.agent.capabilities if getattr(c, "id", None) == "viz"
        )
        assert sql.query_result_store is store
        assert notebook.tool._context.query_result_store is store
        assert viz.tool.query_result_store is store
    finally:
        await saber.close()


@pytest.mark.asyncio
async def test_host_agent_sqltools_and_viz_share_store_and_all_rows(tmp_path) -> None:
    database = tmp_path / "customers.sqlite"
    _wide_customers(database)
    store = InMemoryQueryResultStore()
    sql = SqlTools(database=str(database), query_result_store=store)
    viz = Visualization(
        PluginContext(
            registry=sql.registry,
            knowledge_manager=KnowledgeManager(),
            allow_dangerous=False,
            tool_overrides={},
            config=_settings(),
            main_model_name="anthropic:claude-3-5-sonnet",
            query_result_store=store,
            main_api_key="test-key",
        )
    )

    def respond(messages, info):
        del info
        parts = [part for message in messages for part in message.parts]
        if any(isinstance(part, ToolReturnPart) for part in parts):
            return ModelResponse(parts=[TextPart("Listed the customers.")])
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "execute_sql",
                    {"query": "SELECT id, name, revenue FROM customers ORDER BY id"},
                    tool_call_id="sql1",
                )
            ]
        )

    agent = Agent(
        FunctionModel(respond),
        instructions="You query the customers table.",
        capabilities=[sql, viz],
    )
    try:
        async with agent:
            result = await agent.run("Show every customer")
        execute_sql = next(
            part
            for message in result.new_messages()
            for part in message.parts
            if isinstance(part, ToolReturnPart) and part.tool_name == "execute_sql"
        )
        preview = json.loads(execute_sql.content)
        assert preview["results_truncated"] is True
        assert len(preview.get("preview_rows") or []) < 250
        references = query_result_references_from_messages(result.new_messages())
        loaded = await viz.tool.query_result_store.get(
            references[0].descriptor.id,
            context=QueryResultContext(),
        )
        rows = loaded.rows()
        assert len(rows) == 250
        assert rows[0]["name"].startswith("customer-000-")
        assert rows[-1]["name"].startswith("customer-249-")
        assert viz.tool.query_result_store is sql.query_result_store
    finally:
        await sql.close()
