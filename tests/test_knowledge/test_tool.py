from __future__ import annotations

import json

import pytest

from sqlsaber.knowledge.manager import KnowledgeManager
from sqlsaber.knowledge.sqlite_store import SQLiteKnowledgeStore
from sqlsaber.tools.knowledge_tool import SearchKnowledgeTool


@pytest.mark.asyncio
async def test_search_knowledge_tool_returns_scoped_results(temp_dir):
    manager = KnowledgeManager(store=SQLiteKnowledgeStore(db_path=temp_dir / "tool.db"))
    await manager.add_knowledge(
        database_name="db_a",
        name="Revenue",
        description="Monthly revenue aggregation",
    )
    await manager.add_knowledge(
        database_name="db_b",
        name="Revenue",
        description="Should not be visible in db_a",
    )

    tool = SearchKnowledgeTool()
    tool.set_context("db_a", manager)

    data = json.loads(await tool.execute("revenue"))
    assert data["total_results"] == 1
    assert data["results"][0]["description"] == "Monthly revenue aggregation"


@pytest.mark.asyncio
async def test_search_knowledge_tool_validates_input_and_context():
    tool = SearchKnowledgeTool()

    no_query = json.loads(await tool.execute("   "))
    assert "error" in no_query

    no_context = json.loads(await tool.execute("revenue"))
    assert "error" in no_context


@pytest.mark.asyncio
async def test_search_knowledge_tool_handles_store_errors(temp_dir, monkeypatch):
    manager = KnowledgeManager(store=SQLiteKnowledgeStore(db_path=temp_dir / "tool.db"))
    tool = SearchKnowledgeTool()
    tool.set_context("db_a", manager)

    async def _raise_search_error(*_: object, **__: object) -> list[object]:
        raise OSError("storage unavailable")

    monkeypatch.setattr(manager, "search_knowledge", _raise_search_error)

    data = json.loads(await tool.execute("revenue"))
    assert data["error"] == "Error searching knowledge: storage unavailable"
