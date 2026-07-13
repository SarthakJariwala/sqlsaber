"""Tests for the knowledge capability."""

import json

import pytest

from sqlsaber.capabilities import Knowledge
from sqlsaber.database.registry import DatabaseEntry, DatabaseRegistry
from sqlsaber.database.sqlite import SQLiteConnection
from sqlsaber.knowledge.manager import KnowledgeManager
from sqlsaber.knowledge.sqlite_store import SQLiteKnowledgeStore


def _multi_db_registry() -> DatabaseRegistry:
    return DatabaseRegistry(
        [
            DatabaseEntry.from_connection(
                name=name,
                connection=SQLiteConnection("sqlite:///:memory:"),
                description=None,
                excluded_schemas=[],
            )
            for name in ("prod", "staging")
        ]
    )


@pytest.mark.asyncio
async def test_multi_db_knowledge_uses_supplied_manager(temp_dir) -> None:
    manager = KnowledgeManager(
        store=SQLiteKnowledgeStore(db_path=temp_dir / "knowledge.db")
    )
    await manager.add_knowledge(
        database_name="prod",
        name="Revenue",
        description="production revenue definition",
    )
    await manager.add_knowledge(
        database_name="staging",
        name="Revenue",
        description="staging revenue definition",
    )
    capability = Knowledge(
        knowledge_manager=manager,
        registry=_multi_db_registry(),
    )
    tool = capability.display_specs["search_knowledge"]

    try:
        result = json.loads(await tool.execute("revenue", db_name="prod"))
    finally:
        await manager.close()

    assert result["total_results"] == 1
    assert result["results"][0]["description"] == "production revenue definition"
