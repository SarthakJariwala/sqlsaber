from __future__ import annotations

import pytest

from sqlsaber import SQLSaber, SQLSaberOptions
from sqlsaber.config.settings import Config
from sqlsaber.knowledge.manager import KnowledgeManager
from sqlsaber.knowledge.sqlite_store import SQLiteKnowledgeStore
from sqlsaber.tools.knowledge_tool import SearchKnowledgeTool


@pytest.mark.asyncio
async def test_api_knowledge_manager_is_passed_to_agent(temp_dir):
    manager = KnowledgeManager(
        store=SQLiteKnowledgeStore(db_path=temp_dir / "knowledge.db")
    )

    saber = SQLSaber(
        options=SQLSaberOptions(
            database="sqlite:///:memory:",
            settings=Config.in_memory(
                model_name="anthropic:claude-3-5-sonnet",
                api_keys={"anthropic": "test-key"},
            ),
            knowledge_manager=manager,
        )
    )

    try:
        assert saber.agent.knowledge_manager is manager
        tool = saber.agent._tools.get("search_knowledge")
        assert isinstance(tool, SearchKnowledgeTool)
        assert tool.knowledge_manager is manager
        assert tool.database_name == saber.db_name
    finally:
        await saber.close()
