"""Knowledge-base search tool for agent context retrieval."""

from __future__ import annotations

from sqlsaber.knowledge.manager import KnowledgeManager
from sqlsaber.utils.json_utils import json_dumps

from .base import Tool
from .display import (
    ColumnDef,
    DisplayMetadata,
    ExecutingConfig,
    FieldMappings,
    ResultConfig,
    TableConfig,
    ToolDisplaySpec,
)
from .registry import register_tool


class KnowledgeTool(Tool):
    """Base class for tools that require knowledge manager context."""

    def __init__(
        self,
        database_name: str | None = None,
        knowledge_manager: KnowledgeManager | None = None,
    ):
        super().__init__()
        self.database_name = database_name
        self.knowledge_manager = knowledge_manager

    def set_context(
        self, database_name: str | None, knowledge_manager: KnowledgeManager
    ) -> None:
        """Set contextual dependencies after tool construction."""
        self.database_name = database_name
        self.knowledge_manager = knowledge_manager


@register_tool
class SearchKnowledgeTool(KnowledgeTool):
    """Search saved database knowledge by keyword."""

    display_spec = ToolDisplaySpec(
        executing=ExecutingConfig(
            message="Searching knowledge base",
            icon="🔎",
            show_args=["query"],
        ),
        result=ResultConfig(
            format="table",
            title="Knowledge Matches ({total_results} total)",
            fields=FieldMappings(items="results", error="error"),
            table=TableConfig(
                columns=[
                    ColumnDef(field="name", header="Name", style="column.name"),
                    ColumnDef(
                        field="description", header="Description", style="column.type"
                    ),
                    ColumnDef(field="sql", header="SQL", style="muted"),
                    ColumnDef(field="source", header="Source", style="info"),
                ],
                max_rows=20,
            ),
        ),
        metadata=DisplayMetadata(display_name="Search Knowledge"),
    )

    @property
    def name(self) -> str:
        return "search_knowledge"

    async def execute(self, query: str) -> str:
        """Search existing sql and knowledge about active database.

        When to use this tool:
            - Whenever user asks a question about their data
            - To look for existing query patterns
            - To understand metrics, terminology, and references user makes in their request

        Args:
            query: The keyword search query to execute.
        """

        if not query.strip():
            return json_dumps({"error": "No query provided."})

        if self.knowledge_manager is None or not self.database_name:
            return json_dumps(
                {
                    "error": (
                        "Knowledge context is unavailable for this session. "
                        "Set an active database first."
                    )
                }
            )

        try:
            entries = await self.knowledge_manager.search_knowledge(
                self.database_name, query, limit=10
            )
        except Exception as exc:
            return json_dumps({"error": f"Error searching knowledge: {str(exc)}"})

        return json_dumps(
            {
                "total_results": len(entries),
                "results": [
                    {
                        "id": entry.id,
                        "name": entry.name,
                        "description": entry.description,
                        "sql": entry.sql or "",
                        "source": entry.source or "",
                    }
                    for entry in entries
                ],
            }
        )
