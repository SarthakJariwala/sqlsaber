"""High-level knowledge management operations."""

from __future__ import annotations

import asyncio
import time
import uuid

from sqlsaber.knowledge.base_store import BaseKnowledgeStore
from sqlsaber.knowledge.models import KnowledgeEntry
from sqlsaber.knowledge.sqlite_store import SQLiteKnowledgeStore


class KnowledgeManager:
    """Orchestrates knowledge operations via a pluggable store backend."""

    def __init__(self, store: BaseKnowledgeStore | None = None):
        self.store = store or SQLiteKnowledgeStore()
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize the configured store."""
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            await self.store.initialize()
            self._initialized = True

    async def close(self) -> None:
        """Close store resources."""
        await self.store.close()

    async def add_knowledge(
        self,
        database_name: str,
        name: str,
        description: str,
        sql: str | None = None,
        source: str | None = None,
    ) -> KnowledgeEntry:
        """Add a knowledge entry for a database."""
        await self._ensure_initialized()

        normalized_name = self._normalize_required(name, field="name")
        normalized_description = self._normalize_required(
            description, field="description"
        )
        normalized_sql = self._normalize_optional(sql)
        normalized_source = self._normalize_optional(source)
        now = time.time()

        entry = KnowledgeEntry(
            id=str(uuid.uuid4()),
            database_name=database_name,
            name=normalized_name,
            description=normalized_description,
            sql=normalized_sql,
            source=normalized_source,
            created_at=now,
            updated_at=now,
        )
        await self.store.add(entry)
        return entry

    async def search_knowledge(
        self, database_name: str, query: str, limit: int = 10
    ) -> list[KnowledgeEntry]:
        """Search knowledge entries for a database."""
        await self._ensure_initialized()
        if not query.strip():
            return []
        return await self.store.search(database_name, query, max(1, limit))

    async def get_knowledge(
        self, database_name: str, entry_id: str
    ) -> KnowledgeEntry | None:
        """Get a knowledge entry by id."""
        await self._ensure_initialized()
        return await self.store.get(database_name, entry_id)

    async def list_knowledge(self, database_name: str) -> list[KnowledgeEntry]:
        """List all knowledge entries for a database."""
        await self._ensure_initialized()
        return await self.store.list_all(database_name)

    async def remove_knowledge(self, database_name: str, entry_id: str) -> bool:
        """Remove a knowledge entry by id."""
        await self._ensure_initialized()
        return await self.store.remove(database_name, entry_id)

    async def clear_knowledge(self, database_name: str) -> int:
        """Remove all knowledge entries for a database."""
        await self._ensure_initialized()
        return await self.store.clear(database_name)

    async def update_knowledge(
        self,
        database_name: str,
        entry_id: str,
        name: str,
        description: str,
        sql: str | None = None,
        source: str | None = None,
    ) -> KnowledgeEntry | None:
        """Update a knowledge entry. Returns updated entry or None if not found."""
        await self._ensure_initialized()
        current = await self.store.get(database_name, entry_id)
        if current is None:
            return None

        current.name = self._normalize_required(name, field="name")
        current.description = self._normalize_required(description, field="description")
        current.sql = self._normalize_optional(sql)
        current.source = self._normalize_optional(source)
        current.updated_at = time.time()

        updated = await self.store.update(current)
        if not updated:
            return None
        return current

    def format_search_results_for_prompt(self, entries: list[KnowledgeEntry]) -> str:
        """Format entries for inclusion in model prompts."""
        if not entries:
            return ""

        lines: list[str] = []
        for entry in entries:
            lines.append(f"- {entry.name}: {entry.description}")
            if entry.sql:
                lines.append(f"  SQL: {entry.sql}")
            if entry.source:
                lines.append(f"  Source: {entry.source}")
        return "\n".join(lines)

    async def _ensure_initialized(self) -> None:
        if not self._initialized:
            await self.initialize()

    def _normalize_required(self, value: str, *, field: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError(f"Knowledge {field} cannot be empty.")
        return normalized

    def _normalize_optional(self, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None
