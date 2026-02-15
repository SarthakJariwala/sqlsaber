from __future__ import annotations

from dataclasses import replace

import pytest

from sqlsaber.knowledge.base_store import BaseKnowledgeStore
from sqlsaber.knowledge.manager import KnowledgeManager
from sqlsaber.knowledge.models import KnowledgeEntry


class InMemoryKnowledgeStore(BaseKnowledgeStore):
    def __init__(self):
        self.entries: dict[tuple[str, str], KnowledgeEntry] = {}
        self.initialized = False
        self.closed = False

    async def initialize(self) -> None:
        self.initialized = True

    async def close(self) -> None:
        self.closed = True

    async def add(self, entry: KnowledgeEntry) -> None:
        self.entries[(entry.database_name, entry.id)] = entry

    async def get(self, database_name: str, entry_id: str) -> KnowledgeEntry | None:
        return self.entries.get((database_name, entry_id))

    async def search(
        self, database_name: str, query: str, limit: int = 10
    ) -> list[KnowledgeEntry]:
        lowered = query.lower()
        matches = [
            entry
            for (db_name, _), entry in self.entries.items()
            if db_name == database_name
            and (
                lowered in entry.name.lower()
                or lowered in entry.description.lower()
                or (entry.sql and lowered in entry.sql.lower())
            )
        ]
        return matches[:limit]

    async def list_all(self, database_name: str) -> list[KnowledgeEntry]:
        return [
            entry
            for (db_name, _), entry in self.entries.items()
            if db_name == database_name
        ]

    async def remove(self, database_name: str, entry_id: str) -> bool:
        return self.entries.pop((database_name, entry_id), None) is not None

    async def clear(self, database_name: str) -> int:
        keys = [key for key in self.entries if key[0] == database_name]
        for key in keys:
            del self.entries[key]
        return len(keys)

    async def update(self, entry: KnowledgeEntry) -> bool:
        key = (entry.database_name, entry.id)
        if key not in self.entries:
            return False
        self.entries[key] = entry
        return True


@pytest.mark.asyncio
async def test_manager_add_generates_uuid_and_timestamps():
    store = InMemoryKnowledgeStore()
    manager = KnowledgeManager(store=store)

    entry = await manager.add_knowledge(
        database_name="analytics",
        name="Revenue",
        description="Monthly revenue KPI",
        source="wiki",
    )

    assert store.initialized is True
    assert entry.id
    assert entry.created_at > 0
    assert entry.updated_at == entry.created_at
    assert entry.database_name == "analytics"


@pytest.mark.asyncio
async def test_manager_operations_are_database_scoped():
    store = InMemoryKnowledgeStore()
    manager = KnowledgeManager(store=store)

    first = await manager.add_knowledge(
        database_name="db_a",
        name="Orders",
        description="Order aggregation query",
    )
    second = await manager.add_knowledge(
        database_name="db_b",
        name="Orders",
        description="Different DB entry",
    )

    db_a_entries = await manager.list_knowledge("db_a")
    db_b_entries = await manager.list_knowledge("db_b")
    assert [entry.id for entry in db_a_entries] == [first.id]
    assert [entry.id for entry in db_b_entries] == [second.id]

    db_a_search = await manager.search_knowledge("db_a", "order")
    assert [entry.id for entry in db_a_search] == [first.id]

    removed = await manager.remove_knowledge("db_a", first.id)
    assert removed is True
    assert await manager.get_knowledge("db_a", first.id) is None


@pytest.mark.asyncio
async def test_manager_update_and_format_results():
    store = InMemoryKnowledgeStore()
    manager = KnowledgeManager(store=store)

    entry = await manager.add_knowledge(
        database_name="db_a",
        name="Retention Cohorts",
        description="Original description",
    )

    updated = await manager.update_knowledge(
        database_name="db_a",
        entry_id=entry.id,
        name="Retention Cohorts v2",
        description="Updated cohort query notes",
        sql="SELECT cohort_month, retained_users FROM cohorts",
        source="notion",
    )

    assert updated is not None
    assert updated.name == "Retention Cohorts v2"
    assert updated.sql is not None

    rendered = manager.format_search_results_for_prompt([replace(updated)])
    assert "Retention Cohorts v2" in rendered
    assert "SQL:" in rendered

    assert await manager.clear_knowledge("db_a") == 1
    await manager.close()
    assert store.closed is True
