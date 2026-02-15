from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

import aiosqlite
import pytest

from sqlsaber.knowledge.models import KnowledgeEntry
from sqlsaber.knowledge.sqlite_store import SQLiteKnowledgeStore


def _entry(
    *,
    database_name: str,
    name: str,
    description: str,
    sql: str | None = None,
    source: str | None = None,
    created_at: float = 100.0,
    updated_at: float = 100.0,
) -> KnowledgeEntry:
    return KnowledgeEntry(
        id=str(uuid.uuid4()),
        database_name=database_name,
        name=name,
        description=description,
        sql=sql,
        source=source,
        created_at=created_at,
        updated_at=updated_at,
    )


@pytest.mark.asyncio
async def test_sqlite_store_add_get_list_remove_clear(temp_dir):
    store = SQLiteKnowledgeStore(db_path=temp_dir / "knowledge.db")
    await store.initialize()

    first = _entry(
        database_name="db_a",
        name="Revenue KPI",
        description="Tracks monthly revenue totals.",
        source="manual",
        updated_at=100.0,
    )
    second = _entry(
        database_name="db_a",
        name="Order Count KPI",
        description="Tracks order volume by day.",
        updated_at=200.0,
    )

    await store.add(first)
    await store.add(second)

    fetched = await store.get("db_a", first.id)
    assert fetched is not None
    assert fetched.id == first.id
    assert fetched.database_name == "db_a"

    listed = await store.list_all("db_a")
    assert [entry.id for entry in listed] == [second.id, first.id]

    assert await store.remove("db_a", first.id) is True
    assert await store.remove("db_a", first.id) is False

    cleared = await store.clear("db_a")
    assert cleared == 1
    assert await store.list_all("db_a") == []


@pytest.mark.asyncio
async def test_sqlite_store_search_or_mode_and_db_isolation(temp_dir):
    store = SQLiteKnowledgeStore(db_path=temp_dir / "knowledge.db")

    in_scope_best = _entry(
        database_name="db_a",
        name="Monthly Revenue Report",
        description="Monthly revenue trend and totals.",
        updated_at=20.0,
    )
    in_scope_partial = _entry(
        database_name="db_a",
        name="Revenue Glossary",
        description="Definitions for revenue metrics.",
        updated_at=10.0,
    )
    out_of_scope = _entry(
        database_name="db_b",
        name="Monthly Revenue Report",
        description="Should not leak to db_a search.",
    )

    await store.add(in_scope_best)
    await store.add(in_scope_partial)
    await store.add(out_of_scope)

    results = await store.search("db_a", "revenue monthly", limit=10)
    assert len(results) == 2
    assert results[0].id == in_scope_best.id
    assert {entry.id for entry in results} == {in_scope_best.id, in_scope_partial.id}

    db_b_results = await store.search("db_b", "revenue monthly", limit=10)
    assert len(db_b_results) == 1
    assert db_b_results[0].id == out_of_scope.id


@pytest.mark.asyncio
async def test_sqlite_store_update_entry(temp_dir):
    store = SQLiteKnowledgeStore(db_path=temp_dir / "knowledge.db")
    original = _entry(
        database_name="db_a",
        name="Sales Notes",
        description="Baseline description.",
        updated_at=5.0,
    )
    await store.add(original)

    original.name = "Sales Runbook"
    original.description = "Updated process for sales reporting."
    original.sql = "SELECT * FROM sales"
    original.source = "wiki"
    original.updated_at = 50.0

    updated = await store.update(original)
    assert updated is True

    fetched = await store.get("db_a", original.id)
    assert fetched is not None
    assert fetched.name == "Sales Runbook"
    assert fetched.sql == "SELECT * FROM sales"

    not_found = _entry(
        database_name="db_a",
        name="ghost",
        description="ghost",
    )
    assert await store.update(not_found) is False


@pytest.mark.asyncio
async def test_sqlite_store_handles_concurrent_adds(temp_dir):
    store = SQLiteKnowledgeStore(db_path=temp_dir / "knowledge.db")

    entries = [
        _entry(
            database_name="db_a",
            name=f"entry-{i}",
            description=f"description-{i}",
            updated_at=float(i),
        )
        for i in range(12)
    ]

    await asyncio.gather(*(store.add(entry) for entry in entries))
    listed = await store.list_all("db_a")
    assert len(listed) == 12


@pytest.mark.asyncio
async def test_sqlite_store_initialize_skips_rebuild_when_fts_present(temp_dir):
    db_path = temp_dir / "knowledge.db"
    seeded_store = SQLiteKnowledgeStore(db_path=db_path)
    await seeded_store.add(
        _entry(
            database_name="db_a",
            name="Revenue",
            description="Revenue notes.",
        )
    )

    async with aiosqlite.connect(db_path) as db:
        assert await seeded_store._needs_fts_rebuild(db) is False


@pytest.mark.asyncio
async def test_sqlite_store_initialize_rebuilds_fts_when_missing(temp_dir):
    db_path = temp_dir / "knowledge.db"
    seeded_store = SQLiteKnowledgeStore(db_path=db_path)
    entry = _entry(
        database_name="db_a",
        name="Revenue",
        description="Revenue notes.",
    )
    await seeded_store.add(entry)

    async with aiosqlite.connect(db_path) as db:
        await db.execute("DELETE FROM knowledge_fts")
        await db.commit()
        assert await seeded_store._needs_fts_rebuild(db) is True

    store = SQLiteKnowledgeStore(db_path=db_path)
    await store.initialize()
    results = await store.search("db_a", "revenue", limit=10)

    assert [item.id for item in results] == [entry.id]


@pytest.mark.asyncio
async def test_sqlite_store_initialize_skips_chmod_for_existing_parent_dir(
    temp_dir, monkeypatch
):
    db_path = temp_dir / "knowledge.db"
    store = SQLiteKnowledgeStore(db_path=db_path)
    calls: list[tuple[Path, bool]] = []

    def _track_permissions(path: Path, is_directory: bool = False) -> None:
        calls.append((Path(path), is_directory))

    monkeypatch.setattr(store, "_set_secure_permissions", _track_permissions)

    await store.initialize()

    assert (db_path.parent, True) not in calls
    assert (db_path, False) in calls


@pytest.mark.asyncio
async def test_sqlite_store_initialize_chmods_newly_created_parent_dir(
    temp_dir, monkeypatch
):
    db_path = temp_dir / "nested" / "knowledge.db"
    store = SQLiteKnowledgeStore(db_path=db_path)
    calls: list[tuple[Path, bool]] = []

    def _track_permissions(path: Path, is_directory: bool = False) -> None:
        calls.append((Path(path), is_directory))

    monkeypatch.setattr(store, "_set_secure_permissions", _track_permissions)

    await store.initialize()

    assert (db_path.parent, True) in calls
    assert (db_path, False) in calls
