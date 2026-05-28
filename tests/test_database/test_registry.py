"""Tests for the multi-database registry that backs a SQLSaber session."""

from unittest.mock import AsyncMock

import pytest

from sqlsaber.database.registry import (
    DatabaseEntry,
    DatabaseRegistry,
    UnknownDatabaseError,
)
from sqlsaber.database.resolver import ResolvedDatabase
from sqlsaber.database.sqlite import SQLiteConnection


def _entry(name: str, *, description: str | None = None) -> DatabaseEntry:
    connection = SQLiteConnection("sqlite:///:memory:")
    return DatabaseEntry.from_connection(
        name=name,
        connection=connection,
        description=description,
        excluded_schemas=[],
    )


class TestDatabaseRegistry:
    def test_single_entry_basics(self):
        entry = _entry("solo")
        registry = DatabaseRegistry([entry])

        assert len(registry) == 1
        assert registry.names() == ["solo"]
        assert registry.primary() == "solo"
        assert "solo" in registry
        assert registry.get("solo") is entry
        assert registry.connection("solo") is entry.connection
        assert registry.schema_manager("solo") is entry.schema_manager
        assert registry.dialect("solo") == "sqlite"

    def test_ordered_names_match_construction(self):
        registry = DatabaseRegistry([_entry("prod"), _entry("staging"), _entry("dev")])
        assert registry.names() == ["prod", "staging", "dev"]
        assert registry.primary() == "prod"

    def test_get_unknown_raises_with_valid_names(self):
        registry = DatabaseRegistry([_entry("prod"), _entry("staging")])
        with pytest.raises(UnknownDatabaseError) as exc:
            registry.get("nope")
        assert "nope" in str(exc.value)
        assert "prod" in str(exc.value)
        assert "staging" in str(exc.value)

    def test_catalog_includes_metadata(self):
        registry = DatabaseRegistry(
            [
                _entry("prod", description="production OLTP"),
                _entry("warehouse"),
            ]
        )
        catalog = registry.catalog()
        assert catalog[0]["name"] == "prod"
        assert catalog[0]["dialect"] == "sqlite"
        assert catalog[0]["description"] == "production OLTP"
        assert catalog[1]["name"] == "warehouse"
        assert catalog[1]["description"] is None

    def test_duplicate_names_rejected(self):
        with pytest.raises(ValueError, match="duplicate"):
            DatabaseRegistry([_entry("prod"), _entry("prod")])

    def test_empty_registry_rejected(self):
        with pytest.raises(ValueError, match="at least one"):
            DatabaseRegistry([])

    @pytest.mark.asyncio
    async def test_close_closes_every_connection(self):
        entry_a = _entry("a")
        entry_b = _entry("b")
        entry_a.connection.close = AsyncMock()
        entry_b.connection.close = AsyncMock()

        registry = DatabaseRegistry([entry_a, entry_b])
        await registry.close()

        entry_a.connection.close.assert_awaited_once()
        entry_b.connection.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_aggregates_errors_but_still_closes_all(self):
        entry_a = _entry("a")
        entry_b = _entry("b")
        entry_c = _entry("c")
        entry_a.connection.close = AsyncMock(side_effect=RuntimeError("boom"))
        entry_b.connection.close = AsyncMock()
        entry_c.connection.close = AsyncMock()

        registry = DatabaseRegistry([entry_a, entry_b, entry_c])
        with pytest.raises(RuntimeError, match="boom"):
            await registry.close()

        entry_a.connection.close.assert_awaited_once()
        entry_b.connection.close.assert_awaited_once()
        entry_c.connection.close.assert_awaited_once()


class TestDatabaseRegistryFromResolved:
    def test_from_resolved_builds_connections(self):
        resolved = [
            ResolvedDatabase(
                name="solo",
                connection_string="sqlite:///:memory:",
                excluded_schemas=[],
                description="local sqlite",
            )
        ]
        registry = DatabaseRegistry.from_resolved(resolved)
        assert registry.names() == ["solo"]
        assert registry.dialect("solo") == "sqlite"
        assert registry.catalog()[0]["description"] == "local sqlite"
