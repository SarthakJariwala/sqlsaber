"""Registry of databases connected to a single SQLSaber session."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from sqlsaber.database import DatabaseConnection
from sqlsaber.database.base import BaseDatabaseConnection
from sqlsaber.database.resolver import ResolvedDatabase
from sqlsaber.database.schema import SchemaManager


class UnknownDatabaseError(KeyError):
    """Raised when a registry lookup uses a name that is not registered."""


@dataclass
class DatabaseEntry:
    """A live, registered database connection together with its metadata."""

    name: str
    connection: BaseDatabaseConnection
    schema_manager: SchemaManager
    description: str | None
    excluded_schemas: list[str]

    @property
    def dialect(self) -> str:
        return self.connection.sqlglot_dialect

    @property
    def display_name(self) -> str:
        return self.connection.display_name

    @classmethod
    def from_connection(
        cls,
        *,
        name: str,
        connection: BaseDatabaseConnection,
        description: str | None,
        excluded_schemas: list[str],
    ) -> "DatabaseEntry":
        return cls(
            name=name,
            connection=connection,
            schema_manager=SchemaManager(connection),
            description=description,
            excluded_schemas=list(excluded_schemas),
        )


class DatabaseRegistry:
    """Ordered, name-keyed collection of database entries for one session."""

    def __init__(self, entries: list[DatabaseEntry]):
        if not entries:
            raise ValueError("DatabaseRegistry requires at least one entry.")

        seen: set[str] = set()
        for entry in entries:
            if entry.name in seen:
                raise ValueError(
                    f"Cannot register duplicate database name '{entry.name}'."
                )
            seen.add(entry.name)

        self._entries: dict[str, DatabaseEntry] = {e.name: e for e in entries}

    @classmethod
    def from_resolved(cls, resolved: list[ResolvedDatabase]) -> "DatabaseRegistry":
        """Build a registry from a list of resolver outputs (eager connect)."""
        entries: list[DatabaseEntry] = []
        for item in resolved:
            connection = DatabaseConnection(
                item.connection_string, excluded_schemas=item.excluded_schemas
            )
            entries.append(
                DatabaseEntry.from_connection(
                    name=item.name,
                    connection=connection,
                    description=item.description,
                    excluded_schemas=list(item.excluded_schemas),
                )
            )
        return cls(entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._entries

    def __iter__(self) -> Iterator[DatabaseEntry]:
        return iter(self._entries.values())

    def names(self) -> list[str]:
        return list(self._entries.keys())

    def primary(self) -> str:
        """Return the name of the first-registered database."""
        return next(iter(self._entries))

    def get(self, name: str) -> DatabaseEntry:
        entry = self._entries.get(name)
        if entry is None:
            valid = ", ".join(self._entries.keys())
            raise UnknownDatabaseError(
                f"Unknown database '{name}'. Valid names: {valid}."
            )
        return entry

    def connection(self, name: str) -> BaseDatabaseConnection:
        return self.get(name).connection

    def schema_manager(self, name: str) -> SchemaManager:
        return self.get(name).schema_manager

    def dialect(self, name: str) -> str:
        return self.get(name).dialect

    def catalog(self) -> list[dict[str, str | None]]:
        """Return a serializable description of every registered database.

        Used by `list_dbs` and the multi-DB system prompt.
        """
        return [
            {
                "name": entry.name,
                "display_name": entry.display_name,
                "dialect": entry.dialect,
                "description": entry.description,
            }
            for entry in self._entries.values()
        ]

    async def close(self) -> None:
        """Close every connection. Aggregates errors and re-raises the first."""
        errors: list[BaseException] = []
        for entry in self._entries.values():
            try:
                await entry.connection.close()
            except BaseException as exc:
                errors.append(exc)
        if errors:
            raise errors[0]
