"""Base protocol for knowledge storage backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

from sqlsaber.knowledge.models import KnowledgeEntry


class BaseKnowledgeStore(ABC):
    """Abstract interface for knowledge stores."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize store resources and schema."""

    @abstractmethod
    async def close(self) -> None:
        """Close any open resources."""

    @abstractmethod
    async def add(self, entry: KnowledgeEntry) -> None:
        """Persist a new knowledge entry."""

    @abstractmethod
    async def get(self, database_name: str, entry_id: str) -> KnowledgeEntry | None:
        """Get a single entry by id scoped to a database."""

    @abstractmethod
    async def search(
        self, database_name: str, query: str, limit: int = 10
    ) -> list[KnowledgeEntry]:
        """Search entries for a database."""

    @abstractmethod
    async def list_all(self, database_name: str) -> list[KnowledgeEntry]:
        """List all entries for a database."""

    @abstractmethod
    async def remove(self, database_name: str, entry_id: str) -> bool:
        """Remove an entry by id scoped to a database."""

    @abstractmethod
    async def clear(self, database_name: str) -> int:
        """Clear all entries for a database and return deleted count."""

    @abstractmethod
    async def update(self, entry: KnowledgeEntry) -> bool:
        """Update an entry and return whether it existed."""
