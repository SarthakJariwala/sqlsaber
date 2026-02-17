"""Knowledge entry models."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class KnowledgeEntry:
    """Represents a single knowledge entry for a database."""

    id: str
    database_name: str
    name: str
    description: str
    sql: str | None = None
    source: str | None = None
    created_at: float = 0.0
    updated_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert the entry to a JSON-serializable dictionary."""
        return {
            "id": self.id,
            "database_name": self.database_name,
            "name": self.name,
            "description": self.description,
            "sql": self.sql,
            "source": self.source,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KnowledgeEntry:
        """Create a knowledge entry from serialized data."""
        return cls(
            id=str(data["id"]),
            database_name=str(data["database_name"]),
            name=str(data["name"]),
            description=str(data["description"]),
            sql=str(data["sql"]) if data.get("sql") is not None else None,
            source=str(data["source"]) if data.get("source") is not None else None,
            created_at=float(data["created_at"]),
            updated_at=float(data["updated_at"]),
        )

    def formatted_created_at(self) -> str:
        """Get a human-readable created timestamp."""
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.created_at))

    def formatted_updated_at(self) -> str:
        """Get a human-readable updated timestamp."""
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.updated_at))
