"""Knowledge management primitives and storage backends."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlsaber.knowledge.base_store import BaseKnowledgeStore
    from sqlsaber.knowledge.manager import KnowledgeManager
    from sqlsaber.knowledge.models import KnowledgeEntry
    from sqlsaber.knowledge.sqlite_store import SQLiteKnowledgeStore

__all__ = [
    "KnowledgeEntry",
    "BaseKnowledgeStore",
    "SQLiteKnowledgeStore",
    "KnowledgeManager",
]


def __getattr__(name: str):
    if name == "KnowledgeEntry":
        from sqlsaber.knowledge.models import KnowledgeEntry

        return KnowledgeEntry
    if name == "BaseKnowledgeStore":
        from sqlsaber.knowledge.base_store import BaseKnowledgeStore

        return BaseKnowledgeStore
    if name == "SQLiteKnowledgeStore":
        from sqlsaber.knowledge.sqlite_store import SQLiteKnowledgeStore

        return SQLiteKnowledgeStore
    if name == "KnowledgeManager":
        from sqlsaber.knowledge.manager import KnowledgeManager

        return KnowledgeManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
