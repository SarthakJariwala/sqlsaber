"""SQLite-backed knowledge store with FTS5 search."""

from __future__ import annotations

import asyncio
import os
import platform
import stat
from pathlib import Path

import aiosqlite
import platformdirs

from sqlsaber.knowledge.base_store import BaseKnowledgeStore
from sqlsaber.knowledge.models import KnowledgeEntry

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS knowledge (
    id TEXT PRIMARY KEY,
    database_name TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    sql TEXT,
    source TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_knowledge_database_name
    ON knowledge(database_name);

CREATE INDEX IF NOT EXISTS idx_knowledge_database_updated
    ON knowledge(database_name, updated_at DESC);

CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts
USING fts5(
    name,
    description,
    sql,
    content='knowledge',
    content_rowid='rowid'
);

CREATE TRIGGER IF NOT EXISTS knowledge_ai AFTER INSERT ON knowledge BEGIN
    INSERT INTO knowledge_fts(rowid, name, description, sql)
    VALUES (new.rowid, new.name, new.description, COALESCE(new.sql, ''));
END;

CREATE TRIGGER IF NOT EXISTS knowledge_ad AFTER DELETE ON knowledge BEGIN
    INSERT INTO knowledge_fts(knowledge_fts, rowid, name, description, sql)
    VALUES ('delete', old.rowid, old.name, old.description, COALESCE(old.sql, ''));
END;

CREATE TRIGGER IF NOT EXISTS knowledge_au AFTER UPDATE ON knowledge BEGIN
    INSERT INTO knowledge_fts(knowledge_fts, rowid, name, description, sql)
    VALUES ('delete', old.rowid, old.name, old.description, COALESCE(old.sql, ''));
    INSERT INTO knowledge_fts(rowid, name, description, sql)
    VALUES (new.rowid, new.name, new.description, COALESCE(new.sql, ''));
END;
"""


class SQLiteKnowledgeStore(BaseKnowledgeStore):
    """Knowledge store using SQLite with FTS5 BM25 ranking."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        default_path = Path(platformdirs.user_data_dir("sqlsaber")) / "knowledge.db"
        self.db_path = Path(db_path) if db_path is not None else default_path

        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize schema and FTS index."""
        if self._initialized:
            return

        async with self._init_lock:
            if self._initialized:
                return

            parent_dir_existed = self.db_path.parent.exists()
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            if not parent_dir_existed:
                self._set_secure_permissions(self.db_path.parent, is_directory=True)

            async with aiosqlite.connect(self.db_path) as db:
                await db.executescript(SCHEMA_SQL)
                await self._maybe_rebuild_fts_index(db)
                await db.commit()

            self._set_secure_permissions(self.db_path, is_directory=False)
            self._initialized = True

    async def _maybe_rebuild_fts_index(self, db: aiosqlite.Connection) -> None:
        """Rebuild FTS only for legacy databases where index rows are missing."""
        if not await self._needs_fts_rebuild(db):
            return

        await db.execute("INSERT INTO knowledge_fts(knowledge_fts) VALUES ('rebuild')")

    async def _needs_fts_rebuild(self, db: aiosqlite.Connection) -> bool:
        """Detect legacy databases where FTS index has not been built yet."""
        async with db.execute("SELECT EXISTS(SELECT 1 FROM knowledge LIMIT 1)") as cur:
            knowledge_row = await cur.fetchone()
        if not knowledge_row or not bool(knowledge_row[0]):
            return False

        try:
            async with db.execute(
                "SELECT EXISTS(SELECT 1 FROM knowledge_fts_docsize LIMIT 1)"
            ) as cur:
                docsize_row = await cur.fetchone()
        except aiosqlite.OperationalError:
            # Unexpected FTS schema shape: safest fallback is a one-time rebuild.
            return True

        return not bool(docsize_row and docsize_row[0])

    async def close(self) -> None:
        """Close store resources (no-op for per-operation SQLite connections)."""
        pass

    async def add(self, entry: KnowledgeEntry) -> None:
        """Insert a new knowledge entry."""
        await self.initialize()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO knowledge (
                    id, database_name, name, description, sql, source,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.id,
                    entry.database_name,
                    entry.name,
                    entry.description,
                    entry.sql,
                    entry.source,
                    entry.created_at,
                    entry.updated_at,
                ),
            )
            await db.commit()

    async def get(self, database_name: str, entry_id: str) -> KnowledgeEntry | None:
        """Get an entry by id for a database."""
        await self.initialize()
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """
                SELECT id, database_name, name, description, sql, source,
                       created_at, updated_at
                FROM knowledge
                WHERE database_name = ? AND id = ?
                """,
                (database_name, entry_id),
            ) as cursor:
                row = await cursor.fetchone()
                if row is None:
                    return None
                return self._row_to_entry(row)

    async def search(
        self, database_name: str, query: str, limit: int = 10
    ) -> list[KnowledgeEntry]:
        """Search entries via FTS5 MATCH and BM25 ranking."""
        await self.initialize()

        if not query.strip():
            return []

        final_query = self._prepare_fts_query(query)
        if not final_query:
            return []

        max_results = max(1, limit)
        rows = await self._run_fts_query(database_name, final_query, max_results)
        if rows:
            return [self._row_to_entry(row) for row in rows]

        # If user query includes unsupported FTS syntax, retry with quoted tokens.
        fallback_query = self._quoted_token_query(query)
        if not fallback_query or fallback_query == final_query:
            return []

        fallback_rows = await self._run_fts_query(
            database_name, fallback_query, max_results
        )
        return [self._row_to_entry(row) for row in fallback_rows]

    async def list_all(self, database_name: str) -> list[KnowledgeEntry]:
        """List all entries for a database."""
        await self.initialize()
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """
                SELECT id, database_name, name, description, sql, source,
                       created_at, updated_at
                FROM knowledge
                WHERE database_name = ?
                ORDER BY updated_at DESC
                """,
                (database_name,),
            ) as cursor:
                rows = await cursor.fetchall()
                return [self._row_to_entry(row) for row in rows]

    async def remove(self, database_name: str, entry_id: str) -> bool:
        """Delete an entry by id for a database."""
        await self.initialize()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "DELETE FROM knowledge WHERE database_name = ? AND id = ?",
                (database_name, entry_id),
            )
            changed = db.total_changes
            await db.commit()
            return changed > 0

    async def clear(self, database_name: str) -> int:
        """Delete all entries for a database and return deleted count."""
        await self.initialize()
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT COUNT(*) FROM knowledge WHERE database_name = ?",
                (database_name,),
            ) as cursor:
                row = await cursor.fetchone()
                count = int(row[0]) if row else 0

            if count > 0:
                await db.execute(
                    "DELETE FROM knowledge WHERE database_name = ?",
                    (database_name,),
                )
                await db.commit()

            return count

    async def update(self, entry: KnowledgeEntry) -> bool:
        """Update an existing entry."""
        await self.initialize()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                UPDATE knowledge
                SET name = ?,
                    description = ?,
                    sql = ?,
                    source = ?,
                    updated_at = ?
                WHERE database_name = ? AND id = ?
                """,
                (
                    entry.name,
                    entry.description,
                    entry.sql,
                    entry.source,
                    entry.updated_at,
                    entry.database_name,
                    entry.id,
                ),
            )
            changed = db.total_changes
            await db.commit()
            return changed > 0

    async def _run_fts_query(
        self, database_name: str, query: str, limit: int
    ) -> list[aiosqlite.Row]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            try:
                async with db.execute(
                    """
                    SELECT k.id, k.database_name, k.name, k.description,
                           k.sql, k.source, k.created_at, k.updated_at
                    FROM knowledge_fts
                    JOIN knowledge AS k ON k.rowid = knowledge_fts.rowid
                    WHERE knowledge_fts MATCH ?
                      AND k.database_name = ?
                    ORDER BY bm25(knowledge_fts), k.updated_at DESC
                    LIMIT ?
                    """,
                    (query, database_name, limit),
                ) as cursor:
                    return list(await cursor.fetchall())
            except aiosqlite.OperationalError:
                return []

    def _prepare_fts_query(self, raw_query: str) -> str:
        """Convert free-text input into OR-mode FTS terms."""
        stripped = raw_query.strip()
        if not stripped:
            return ""

        upper = stripped.upper()
        has_fts_operators = any(
            token in upper for token in (" OR ", " AND ", " NOT ", " NEAR ", '"')
        )
        if has_fts_operators or "(" in stripped or ")" in stripped:
            return stripped

        tokens = [token for token in stripped.split() if token]
        if not tokens:
            return ""
        if len(tokens) == 1:
            return tokens[0]
        return " OR ".join(tokens)

    def _quoted_token_query(self, raw_query: str) -> str:
        tokens = [token.strip() for token in raw_query.split() if token.strip()]
        if not tokens:
            return ""
        if len(tokens) == 1:
            token = self._sanitize_token(tokens[0])
            return f'"{token}"' if token else ""
        quoted = [f'"{self._sanitize_token(token)}"' for token in tokens]
        return " OR ".join(token for token in quoted if token != '""')

    def _set_secure_permissions(self, path: Path, is_directory: bool = False) -> None:
        try:
            if platform.system() == "Windows":
                return
            if is_directory:
                os.chmod(path, stat.S_IRWXU)
            else:
                os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
        except (OSError, PermissionError):
            pass

    def _row_to_entry(self, row: aiosqlite.Row) -> KnowledgeEntry:
        return KnowledgeEntry(
            id=str(row["id"]),
            database_name=str(row["database_name"]),
            name=str(row["name"]),
            description=str(row["description"]),
            sql=str(row["sql"]) if row["sql"] is not None else None,
            source=str(row["source"]) if row["source"] is not None else None,
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
        )

    def _sanitize_token(self, token: str) -> str:
        return token.replace('"', "")
