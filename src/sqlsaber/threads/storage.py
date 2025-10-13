"""SQLite storage for pydantic-ai thread snapshots.

Each thread represents a session (interactive or non-interactive) and stores the
complete pydantic-ai message history as a snapshot. On every completed run in the
same session, we overwrite the snapshot with the new full history.

This design intentionally avoids per-run append logs and mirrors pydantic-ai's
recommended approach of serializing ModelMessage[] with ModelMessagesTypeAdapter.
"""

import asyncio
import time
import uuid
from pathlib import Path
from typing import Any

import aiosqlite
import platformdirs
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter

from sqlsaber.config.logging import get_logger

logger = get_logger(__name__)


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS threads (
    id TEXT PRIMARY KEY,
    database_name TEXT,
    title TEXT,
    created_at REAL NOT NULL,
    ended_at REAL,
    last_activity_at REAL NOT NULL,
    model_name TEXT,
    messages_json BLOB NOT NULL,
    extra_metadata TEXT
);

CREATE INDEX IF NOT EXISTS idx_threads_dbname ON threads(database_name);
CREATE INDEX IF NOT EXISTS idx_threads_activity ON threads(last_activity_at);
"""


class Thread:
    """Thread metadata."""

    def __init__(
        self,
        id: str,
        database_name: str | None,
        title: str | None,
        created_at: float,
        ended_at: float | None,
        last_activity_at: float,
        model_name: str | None,
    ) -> None:
        self.id = id
        self.database_name = database_name
        self.title = title
        self.created_at = created_at
        self.ended_at = ended_at
        self.last_activity_at = last_activity_at
        self.model_name = model_name


class ThreadStorage:
    """Handles SQLite storage for pydantic-ai thread snapshots."""

    def __init__(self) -> None:
        self.db_path = Path(platformdirs.user_config_dir("sqlsaber")) / "threads.db"
        self._lock = asyncio.Lock()
        self._initialized = False

    async def _init_db(self) -> None:
        if self._initialized:
            return
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            async with aiosqlite.connect(self.db_path) as db:
                await db.executescript(SCHEMA_SQL)
                await db.commit()
            self._initialized = True
            logger.info("threads.db.init", path=str(self.db_path))
        except Exception as e:  # pragma: no cover - best-effort persistence
            logger.warning("threads.db.init_failed", error=str(e))

    async def save_snapshot(
        self,
        *,
        messages_json: bytes,
        database_name: str | None,
        thread_id: str | None = None,
        extra_metadata: str | None = None,
    ) -> str:
        """Create or update a thread snapshot."""
        await self._init_db()
        now = time.time()

        if thread_id is None:
            thread_id = str(uuid.uuid4())
            try:
                async with self._lock, aiosqlite.connect(self.db_path) as db:
                    await db.execute(
                        """
                        INSERT INTO threads (
                            id, database_name, created_at,
                            last_activity_at, messages_json, extra_metadata
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            thread_id,
                            database_name,
                            now,
                            now,
                            messages_json,
                            extra_metadata,
                        ),
                    )
                    await db.commit()
                logger.info("threads.create", thread_id=thread_id)
                return thread_id
            except Exception as e:  # pragma: no cover
                logger.warning("threads.create_failed", error=str(e))
                return thread_id
        else:
            try:
                async with self._lock, aiosqlite.connect(self.db_path) as db:
                    await db.execute(
                        """
                        UPDATE threads
                        SET last_activity_at = ?,
                            messages_json = ?,
                            extra_metadata = COALESCE(?, extra_metadata)
                        WHERE id = ?
                        """,
                        (
                            now,
                            messages_json,
                            extra_metadata,
                            thread_id,
                        ),
                    )
                    await db.commit()
                logger.info("threads.update_snapshot", thread_id=thread_id)
                return thread_id
            except Exception as e:  # pragma: no cover
                logger.warning(
                    "threads.update_snapshot_failed", thread_id=thread_id, error=str(e)
                )
                return thread_id

    async def save_metadata(
        self,
        *,
        thread_id: str,
        title: str | None = None,
        model_name: str | None = None,
    ) -> bool:
        """Update thread metadata (title/model/extra). Only provided fields are updated."""
        await self._init_db()

        try:
            async with self._lock, aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    """
                    UPDATE threads
                    SET title = ?, model_name = ?
                    WHERE id = ?
                    """,
                    (title, model_name, thread_id),
                )
                await db.commit()
            logger.info("threads.update_metadata", thread_id=thread_id)
            return True
        except Exception as e:  # pragma: no cover
            logger.warning(
                "threads.update_metadata_failed", thread_id=thread_id, error=str(e)
            )
            return False

    async def end_thread(self, thread_id: str) -> bool:
        await self._init_db()
        try:
            async with self._lock, aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    "UPDATE threads SET ended_at = ?, last_activity_at = ? WHERE id = ?",
                    (time.time(), time.time(), thread_id),
                )
                await db.commit()
            logger.info("threads.end", thread_id=thread_id)
            return True
        except Exception as e:  # pragma: no cover
            logger.warning("threads.end_failed", thread_id=thread_id, error=str(e))
            return False

    async def get_thread(self, thread_id: str) -> Thread | None:
        await self._init_db()
        try:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(
                    """
                    SELECT id, database_name, title, created_at, ended_at,
                           last_activity_at, model_name
                    FROM threads WHERE id = ?
                    """,
                    (thread_id,),
                ) as cur:
                    row = await cur.fetchone()
                    if not row:
                        return None
                    return Thread(
                        id=row[0],
                        database_name=row[1],
                        title=row[2],
                        created_at=row[3],
                        ended_at=row[4],
                        last_activity_at=row[5],
                        model_name=row[6],
                    )
        except Exception as e:  # pragma: no cover
            logger.warning("threads.get_failed", thread_id=thread_id, error=str(e))
            return None

    async def get_thread_messages(self, thread_id: str) -> list[ModelMessage]:
        """Load the full message history for a thread as ModelMessage[]."""
        await self._init_db()
        try:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(
                    "SELECT messages_json FROM threads WHERE id = ?",
                    (thread_id,),
                ) as cur:
                    row = await cur.fetchone()
                    if not row:
                        return []
                    messages_blob: bytes = row[0]
                    return ModelMessagesTypeAdapter.validate_json(messages_blob)
        except Exception as e:  # pragma: no cover
            logger.warning(
                "threads.get_messages_failed", thread_id=thread_id, error=str(e)
            )
            return []

    async def list_threads(
        self, *, database_name: str | None = None, limit: int = 50
    ) -> list[Thread]:
        await self._init_db()
        try:
            query = (
                "SELECT id, database_name, title, created_at, ended_at, last_activity_at, model_name"
                " FROM threads"
            )
            params: list[Any] = []
            if database_name:
                query += " WHERE database_name = ?"
                params.append(database_name)
            query += " ORDER BY last_activity_at DESC LIMIT ?"
            params.append(limit)

            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(query, params) as cur:
                    threads: list[Thread] = []
                    async for row in cur:
                        threads.append(
                            Thread(
                                id=row[0],
                                database_name=row[1],
                                title=row[2],
                                created_at=row[3],
                                ended_at=row[4],
                                last_activity_at=row[5],
                                model_name=row[6],
                            )
                        )
                    return threads
        except Exception as e:  # pragma: no cover
            logger.warning("threads.list_failed", error=str(e))
            return []

    async def delete_thread(self, thread_id: str) -> bool:
        await self._init_db()
        try:
            async with self._lock, aiosqlite.connect(self.db_path) as db:
                cur = await db.execute("DELETE FROM threads WHERE id = ?", (thread_id,))
                await db.commit()
                deleted = cur.rowcount > 0
                if deleted:
                    logger.info("threads.delete", thread_id=thread_id)
                return deleted
        except Exception as e:  # pragma: no cover
            logger.warning("threads.delete_failed", thread_id=thread_id, error=str(e))
            return False

    async def prune_threads(self, older_than_days: int = 30) -> int:
        """Delete threads whose last_activity_at is older than the cutoff.

        Args:
            older_than_days: Threads with last_activity_at older than this many days ago will be deleted.

        Returns:
            The number of rows deleted (best-effort; 0 on failure).
        """
        await self._init_db()
        cutoff = time.time() - older_than_days * 24 * 3600
        try:
            async with self._lock, aiosqlite.connect(self.db_path) as db:
                cur = await db.execute(
                    "DELETE FROM threads WHERE last_activity_at < ?",
                    (cutoff,),
                )
                await db.commit()
                deleted = cur.rowcount or 0
                logger.info("threads.prune", days=older_than_days, deleted=deleted)
                return deleted
        except Exception as e:  # pragma: no cover
            logger.warning("threads.prune_failed", days=older_than_days, error=str(e))
            return 0
