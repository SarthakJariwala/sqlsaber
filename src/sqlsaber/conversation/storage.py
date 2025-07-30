"""SQLite storage implementation for conversation history."""

import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

import aiosqlite
import platformdirs

from .models import Conversation, ConversationMessage

logger = logging.getLogger(__name__)

# Database schema
SCHEMA_SQL = """
-- Conversations table
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    database_name TEXT NOT NULL,
    started_at REAL NOT NULL,
    ended_at REAL
);

-- Messages table
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    index_in_conv INTEGER NOT NULL,
    created_at REAL NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);

-- Indices for performance
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, index_in_conv);
CREATE INDEX IF NOT EXISTS idx_conv_dbname ON conversations(database_name);

-- Store schema version for future migrations
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);

INSERT OR IGNORE INTO schema_version (version) VALUES (1);
"""


class ConversationStorage:
    """Handles SQLite storage and retrieval of conversation history."""

    _DB_VERSION = 1

    def __init__(self):
        """Initialize conversation storage."""
        self.db_path = (
            Path(platformdirs.user_config_dir("sqlsaber")) / "conversations.db"
        )
        self._lock = asyncio.Lock()
        self._initialized: bool = False

    async def _init_db(self) -> None:
        """Initialize the database with schema if needed."""
        if self._initialized:
            return

        try:
            # Ensure parent directory exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            async with aiosqlite.connect(self.db_path) as db:
                await db.executescript(SCHEMA_SQL)
                await db.commit()

            self._initialized = True
            logger.debug(f"Initialized conversation database at {self.db_path}")

        except Exception as e:
            logger.warning(f"Failed to initialize conversation database: {e}")
            # Don't raise - let the system continue without persistence

    async def create_conversation(self, database_name: str) -> str:
        """Create a new conversation record.

        Args:
            database_name: Name of the database for this conversation

        Returns:
            Conversation ID (UUID)
        """
        await self._init_db()

        conversation_id = str(uuid.uuid4())
        started_at = time.time()

        try:
            async with self._lock, aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    """
                    INSERT INTO conversations (id, database_name, started_at)
                    VALUES (?, ?, ?)
                    """,
                    (conversation_id, database_name, started_at),
                )
                await db.commit()

            logger.debug(
                f"Created conversation {conversation_id} for db {database_name}"
            )
            return conversation_id

        except Exception as e:
            logger.warning(f"Failed to create conversation: {e}")
            return conversation_id  # Return ID anyway to allow in-memory operation

    async def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str | list[dict[str, Any]] | dict[str, Any],
        index_in_conv: int,
    ) -> str:
        """Add a message to a conversation.

        Args:
            conversation_id: ID of the conversation
            role: Message role (user, assistant, system, tool)
            content: Message content (will be JSON-serialized)
            index_in_conv: Sequential index of message in conversation

        Returns:
            Message ID (UUID)
        """
        await self._init_db()

        message_id = str(uuid.uuid4())
        created_at = time.time()

        try:
            async with self._lock, aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    """
                    INSERT INTO messages (id, conversation_id, role, content, index_in_conv, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        message_id,
                        conversation_id,
                        role,
                        json.dumps(content),
                        index_in_conv,
                        created_at,
                    ),
                )
                await db.commit()

            logger.debug(
                f"Added message {message_id} to conversation {conversation_id}"
            )
            return message_id

        except Exception as e:
            logger.warning(f"Failed to add message: {e}")
            return message_id  # Return ID anyway

    async def end_conversation(self, conversation_id: str) -> bool:
        """Mark a conversation as ended.

        Args:
            conversation_id: ID of the conversation to end

        Returns:
            True if successfully updated, False otherwise
        """
        await self._init_db()

        try:
            async with self._lock, aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    "UPDATE conversations SET ended_at = ? WHERE id = ?",
                    (time.time(), conversation_id),
                )
                await db.commit()

            logger.debug(f"Ended conversation {conversation_id}")
            return True

        except Exception as e:
            logger.warning(f"Failed to end conversation: {e}")
            return False

    async def get_conversation(self, conversation_id: str) -> Conversation | None:
        """Get a conversation by ID.

        Args:
            conversation_id: ID of the conversation

        Returns:
            Conversation object or None if not found
        """
        await self._init_db()

        try:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(
                    "SELECT id, database_name, started_at, ended_at FROM conversations WHERE id = ?",
                    (conversation_id,),
                ) as cursor:
                    row = await cursor.fetchone()
                    if not row:
                        return None

                    return Conversation(
                        id=row[0],
                        database_name=row[1],
                        started_at=row[2],
                        ended_at=row[3],
                    )

        except Exception as e:
            logger.warning(f"Failed to get conversation {conversation_id}: {e}")
            return None

    async def get_conversation_messages(
        self, conversation_id: str
    ) -> list[ConversationMessage]:
        """Get all messages for a conversation.

        Args:
            conversation_id: ID of the conversation

        Returns:
            List of ConversationMessage objects ordered by index
        """
        await self._init_db()

        try:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(
                    """
                    SELECT id, conversation_id, role, content, index_in_conv, created_at
                    FROM messages
                    WHERE conversation_id = ?
                    ORDER BY index_in_conv
                    """,
                    (conversation_id,),
                ) as cursor:
                    messages = []
                    async for row in cursor:
                        message = ConversationMessage.from_storage_data(
                            id_=row[0],
                            conversation_id=row[1],
                            role=row[2],
                            content_json=row[3],
                            index_in_conv=row[4],
                            created_at=row[5],
                        )
                        messages.append(message)

                    return messages

        except Exception as e:
            logger.warning(
                f"Failed to get messages for conversation {conversation_id}: {e}"
            )
            return []

    async def list_conversations(
        self, database_name: str | None = None, limit: int = 50
    ) -> list[Conversation]:
        """List conversations, optionally filtered by database.

        Args:
            database_name: Optional database name filter
            limit: Maximum number of conversations to return

        Returns:
            List of Conversation objects ordered by start time (newest first)
        """
        await self._init_db()

        try:
            query = """
                SELECT id, database_name, started_at, ended_at
                FROM conversations
            """
            params = []

            if database_name:
                query += " WHERE database_name = ?"
                params.append(database_name)

            query += " ORDER BY started_at DESC LIMIT ?"
            params.append(limit)

            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(query, params) as cursor:
                    conversations = []
                    async for row in cursor:
                        conversation = Conversation(
                            id=row[0],
                            database_name=row[1],
                            started_at=row[2],
                            ended_at=row[3],
                        )
                        conversations.append(conversation)

                    return conversations

        except Exception as e:
            logger.warning(f"Failed to list conversations: {e}")
            return []

    async def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and all its messages.

        Args:
            conversation_id: ID of the conversation to delete

        Returns:
            True if successfully deleted, False otherwise
        """
        await self._init_db()

        try:
            async with self._lock, aiosqlite.connect(self.db_path) as db:
                # Delete conversation (messages will be deleted by CASCADE)
                cursor = await db.execute(
                    "DELETE FROM conversations WHERE id = ?", (conversation_id,)
                )
                await db.commit()

                deleted = cursor.rowcount > 0
                if deleted:
                    logger.debug(f"Deleted conversation {conversation_id}")
                return deleted

        except Exception as e:
            logger.warning(f"Failed to delete conversation {conversation_id}: {e}")
            return False

    async def get_database_names(self) -> list[str]:
        """Get list of all database names that have conversations.

        Returns:
            List of unique database names
        """
        await self._init_db()

        try:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(
                    "SELECT DISTINCT database_name FROM conversations ORDER BY database_name"
                ) as cursor:
                    return [row[0] async for row in cursor]

        except Exception as e:
            logger.warning(f"Failed to get database names: {e}")
            return []
