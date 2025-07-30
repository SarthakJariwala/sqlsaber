"""Manager for conversation storage operations."""

import logging
import uuid
from typing import Any

from .models import Conversation, ConversationMessage
from .storage import ConversationStorage

logger = logging.getLogger(__name__)


class ConversationManager:
    """High-level manager for conversation storage operations."""

    def __init__(self):
        """Initialize conversation manager."""
        self._storage = ConversationStorage()

    async def start_conversation(self, database_name: str) -> str:
        """Start a new conversation.

        Args:
            database_name: Name of the database for this conversation

        Returns:
            Conversation ID
        """
        try:
            return await self._storage.create_conversation(database_name)
        except Exception as e:
            logger.warning(f"Failed to start conversation: {e}")
            return str(uuid.uuid4())

    async def add_user_message(
        self, conversation_id: str, content: str | dict[str, Any], index: int
    ) -> str:
        """Add a user message to the conversation.

        Args:
            conversation_id: ID of the conversation
            content: Message content
            index: Sequential index in conversation

        Returns:
            Message ID
        """
        try:
            return await self._storage.add_message(
                conversation_id, "user", content, index
            )
        except Exception as e:
            logger.warning(f"Failed to add user message: {e}")
            return str(uuid.uuid4())

    async def add_assistant_message(
        self,
        conversation_id: str,
        content: list[dict[str, Any]] | dict[str, Any],
        index: int,
    ) -> str:
        """Add an assistant message to the conversation.

        Args:
            conversation_id: ID of the conversation
            content: Message content (typically ContentBlock list)
            index: Sequential index in conversation

        Returns:
            Message ID
        """
        try:
            return await self._storage.add_message(
                conversation_id, "assistant", content, index
            )
        except Exception as e:
            logger.warning(f"Failed to add assistant message: {e}")
            return str(uuid.uuid4())

    async def add_tool_message(
        self,
        conversation_id: str,
        content: list[dict[str, Any]] | dict[str, Any],
        index: int,
    ) -> str:
        """Add a tool/system message to the conversation.

        Args:
            conversation_id: ID of the conversation
            content: Message content (typically tool results)
            index: Sequential index in conversation

        Returns:
            Message ID
        """
        try:
            return await self._storage.add_message(
                conversation_id, "tool", content, index
            )
        except Exception as e:
            logger.warning(f"Failed to add tool message: {e}")
            return str(uuid.uuid4())

    async def end_conversation(self, conversation_id: str) -> bool:
        """End a conversation.

        Args:
            conversation_id: ID of the conversation to end

        Returns:
            True if successfully ended, False otherwise
        """
        try:
            return await self._storage.end_conversation(conversation_id)
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
        try:
            return await self._storage.get_conversation(conversation_id)
        except Exception as e:
            logger.warning(f"Failed to get conversation: {e}")
            return None

    async def get_conversation_messages(
        self, conversation_id: str
    ) -> list[ConversationMessage]:
        """Get all messages for a conversation.

        Args:
            conversation_id: ID of the conversation

        Returns:
            List of messages ordered by index
        """
        try:
            return await self._storage.get_conversation_messages(conversation_id)
        except Exception as e:
            logger.warning(f"Failed to get conversation messages: {e}")
            return []

    async def list_conversations(
        self, database_name: str | None = None, limit: int = 50
    ) -> list[Conversation]:
        """List conversations.

        Args:
            database_name: Optional database name filter
            limit: Maximum number of conversations to return

        Returns:
            List of conversations ordered by start time (newest first)
        """
        try:
            return await self._storage.list_conversations(database_name, limit)
        except Exception as e:
            logger.warning(f"Failed to list conversations: {e}")
            return []

    async def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation.

        Args:
            conversation_id: ID of the conversation to delete

        Returns:
            True if successfully deleted, False otherwise
        """
        try:
            return await self._storage.delete_conversation(conversation_id)
        except Exception as e:
            logger.warning(f"Failed to delete conversation: {e}")
            return False

    async def get_database_names(self) -> list[str]:
        """Get list of database names with conversations.

        Returns:
            List of unique database names
        """
        try:
            return await self._storage.get_database_names()
        except Exception as e:
            logger.warning(f"Failed to get database names: {e}")
            return []

    async def restore_conversation_to_agent(
        self, conversation_id: str, agent_history: list[dict[str, Any]]
    ) -> bool:
        """Restore a conversation's messages to an agent's in-memory history.

        Args:
            conversation_id: ID of the conversation to restore
            agent_history: Agent's conversation_history list to populate

        Returns:
            True if successfully restored, False otherwise
        """
        try:
            messages = await self.get_conversation_messages(conversation_id)

            # Clear existing history
            agent_history.clear()

            # Convert messages back to agent format
            for msg in messages:
                if msg.role in ("user", "assistant", "tool"):
                    agent_history.append({"role": msg.role, "content": msg.content})

            logger.debug(f"Restored {len(messages)} messages to agent history")
            return True

        except Exception as e:
            logger.warning(f"Failed to restore conversation to agent: {e}")
            return False
