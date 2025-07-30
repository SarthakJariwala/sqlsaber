"""Test conversation storage functionality."""

import tempfile
from pathlib import Path

import pytest

from sqlsaber.conversation.manager import ConversationManager
from sqlsaber.conversation.storage import ConversationStorage


@pytest.fixture
def temp_storage():
    """Create a temporary conversation storage for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        storage = ConversationStorage()
        storage.db_path = Path(tmp_dir) / "test_conversations.db"
        yield storage


@pytest.mark.asyncio
async def test_conversation_creation(temp_storage):
    """Test creating a new conversation."""
    conversation_id = await temp_storage.create_conversation("test_db")

    assert conversation_id is not None
    assert len(conversation_id) > 0

    # Verify conversation exists
    conversation = await temp_storage.get_conversation(conversation_id)
    assert conversation is not None
    assert conversation.database_name == "test_db"
    assert conversation.ended_at is None


@pytest.mark.asyncio
async def test_message_storage(temp_storage):
    """Test storing messages in a conversation."""
    conversation_id = await temp_storage.create_conversation("test_db")

    # Add user message
    user_msg_id = await temp_storage.add_message(
        conversation_id, "user", "What tables do we have?", 0
    )
    assert user_msg_id is not None

    # Add assistant message
    assistant_content = [{"type": "text", "text": "Here are the tables..."}]
    assistant_msg_id = await temp_storage.add_message(
        conversation_id, "assistant", assistant_content, 1
    )
    assert assistant_msg_id is not None

    # Retrieve messages
    messages = await temp_storage.get_conversation_messages(conversation_id)
    assert len(messages) == 2

    # Check order and content
    assert messages[0].role == "user"
    assert messages[0].content == "What tables do we have?"
    assert messages[0].index_in_conv == 0

    assert messages[1].role == "assistant"
    assert messages[1].content == assistant_content
    assert messages[1].index_in_conv == 1


@pytest.mark.asyncio
async def test_conversation_manager():
    """Test the conversation manager."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        manager = ConversationManager()
        manager._storage.db_path = Path(tmp_dir) / "test_conversations.db"

        # Start conversation
        conv_id = await manager.start_conversation("test_db")
        assert conv_id is not None

        # Add messages
        await manager.add_user_message(conv_id, "Hello", 0)
        await manager.add_assistant_message(
            conv_id, [{"type": "text", "text": "Hi there!"}], 1
        )

        # End conversation
        success = await manager.end_conversation(conv_id)
        assert success

        # Verify conversation is ended
        conversation = await manager.get_conversation(conv_id)
        assert conversation is not None
        assert conversation.ended_at is not None


@pytest.mark.asyncio
async def test_conversation_listing(temp_storage):
    """Test listing conversations."""
    # Create multiple conversations
    conv1 = await temp_storage.create_conversation("db1")
    conv2 = await temp_storage.create_conversation("db2")
    conv3 = await temp_storage.create_conversation("db1")

    # List all conversations
    all_conversations = await temp_storage.list_conversations()
    assert len(all_conversations) >= 3

    # List conversations for specific database
    db1_conversations = await temp_storage.list_conversations("db1")
    assert len(db1_conversations) == 2

    db2_conversations = await temp_storage.list_conversations("db2")
    assert len(db2_conversations) == 1


@pytest.mark.asyncio
async def test_conversation_restoration():
    """Test restoring conversation to agent history."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        manager = ConversationManager()
        manager._storage.db_path = Path(tmp_dir) / "test_conversations.db"

        # Create conversation with messages
        conv_id = await manager.start_conversation("test_db")
        await manager.add_user_message(conv_id, "Show me tables", 0)
        await manager.add_assistant_message(
            conv_id, [{"type": "text", "text": "Here are the tables..."}], 1
        )

        # Restore to agent history
        agent_history = []
        success = await manager.restore_conversation_to_agent(conv_id, agent_history)
        assert success
        assert len(agent_history) == 2
        assert agent_history[0]["role"] == "user"
        assert agent_history[0]["content"] == "Show me tables"
        assert agent_history[1]["role"] == "assistant"
