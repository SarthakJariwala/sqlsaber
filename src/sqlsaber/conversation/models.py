"""Data models for conversation storage."""

import json
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class Conversation:
    """Represents a conversation session."""

    id: str
    database_name: str
    started_at: float
    ended_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "database_name": self.database_name,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Conversation":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            database_name=data["database_name"],
            started_at=data["started_at"],
            ended_at=data.get("ended_at"),
        )

    def formatted_start_time(self) -> str:
        """Get human-readable start timestamp."""
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.started_at))

    def formatted_end_time(self) -> str | None:
        """Get human-readable end timestamp."""
        if self.ended_at is None:
            return None
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.ended_at))

    def duration_seconds(self) -> float | None:
        """Get conversation duration in seconds."""
        if self.ended_at is None:
            return None
        return self.ended_at - self.started_at


@dataclass
class ConversationMessage:
    """Represents a single message in a conversation."""

    id: str
    conversation_id: str
    role: str
    content: dict[str, Any] | str
    index_in_conv: int
    created_at: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "role": self.role,
            "content": self.content,
            "index_in_conv": self.index_in_conv,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConversationMessage":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            conversation_id=data["conversation_id"],
            role=data["role"],
            content=data["content"],
            index_in_conv=data["index_in_conv"],
            created_at=data["created_at"],
        )

    def formatted_timestamp(self) -> str:
        """Get human-readable timestamp."""
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.created_at))

    def content_json(self) -> str:
        """Get content as JSON string for storage."""
        return json.dumps(self.content)

    @classmethod
    def from_storage_data(
        cls,
        id_: str,
        conversation_id: str,
        role: str,
        content_json: str,
        index_in_conv: int,
        created_at: float,
    ) -> "ConversationMessage":
        """Create from SQLite storage data."""
        try:
            content = json.loads(content_json)
        except json.JSONDecodeError:
            # Fallback to string content for malformed JSON
            content = content_json

        return cls(
            id=id_,
            conversation_id=conversation_id,
            role=role,
            content=content,
            index_in_conv=index_in_conv,
            created_at=created_at,
        )
