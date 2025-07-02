"""Memory storage implementation for database-specific memories."""

import json
import os
import platform
import stat
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import platformdirs


@dataclass
class Memory:
    """Represents a single memory entry."""

    id: str
    content: str
    timestamp: float

    def to_dict(self) -> dict:
        """Convert memory to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "content": self.content,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Memory":
        """Create Memory from dictionary."""
        return cls(
            id=data["id"],
            content=data["content"],
            timestamp=data["timestamp"],
        )

    def formatted_timestamp(self) -> str:
        """Get human-readable timestamp."""
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp))


class MemoryStorage:
    """Handles storage and retrieval of database-specific memories."""

    def __init__(self):
        self.memory_dir = Path(platformdirs.user_config_dir("sqlsaber")) / "memories"
        self._ensure_memory_dir()

    def _ensure_memory_dir(self) -> None:
        """Ensure memory directory exists with proper permissions."""
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self._set_secure_permissions(self.memory_dir, is_directory=True)

    def _set_secure_permissions(self, path: Path, is_directory: bool = False) -> None:
        """Set secure permissions cross-platform."""
        try:
            if platform.system() == "Windows":
                # On Windows, rely on NTFS permissions and avoid chmod
                return
            else:
                # Unix-like systems (Linux, macOS)
                if is_directory:
                    os.chmod(
                        path, stat.S_IRWXU
                    )  # 0o700 - owner read/write/execute only
                else:
                    os.chmod(
                        path, stat.S_IRUSR | stat.S_IWUSR
                    )  # 0o600 - owner read/write only
        except (OSError, PermissionError):
            # If we can't set permissions, continue anyway
            pass

    def _get_memory_file(self, database_name: str) -> Path:
        """Get the memory file path for a specific database."""
        return self.memory_dir / f"{database_name}.json"

    def _load_memories(self, database_name: str) -> list[Memory]:
        """Load memories for a specific database."""
        memory_file = self._get_memory_file(database_name)

        if not memory_file.exists():
            return []

        try:
            with open(memory_file, "r") as f:
                data = json.load(f)
                return [
                    Memory.from_dict(memory_data)
                    for memory_data in data.get("memories", [])
                ]
        except (json.JSONDecodeError, IOError, KeyError):
            return []

    def _save_memories(self, database_name: str, memories: list[Memory]) -> None:
        """Save memories for a specific database."""
        memory_file = self._get_memory_file(database_name)

        data = {
            "database": database_name,
            "memories": [memory.to_dict() for memory in memories],
        }

        with open(memory_file, "w") as f:
            json.dump(data, f, indent=2)

        # Set secure permissions
        self._set_secure_permissions(memory_file, is_directory=False)

    def add_memory(self, database_name: str, content: str) -> Memory:
        """Add a new memory for the specified database."""
        memory = Memory(
            id=str(uuid.uuid4()),
            content=content.strip(),
            timestamp=time.time(),
        )

        memories = self._load_memories(database_name)
        memories.append(memory)
        self._save_memories(database_name, memories)

        return memory

    def get_memories(self, database_name: str) -> list[Memory]:
        """Get all memories for the specified database."""
        return self._load_memories(database_name)

    def remove_memory(self, database_name: str, memory_id: str) -> bool:
        """Remove a specific memory by ID."""
        memories = self._load_memories(database_name)
        original_count = len(memories)

        memories = [m for m in memories if m.id != memory_id]

        if len(memories) < original_count:
            self._save_memories(database_name, memories)
            return True

        return False

    def clear_memories(self, database_name: str) -> int:
        """Clear all memories for the specified database."""
        memories = self._load_memories(database_name)
        count = len(memories)

        if count > 0:
            self._save_memories(database_name, [])

        return count

    def get_memory_by_id(self, database_name: str, memory_id: str) -> Memory | None:
        """Get a specific memory by ID."""
        memories = self._load_memories(database_name)
        return next((m for m in memories if m.id == memory_id), None)

    def has_memories(self, database_name: str) -> bool:
        """Check if database has any memories."""
        return len(self._load_memories(database_name)) > 0

    def list_databases_with_memories(self) -> list[str]:
        """List all databases that have memories."""
        databases = []

        if not self.memory_dir.exists():
            return databases

        for memory_file in self.memory_dir.glob("*.json"):
            database_name = memory_file.stem
            if self.has_memories(database_name):
                databases.append(database_name)

        return databases
