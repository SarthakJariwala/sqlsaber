"""Memory manager for handling database-specific context and memories."""

from sqlsaber.memory.storage import Memory, MemoryStorage


class MemoryManager:
    """Manages database-specific memories and context."""

    def __init__(self):
        self.storage = MemoryStorage()

    def add_memory(self, database_name: str, content: str) -> Memory:
        """Add a new memory for the specified database."""
        return self.storage.add_memory(database_name, content)

    def get_memories(self, database_name: str) -> list[Memory]:
        """Get all memories for the specified database."""
        return self.storage.get_memories(database_name)

    def remove_memory(self, database_name: str, memory_id: str) -> bool:
        """Remove a specific memory by ID."""
        return self.storage.remove_memory(database_name, memory_id)

    def clear_memories(self, database_name: str) -> int:
        """Clear all memories for the specified database."""
        return self.storage.clear_memories(database_name)

    def get_memory_by_id(self, database_name: str, memory_id: str) -> Memory | None:
        """Get a specific memory by ID."""
        return self.storage.get_memory_by_id(database_name, memory_id)

    def has_memories(self, database_name: str) -> bool:
        """Check if database has any memories."""
        return self.storage.has_memories(database_name)

    def format_memories_for_prompt(self, database_name: str) -> str:
        """Format memories for inclusion in system prompt."""
        memories = self.get_memories(database_name)

        if not memories:
            return ""

        formatted_memories = []
        for memory in memories:
            formatted_memories.append(f"- {memory.content}")

        return f"""
Previous context from user:
{chr(10).join(formatted_memories)}

Use this context to better understand the user's needs and provide more relevant responses.
"""

    def get_memories_summary(self, database_name: str) -> dict:
        """Get a summary of memories for a database."""
        memories = self.get_memories(database_name)

        return {
            "database": database_name,
            "total_memories": len(memories),
            "memories": [
                {
                    "id": memory.id,
                    "content": memory.content[:100] + "..."
                    if len(memory.content) > 100
                    else memory.content,
                    "timestamp": memory.formatted_timestamp(),
                }
                for memory in memories
            ],
        }

    def list_databases_with_memories(self) -> list[str]:
        """List all databases that have memories."""
        return self.storage.list_databases_with_memories()
