"""Database connection management."""

import os
from typing import Any, Dict, Optional

import asyncpg


class DatabaseConnection:
    """Manages database connections for the SQL agent."""

    def __init__(self, connection_string: Optional[str] = None):
        self.connection_string = connection_string or os.getenv(
            "DATABASE_URL", "postgresql://localhost:5432/postgres"
        )
        self._pool: Optional[asyncpg.Pool] = None

    async def get_pool(self) -> asyncpg.Pool:
        """Get or create connection pool."""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                self.connection_string, min_size=1, max_size=10
            )
        return self._pool

    async def close(self):
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None

    async def execute_query(self, query: str, *args) -> list[Dict[str, Any]]:
        """Execute a query and return results as list of dicts.

        All queries run in a transaction that is rolled back at the end,
        ensuring no changes are persisted to the database.
        """
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            # Start a transaction that we'll always rollback
            transaction = conn.transaction()
            await transaction.start()

            try:
                rows = await conn.fetch(query, *args)
                return [dict(row) for row in rows]
            finally:
                # Always rollback to ensure no changes are committed
                await transaction.rollback()
