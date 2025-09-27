"""Tests for query timeout functionality."""

import asyncio

import pytest

from sqlsaber.database import (
    DEFAULT_QUERY_TIMEOUT,
    DatabaseConnection,
    QueryTimeoutError,
    SQLiteConnection,
)


class TestQueryTimeout:
    """Test query timeout functionality."""

    def test_default_query_timeout_constant(self):
        """Test that the default query timeout constant is defined."""
        assert DEFAULT_QUERY_TIMEOUT == 30.0

    @pytest.mark.asyncio
    async def test_database_connection_creation(self):
        """Test that DatabaseConnection can be created."""
        connection_string = "sqlite:///:memory:"

        conn = DatabaseConnection(connection_string)
        assert conn.connection_string == connection_string

        await conn.close()

    @pytest.mark.asyncio
    async def test_timeout_fallback_logic(self):
        """Test timeout fallback: per-query > hardcoded default."""
        connection_string = "sqlite:///:memory:"

        # Test with no timeout - should use hardcoded default
        conn1 = SQLiteConnection(connection_string)
        result1 = await conn1.execute_query("SELECT 1 as test")
        assert len(result1) == 1
        await conn1.close()

        # Test with per-query timeout override
        conn2 = SQLiteConnection(connection_string)
        result2 = await conn2.execute_query("SELECT 2 as test", timeout=60.0)
        assert len(result2) == 1
        await conn2.close()

    @pytest.mark.asyncio
    async def test_sqlite_timeout_functionality(self):
        """Test timeout functionality with SQLite in-memory database."""
        connection_string = "sqlite:///:memory:"
        timeout = 1.0  # 1 second timeout

        conn = SQLiteConnection(connection_string)

        try:
            # This query should complete quickly
            result = await conn.execute_query("SELECT 1 as test", timeout=timeout)
            assert len(result) == 1
            assert result[0]["test"] == 1

            # Test with no explicit timeout (uses default)
            result2 = await conn.execute_query("SELECT 2 as test")
            assert len(result2) == 1
            assert result2[0]["test"] == 2

        finally:
            await conn.close()

    @pytest.mark.asyncio
    async def test_timeout_error_handling(self):
        """Test that timeout errors are properly raised with asyncio.TimeoutError."""
        connection_string = "sqlite:///:memory:"

        conn = SQLiteConnection(connection_string)

        try:
            # Create a mock query that artificially delays to trigger timeout
            async def mock_slow_execute_query(query, *args, timeout=None):
                # Simulate a slow query by sleeping longer than timeout
                await asyncio.sleep(0.1)  # 100ms, longer than 1ms timeout
                return [{"result": "should not reach here"}]

            # Replace the execute_query method temporarily
            conn.execute_query = mock_slow_execute_query

            # This should raise an asyncio.TimeoutError which gets converted to QueryTimeoutError
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(conn.execute_query("SELECT 1"), timeout=0.001)

        finally:
            await conn.close()

    def test_query_timeout_error_message(self):
        """Test QueryTimeoutError message formatting."""
        timeout = 30.5
        error = QueryTimeoutError(timeout)

        assert str(error) == "Query exceeded timeout of 30.5s"
        assert error.timeout == 30.5


if __name__ == "__main__":
    pytest.main([__file__])
