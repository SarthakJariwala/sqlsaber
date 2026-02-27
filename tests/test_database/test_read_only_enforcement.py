"""Regression tests for read-only enforcement plumbing."""

import inspect

import pytest

from sqlsaber.database import (
    BaseDatabaseConnection,
    CSVConnection,
    CSVsConnection,
    DuckDBConnection,
    MySQLConnection,
    PostgreSQLConnection,
    SQLiteConnection,
)


def test_base_connection_contract_exposes_read_only_flag():
    """Base execute_query contract should include read_only kwarg."""
    signature = inspect.signature(BaseDatabaseConnection.execute_query)

    assert "read_only" in signature.parameters


@pytest.mark.parametrize(
    "connection_cls",
    [
        PostgreSQLConnection,
        MySQLConnection,
        SQLiteConnection,
        DuckDBConnection,
        CSVConnection,
        CSVsConnection,
    ],
)
def test_concrete_connections_expose_read_only_flag(connection_cls):
    """All connection implementations should expose read_only kwarg."""
    signature = inspect.signature(connection_cls.execute_query)

    assert "read_only" in signature.parameters
