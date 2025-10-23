"""Tests for MySQL schema exclusion behavior."""

import pytest

from sqlsaber.database.mysql import MySQLSchemaIntrospector


class _FakeMySQLConnection:
    def __init__(self):
        self._excluded_schemas: list[str] = []

    @property
    def excluded_schemas(self) -> list[str]:
        return self._excluded_schemas

    def set_excluded_schemas(self, schemas: list[str]) -> None:
        self._excluded_schemas = schemas


@pytest.fixture
def connection():
    return _FakeMySQLConnection()


def test_mysql_excluded_defaults(connection, monkeypatch):
    """Default exclusions should include MySQL system schemas."""
    monkeypatch.delenv("SQLSABER_MYSQL_EXCLUDE_SCHEMAS", raising=False)
    insp = MySQLSchemaIntrospector()
    excluded = insp._get_excluded_schemas(connection)
    for schema in ["information_schema", "performance_schema", "mysql", "sys"]:
        assert schema in excluded


def test_mysql_excluded_env_extension(connection, monkeypatch):
    """Environment variable should extend excluded schemas."""
    monkeypatch.setenv("SQLSABER_MYSQL_EXCLUDE_SCHEMAS", "foo,bar")
    insp = MySQLSchemaIntrospector()
    excluded = insp._get_excluded_schemas(connection)
    assert "foo" in excluded
    assert "bar" in excluded


def test_mysql_excluded_connection_extension(connection, monkeypatch):
    """Connection-level configuration should merge with defaults."""
    monkeypatch.delenv("SQLSABER_MYSQL_EXCLUDE_SCHEMAS", raising=False)
    connection.set_excluded_schemas(["custom_db"])
    insp = MySQLSchemaIntrospector()
    excluded = insp._get_excluded_schemas(connection)
    assert "custom_db" in excluded
