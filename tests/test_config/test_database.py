"""Tests for database configuration management."""

import pytest

from sqlsaber.config.database import DatabaseConfig


class TestDatabaseConfig:
    """Test the DatabaseConfig class."""

    def test_postgresql_connection_string(self):
        """Test PostgreSQL connection string generation."""
        config = DatabaseConfig(
            name="postgres_db",
            type="postgresql",
            host="localhost",
            port=5432,
            username="user",
            password="pass123",
            database="mydb",
        )

        expected = "postgresql://user:pass123@localhost:5432/mydb"
        assert config.to_connection_string() == expected

    def test_mysql_connection_string(self):
        """Test MySQL connection string generation."""
        config = DatabaseConfig(
            name="mysql_db",
            type="mysql",
            host="localhost",
            port=3306,
            username="root",
            password="secret",
            database="testdb",
        )

        expected = "mysql://root:secret@localhost:3306/testdb"
        assert config.to_connection_string() == expected

    def test_sqlite_connection_string(self):
        """Test SQLite connection string generation."""
        config = DatabaseConfig(
            name="sqlite_db",
            type="sqlite",
            host="localhost",
            port=None,
            username=None,
            password=None,
            database="/path/to/database.db",
        )

        expected = "sqlite:////path/to/database.db"
        assert config.to_connection_string() == expected

    def test_invalid_database_type(self):
        """Test error for invalid database type."""
        config = DatabaseConfig(
            name="invalid_db",
            type="mongodb",  # Not supported
            host="localhost",
            port=27017,
            username="user",
            password="pass",
            database="db",
        )

        with pytest.raises(ValueError, match="Unsupported database type"):
            config.to_connection_string()

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = DatabaseConfig(
            name="test_db",
            type="postgresql",
            host="localhost",
            port=5432,
            username="user",
            password="pass",
            database="db",
        )

        result = config.to_dict()
        assert result == {
            "name": "test_db",
            "type": "postgresql",
            "host": "localhost",
            "port": 5432,
            "username": "user",
            "database": "db",
            "schema": None,
            "ssl_mode": None,
            "ssl_ca": None,
            "ssl_cert": None,
            "ssl_key": None,
        }

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            "name": "test_db",
            "type": "mysql",
            "host": "192.168.1.1",
            "port": 3306,
            "username": "admin",
            "database": "production",
        }

        config = DatabaseConfig.from_dict(data)
        assert config.name == "test_db"
        assert config.type == "mysql"
        assert config.host == "192.168.1.1"
        assert config.port == 3306
        assert config.username == "admin"
        assert config.database == "production"


class TestDatabaseConfigManager:
    """Test the DatabaseConfigManager class."""

    @pytest.fixture
    def db_manager(self, mock_config_manager):
        """Use the mock config manager from conftest."""
        return mock_config_manager

    def test_add_database(self, db_manager):
        """Test adding a new database configuration."""
        config = DatabaseConfig(
            name="new_db",
            type="postgresql",
            host="localhost",
            port=5432,
            username="user",
            password="pass",
            database="testdb",
        )

        db_manager.add_database(config)

        # Verify it was added
        retrieved = db_manager.get_database("new_db")
        assert retrieved is not None
        assert retrieved.name == "new_db"
        assert retrieved.type == "postgresql"

    def test_add_duplicate_database(self, db_manager):
        """Test error when adding duplicate database name."""
        config1 = DatabaseConfig(
            name="duplicate",
            type="postgresql",
            host="host1",
            port=5432,
            username="user",
            password="pass",
            database="db1",
        )

        config2 = DatabaseConfig(
            name="duplicate",
            type="mysql",
            host="host2",
            port=3306,
            username="user",
            password="pass",
            database="db2",
        )

        db_manager.add_database(config1)

        with pytest.raises(ValueError, match="already exists"):
            db_manager.add_database(config2)

    def test_remove_database(self, db_manager):
        """Test removing a database configuration."""
        config = DatabaseConfig(
            name="to_remove",
            type="sqlite",
            host="localhost",
            port=None,
            username=None,
            password=None,
            database="/tmp/test.db",
        )

        db_manager.add_database(config)
        assert db_manager.get_database("to_remove") is not None

        # Remove it
        result = db_manager.remove_database("to_remove")
        assert result is True
        assert db_manager.get_database("to_remove") is None

        # Try removing non-existent
        result = db_manager.remove_database("non_existent")
        assert result is False

    def test_list_databases(self, db_manager):
        """Test listing all database configurations."""
        # Add multiple databases
        configs = [
            DatabaseConfig("db1", "postgresql", "host1", 5432, "user", "pass", "db"),
            DatabaseConfig("db2", "mysql", "host2", 3306, "user", "pass", "db"),
            DatabaseConfig(
                "db3", "sqlite", "localhost", None, "/tmp/db.sqlite", None, None
            ),
        ]

        for config in configs:
            db_manager.add_database(config)

        databases = db_manager.list_databases()
        assert len(databases) >= 3  # May have more from other tests

        names = [db.name for db in databases]
        assert "db1" in names
        assert "db2" in names
        assert "db3" in names

    def test_set_default_database(self, db_manager):
        """Test setting and getting default database."""
        config = DatabaseConfig(
            name="default_db",
            type="postgresql",
            host="localhost",
            port=5432,
            username="user",
            password="pass",
            database="db",
        )

        db_manager.add_database(config)
        db_manager.set_default_database("default_db")

        default = db_manager.get_default_database()
        assert default is not None
        assert default.name == "default_db"
