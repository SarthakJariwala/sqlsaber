"""Global pytest configuration and fixtures."""

import tempfile
from pathlib import Path

import pytest

from sqlsaber.config.database import DatabaseConfig, DatabaseConfigManager


@pytest.fixture
def temp_dir():
    """Provide a temporary directory that's cleaned up after the test."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_database_config():
    """Provide a mock database configuration."""
    return DatabaseConfig(
        name="test_db",
        type="postgresql",
        host="localhost",
        port=5432,
        username="test_user",
        password="test_password",
        database="test_database",
    )


@pytest.fixture
def mock_config_manager(temp_dir, monkeypatch):
    """Provide a mock database config manager with temp directory."""
    # Monkey patch the config directory to use temp directory
    config_dir = temp_dir / "config"
    monkeypatch.setattr(
        "platformdirs.user_config_dir", lambda *args, **kwargs: str(config_dir)
    )

    manager = DatabaseConfigManager()
    return manager
