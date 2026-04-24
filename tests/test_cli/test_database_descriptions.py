import pytest

from sqlsaber.cli.database import add, describe, list_databases
from sqlsaber.config.database import DatabaseConfigManager


@pytest.fixture
def isolated_db_config(temp_dir, monkeypatch):
    config_dir = temp_dir / "config"
    monkeypatch.setattr(
        "platformdirs.user_config_dir", lambda *args, **kwargs: str(config_dir)
    )
    manager = DatabaseConfigManager()
    monkeypatch.setattr("sqlsaber.cli.database.config_manager", manager)
    return manager


def test_db_add_accepts_description(isolated_db_config) -> None:
    add(
        name="warehouse",
        type="sqlite",
        database="/tmp/warehouse.db",
        interactive=False,
        description="Analytics warehouse",
    )

    config = isolated_db_config.get_database("warehouse")
    assert config is not None
    assert config.description == "Analytics warehouse"


def test_db_describe_sets_and_clears_description(isolated_db_config) -> None:
    add(
        name="warehouse",
        type="sqlite",
        database="/tmp/warehouse.db",
        interactive=False,
    )

    describe("warehouse", set_description="Analytics warehouse")
    config = isolated_db_config.get_database("warehouse")
    assert config is not None
    assert config.description == "Analytics warehouse"

    describe("warehouse", clear=True)
    config = isolated_db_config.get_database("warehouse")
    assert config is not None
    assert config.description is None


def test_db_list_shows_description(isolated_db_config, capsys) -> None:
    add(
        name="warehouse",
        type="sqlite",
        database="/tmp/warehouse.db",
        interactive=False,
        description="Analytics warehouse",
    )

    list_databases()

    captured = capsys.readouterr()
    assert "Analytics warehouse" in captured.out
