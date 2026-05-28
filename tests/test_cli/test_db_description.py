"""Tests for `db add` description plumbing."""

from __future__ import annotations

import importlib


def _patched_db_module(monkeypatch, temp_dir):
    """Reload `sqlsaber.cli.database` so it points at a temp config dir."""
    config_dir = temp_dir / "config"
    monkeypatch.setattr(
        "platformdirs.user_config_dir", lambda *args, **kwargs: str(config_dir)
    )

    from sqlsaber.cli import database as db_module

    return importlib.reload(db_module)


def test_db_add_persists_description_non_interactive(temp_dir, monkeypatch):
    sqlite_path = temp_dir / "data.db"
    sqlite_path.write_text("")

    db_module = _patched_db_module(monkeypatch, temp_dir)

    db_module.add(
        name="warehouse",
        type="sqlite",
        database=str(sqlite_path),
        description="analytics warehouse, read-only",
        interactive=False,
    )

    saved = db_module.config_manager.get_database("warehouse")
    assert saved is not None
    assert saved.description == "analytics warehouse, read-only"


def test_db_add_without_description_defaults_to_none(temp_dir, monkeypatch):
    sqlite_path = temp_dir / "data.db"
    sqlite_path.write_text("")

    db_module = _patched_db_module(monkeypatch, temp_dir)

    db_module.add(
        name="warehouse",
        type="sqlite",
        database=str(sqlite_path),
        interactive=False,
    )

    saved = db_module.config_manager.get_database("warehouse")
    assert saved is not None
    assert saved.description is None
