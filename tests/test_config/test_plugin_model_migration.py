"""Legacy subagents.<plugin> values move into plugin settings once."""

from __future__ import annotations

from types import SimpleNamespace

from sqlsaber.config.plugins import (
    PluginConfigStore,
    SavedPlugin,
    migrate_legacy_plugin_models,
)
from sqlsaber.config.settings import ModelConfigManager
from sqlsaber.plugin_settings import PluginSettings, model_setting


def test_migrate_legacy_plugin_models_adopts_plugin_keys_once(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        "platformdirs.user_config_dir", lambda *args, **kwargs: str(tmp_path)
    )
    declaration = PluginSettings(
        fields=(model_setting(),),
        validate=lambda values: None,
        bind=lambda values, secrets: object(),
    )
    monkeypatch.setattr(
        "sqlsaber.config.plugins.installed_plugins",
        lambda: {"notebook": SimpleNamespace(), "handoff": SimpleNamespace()},
    )
    monkeypatch.setattr(
        "sqlsaber.config.plugins.load_plugin_settings",
        lambda name: declaration if name == "notebook" else None,
    )
    models = ModelConfigManager()
    models.set_subagent_model("notebook", "openai:gpt-5-mini")
    models.set_subagent_model("handoff", "openai:gpt-5")
    store = PluginConfigStore()

    adopted = migrate_legacy_plugin_models(store, models=models)
    assert adopted == ("notebook",)
    assert store.get("notebook").settings == {"model": "openai:gpt-5-mini"}
    assert models.get_subagent_model("notebook") is None
    assert models.get_subagent_model("handoff") == "openai:gpt-5"

    store.save(
        "notebook",
        SavedPlugin(True, {"model": "anthropic:claude-haiku-4-5"}),
    )
    models.set_subagent_model("notebook", "openai:should-not-overwrite")
    assert migrate_legacy_plugin_models(store, models=models) == ("notebook",)
    assert store.get("notebook").settings == {"model": "anthropic:claude-haiku-4-5"}
    assert models.get_subagent_model("notebook") is None


def test_migrate_does_not_load_settings_without_a_legacy_key(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        "platformdirs.user_config_dir", lambda *args, **kwargs: str(tmp_path)
    )
    monkeypatch.setattr(
        "sqlsaber.config.plugins.installed_plugins",
        lambda: {"notebook": SimpleNamespace()},
    )

    def boom(name: str) -> None:
        raise AssertionError(f"should not load settings for {name}")

    monkeypatch.setattr("sqlsaber.config.plugins.load_plugin_settings", boom)
    models = ModelConfigManager()
    assert migrate_legacy_plugin_models(PluginConfigStore(), models=models) == ()
