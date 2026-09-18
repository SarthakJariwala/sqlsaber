"""CLI plugin model settings round-trip through plugins setup."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock


from sqlsaber.cli import plugins as cli
from sqlsaber.cli.session import configured_capabilities
from sqlsaber.config import plugins as config
from sqlsaber.nested_model import INHERIT
from sqlsaber.plugin_settings import PluginSettings, configured_model, model_setting


def test_model_field_round_trips_and_shows_inherit(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        "platformdirs.user_config_dir", lambda *args, **kwargs: str(tmp_path)
    )
    factory = Mock()
    declaration = PluginSettings(
        fields=(model_setting(label="Analyst model"),),
        validate=lambda values: None,
        bind=Mock(return_value=factory),
    )
    entries = {"viz": SimpleNamespace(load=Mock())}
    monkeypatch.setattr(cli, "installed_plugins", lambda: entries)
    monkeypatch.setattr(config, "installed_plugins", lambda: entries)
    monkeypatch.setattr(cli, "load_plugin_settings", lambda name: declaration)
    monkeypatch.setattr(config, "load_plugin_settings", lambda name: declaration)

    cli.show("viz")
    assert "unset (uses main model)" in capsys.readouterr().out

    cli.setup("viz", set_values=["model=openai:gpt-5-mini"])
    assert config.PluginConfigStore().get("viz").settings == {
        "model": "openai:gpt-5-mini"
    }
    cli.show("viz")
    assert "openai:gpt-5-mini" in capsys.readouterr().out

    cli.unset("viz", "model")
    configured_capabilities()
    values = declaration.bind.call_args.args[0]
    assert values["model"] is None
    assert configured_model(values) is INHERIT
