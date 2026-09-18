"""Core models CLI no longer owns plugin nested-model names."""

from __future__ import annotations

import pytest

from sqlsaber.cli.models import models_app


def test_models_set_plugin_agent_points_at_plugins_setup(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "sqlsaber.config.plugins.installed_plugins",
        lambda: {"viz": object(), "notebook": object()},
    )
    with pytest.raises(SystemExit) as exc:
        models_app(["set", "openai:gpt-5-mini", "--agent", "viz"])
    assert exc.value.code == 2
    output = capsys.readouterr()
    text = output.out + output.err
    assert "plugins setup viz" in text
    assert "model=PROVIDER:MODEL" in text


def test_models_current_lists_handoff_and_points_at_plugins(
    monkeypatch, tmp_path, capsys
) -> None:
    monkeypatch.setattr(
        "platformdirs.user_config_dir", lambda *args, **kwargs: str(tmp_path)
    )
    with pytest.raises(SystemExit) as exc:
        models_app(["current"])
    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "handoff" in output
    assert "saber plugins list" in output
    assert "--agent viz" not in output
