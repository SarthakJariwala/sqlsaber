"""PluginSettings model field parsing and declaration invariants."""

from __future__ import annotations

import pytest

from sqlsaber.nested_model import INHERIT, pin
from sqlsaber.plugin_settings import (
    PluginSettings,
    Setting,
    configured_model,
    model_setting,
)


def test_model_setting_parses_canonical_id_and_blank_inherits() -> None:
    field = model_setting()
    assert field.parse("google-gla:gemini-2.5-pro") == "google:gemini-2.5-pro"
    assert field.parse(None) is None
    assert field.parse("  ") is None
    with pytest.raises(ValueError, match="PROVIDER:MODEL"):
        field.parse("gpt-5-mini")


def test_configured_model_reads_saved_or_inherit() -> None:
    assert configured_model({}) is INHERIT
    assert configured_model({"model": None}) is INHERIT
    assert configured_model({"model": "openai:gpt-5-mini"}) == pin("openai:gpt-5-mini")


def test_plugin_settings_reject_a_second_model_field() -> None:
    with pytest.raises(ValueError, match="at most one model field"):
        PluginSettings(
            fields=(
                model_setting(),
                Setting(name="analyst", label="Analyst", kind="model"),
            ),
            validate=lambda values: None,
            bind=lambda values, secrets: object(),
        )
