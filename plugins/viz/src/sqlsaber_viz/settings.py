"""Side-effect-free CLI settings declaration for the visualization plugin."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial

from sqlsaber.plugin_settings import (
    PluginSettings,
    SettingsValues,
    configured_model,
    model_setting,
)

from .config import VizConfig


def _config(values: SettingsValues) -> VizConfig:
    return VizConfig(model=configured_model(values))


def _validate(values: SettingsValues) -> None:
    _config(values)


def _bind(values: SettingsValues, secrets: Mapping[str, str]) -> object:
    del secrets
    from .capability import capability

    return partial(capability, config=_config(values))


settings = PluginSettings(
    fields=(model_setting(label="Spec agent model"),),
    validate=_validate,
    bind=_bind,
)
