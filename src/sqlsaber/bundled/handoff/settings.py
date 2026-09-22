"""Side-effect-free handoff settings for the standard plugin editor."""

from collections.abc import Mapping
from functools import partial

from sqlsaber.plugin_settings import PluginSettings, Setting, SettingsValues


def _validate(values: SettingsValues) -> None:
    model = values.get("model")
    if model is not None and (not isinstance(model, str) or not model.strip()):
        raise ValueError("model must be nonempty text")


def _bind(values: SettingsValues, secrets: Mapping[str, str]) -> object:
    from . import capability

    model = values.get("model")
    return partial(capability, model_name=model if isinstance(model, str) else None)


settings = PluginSettings(
    fields=(
        Setting(
            "model",
            "Handoff model",
            env="SQLSABER_HANDOFF_MODEL",
            help="Unset: use the legacy handoff model override, otherwise the active session model.",
        ),
    ),
    validate=_validate,
    bind=_bind,
)
