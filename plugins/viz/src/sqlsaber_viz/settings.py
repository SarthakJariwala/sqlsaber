"""Side-effect-free visualization settings for the plugin editor."""

from collections.abc import Mapping
from functools import partial

from sqlsaber.plugin_settings import PluginSettings, Setting, SettingsValues


def _validate(values: SettingsValues) -> None:
    model = values.get("model")
    if model is not None and (not isinstance(model, str) or not model.strip()):
        raise ValueError("model must be nonempty text")


def _bind(values: SettingsValues, secrets: Mapping[str, str]) -> object:
    del secrets
    from .capability import capability

    model = values.get("model")
    return partial(capability, model_name=model if isinstance(model, str) else None)


settings = PluginSettings(
    fields=(
        Setting(
            "model",
            "Visualization model",
            env="SQLSABER_VIZ_MODEL",
            help="Unset uses the active session model.",
        ),
    ),
    validate=_validate,
    bind=_bind,
)
