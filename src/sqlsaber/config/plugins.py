"""CLI plugin discovery, persistent settings, and credential resolution."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from importlib.metadata import EntryPoint, entry_points
from pathlib import Path
from typing import Protocol

import platformdirs

from sqlsaber.plugin_settings import (
    PluginSettings,
    Setting,
    SettingsValues,
    SettingValue,
)


@dataclass(frozen=True)
class SavedPlugin:
    enabled: bool = True
    settings: dict[str, SettingValue] = field(default_factory=dict)


@dataclass(frozen=True)
class ResolvedSettings:
    values: dict[str, SettingValue]
    secrets: dict[str, str] = field(repr=False)
    sources: dict[str, str]


class PluginSetupRequired(ValueError):
    """An installed plugin needs an explicit configuration choice."""


class _SubagentModelStore(Protocol):
    def get_subagent_model(self, agent: str) -> str | None: ...

    def set_subagent_model(self, agent: str, model: str | None) -> None: ...


def installed_plugins() -> dict[str, EntryPoint]:
    return {ep.name: ep for ep in entry_points(group="sqlsaber.capabilities")}


def load_plugin_settings(name: str) -> PluginSettings | None:
    for ep in entry_points(group="sqlsaber.plugin_settings"):
        if ep.name == name:
            try:
                declaration = ep.load()
            except Exception as exc:
                raise ValueError(f"Cannot load settings for plugin '{name}'") from exc
            if not isinstance(declaration, PluginSettings):
                raise ValueError(f"Plugin '{name}' did not declare PluginSettings")
            return declaration
    return None


class PluginConfigStore:
    def __init__(self, path: Path | None = None):
        self.path = path or (
            Path(platformdirs.user_config_dir("sqlsaber", "sqlsaber"))
            / "plugin_config.json"
        )

    def _read(self) -> dict[str, SavedPlugin]:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text())
            if not isinstance(data, dict) or data.get("version") != 1:
                raise ValueError("unsupported format version")
            plugins = data["plugins"]
            if not isinstance(plugins, dict):
                raise ValueError("plugins must be an object")
            result: dict[str, SavedPlugin] = {}
            for name, entry in plugins.items():
                if not isinstance(entry, dict):
                    raise ValueError(f"{name} must be an object")
                enabled = entry.get("enabled", True)
                values = entry.get("settings", {})
                if not isinstance(enabled, bool) or not isinstance(values, dict):
                    raise ValueError(f"invalid settings for {name}")
                if any(
                    value is not None and not isinstance(value, (str, int, float, bool))
                    for value in values.values()
                ):
                    raise ValueError(f"{name} settings must be scalar values")
                result[name] = SavedPlugin(enabled, values)
            return result
        except (OSError, ValueError, KeyError) as exc:
            raise ValueError(
                f"Cannot read plugin configuration at {self.path}. "
                "Repair the file before starting a session."
            ) from exc

    def get(self, name: str) -> SavedPlugin:
        return self._read().get(name, SavedPlugin())

    def save(self, name: str, plugin: SavedPlugin) -> None:
        plugins = self._read()
        plugins[name] = plugin
        data = {
            "version": 1,
            "plugins": {
                key: {"enabled": value.enabled, "settings": value.settings}
                for key, value in plugins.items()
            },
        }
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", dir=self.path.parent, delete=False
            ) as stream:
                temporary = Path(stream.name)
                json.dump(data, stream, indent=2, allow_nan=False)
                stream.write("\n")
            temporary.replace(self.path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def _credential_id(name: str, setting: Setting) -> str:
    return setting.credential or f"{name}.{setting.name}"


def save_secret(name: str, setting: Setting, value: str) -> None:
    import keyring

    identity = _credential_id(name, setting)
    try:
        keyring.set_password("sqlsaber-plugins", identity, value)
        if keyring.get_password("sqlsaber-plugins", identity) != value:
            raise ValueError("credential backend did not retain the value")
    except Exception:
        raise ValueError(
            f"Could not save {setting.name} in the OS credential store. "
            "Configure a working keyring or use the declared environment variable. "
            "No secret was written to the settings file."
        ) from None


def resolve_settings(
    name: str,
    declaration: PluginSettings,
    saved: SettingsValues,
    *,
    overrides: SettingsValues | None = None,
    use_environment: bool = True,
    validate: bool = True,
    load_secrets: bool = True,
) -> ResolvedSettings:
    """Resolve flag > environment > saved > default, then validate active fields."""
    overrides = overrides or {}
    for key in {*saved, *overrides}:
        setting = declaration.field(key)
        if setting.kind == "secret":
            raise ValueError(
                f"{name}.{key} is a secret; use masked setup or --secret-stdin {key}, "
                "never the settings file or --set."
            )
    values: dict[str, SettingValue] = {}
    sources: dict[str, str] = {}
    raw: dict[str, SettingValue] = {}
    for setting in declaration.fields:
        if setting.kind == "secret":
            continue
        key = setting.name
        value = saved.get(key, setting.default)
        sources[key] = "saved" if key in saved else "default"
        if use_environment and setting.env and setting.env in os.environ:
            value = os.environ[setting.env]
            sources[key] = f"environment ({setting.env})"
        if key in overrides:
            value = overrides[key]
            sources[key] = "option"
        raw[key] = value
        if setting.when is None:
            values[key] = setting.parse(value) if value is not None else None
    for setting in declaration.fields:
        if setting.kind == "secret" or not setting.active(values):
            continue
        value = raw[setting.name]
        values[setting.name] = setting.parse(value) if value is not None else None
        if validate and setting.required and values[setting.name] is None:
            raise PluginSetupRequired(
                f"{name}.{setting.name} is required. Run: saber plugins setup {name}"
            )
    if validate:
        try:
            declaration.validate(values)
        except ValueError as exc:
            raise ValueError(
                f"Invalid settings for {name}: {exc}. Run: saber plugins setup {name}"
            ) from None
    secrets: dict[str, str] = {}
    for setting in declaration.fields:
        if setting.kind != "secret" or not setting.active(values):
            continue
        value = os.getenv(setting.env) if use_environment and setting.env else None
        source = f"environment ({setting.env})" if value else "not configured"
        if not value and load_secrets:
            import keyring

            try:
                value = keyring.get_password(
                    "sqlsaber-plugins", _credential_id(name, setting)
                )
            except Exception:
                source = "credential store unavailable"
            else:
                if value:
                    source = "credential store"
        if value:
            secrets[setting.name] = value
        sources[setting.name] = source
    return ResolvedSettings(values, secrets, sources)


def migrate_legacy_plugin_models(
    store: PluginConfigStore | None = None,
    *,
    models: _SubagentModelStore | None = None,
) -> tuple[str, ...]:
    """Move leftover ``subagents.<plugin>`` values into plugin settings once."""
    from sqlsaber.config.logging import get_logger
    from sqlsaber.config.settings import CoreAgent, ModelConfigManager

    store = store or PluginConfigStore()
    manager: _SubagentModelStore = (
        models if models is not None else ModelConfigManager()
    )
    adopted: list[str] = []
    for name in installed_plugins():
        if name == CoreAgent.HANDOFF.value:
            continue
        declaration = load_plugin_settings(name)
        if declaration is None:
            continue
        model_fields = [field for field in declaration.fields if field.kind == "model"]
        if len(model_fields) != 1:
            continue
        setting = model_fields[0]
        legacy = manager.get_subagent_model(name)
        if not legacy:
            continue
        saved = store.get(name)
        settings = dict(saved.settings)
        if setting.name not in settings:
            try:
                settings[setting.name] = setting.parse(legacy)
            except ValueError:
                get_logger(__name__).warning(
                    "Dropping unparseable legacy %s model %r", name, legacy
                )
            else:
                store.save(name, SavedPlugin(saved.enabled, settings))
        manager.set_subagent_model(name, None)
        adopted.append(name)
    return tuple(adopted)
