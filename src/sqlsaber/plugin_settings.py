"""Plugin-owned declarations for the CLI settings editor.

Declarations must be side-effect free: no prompts, network calls, or capability
construction. Only ``bind`` loads runtime code. SDK factories do not read this
store; the CLI binds resolved settings before constructing SQLSaberOptions.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

type SettingValue = str | int | float | bool | None
type SettingsValues = Mapping[str, SettingValue]


@dataclass(frozen=True, slots=True)
class Setting:
    """One explicitly exposed CLI field.

    ``when`` controls applicability, not validation. Inactive saved values are
    retained but never passed to the plugin. Secret values live only in the
    credential store under ``credential`` (or the plugin/field name), never JSON.
    Environment aliases override saved values. Defaults remain plugin-owned.
    """

    name: str
    label: str
    kind: Literal["text", "integer", "number", "boolean", "secret"] = "text"
    default: SettingValue = None
    choices: tuple[str, ...] = ()
    env: str | None = None
    advanced: bool = False
    required: bool = False
    when: Callable[[SettingsValues], bool] | None = None
    credential: str | None = None
    help: str = ""

    def active(self, values: SettingsValues) -> bool:
        return self.when is None or self.when(values)

    def parse(self, value: object) -> SettingValue:
        if value is None or (value == "null" and self.kind not in {"text", "secret"}):
            if self.required:
                raise ValueError(f"{self.name} is required")
            return None
        if self.kind in {"text", "secret"}:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{self.name} must be nonempty text")
            parsed: SettingValue = value.strip()
        elif self.kind == "boolean":
            if isinstance(value, bool):
                parsed = value
            elif isinstance(value, str) and value.lower() in {"true", "false"}:
                parsed = value.lower() == "true"
            else:
                raise ValueError(f"{self.name} must be true or false")
        else:
            if isinstance(value, bool) or not isinstance(value, (str, int, float)):
                raise ValueError(f"{self.name} must be a {self.kind}")
            try:
                if self.kind == "integer":
                    if isinstance(value, float) and not value.is_integer():
                        raise ValueError
                    parsed = int(value)
                else:
                    parsed = float(value)
                if not math.isfinite(parsed):
                    raise ValueError
            except (ValueError, OverflowError):
                raise ValueError(f"{self.name} must be a finite {self.kind}") from None
        if self.choices and parsed not in self.choices:
            raise ValueError(f"{self.name} must be one of: {', '.join(self.choices)}")
        return parsed


@dataclass(frozen=True, slots=True)
class PluginSettings:
    """Optional ``sqlsaber.plugin_settings`` entry-point value.

    Field conditions depend on unconditional selector fields, such as provider.
    ``validate`` receives active non-secret values and raises ValueError for
    invalid configuration, reusing the plugin's runtime config validation.
    ``bind`` receives those values and active credentials and returns a callable
    accepting PluginContext. It must not provision resources or change os.environ.
    ``notice`` describes remote data transfer, charges, or native login steps.
    """

    fields: tuple[Setting, ...]
    validate: Callable[[SettingsValues], None]
    bind: Callable[[SettingsValues, Mapping[str, str]], object]
    notice: Callable[[SettingsValues], str | None] = lambda values: None

    def __post_init__(self) -> None:
        names = [field.name for field in self.fields]
        if len(names) != len(set(names)):
            raise ValueError("Plugin settings contain duplicate field names")

    def field(self, name: str) -> Setting:
        for field in self.fields:
            if field.name == name:
                return field
        raise ValueError(f"Unknown setting '{name}'. Use 'saber plugins show NAME'.")
