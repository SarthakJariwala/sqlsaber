"""Runtime override models for SQLSaber API and tools."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from sqlsaber.nested_model import Pinned, pin


@dataclass(frozen=True, slots=True)
class ModelOverides:
    """Model and credential overrides for a tool.

    Prefer ``pin("provider:model")`` for new code. This type remains accepted
    as ``tool_overrides`` input.
    """

    model_name: str | None = None
    api_key: str | None = None


type ModelOverideInput = Pinned | str | ModelOverides | Mapping[str, str | None]
type ToolOveridesInput = Mapping[str, ModelOverideInput | None]


def normalize_tool_overides(
    tool_overides: ToolOveridesInput | None,
) -> dict[str, Pinned]:
    """Normalize user-provided tool override inputs to pinned models."""
    if not tool_overides:
        return {}

    normalized: dict[str, Pinned] = {}
    for raw_tool_name, raw_override in tool_overides.items():
        tool_name = _normalize_tool_name(raw_tool_name)
        override = normalize_model_overide(raw_override)
        if override is not None:
            normalized[tool_name] = override

    return normalized


def normalize_model_overide(
    value: ModelOverideInput | None,
) -> Pinned | None:
    """Normalize a model override into a pin, or None when the entry is empty."""
    if value is None:
        return None
    if isinstance(value, Pinned):
        return value
    if isinstance(value, str):
        return pin(value)

    model_name: str | None
    api_key: str | None

    if isinstance(value, ModelOverides):
        model_name = _normalize_optional_text(value.model_name)
        api_key = _normalize_optional_text(value.api_key)
    elif isinstance(value, Mapping):
        _validate_override_keys(value)
        model_name = _normalize_optional_text(value.get("model_name"))
        api_key = _normalize_optional_text(value.get("api_key"))
    else:
        raise TypeError(
            "Tool override value must be a pin, model id, ModelOverides, or mapping."
        )

    if api_key and not model_name:
        raise ValueError(
            "api_key override requires model_name so provider can be determined."
        )

    if model_name is None:
        return None

    return pin(model_name, api_key=api_key)


def _validate_override_keys(value: Mapping[str, str | None]) -> None:
    allowed = {"model_name", "api_key"}
    unknown = set(value.keys()) - allowed
    if unknown:
        unknown_fields = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown override fields: {unknown_fields}")


def _normalize_tool_name(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("Tool override keys must be strings.")
    normalized = value.strip()
    if not normalized:
        raise ValueError("Tool override key cannot be empty.")
    return normalized


def _normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("Override values must be strings or None.")
    normalized = value.strip()
    return normalized or None
