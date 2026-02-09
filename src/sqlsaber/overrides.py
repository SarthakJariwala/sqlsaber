"""Runtime override models for SQLSaber API and tools."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class ModelOverides:
    """Model and credential overrides for a tool."""

    model_name: str | None = None
    api_key: str | None = None


type ModelOverideInput = ModelOverides | Mapping[str, str | None]
type ToolOveridesInput = Mapping[str, ModelOverideInput | None]


@dataclass(frozen=True, slots=True)
class ToolRunDeps:
    """Run-scoped dependency payload passed to pydantic-ai tools."""

    tool_overides: Mapping[str, ModelOverides]


def normalize_tool_overides(
    tool_overides: ToolOveridesInput | None,
) -> dict[str, ModelOverides]:
    """Normalize user-provided tool override inputs."""
    if not tool_overides:
        return {}

    normalized: dict[str, ModelOverides] = {}
    for raw_tool_name, raw_override in tool_overides.items():
        tool_name = _normalize_tool_name(raw_tool_name)
        override = normalize_model_overide(raw_override)
        if override is not None:
            normalized[tool_name] = override

    return normalized


def build_tool_run_deps(tool_overides: Mapping[str, ModelOverides]) -> ToolRunDeps:
    """Build immutable run-scoped deps for pydantic-ai tool contexts."""
    return ToolRunDeps(tool_overides=MappingProxyType(dict(tool_overides)))


def get_tool_model_overide_from_ctx(
    ctx: object,
    tool_name: str,
) -> ModelOverides | None:
    """Resolve a tool model override from pydantic-ai run context deps."""
    deps = getattr(ctx, "deps", None)
    if not isinstance(deps, ToolRunDeps):
        return None
    return deps.tool_overides.get(tool_name)


def normalize_model_overide(
    value: ModelOverideInput | None,
) -> ModelOverides | None:
    """Normalize a model override mapping or dataclass."""
    if value is None:
        return None

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
            "Tool override value must be a ModelOverides instance or mapping."
        )

    if api_key and not model_name:
        raise ValueError(
            "api_key override requires model_name so provider can be determined."
        )

    if model_name is None and api_key is None:
        return None

    return ModelOverides(model_name=model_name, api_key=api_key)


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
