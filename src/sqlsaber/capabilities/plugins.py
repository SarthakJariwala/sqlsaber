"""Third-party SQLSaber capability discovery."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Any

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models import Model

from sqlsaber.config import providers
from sqlsaber.config.logging import get_logger
from sqlsaber.config.settings import Config
from sqlsaber.database.registry import DatabaseRegistry
from sqlsaber.knowledge.manager import KnowledgeManager
from sqlsaber.overrides import ModelOverides

if TYPE_CHECKING:
    from sqlsaber.artifacts import ArtifactFailureMode, ArtifactPublisher

logger = get_logger(__name__)
PLUGIN_GROUP = "sqlsaber.capabilities"


@dataclass(frozen=True, slots=True)
class PluginContext:
    """Managed resources and options passed to capability plugin factories."""

    registry: DatabaseRegistry
    knowledge_manager: KnowledgeManager
    allow_dangerous: bool
    tool_overrides: Mapping[str, ModelOverides]
    config: Config
    main_model_name: str
    main_api_key: str | None = None
    artifact_publisher: ArtifactPublisher | None = None
    artifact_failure_mode: ArtifactFailureMode = "required"

    def resolve_subagent_model(
        self,
        name: str,
        *,
        tool_name: str | None = None,
    ) -> tuple[str, Model | str, str]:
        """Resolve a child model from tool, subagent, and main-agent settings."""

        override = self.tool_overrides.get(tool_name) if tool_name else None
        subagent_model = self.config.model.get_subagent_model(name)
        model_name = (
            (override.model_name if override else None)
            or subagent_model
            or self.main_model_name
        )

        explicit_key = override.api_key if override else None
        use_main_key = override is None and subagent_model is None
        api_key = explicit_key or (self.main_api_key if use_main_key else None)
        if api_key is None:
            self.config.auth.validate(model_name)
            api_key = self.config.auth.get_api_key(model_name)

        provider = providers.provider_from_model(model_name)
        if provider is None:
            provider = model_name.partition(":")[0].strip().lower()

        # Import lazily to avoid loading the managed-agent package while plugin
        # discovery types themselves are being imported.
        from sqlsaber.agents.model_factory import build_model

        return model_name, build_model(model_name, api_key), provider


def _select_entry_points(group: str) -> Iterable[Any]:
    discovered = entry_points()
    if hasattr(discovered, "select"):
        return discovered.select(group=group)
    return discovered.get(group, [])


def _normalize_capabilities(result: object) -> list[AbstractCapability[Any]]:
    if result is None:
        return []
    if isinstance(result, AbstractCapability):
        return [result]
    if isinstance(result, Sequence) and not isinstance(result, (str, bytes)):
        capabilities: list[AbstractCapability[Any]] = []
        for item in result:
            if isinstance(item, AbstractCapability):
                capabilities.append(item)
            else:
                logger.warning("Plugin returned non-capability entry: %r", item)
        return capabilities
    logger.warning("Plugin returned unsupported result: %r", result)
    return []


def discover_capabilities(context: PluginContext) -> list[AbstractCapability[Any]]:
    """Load capability factories from the ``sqlsaber.capabilities`` entry group."""
    capabilities: list[AbstractCapability[Any]] = []
    discovered = sorted(
        _select_entry_points(PLUGIN_GROUP),
        key=lambda entry_point: entry_point.name,
    )
    for entry_point in discovered:
        try:
            factory = entry_point.load()
            if not callable(factory):
                logger.warning(
                    "Plugin '%s' is not a capability factory", entry_point.name
                )
                continue
            loaded = _normalize_capabilities(factory(context))
            capabilities.extend(loaded)
            logger.debug("Loaded capability plugin: %s", entry_point.name)
        except Exception as exc:
            logger.warning(
                "Failed to load capability plugin '%s': %s", entry_point.name, exc
            )
    return capabilities
