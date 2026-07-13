"""Third-party SQLSaber capability discovery."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Any

from pydantic_ai.capabilities import AbstractCapability

from sqlsaber.config.logging import get_logger
from sqlsaber.database.registry import DatabaseRegistry
from sqlsaber.knowledge.manager import KnowledgeManager
from sqlsaber.overrides import ModelOverides

logger = get_logger(__name__)
PLUGIN_GROUP = "sqlsaber.capabilities"


@dataclass(frozen=True, slots=True)
class PluginContext:
    """Managed resources and options passed to capability plugin factories."""

    registry: DatabaseRegistry
    knowledge_manager: KnowledgeManager
    allow_dangerous: bool
    tool_overrides: Mapping[str, ModelOverides]


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
    for entry_point in _select_entry_points(PLUGIN_GROUP):
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
