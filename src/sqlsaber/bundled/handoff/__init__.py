"""Bundled handoff capability factory (runtime imports are lazy)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlsaber.capabilities.plugins import PluginContext
    from .runtime import Handoff


def capability(
    context: PluginContext,
    *,
    model_name: str | None = None,
    api_key: str | None = None,
) -> Handoff:
    """Create a handoff plugin without initializing its child model."""
    from .runtime import Handoff

    return Handoff(context, model_name=model_name, api_key=api_key)
