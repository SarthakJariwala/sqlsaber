"""Shared SQLSaberOptions construction for CLI sessions."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeGuard

if TYPE_CHECKING:
    from sqlsaber import SQLSaberOptions
    from sqlsaber.capabilities.plugins import CapabilityFactory, PluginContext


def _is_capability_factory(
    value: object,
) -> TypeGuard[Callable[[PluginContext], object]]:
    return callable(value)


def cli_sqlsaber_options(**kwargs: Any) -> SQLSaberOptions:
    """Build CLI session options with installed plugin factories."""
    from sqlsaber import SQLSaberOptions

    if "capabilities" not in kwargs:
        kwargs["capabilities"] = configured_capabilities()
    return SQLSaberOptions(**kwargs)


def configured_capabilities() -> tuple[CapabilityFactory, ...]:
    from sqlsaber.capabilities.plugins import CapabilityFactory
    from sqlsaber.cli.output import err
    from sqlsaber.config.plugins import (
        PluginConfigStore,
        PluginSetupRequired,
        installed_plugins,
        load_plugin_settings,
        resolve_settings,
    )
    from sqlsaber.render import blocks as b

    store = PluginConfigStore()
    factories: list[CapabilityFactory] = []
    for name, entry_point in sorted(installed_plugins().items()):
        saved = store.get(name)
        if not saved.enabled:
            continue
        declaration = load_plugin_settings(name)
        if declaration is not None:
            try:
                resolved = resolve_settings(name, declaration, saved.settings)
            except PluginSetupRequired as exc:
                if saved.settings:
                    raise
                err(b.warn(str(exc)))
                continue
            factory = declaration.bind(resolved.values, resolved.secrets)
        else:
            try:
                factory = entry_point.load()
            except Exception:
                err(b.warn(f"Could not load plugin '{name}'. Check its installation."))
                continue
        if not _is_capability_factory(factory):
            raise ValueError(f"Plugin '{name}' did not return a capability factory")
        factories.append(CapabilityFactory(name, factory))
    return tuple(factories)
