"""SQLSaber visualization capability plugin."""

from typing import Any

from .config import VIZ, VizConfig

_LAZY_EXPORTS = {
    "Visualization": ("capability", "Visualization"),
    "capability": ("capability", "capability"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(name)
    from importlib import import_module

    module, attribute = _LAZY_EXPORTS[name]
    value = getattr(import_module(f"sqlsaber_viz.{module}"), attribute)
    globals()[name] = value
    return value


__all__ = ["VIZ", "Visualization", "VizConfig", "capability"]
