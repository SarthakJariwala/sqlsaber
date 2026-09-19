"""Host-selected visualization configuration."""

from __future__ import annotations

from dataclasses import dataclass

from sqlsaber.nested_model import INHERIT, NestedModel

VIZ = "viz"


@dataclass(frozen=True, slots=True)
class VizConfig:
    """Nested-model choice for the visualization spec agent."""

    model: NestedModel = INHERIT


DEFAULT_VIZ_CONFIG = VizConfig()
