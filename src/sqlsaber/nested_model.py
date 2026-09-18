"""Which model a nested SQLSaber agent uses."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Final

from sqlsaber.config import providers
from sqlsaber.config.providers import AuthKind


class Inherit(Enum):
    """The deferring branch of ``NestedModel``."""

    INHERIT = auto()

    def __str__(self) -> str:
        return "inherit"


INHERIT: Final = Inherit.INHERIT


@dataclass(frozen=True, slots=True)
class ModelId:
    """A canonical ``provider:model`` identifier."""

    provider: str
    model: str

    def __post_init__(self) -> None:
        if providers.canonical(self.provider) != self.provider:
            raise ValueError(f"Unsupported provider '{self.provider}'")
        if not self.model or self.model != self.model.strip():
            raise ValueError("Model id must be a nonempty catalog id")

    def __str__(self) -> str:
        return f"{self.provider}:{self.model}"


@dataclass(frozen=True, slots=True)
class Pinned:
    """The explicit branch of ``NestedModel``.

    ``api_key is None`` means resolve this provider's credential the normal way.
    """

    id: ModelId
    api_key: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.api_key is not None and not self.api_key.strip():
            object.__setattr__(self, "api_key", None)
        if (
            self.api_key is not None
            and providers.auth_kind(self.id.provider) is AuthKind.OPENAI_CODEX
        ):
            raise ValueError("OpenAI Codex subscription models do not accept API keys.")


type NestedModel = Inherit | Pinned


def parse_model_id(spec: str) -> ModelId:
    """Parse ``provider:model``, canonicalizing provider aliases.

    Raises:
        ValueError: naming the supported providers.
    """
    text = spec.strip()
    provider_raw, separator, model = text.partition(":")
    canonical = providers.canonical(provider_raw)
    if not separator or not model.strip() or canonical is None:
        keys = ", ".join(providers.all_keys())
        raise ValueError(
            f"MODEL must use a supported PROVIDER:MODEL ID. Providers: {keys}"
        )
    return ModelId(provider=canonical, model=model.strip())


def pin(spec: str, *, api_key: str | None = None) -> Pinned:
    """Pin a nested agent to ``provider:model``."""
    return Pinned(id=parse_model_id(spec), api_key=api_key)


def parse_nested_model(value: str | None) -> NestedModel:
    """Read a stored or typed value. Absent, empty, or blank is ``INHERIT``."""
    if value is None or not str(value).strip():
        return INHERIT
    return pin(str(value))


def most_specific(*layers: NestedModel) -> NestedModel:
    """Fold configuration layers outermost-first. The first pin wins."""
    for layer in layers:
        if isinstance(layer, Pinned):
            return layer
    return INHERIT
