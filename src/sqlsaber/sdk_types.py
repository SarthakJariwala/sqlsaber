"""Immutable public values returned by the SQLSaber SDK."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlsaber.config.settings import ThinkingLevel


@dataclass(frozen=True, slots=True)
class ThinkingState:
    """Current reasoning configuration for the managed model."""

    enabled: bool
    level: ThinkingLevel


@dataclass(frozen=True, slots=True)
class SQLSaberInfo:
    """Read-only metadata for a live SQLSaber conversation."""

    database_names: tuple[str, ...]
    primary_database_name: str
    primary_database_type: str
    model_name: str
    model_id: str | None
    thinking: ThinkingState
    dangerous_mode: bool
    thread_id: str | None
    is_new_thread: bool

    @property
    def databases(self) -> tuple[str, ...]:
        return self.database_names

    @property
    def primary_database(self) -> str:
        return self.primary_database_name

    @property
    def database_type(self) -> str:
        return self.primary_database_type

    @property
    def allow_dangerous(self) -> bool:
        return self.dangerous_mode

    @property
    def new_thread(self) -> bool:
        return self.is_new_thread


@dataclass(frozen=True, slots=True)
class TableInfo:
    """One database table exposed for inspection and completion."""

    database_name: str
    schema_name: str
    name: str
    kind: str
    qualified_name: str
    completion_name: str
