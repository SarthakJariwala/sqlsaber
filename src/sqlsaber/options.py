"""Session options for SQLSaber SDK usage."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from sqlsaber.config.settings import ThinkingLevel
from sqlsaber.overrides import ToolOveridesInput

if TYPE_CHECKING:
    from sqlsaber.config.settings import Config
    from sqlsaber.knowledge.manager import KnowledgeManager
    from sqlsaber.threads.manager import ThreadManager


@dataclass(slots=True)
class SQLSaberOptions:
    """Typed options bag for SQLSaber session construction."""

    # Database
    database: str | list[str] | tuple[str, ...] | None = None

    # Model
    model_name: str | None = None
    api_key: str | None = None
    thinking_enabled: bool | None = None
    thinking_level: ThinkingLevel | str | None = None

    # Prompt
    system_prompt: str | Path | None = None

    # Injectable components
    settings: Config | None = None
    tools: object | None = None
    providers: object | None = None
    knowledge_manager: KnowledgeManager | None = None
    thread_manager: ThreadManager | None = None
    hooks: Sequence[object] = ()

    # Tool overrides
    tool_overrides: ToolOveridesInput | None = None
    allow_dangerous: bool = False
