"""Session options for SQLSaber SDK usage."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from sqlsaber.config.settings import ThinkingLevel
from sqlsaber.overrides import ToolOveridesInput

if TYPE_CHECKING:
    from sqlsaber.artifacts import ArtifactFailureMode, ArtifactStore
    from sqlsaber.capabilities.plugins import CapabilitySpec
    from sqlsaber.config.settings import Config
    from sqlsaber.knowledge.manager import KnowledgeManager
    from sqlsaber.query_results import QueryResultStore
    from sqlsaber.threads.manager import ThreadManager
    from sqlsaber.workspace_inputs import WorkspaceInputResolver


@dataclass(slots=True)
class SQLSaberOptions:
    """Typed options bag for SQLSaber session construction.

    ``capabilities`` is an explicit list of pydantic-ai capabilities and
    PluginContext factories. It defaults to empty. SQLSaber always includes
    Knowledge and SqlTools. Pass ``load_capability_factories()`` to opt into
    installed ``sqlsaber.capabilities`` plugins, as the CLI does.
    """

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
    knowledge_manager: KnowledgeManager | None = None
    thread_manager: ThreadManager | None = None
    capabilities: Sequence[CapabilitySpec] = ()
    artifact_store: ArtifactStore | None = None
    artifact_failure_mode: ArtifactFailureMode = "required"
    query_result_store: QueryResultStore | None = None
    workspace_input_resolver: WorkspaceInputResolver | None = None

    # Tool overrides
    tool_overrides: ToolOveridesInput | None = None
    allow_dangerous: bool = False
    csv_tool_results: bool = False
