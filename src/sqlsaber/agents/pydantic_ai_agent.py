"""Pydantic-AI Agent for SQLSaber.

This replaces the custom AnthropicSQLAgent and uses pydantic-ai's Agent,
function tools, and streaming event types directly.
"""

import inspect
from collections.abc import AsyncIterable, Awaitable, Sequence
from typing import Any, Callable, Literal, cast

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import AgentStreamEvent, ModelMessage
from pydantic_ai.models import Model

from sqlsaber.agents.provider_factory import ProviderFactory
from sqlsaber.config import providers
from sqlsaber.config.settings import Config, ThinkingLevel
from sqlsaber.database import BaseDatabaseConnection
from sqlsaber.database.registry import DatabaseEntry, DatabaseRegistry
from sqlsaber.database.schema import SchemaManager
from sqlsaber.knowledge.manager import KnowledgeManager
from sqlsaber.overrides import (
    ToolOveridesInput,
    ToolRunDeps,
    build_tool_run_deps,
    normalize_tool_overides,
)
from sqlsaber.prompts.claude import CLAUDE, CLAUDE_MULTI
from sqlsaber.prompts.dangerous_mode import DANGEROUS_MODE
from sqlsaber.prompts.openai import GPT_5, GPT_5_MULTI
from sqlsaber.tools.base import Tool
from sqlsaber.tools.knowledge_tool import KnowledgeTool
from sqlsaber.tools.registry import tool_registry
from sqlsaber.tools.sql_tools import SQLTool


def _wrap_strip_db_name(tool: Tool) -> Callable[..., Awaitable[str]]:
    """Wrap `tool.execute` so its public signature no longer includes `db_name`.

    Used in single-DB mode so the tool's JSON schema is identical to today's.
    The underlying tool still accepts `db_name=None` via its single-DB path.
    """
    raw = tool.execute
    raw_sig = inspect.signature(raw)
    new_params = [p for name, p in raw_sig.parameters.items() if name != "db_name"]
    new_sig = raw_sig.replace(parameters=new_params)

    if tool.requires_ctx:

        async def wrapper(ctx: RunContext, *args, **kwargs) -> str:
            return await raw(ctx, *args, **kwargs)
    else:

        async def wrapper(*args, **kwargs) -> str:
            return await raw(*args, **kwargs)

    wrapper.__signature__ = new_sig  # type: ignore
    wrapper.__name__ = getattr(raw, "__name__", tool.name)
    wrapper.__doc__ = raw.__doc__
    wrapper.__annotations__ = {
        k: v for k, v in getattr(raw, "__annotations__", {}).items() if k != "db_name"
    }
    return wrapper


def _wrap_add_db_name(
    tool: Tool, names: tuple[str, ...]
) -> Callable[..., Awaitable[str]]:
    """Wrap `tool.execute` so the public schema requires `db_name: Literal[...]`.

    Used in multi-DB mode. The Literal binds the LLM to a registered database
    name, so an invalid value is rejected at schema validation time rather than
    surfaced as a runtime tool error.
    """
    raw = tool.execute
    raw_sig = inspect.signature(raw)
    db_lit = Literal[names]  # type: ignore

    new_params = []
    for name, p in raw_sig.parameters.items():
        if name == "db_name":
            new_params.append(
                inspect.Parameter(
                    "db_name",
                    inspect.Parameter.KEYWORD_ONLY,
                    annotation=db_lit,
                )
            )
        else:
            new_params.append(p)
    if not any(p.name == "db_name" for p in new_params):
        new_params.append(
            inspect.Parameter(
                "db_name",
                inspect.Parameter.KEYWORD_ONLY,
                annotation=db_lit,
            )
        )
    new_sig = raw_sig.replace(parameters=new_params)

    if tool.requires_ctx:

        async def wrapper(ctx: RunContext, *args, db_name, **kwargs) -> str:
            return await raw(ctx, *args, db_name=db_name, **kwargs)
    else:

        async def wrapper(*args, db_name, **kwargs) -> str:
            return await raw(*args, db_name=db_name, **kwargs)

    wrapper.__signature__ = new_sig  # type: ignore
    wrapper.__name__ = getattr(raw, "__name__", tool.name)
    wrapper.__doc__ = _docstring_with_db_name(raw.__doc__)
    wrapper.__annotations__ = {
        **getattr(raw, "__annotations__", {}),
        "db_name": db_lit,
    }
    return wrapper


_DB_NAME_DOC_TEXT = "db_name: which connected database to target."


def _dedent_docstring(doc: str) -> str:
    """Dedent a docstring per PEP 257 (first line untouched).

    `textwrap.dedent` does nothing when the first line has no leading whitespace
    but later lines do. PEP 257 dedents based on the minimum indent of all
    lines after the first, then leaves the first line untouched.
    """
    lines = doc.splitlines()
    if not lines:
        return doc

    rest = lines[1:]
    indents = [len(line) - len(line.lstrip(" ")) for line in rest if line.strip()]
    if not indents:
        return doc.strip("\n")

    common = min(indents)
    dedented_rest = [line[common:] if line.strip() else line for line in rest]
    return "\n".join([lines[0].lstrip(), *dedented_rest]).strip("\n")


def _docstring_with_db_name(doc: str | None) -> str:
    """Add a `db_name` entry to a tool docstring's Args section, in place.

    Pydantic-ai parses the function docstring (via griffe) to populate
    per-parameter descriptions. Two `Args:` blocks confuse the parser and wipe
    earlier descriptions, so we splice `db_name` into the existing block when
    one exists. We dedent first so the new entry inherits the original
    docstring's indentation level rather than guessing it.
    """
    if not doc:
        return f"Args:\n    {_DB_NAME_DOC_TEXT}\n"

    dedented = _dedent_docstring(doc)
    lines = dedented.splitlines()
    args_idx = next(
        (i for i, line in enumerate(lines) if line.strip().startswith("Args:")),
        None,
    )
    if args_idx is None:
        return dedented + f"\n\nArgs:\n    {_DB_NAME_DOC_TEXT}\n"

    insert_at = args_idx + 1
    # Skip any existing entries already inside the Args block.
    while insert_at < len(lines) and (
        lines[insert_at].startswith(" ") or lines[insert_at] == ""
    ):
        insert_at += 1
    lines.insert(insert_at, f"    {_DB_NAME_DOC_TEXT}")
    return "\n".join(lines) + "\n"


def _build_db_catalog(registry: DatabaseRegistry) -> str:
    """Render the registry as a markdown bullet list for the system prompt."""
    lines: list[str] = []
    for entry in registry:
        description = entry.description or "no description"
        lines.append(
            f"- {entry.name} ({entry.display_name}, dialect={entry.dialect}) "
            f"— {description}"
        )
    return "\n".join(lines)


def _make_single_db_registry(
    connection: BaseDatabaseConnection, name: str | None
) -> DatabaseRegistry:
    """Build a single-entry registry from a connection (used as a thin shim)."""
    entry = DatabaseEntry.from_connection(
        name=name or connection.display_name or "database",
        connection=connection,
        description=None,
        excluded_schemas=list(getattr(connection, "excluded_schemas", []) or []),
    )
    return DatabaseRegistry([entry])


class SQLSaberAgent:
    """Pydantic-AI Agent wrapper for SQLSaber with enhanced state management."""

    def __init__(
        self,
        db_connection: BaseDatabaseConnection | None = None,
        database_name: str | None = None,
        *,
        registry: DatabaseRegistry | None = None,
        settings: Config | None = None,
        knowledge_manager: KnowledgeManager | None = None,
        thinking_enabled: bool | None = None,
        thinking_level: ThinkingLevel | None = None,
        model_name: str | None = None,
        api_key: str | None = None,
        allow_dangerous: bool = False,
        system_prompt: str | None = None,
        tool_overides: ToolOveridesInput | None = None,
    ):
        if registry is None:
            if db_connection is None:
                raise ValueError(
                    "SQLSaberAgent requires either `registry` or `db_connection`."
                )
            registry = _make_single_db_registry(db_connection, database_name)

        self.registry = registry
        primary = self.registry.get(self.registry.primary())
        self.db_connection: BaseDatabaseConnection = primary.connection
        self.database_name = primary.name
        self.config = settings or Config.default()
        self._owns_knowledge_manager = knowledge_manager is None
        self.knowledge_manager = knowledge_manager or KnowledgeManager()
        if system_prompt is not None and not system_prompt.strip():
            self.system_prompt_override = None
        else:
            self.system_prompt_override = system_prompt
        self._model_name_override = model_name
        self._api_key_override = api_key
        self.db_type = self.db_connection.display_name
        self.allow_dangerous = allow_dangerous
        self._tool_overides = normalize_tool_overides(tool_overides)

        self.schema_manager: SchemaManager = primary.schema_manager

        self.thinking_enabled = (
            thinking_enabled
            if thinking_enabled is not None
            else self.config.model.thinking_enabled
        )

        self.thinking_level = (
            thinking_level
            if thinking_level is not None
            else self.config.model.thinking_level
        )

        self._tools: dict[str, Tool] = {}
        for name in tool_registry.list_tools():
            cls = tool_registry.get_tool_class(name)
            if cls.multi_db_only and len(self.registry) <= 1:
                continue
            self._tools[name] = tool_registry.create_tool(name)

        self._configure_sql_tools()
        self._configure_knowledge_tools()
        self.agent = self._build_agent()

    @property
    def is_multi_db(self) -> bool:
        return len(self.registry) > 1

    def _configure_sql_tools(self) -> None:
        """Ensure SQL tools receive the active database state and session config."""
        for tool in self._tools.values():
            if isinstance(tool, SQLTool):
                if self.is_multi_db:
                    tool.set_registry(self.registry)
                else:
                    tool.set_connection(self.db_connection, self.schema_manager)
                tool.allow_dangerous = self.allow_dangerous

    def _configure_knowledge_tools(self) -> None:
        """Ensure knowledge tools receive database and manager context."""
        for tool in self._tools.values():
            if isinstance(tool, KnowledgeTool):
                if self.is_multi_db:
                    tool.set_registry(self.registry)
                    tool.knowledge_manager = self.knowledge_manager
                else:
                    tool.set_context(self.database_name, self.knowledge_manager)

    def _build_agent(self) -> Agent:
        """Create and configure the pydantic-ai Agent."""
        if self._api_key_override and not self._model_name_override:
            raise ValueError(
                "Model name is required when providing an api_key override."
            )

        model_name = self._model_name_override or self.config.model.name
        model_name_only = (
            model_name.split(":", 1)[1] if ":" in model_name else model_name
        )

        if not (self._model_name_override and self._api_key_override):
            self.config.auth.validate(model_name)

        provider = providers.provider_from_model(model_name) or ""

        api_key = self._api_key_override or self.config.auth.get_api_key(model_name)

        factory = ProviderFactory()
        agent = factory.create_agent(
            provider=provider,
            model_name=model_name_only,
            full_model_str=model_name,
            api_key=api_key,
            thinking_enabled=self.thinking_enabled,
            thinking_level=self.thinking_level,
        )

        self._setup_system_prompt(agent)
        self._register_tools(agent)
        return agent

    def _setup_system_prompt(self, agent: Agent) -> None:
        """Configure the agent's system prompt using a simple prompt string."""

        @agent.system_prompt(dynamic=True)
        async def sqlsaber_system_prompt(ctx: RunContext) -> str:
            use_gpt5 = (
                isinstance(agent.model, Model) and "gpt-5" in agent.model.model_name
            )
            base = self._base_system_prompt(use_gpt5=use_gpt5)
            return self._apply_prompt_extras(base)

    def _base_system_prompt(self, *, use_gpt5: bool) -> str:
        if self.system_prompt_override is not None:
            return self.system_prompt_override
        if self.is_multi_db:
            catalog = _build_db_catalog(self.registry)
            if use_gpt5:
                return GPT_5_MULTI.format(db_catalog=catalog)
            return CLAUDE_MULTI.format(db_catalog=catalog)
        if use_gpt5:
            return GPT_5.format(db=self.db_type)
        return CLAUDE.format(db=self.db_type)

    def _apply_prompt_extras(self, base: str) -> str:
        prompt = base
        if self.allow_dangerous:
            prompt += DANGEROUS_MODE
        return prompt

    def system_prompt_text(self) -> str:
        """Return the SQLSaber system prompt as a single string."""
        base = self._base_system_prompt(use_gpt5=False)
        return self._apply_prompt_extras(base)

    def _register_tools(self, agent: Agent) -> None:
        """Register tools, applying the right wrapper for the session shape."""
        names = tuple(self.registry.names())
        for tool in self._tools.values():
            sig = inspect.signature(tool.execute)
            has_db_name = "db_name" in sig.parameters
            register = agent.tool if tool.requires_ctx else agent.tool_plain

            if not has_db_name:
                # e.g. ListDatabasesTool — no per-call DB routing
                register(name=tool.name)(tool.execute)
                continue

            if self.is_multi_db:
                wrapper = _wrap_add_db_name(tool, names)
            else:
                wrapper = _wrap_strip_db_name(tool)
            register(name=tool.name)(wrapper)

    def set_thinking(self, enabled: bool, level: ThinkingLevel | None = None) -> None:
        """Update thinking settings and rebuild the agent.

        Args:
            enabled: Whether thinking is enabled.
            level: Optional thinking level to set. If not provided, keeps current level.
        """
        self.thinking_enabled = enabled
        if level is not None:
            self.thinking_level = level
        self.agent = self._build_agent()

    async def run(
        self,
        prompt: str,
        message_history: Sequence[ModelMessage] | None = None,
        event_stream_handler: Callable[
            [RunContext[Any], AsyncIterable[AgentStreamEvent]],
            Awaitable[None],
        ]
        | None = None,
    ) -> Any:
        """Run the agent."""
        run_agent = cast(Agent[ToolRunDeps, Any], self.agent)
        return await run_agent.run(
            prompt,
            message_history=message_history,
            event_stream_handler=event_stream_handler,
            deps=build_tool_run_deps(self._tool_overides),
        )

    async def close(self) -> None:
        """Close resources owned by the agent."""
        if self._owns_knowledge_manager:
            await self.knowledge_manager.close()
            self._owns_knowledge_manager = False
