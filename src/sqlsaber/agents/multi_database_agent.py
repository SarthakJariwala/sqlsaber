"""Coordinator agent for routing questions across multiple database agents."""

from __future__ import annotations

import inspect
import json
from collections.abc import AsyncIterable, Awaitable, Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, cast

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import AgentStreamEvent, ModelMessage

from sqlsaber.agents.provider_factory import ProviderFactory
from sqlsaber.config import providers
from sqlsaber.config.settings import Config, ThinkingLevel
from sqlsaber.threads.manager import ThreadManager


class DatabaseDescriptor(BaseModel):
    """Database metadata available to the coordinator."""

    id: str
    name: str
    type: str
    description: str | None = None
    summary: str | None = None
    thread_id: str | None = None


@dataclass
class DatabaseChild:
    """Runtime state for one child SQLSaber agent."""

    descriptor: DatabaseDescriptor
    agent: Any
    thread_manager: ThreadManager
    database_name: str
    message_history: list[ModelMessage] = field(default_factory=list)


@dataclass
class MultiDatabaseDeps:
    """Dependencies injected into coordinator tools."""

    children: dict[str, DatabaseChild]


class MultiDatabaseCoordinator:
    """Pydantic-AI coordinator that routes work to one child per database."""

    def __init__(
        self,
        children: dict[str, DatabaseChild],
        *,
        database_label: str,
        settings: Config | None = None,
        thinking_enabled: bool | None = None,
        thinking_level: ThinkingLevel | None = None,
        model_name: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.children = children
        self.database_label = database_label
        self.config = settings or Config.default()
        self._model_name_override = model_name
        self._api_key_override = api_key

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

        self.agent = self._build_agent()

    def _build_agent(self) -> Agent:
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

        agent = ProviderFactory().create_agent(
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
        typed_agent = cast(Agent[MultiDatabaseDeps, Any], agent)

        @typed_agent.system_prompt(dynamic=True)
        async def multi_database_system_prompt(
            ctx: RunContext[MultiDatabaseDeps],
        ) -> str:
            _ = ctx
            return self.system_prompt_text()

    def _register_tools(self, agent: Agent) -> None:
        typed_agent = cast(Agent[MultiDatabaseDeps, Any], agent)

        @typed_agent.tool
        async def list_connected_databases(
            ctx: RunContext[MultiDatabaseDeps],
        ) -> list[DatabaseDescriptor]:
            _ = ctx
            return self.descriptors()

        @typed_agent.tool
        async def ask_database(
            ctx: RunContext[MultiDatabaseDeps],
            database_id: str,
            question: str,
        ) -> str:
            return await self.ask_database_direct(
                database_id,
                question,
                usage=ctx.usage,
            )

    def descriptors(self) -> list[DatabaseDescriptor]:
        """Return current descriptors for connected child databases."""
        return [child.descriptor for child in self.children.values()]

    def system_prompt_text(self) -> str:
        """Return the coordinator system prompt as a single string."""
        database_lines = "\n".join(
            (
                f"- {descriptor.id}: {descriptor.name} "
                f"({descriptor.type})"
                f"{f' - {descriptor.description}' if descriptor.description else ''}"
                f"{f' Summary: {descriptor.summary}' if descriptor.summary else ''}"
                f"{f' [thread ID: {descriptor.thread_id}]' if descriptor.thread_id else ''}"
            )
            for descriptor in self.descriptors()
        )
        if not database_lines:
            database_lines = "- No databases are currently connected."

        return (
            "You are the SQLSaber multi-database coordinator for "
            f"'{self.database_label}'. Route questions to the database-specific "
            "child agents using the available tools.\n\n"
            "Connected databases:\n"
            f"{database_lines}\n\n"
            "cross-database SQL joins cannot be executed. query databases "
            "independently and state limitations when an answer requires comparing "
            "or combining data across databases.\n"
            "Thread IDs are traceability metadata. Include relevant child thread "
            "IDs in answers, but do not treat transcript references as evidence."
        )

    async def ask_database_direct(
        self,
        database_id: str,
        question: str,
        *,
        usage: Any | None = None,
    ) -> str:
        """Ask a specific child database agent and return its text answer."""
        child = self.children.get(database_id)
        if child is None:
            return self._format_child_answer(
                database_id,
                database_id,
                thread_id=None,
                answer_text=(
                    "## Answer\n"
                    f"Unable to answer from database '{database_id}'.\n\n"
                    "## Limitations\n"
                    f"- Unknown database id '{database_id}'."
                ),
            )

        model_name = self._child_model_name(child)
        ensure_thread_kwargs: dict[str, str | None] = {
            "database_name": child.database_name,
        }
        if child.thread_manager.current_thread_id is None:
            ensure_thread_kwargs.update(
                title=f"[{child.database_name}] {question}",
                model_name=model_name,
                extra_metadata=json.dumps(
                    {
                        "kind": "multi_database_child",
                        "database_id": database_id,
                    }
                ),
            )

        thread_id = await child.thread_manager.ensure_thread(**ensure_thread_kwargs)
        child.descriptor.thread_id = thread_id

        prompt = self._child_prompt(child.descriptor, question)
        try:
            run_result = await child.agent.run(
                prompt,
                message_history=child.message_history,
                usage=usage,
            )
        except Exception as exc:
            return self._format_child_answer(
                child.descriptor.id,
                child.database_name,
                thread_id=thread_id,
                answer_text=(
                    "## Answer\n"
                    f"Unable to answer from database '{child.descriptor.id}'.\n\n"
                    "## Limitations\n"
                    f"- Child agent failed: {exc}"
                ),
            )

        answer_text = self._coerce_child_answer_text(self._run_output(run_result))

        child.message_history = await child.thread_manager.save_run(
            run_result=run_result,
            database_name=child.database_name,
            user_query=question,
            model_name=model_name,
        )

        return self._format_child_answer(
            child.descriptor.id,
            child.database_name,
            thread_id=thread_id,
            answer_text=answer_text,
        )

    def _child_prompt(self, descriptor: DatabaseDescriptor, question: str) -> str:
        return (
            f"Answer using only database '{descriptor.name}' "
            f"(id: {descriptor.id}, type: {descriptor.type}).\n"
            "Return plain markdown with these sections when they apply:\n"
            "## Answer\n"
            "## Evidence\n"
            "## SQL\n"
            "## Limitations\n\n"
            "Evidence should include result excerpts or schema facts used. SQL should "
            "include executed queries, or `None` if no SQL was executed. Limitations "
            "should state missing data, ambiguity, or single-database constraints.\n\n"
            f"Question: {question}"
        )

    def _format_child_answer(
        self,
        database_id: str,
        database_name: str,
        *,
        thread_id: str | None,
        answer_text: str,
    ) -> str:
        thread_label = thread_id or "unavailable"
        return (
            f"Database: {database_name} (id: {database_id})\n"
            f"Child thread ID: {thread_label}\n\n"
            f"{answer_text.strip()}"
        )

    def _coerce_child_answer_text(self, output: Any) -> str:
        if output is None:
            return (
                "## Answer\n"
                "No answer was returned.\n\n"
                "## Limitations\n"
                "- Child agent returned no text."
            )
        text = output if isinstance(output, str) else str(output)
        text = text.strip()
        if text:
            return text
        return (
            "## Answer\n"
            "No answer was returned.\n\n"
            "## Limitations\n"
            "- Child agent returned an empty response."
        )

    def _run_output(self, run_result: Any) -> Any:
        if hasattr(run_result, "output"):
            return run_result.output
        if hasattr(run_result, "data"):
            return run_result.data
        return None

    def _child_model_name(self, child: DatabaseChild) -> str:
        child_pydantic_agent = getattr(child.agent, "agent", None)
        child_model = getattr(child_pydantic_agent, "model", None)
        model_name = getattr(child_model, "model_name", None)
        if isinstance(model_name, str) and model_name:
            return model_name
        return self.config.model.name

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
        """Run the coordinator agent with child-agent dependencies."""
        run_agent = cast(Agent[MultiDatabaseDeps, Any], self.agent)
        return await run_agent.run(
            prompt,
            message_history=message_history,
            event_stream_handler=event_stream_handler,
            deps=MultiDatabaseDeps(children=self.children),
        )

    async def close(self) -> None:
        """Close child agents that expose an async close method."""
        for child in self.children.values():
            close = getattr(child.agent, "close", None)
            if close is None:
                continue
            result = close()
            if inspect.isawaitable(result):
                await result
