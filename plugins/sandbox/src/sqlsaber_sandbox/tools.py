"""Goal-based managed sandbox tools and artifact presentation."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
import logging
from typing import Any, cast

from pydantic_ai import RunContext, ToolReturn
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import UsageLimits

from sqlsaber.artifact_resolution import (
    ResolvedArtifactPublication,
    artifact_context_from_run,
)
from sqlsaber.artifacts import (
    ArtifactContext,
    ArtifactPublication,
    ArtifactPublicationError,
    artifact_publication_from_metadata,
)
from sqlsaber.capabilities.plugins import PluginContext
from sqlsaber.render import blocks as b
from sqlsaber.run_usage import current_usage_limits
from sqlsaber.tools.base import Tool
from sqlsaber.tools.display import (
    ToolDisplaySpec,
    ExecutingConfig,
    ResultConfig,
    FieldMappings,
)
from sqlsaber.tools.renderer import ToolRenderContext
from sqlsaber.utils.text_input import sanitize_terminal_text

from .backends import SandboxBackend
from .config import ANALYZE_IN_SANDBOX, SandboxConfig
from .execution import SandboxError
from .publication import publish_analysis
from .result import AnalysisResult
from .session import SandboxSession
from .workspace import build_workspace_from_history

logger = logging.getLogger(__name__)


async def prepare_analysis(
    ctx: RunContext, definition: ToolDefinition
) -> ToolDefinition:
    """Serialize budgeted delegations, not independent unbudgeted analyses.

    A per-batch barrier prevents two children from admitting requests against
    the same finite remaining budget. No lock is held during unbudgeted work.
    """
    limits = current_usage_limits()
    finite = limits is not None and (
        limits.request_limit is not None or limits.tool_calls_limit is not None
    )
    return replace(definition, sequential=finite)


class AnalyzeSandboxTool(Tool):
    requires_ctx = True
    display_spec = ToolDisplaySpec(
        executing=ExecutingConfig(
            message="Analyzing in persistent sandbox", show_args=["goal", "session_id"]
        ),
        result=ResultConfig(
            format="panel",
            title="Sandbox analysis",
            fields=FieldMappings(output="answer"),
        ),
    )

    def __init__(
        self,
        context: PluginContext,
        config: SandboxConfig,
        *,
        backend_factory: Callable[[], SandboxBackend] | None = None,
    ):
        self.context = context
        self.config = config
        self._backend_factory = backend_factory
        self.sessions: dict[str, tuple[str, SandboxSession]] = {}
        self.results: dict[str, AnalysisResult] = {}
        self.publications: dict[str, ArtifactPublication] = {}
        self._artifact_contexts: dict[str, ArtifactContext] = {}
        self._displays: OrderedDict[str, tuple[bytes, ...]] = OrderedDict()
        self._resolved: Mapping[str, ResolvedArtifactPublication] = {}

    @property
    def name(self) -> str:
        return ANALYZE_IN_SANDBOX

    def _session(self, ctx: RunContext, session_id: str) -> SandboxSession:
        entry = self.sessions.get(session_id)
        if entry is None or entry[0] != ctx.conversation_id:
            raise ValueError("Sandbox session is unavailable in this conversation")
        return entry[1]

    async def execute(
        self,
        ctx: RunContext,
        goal: str,
        files: list[str] | None = None,
        session_id: str | None = None,
    ) -> ToolReturn:
        """Delegate a goal to a persistent sandbox analyst.

        Args:
            goal: Analysis to perform. The subagent writes and iterates on code.
            files: SQL result keys. Omit for recent results on a new session;
                omission adds no files on continuation. An empty list selects none.
            session_id: Omit to create an independent analysis session; provide a
                previous ID to reuse its variables, files, and analyst history.
        """
        return await self.execute_with_attachments(ctx, goal, files, session_id)

    async def execute_with_attachments(
        self,
        ctx: RunContext,
        goal: str,
        files: list[str] | None = None,
        session_id: str | None = None,
        attachment_refs: list[str] | None = None,
    ) -> ToolReturn:
        """Delegate a goal with SQL results and host-authorized attachments.

        Args:
            goal: Analysis to perform; code iteration stays inside the subagent.
            files: SQL result keys. Omission adds recent results only for new sessions.
            session_id: Existing session to continue, or omit to start an independent one.
            attachment_refs: Opaque references authorized by the application's resolver.
        """
        limits = current_usage_limits()
        if limits is not None and limits.tool_calls_limit is not None:
            if ctx.usage.tool_calls >= limits.tool_calls_limit:
                raise UsageLimitExceeded(
                    "No tool budget remains for sandbox delegation"
                )
            limits = replace(limits, tool_calls_limit=limits.tool_calls_limit - 1)
        session = None
        try:
            if not goal.strip():
                raise ValueError("Analysis goal cannot be empty")
            if session_id is not None:
                session = self._session(ctx, session_id)
            workspace = await build_workspace_from_history(
                ctx,
                only=[] if session_id is not None and files is None else files,
                attachment_refs=attachment_refs,
                query_result_store=self.context.query_result_store,
                workspace_input_resolver=self.context.workspace_input_resolver,
                limits=self.config.workspace,
            )
            if session is None:
                if not ctx.conversation_id:
                    raise ValueError(
                        "A conversation identity is required for managed sandbox sessions"
                    )
                child = self.context.resolve_subagent_model(
                    self.config.model, tool=self.name
                )
                session = SandboxSession(
                    model=child.model,
                    model_provider=child.provider,
                    config=self.config,
                    backend=(
                        self._backend_factory()
                        if self._backend_factory is not None
                        else None
                    ),
                )
                self.sessions[session.id] = (ctx.conversation_id, session)
            result = await session.analyze(
                goal,
                workspace=workspace,
                usage_limits=limits or UsageLimits(request_limit=None),
                parent_usage=ctx.usage,
            )
            self.results[session.id] = result
            images = tuple(
                item.data for item in result.files if item.media_type == "image/png"
            )
            self._displays[ctx.tool_call_id or ""] = images
            while len(self._displays) > 2:
                self._displays.popitem(last=False)
            return await self._publish(ctx, result)
        except (ValueError, SandboxError, TimeoutError) as exc:
            return ToolReturn(
                return_value={
                    "error": str(exc),
                    "session_id": session.id if session is not None else session_id,
                    "session_state": "lost"
                    if session is not None and session.lost
                    else "ready"
                    if session is not None and not session.closed
                    else "unavailable",
                }
            )

    async def _publish(
        self, ctx: RunContext | ArtifactContext, result: AnalysisResult
    ) -> ToolReturn:
        context = artifact_context_from_run(ctx)
        self._artifact_contexts[result.session_id] = context
        session = self.sessions[result.session_id][1]
        metadata: dict[str, Any] = {}
        value: dict[str, Any] = {
            "session_id": result.session_id,
            "analysis_id": result.analysis_id,
            "answer": result.answer,
            "session_state": "closed" if session.closed else "ready",
            "publication_state": "not_configured",
        }
        if self.context.artifact_store is not None:
            try:
                publication = self.publications.get(result.analysis_id)
                if publication is None:
                    publication = await publish_analysis(
                        result,
                        store=self.context.artifact_store,
                        context=context,
                    )
                    self.publications[result.analysis_id] = publication
                metadata.update(publication.to_metadata())
                value["publication_state"] = "published"
                value["artifacts"] = [item.to_dict() for item in publication.artifacts]
            except Exception:
                logger.exception("Sandbox artifact publication failed")
                error = "Analysis completed, but artifacts could not be published. Retry publication without rerunning the analysis."
                value.update(
                    publication_state="failed",
                    artifact_failure_mode=self.context.artifact_failure_mode,
                )
                if self.context.artifact_failure_mode == "required":
                    value["error"] = error
                else:
                    metadata["artifact_error"] = error
        return ToolReturn(return_value=value, metadata=metadata)

    async def publish_artifacts(self, ctx: RunContext, session_id: str) -> ToolReturn:
        """Publish the latest completed snapshot without executing Python again."""
        session = self._session(ctx, session_id)
        result = self.results.get(session_id)
        if result is None:
            result = await session.snapshot()
            self.results[session_id] = result
        return await self._publish(ctx, result)

    async def close_session(self, ctx: RunContext, session_id: str) -> ToolReturn:
        """Publish completed artifacts and release a sandbox session.

        Required publication failure leaves the session open for a retry.
        """
        session = self._session(ctx, session_id)
        returned = ToolReturn(return_value={"session_id": session_id})
        result = self.results.get(session_id)
        if result is not None:
            returned = await self._publish(ctx, result)
        value = cast(dict[str, Any], returned.return_value)
        if "error" in value:
            return returned
        await session.close()
        value["session_state"] = "closed"
        return returned

    def set_resolved_artifact_publications(
        self, publications: Mapping[str, ResolvedArtifactPublication]
    ) -> None:
        self._resolved = publications

    def render_result(
        self, result: object, *, context: ToolRenderContext | None = None
    ) -> Sequence[b.Block] | None:
        ctx = context or ToolRenderContext()
        images = self._displays.pop(ctx.tool_call_id or "", ())
        reference = artifact_publication_from_metadata(ctx.metadata)
        if not images and reference is not None:
            publication = self._resolved.get(reference.id)
            if publication is not None:
                images = tuple(
                    item.data
                    for item in publication.artifacts
                    if item.descriptor.media_type == "image/png"
                )
        if not isinstance(result, Mapping):
            return None
        children: list[b.Block] = []
        if result.get("answer"):
            children.append(b.md(sanitize_terminal_text(str(result["answer"]))))
        if result.get("error"):
            children.append(b.error(sanitize_terminal_text(str(result["error"]))))
        children.append(
            b.key_values(
                {
                    "Session": sanitize_terminal_text(
                        str(result.get("session_id", "unavailable"))
                    ),
                    "State": sanitize_terminal_text(
                        str(result.get("session_state", "unavailable"))
                    ),
                    "Artifacts": sanitize_terminal_text(
                        str(result.get("publication_state", "not published"))
                    ),
                }
            )
        )
        if result.get("artifacts"):
            children.append(
                b.table(
                    [
                        {
                            "File": sanitize_terminal_text(str(item["name"])),
                            "Bytes": item["size"],
                        }
                        for item in result["artifacts"]
                    ]
                )
            )
        children.extend(
            b.image(image, "image/png", filename=f"plot_{index}.png")
            for index, image in enumerate(images)
        )
        return (b.panel(children, title="Sandbox analysis"),)

    async def close(self) -> None:
        errors = []
        pending_publication = False
        for _, session in self.sessions.values():
            try:
                result = self.results.get(session.id)
                if result is not None:
                    returned = await self._publish(
                        self._artifact_contexts[session.id], result
                    )
                    value = cast(dict[str, Any], returned.return_value)
                    pending_publication |= value["publication_state"] == "failed"
                    if "error" in value:
                        raise ArtifactPublicationError(value["error"])
            except Exception as exc:
                errors.append(exc)
            finally:
                # Shutdown must release compute even when artifact storage fails.
                try:
                    await session.close()
                except Exception as exc:
                    errors.append(exc)
        self._displays.clear()
        if not pending_publication and not errors:
            self.results.clear()
            self.publications.clear()
            self._artifact_contexts.clear()
        if errors:
            raise ExceptionGroup("Sandbox cleanup failed", errors)
