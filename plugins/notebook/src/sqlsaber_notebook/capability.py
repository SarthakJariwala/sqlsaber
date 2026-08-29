"""Managed SQLsaber capability for notebook data analysis."""

from __future__ import annotations

import hashlib
import logging
import re
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, cast

from pydantic_ai import RunContext, ToolReturn
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.usage import UsageLimits

from sqlsaber.artifact_resolution import (
    ResolvedArtifactPublication,
    artifact_context_from_run,
)
from sqlsaber.artifacts import ArtifactUnavailable, artifact_publication_from_metadata
from sqlsaber.capabilities.base import SqlSaberCapability
from sqlsaber.capabilities.plugins import PluginContext
from sqlsaber.query_result_resolution import (
    find_query_result_reference,
    query_result_context_from_run,
    query_result_references_from_messages,
    resolve_query_result,
)
from sqlsaber.query_results import (
    InMemoryQueryResultStore,
    QueryResultStore,
    QueryResultUnavailable,
)
from sqlsaber.render import blocks as b
from sqlsaber.run_usage import current_usage_limits
from sqlsaber.utils.text_input import sanitize_terminal_text
from sqlsaber.tools.base import Tool
from sqlsaber.tools.renderer import ToolRenderContext
from sqlsaber.utils.json_utils import json_dumps
from sqlsaber.workspace_inputs import (
    WorkspaceInputFile,
    WorkspaceInputResolver,
    WorkspaceResolutionContext,
)

from ._shared import (
    MAX_DEFAULT_RESULTS,
    MAX_WORKSPACE_FILE_BYTES,
    MAX_WORKSPACE_FILES,
    MAX_WORKSPACE_MANIFEST_BYTES,
    MAX_WORKSPACE_TOTAL_BYTES,
)
from .analyst import analyze, supports_notebook_images
from .execution import (
    NotebookExecutionError,
    NotebookInput,
    NotebookLimitExceeded,
    resolve_notebook_backend,
    resolve_notebook_image,
)
from .execution.base import validate_input_name
from .publication import display_from_publication, publish_analysis
from .rendering import limit_output, render_notebook_bytes
from .result import (
    ManifestEntry,
    Workspace,
    WorkspaceFile,
    workspace_manifest_bytes,
)

logger = logging.getLogger(__name__)

_RESULT_FILE_PATTERN = re.compile(r"^result_[A-Za-z0-9._-]+\.json$")
_MAX_DISPLAY_RESULTS = 2


class WorkspaceInputUnavailable(ValueError):
    """A requested workspace input is unknown, invalid, or unauthorized."""


@dataclass(frozen=True, slots=True)
class _NotebookDisplay:
    markdown: str
    images: tuple[bytes, ...]


def _nested_usage_limits() -> UsageLimits:
    """Resolve child limits from the parent run's explicit limit selection."""
    parent_limits = current_usage_limits()
    if parent_limits is None:
        return UsageLimits(request_limit=None)
    if parent_limits.tool_calls_limit is None:
        return parent_limits
    # The successful parent analyze_data call is counted after execute returns.
    # Reserve room for it in the child's derived limit without mutating the parent.
    return replace(
        parent_limits,
        tool_calls_limit=max(0, parent_limits.tool_calls_limit - 1),
    )


class AnalyzeDataTool(Tool):
    """Delegate multi-step analysis to the notebook analyst."""

    requires_ctx = True

    def __init__(self, context: PluginContext) -> None:
        super().__init__()
        self._context = context
        self._display_results: OrderedDict[str, _NotebookDisplay] = OrderedDict()
        self._resolved_publications: Mapping[str, ResolvedArtifactPublication] = {}

    @property
    def name(self) -> str:
        return "analyze_data"

    async def execute(
        self,
        ctx: RunContext,
        goal: str,
        files: list[str] | None = None,
    ) -> ToolReturn | str:
        """Hand a data-analysis goal to a notebook subagent.

        Use this after running SQL when the answer requires multi-step calculations,
        statistical analysis, data transformations, or plots.

        Args:
            goal: The question to answer and analysis to perform.
            files: Optional execute_sql result keys to analyze. When omitted, the
                newest bounded successful query results are included.
        """

        return await self._execute(ctx, goal, files=files)

    async def execute_with_attachments(
        self,
        ctx: RunContext,
        goal: str,
        files: list[str] | None = None,
        attachment_refs: list[str] | None = None,
    ) -> ToolReturn | str:
        """Hand a data-analysis goal to a notebook subagent.

        Use this for multi-step calculations, statistical analysis, data
        transformations, plots, or analysis of application-provided inputs.

        Args:
            goal: The question to answer and analysis to perform.
            files: Optional execute_sql result keys to analyze. When omitted, the
                newest bounded successful query results are included.
            attachment_refs: Optional opaque, application-authorized input references.
                These are resolved by the configured host adapter, never as paths or
                URLs by SQLSaber.
        """

        return await self._execute(
            ctx,
            goal,
            files=files,
            attachment_refs=attachment_refs,
        )

    async def _execute(
        self,
        ctx: RunContext,
        goal: str,
        *,
        files: list[str] | None,
        attachment_refs: list[str] | None = None,
    ) -> ToolReturn | str:
        try:
            workspace = await build_workspace_from_history(
                ctx,
                only=files,
                attachment_refs=attachment_refs,
                workspace_input_resolver=getattr(
                    self._context,
                    "workspace_input_resolver",
                    None,
                ),
                query_result_store=getattr(
                    self._context,
                    "query_result_store",
                    InMemoryQueryResultStore(),
                ),
            )
            model_name, model, provider = self._context.resolve_subagent_model(
                "notebook",
                tool_name=self.name,
            )
            backend = resolve_notebook_backend()
            store = getattr(self._context, "artifact_store", None)
            result = await analyze(
                goal,
                workspace,
                model=model,
                model_provider=provider,
                backend=backend,
                image=resolve_notebook_image(),
                include_snapshot_images=supports_notebook_images(model_name, provider),
                collect_files=store is not None,
                usage_limits=_nested_usage_limits(),
                parent_usage=ctx.usage,
            )
            markdown, notebook_images = render_notebook_bytes(result.notebook)
            display_images = _dedupe_images([*notebook_images, *result.images])
            self._remember_display(
                ctx.tool_call_id,
                _NotebookDisplay(markdown, tuple(display_images)),
            )
            metadata: dict[str, object] = {
                "backend": backend.name,
                "model": model_name,
                "provenance": result.provenance,
                "files": [item.name for item in workspace.files],
            }
            if store is not None:
                try:
                    publication = await publish_analysis(
                        result,
                        store=store,
                        context=artifact_context_from_run(ctx),
                    )
                except Exception:
                    logger.exception("Notebook artifact publication failed")
                    failure_mode = getattr(
                        self._context, "artifact_failure_mode", "required"
                    )
                    public_error = "Artifacts could not be published."
                    if failure_mode == "required":
                        return _error_result(
                            public_error,
                            backend=backend.name,
                            phase="artifact-publication",
                        )
                    metadata["artifact_error"] = public_error
                else:
                    metadata.update(publication.to_metadata())
            return ToolReturn(return_value=result.answer, metadata=metadata)
        except NotebookExecutionError as exc:
            return _error_result(
                str(exc),
                backend=exc.backend,
                phase=exc.phase,
            )
        except (TimeoutError, ValueError) as exc:
            return _error_result(str(exc))

    def render_executing(self, args: dict):
        goal = args.get("goal")
        if isinstance(goal, str) and goal.strip():
            request = limit_output(goal.strip(), 4_000)
            inner = (b.md(f"**Analyzing data**\n\n{request}"),)
        else:
            inner = (b.md("**Analyzing data in notebook**"),)
        return (b.panel(inner),)

    def render_result(
        self, result: object, *, context: ToolRenderContext | None = None
    ):
        ctx = context or ToolRenderContext()
        display, reconstruction_failed = self._display_for(
            ctx.tool_call_id, ctx.metadata
        )
        if display is None:
            if reconstruction_failed:
                answer = sanitize_terminal_text(limit_output(str(result)).strip())
                blocks = [
                    b.md(
                        "*Persisted notebook could not be reconstructed; "
                        "showing its generic artifact references.*"
                    )
                ]
                if answer:
                    blocks.append(b.md(f"## Analysis result\n\n{answer}"))
                return tuple(blocks)
            return None

        notebook_md = sanitize_terminal_text(display.markdown.strip())
        children: list = [
            b.md(
                "## Analysis notebook\n\n"
                + (notebook_md or "*No notebook cells were executed.*")
            )
        ]
        for index, image in enumerate(display.images, start=1):
            children.append(b.md(f"**Plot {index}**"))
            children.append(
                b.image(
                    image,
                    "image/png",
                    filename=f"plot_{index}.png",
                    max_width_cells=None,
                )
            )
        answer = sanitize_terminal_text(limit_output(str(result)).strip())
        if answer:
            children.append(b.md(f"## Analysis result\n\n{answer}"))
        publication = artifact_publication_from_metadata(ctx.metadata)
        if publication is not None:
            artifact_lines = "\n".join(
                f"- `{artifact.name}`: `{artifact.uri}`"
                for artifact in publication.artifacts
            )
            children.append(b.md(f"## Artifacts\n\n{artifact_lines}"))
        return (b.panel(children),)

    def set_resolved_artifact_publications(
        self,
        publications: Mapping[str, ResolvedArtifactPublication],
    ) -> None:
        """Supply verified publications for read-only transcript replay."""

        self._resolved_publications = publications

    async def close(self) -> None:
        self._display_results.clear()
        self._resolved_publications = {}

    def _display_for(
        self,
        tool_call_id: str | None,
        metadata: object,
    ) -> tuple[_NotebookDisplay | None, bool]:
        live = self._display_results.pop(tool_call_id or "", None)
        if live is not None:
            return live, False
        reference = artifact_publication_from_metadata(metadata)
        if reference is None:
            return None, False
        publication = self._resolved_publications.get(reference.id)
        if publication is None:
            return None, False
        try:
            persisted = display_from_publication(publication)
        except ArtifactUnavailable:
            return None, True
        return _NotebookDisplay(persisted.markdown, persisted.images), False

    def _remember_display(
        self, tool_call_id: str | None, display: _NotebookDisplay
    ) -> None:
        key = tool_call_id or ""
        self._display_results[key] = display
        self._display_results.move_to_end(key)
        while len(self._display_results) > _MAX_DISPLAY_RESULTS:
            self._display_results.popitem(last=False)


class Notebook(SqlSaberCapability):
    """Delegate multi-step data analysis to a notebook subagent."""

    id = "notebook"
    description = "Delegate multi-step data analysis to a notebook subagent."

    def __init__(self, context: PluginContext) -> None:
        self.tool = AnalyzeDataTool(context)
        self._toolset = FunctionToolset[Any](id=self.id)
        execute = (
            self.tool.execute_with_attachments
            if getattr(context, "workspace_input_resolver", None) is not None
            else self.tool.execute
        )
        self._toolset.add_function(
            execute,
            name=self.tool.name,
            takes_ctx=True,
            # Notebook delegation must run as a barrier so sibling parent tools
            # are fully accounted before the nested agent checks shared limits.
            sequential=True,
        )

    @property
    def display_specs(self) -> Mapping[str, Tool]:
        return {self.tool.name: self.tool}

    def get_toolset(self) -> FunctionToolset[Any]:
        return self._toolset

    async def close(self) -> None:
        await self.tool.close()


def display_tools() -> Mapping[str, Tool]:
    """Return storage-independent notebook renderers for transcript replay."""

    return {"analyze_data": AnalyzeDataTool(cast(PluginContext, object()))}


def capability(
    context: PluginContext,
) -> AbstractCapability[Any] | Sequence[AbstractCapability[Any]]:
    """Always expose the installed plugin; backend checks happen on use."""

    return Notebook(context)


async def build_workspace_from_history(
    ctx: RunContext,
    only: list[str] | None,
    *,
    query_result_store: QueryResultStore,
    attachment_refs: list[str] | None = None,
    workspace_input_resolver: WorkspaceInputResolver | None = None,
) -> Workspace:
    """Build one bounded workspace from SQL results and authorized inputs."""

    requested = _normalize_requested_files(only)
    if requested is not None and len(requested) > MAX_WORKSPACE_FILES:
        raise NotebookLimitExceeded(
            f"Workspace has {len(requested)} files; maximum is {MAX_WORKSPACE_FILES}",
            backend="notebook",
            phase="input-validation",
        )
    resolved_inputs = await _resolve_workspace_inputs(
        ctx,
        attachment_refs,
        resolver=workspace_input_resolver,
    )
    external_bytes = sum(len(item.data) for item in resolved_inputs)

    if requested is None:
        references = list(reversed(query_result_references_from_messages(ctx.messages)))
        references = references[:MAX_DEFAULT_RESULTS]
    else:
        references = []
        missing: list[str] = []
        for file in requested:
            reference = find_query_result_reference(ctx.messages, file)
            if reference is None:
                missing.append(file)
            else:
                references.append(reference)
        if missing:
            raise ValueError(
                "Requested SQL result files were not found: " + ", ".join(missing)
            )

    if not references and not resolved_inputs:
        raise ValueError(
            "No successful row-returning execute_sql results are available to analyze"
        )

    selected: list[tuple[str, bytes, ManifestEntry]] = []
    total_bytes = external_bytes
    for reference in references:
        if len(selected) + len(resolved_inputs) >= MAX_WORKSPACE_FILES:
            if requested is not None:
                raise NotebookLimitExceeded(
                    f"Workspace has more than {MAX_WORKSPACE_FILES} files",
                    backend="notebook",
                    phase="input-validation",
                )
            break
        try:
            resolved = await resolve_query_result(
                reference,
                store=query_result_store,
                context=query_result_context_from_run(ctx),
            )
        except QueryResultUnavailable as exc:
            raise ValueError(
                f"Complete SQL result is unavailable: {reference.file}"
            ) from exc
        _validate_file_size(reference.file, resolved.data)
        if total_bytes + len(resolved.data) > MAX_WORKSPACE_TOTAL_BYTES:
            if requested is not None:
                raise NotebookLimitExceeded(
                    f"Workspace exceeds {MAX_WORKSPACE_TOTAL_BYTES} total bytes",
                    backend="notebook",
                    phase="input-validation",
                )
            break
        selected.append(
            (
                reference.file,
                resolved.data,
                ManifestEntry(file=reference.file, sql=reference.query),
            )
        )
        total_bytes += len(resolved.data)

    if references and not selected and not resolved_inputs:
        raise ValueError(
            "No complete execute_sql results fit within the notebook workspace limits"
        )

    files = tuple(NotebookInput(key, data) for key, data, _ in selected) + tuple(
        NotebookInput(item.name, item.data) for item in resolved_inputs
    )
    manifest = tuple(entry for _, _, entry in selected) + tuple(
        ManifestEntry(
            item.name,
            media_type=item.media_type,
            provenance=item.provenance,
        )
        for item in resolved_inputs
    )
    _validate_workspace(files)
    workspace = Workspace(files=files, manifest=manifest)
    manifest_size = len(workspace_manifest_bytes(workspace))
    if manifest_size > MAX_WORKSPACE_MANIFEST_BYTES:
        raise NotebookLimitExceeded(
            f"Workspace manifest exceeds {MAX_WORKSPACE_MANIFEST_BYTES} bytes",
            backend="notebook",
            phase="input-validation",
        )
    return workspace


async def _resolve_workspace_inputs(
    ctx: RunContext,
    attachment_refs: list[str] | None,
    *,
    resolver: WorkspaceInputResolver | None,
) -> tuple[WorkspaceFile, ...]:
    refs = _normalize_attachment_refs(attachment_refs)
    if refs is None:
        return ()
    if resolver is None:
        raise ValueError("No workspace input resolver is configured")

    metadata = getattr(ctx, "metadata", None)
    context = WorkspaceResolutionContext(
        run_id=getattr(ctx, "run_id", None),
        conversation_id=getattr(ctx, "conversation_id", None),
        tool_call_id=getattr(ctx, "tool_call_id", None),
        metadata=metadata if isinstance(metadata, Mapping) else {},
    )
    try:
        resolved = await resolver.resolve(refs, context=context)
        if not isinstance(resolved, Sequence) or isinstance(resolved, (str, bytes)):
            raise WorkspaceInputUnavailable(
                "Workspace input resolver returned an invalid result"
            )
        supplied_files = tuple(resolved)
    except WorkspaceInputUnavailable:
        raise
    except Exception as exc:
        raise WorkspaceInputUnavailable(
            "Attachment inputs could not be resolved"
        ) from exc
    if not supplied_files:
        raise WorkspaceInputUnavailable(
            "No attachment inputs were resolved for the requested references"
        )
    files: list[WorkspaceFile] = []
    for item in supplied_files:
        if not isinstance(item, WorkspaceInputFile):
            raise WorkspaceInputUnavailable(
                "Workspace input resolver returned an invalid file"
            )
        try:
            workspace_file = WorkspaceFile(
                item.name,
                item.data,
                media_type=item.media_type,
                provenance=item.provenance,
            )
        except Exception as exc:
            raise WorkspaceInputUnavailable(
                "Workspace input resolver returned an invalid file"
            ) from exc
        _validate_workspace_file(workspace_file)
        files.append(workspace_file)
    _validate_workspace(tuple(NotebookInput(item.name, item.data) for item in files))
    return tuple(files)


def _normalize_attachment_refs(refs: list[str] | None) -> list[str] | None:
    if refs is None:
        return None
    normalized: list[str] = []
    seen: set[str] = set()
    for ref in refs:
        if (
            not isinstance(ref, str)
            or not ref
            or len(ref) > 2_000
            or any(ord(char) < 32 for char in ref)
        ):
            raise ValueError("Invalid attachment reference")
        if ref in seen:
            raise ValueError("Duplicate attachment reference")
        normalized.append(ref)
        seen.add(ref)
    if not normalized:
        raise ValueError("attachment_refs must contain at least one reference")
    if len(normalized) > MAX_WORKSPACE_FILES:
        raise NotebookLimitExceeded(
            f"Too many attachment references; maximum is {MAX_WORKSPACE_FILES}",
            backend="notebook",
            phase="input-validation",
        )
    return normalized


def _normalize_requested_files(files: list[str] | None) -> list[str] | None:
    if files is None:
        return None
    normalized: list[str] = []
    seen: set[str] = set()
    for key in files:
        if not isinstance(key, str) or not _RESULT_FILE_PATTERN.fullmatch(key):
            raise ValueError(f"Invalid SQL result file key: {key!r}")
        if key not in seen:
            normalized.append(key)
            seen.add(key)
    if not normalized:
        raise ValueError("files must contain at least one SQL result key")
    return normalized


def _validate_workspace(files: tuple[NotebookInput, ...]) -> None:
    if len(files) > MAX_WORKSPACE_FILES:
        raise NotebookLimitExceeded(
            f"Workspace has {len(files)} files; maximum is {MAX_WORKSPACE_FILES}",
            backend="notebook",
            phase="input-validation",
        )
    total = 0
    names: set[str] = set()
    for item in files:
        validate_input_name(item.name, backend="notebook")
        if item.name == "manifest.json":
            raise NotebookLimitExceeded(
                "Workspace filename 'manifest.json' is reserved",
                backend="notebook",
                phase="input-validation",
            )
        if item.name in names:
            raise NotebookLimitExceeded(
                f"Duplicate workspace filename: {item.name}",
                backend="notebook",
                phase="input-validation",
            )
        names.add(item.name)
        _validate_file_size(item.name, item.data)
        total += len(item.data)
    if total > MAX_WORKSPACE_TOTAL_BYTES:
        raise NotebookLimitExceeded(
            f"Workspace exceeds {MAX_WORKSPACE_TOTAL_BYTES} total bytes",
            backend="notebook",
            phase="input-validation",
        )


def _validate_workspace_file(item: WorkspaceFile) -> None:
    if item.media_type is not None and (
        not isinstance(item.media_type, str)
        or not item.media_type
        or len(item.media_type) > 255
        or any(ord(char) < 32 for char in item.media_type)
    ):
        raise WorkspaceInputUnavailable(
            f"Invalid media type for workspace file: {item.name}"
        )
    if len(item.provenance) > 32 or any(
        len(key) > 200 or len(value) > 2_000 for key, value in item.provenance.items()
    ):
        raise WorkspaceInputUnavailable(
            f"Invalid provenance for workspace file: {item.name}"
        )


def _validate_file_size(key: str, data: bytes) -> None:
    if len(data) > MAX_WORKSPACE_FILE_BYTES:
        raise NotebookLimitExceeded(
            f"Workspace file exceeds {MAX_WORKSPACE_FILE_BYTES} bytes: {key}",
            backend="notebook",
            phase="input-validation",
        )


def _error_result(
    message: str,
    *,
    backend: str | None = None,
    phase: str | None = None,
) -> str:
    payload: dict[str, object] = {
        "error": limit_output(message or "Notebook analysis failed", 2_000)
    }
    if backend:
        payload["backend"] = backend
    if phase:
        payload["phase"] = phase
    return json_dumps(payload)


def _dedupe_images(images: list[bytes]) -> list[bytes]:
    selected: list[bytes] = []
    hashes: set[str] = set()
    for image in images:
        digest = hashlib.sha256(image).hexdigest()
        if digest in hashes:
            continue
        hashes.add(digest)
        selected.append(image)
    return selected
