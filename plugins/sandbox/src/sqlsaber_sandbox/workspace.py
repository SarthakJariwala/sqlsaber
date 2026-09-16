"""Build bounded sandbox workspaces from authorized analysis inputs."""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping, Sequence

from sqlsaber.query_result_resolution import (
    find_query_result_reference,
    query_result_context_from_run,
    query_result_references_from_messages,
    resolve_query_result,
)
from sqlsaber.query_results import (
    QUERY_RESULT_MEDIA_TYPE,
    QueryResultStore,
    QueryResultUnavailable,
    valid_query_result_file,
)
from sqlsaber.workspace_inputs import (
    WorkspaceInputFile,
    WorkspaceInputResolver,
    WorkspaceResolutionContext,
)

from .config import WorkspaceLimits
from .result import Workspace, WorkspaceFile

_MAX_ATTACHMENT_REFERENCE_CHARS = 2_000
_MAX_PROVENANCE_ENTRIES = 32
_MAX_PROVENANCE_KEY_CHARS = 200
_MAX_PROVENANCE_VALUE_CHARS = 2_000
_MAX_MEDIA_TYPE_CHARS = 255


def _contains_control_characters(value: str) -> bool:
    return any(unicodedata.category(character) == "Cc" for character in value)


def _normalize_requested_files(files: list[str] | None) -> list[str] | None:
    if files is None:
        return None
    normalized: list[str] = []
    seen: set[str] = set()
    for name in files:
        if not isinstance(name, str) or not valid_query_result_file(name):
            raise ValueError("Invalid SQL result file key")
        if name not in seen:
            normalized.append(name)
            seen.add(name)
    return normalized


def _normalize_attachment_refs(
    refs: list[str] | None,
    *,
    limits: WorkspaceLimits,
) -> list[str]:
    if refs is None:
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for ref in refs:
        if (
            not isinstance(ref, str)
            or not ref
            or len(ref) > _MAX_ATTACHMENT_REFERENCE_CHARS
            or _contains_control_characters(ref)
        ):
            raise ValueError("Invalid attachment reference")
        if ref in seen:
            raise ValueError("Duplicate attachment reference")
        normalized.append(ref)
        seen.add(ref)
    if len(normalized) > limits.max_files:
        raise ValueError("Too many attachment references")
    return normalized


def _validate_resolved_metadata(item: WorkspaceFile) -> None:
    if item.media_type is not None and (
        not item.media_type
        or len(item.media_type) > _MAX_MEDIA_TYPE_CHARS
        or _contains_control_characters(item.media_type)
    ):
        raise ValueError("Invalid workspace input media type")
    if len(item.provenance) > _MAX_PROVENANCE_ENTRIES or any(
        len(key) > _MAX_PROVENANCE_KEY_CHARS
        or len(value) > _MAX_PROVENANCE_VALUE_CHARS
        or _contains_control_characters(key)
        for key, value in item.provenance.items()
    ):
        raise ValueError("Invalid workspace input provenance")


async def _resolve_workspace_inputs(
    ctx: object,
    refs: list[str],
    *,
    resolver: WorkspaceInputResolver | None,
) -> tuple[WorkspaceFile, ...]:
    if not refs:
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
            raise ValueError
        supplied = tuple(resolved)
        if not supplied:
            raise ValueError

        files: list[WorkspaceFile] = []
        for item in supplied:
            if not isinstance(item, WorkspaceInputFile):
                raise ValueError
            workspace_file = WorkspaceFile(
                item.name,
                item.data,
                media_type=item.media_type,
                provenance=item.provenance,
            )
            _validate_resolved_metadata(workspace_file)
            files.append(workspace_file)
        return tuple(files)
    except Exception:
        raise ValueError("Attachment inputs could not be resolved") from None


def _safe_file_label(name: str) -> str:
    if len(name) <= 200:
        return name
    return name[:197] + "..."


async def build_workspace_from_history(
    ctx: object,
    *,
    only: list[str] | None,
    attachment_refs: list[str] | None,
    query_result_store: QueryResultStore,
    workspace_input_resolver: WorkspaceInputResolver | None,
    limits: WorkspaceLimits,
) -> Workspace:
    """Build one bounded workspace from SQL history and authorized references."""

    requested = _normalize_requested_files(only)
    if requested is not None and len(requested) > limits.max_files:
        raise ValueError("Requested SQL results exceed the workspace file limit")

    messages = getattr(ctx, "messages", ())
    if requested is None:
        references = list(reversed(query_result_references_from_messages(messages)))
        references = references[: limits.default_results]
    else:
        references = []
        missing: list[str] = []
        for name in requested:
            try:
                reference = find_query_result_reference(messages, name)
            except QueryResultUnavailable:
                reference = None
            if reference is None:
                missing.append(name)
            else:
                references.append(reference)
        if missing:
            listed = ", ".join(missing[:5])
            if len(missing) > 5:
                listed += ", ..."
            raise ValueError(f"Requested SQL result files were not found: {listed}")

    refs = _normalize_attachment_refs(attachment_refs, limits=limits)
    resolved_inputs = await _resolve_workspace_inputs(
        ctx,
        refs,
        resolver=workspace_input_resolver,
    )
    limits.validate(resolved_inputs)

    if (
        requested is not None
        and len(references) + len(resolved_inputs) > limits.max_files
    ):
        raise ValueError("Workspace exceeds the file limit")

    selected: list[WorkspaceFile] = []
    total_bytes = sum(len(item.data) for item in resolved_inputs)
    for reference in references:
        if len(selected) + len(resolved_inputs) >= limits.max_files:
            if requested is not None:
                raise ValueError("Workspace exceeds the file limit")
            break
        try:
            resolved = await resolve_query_result(
                reference,
                store=query_result_store,
                context=query_result_context_from_run(ctx),
            )
        except QueryResultUnavailable:
            label = _safe_file_label(reference.file)
            raise ValueError(f"SQL result is unavailable: {label}") from None

        data_size = len(resolved.data)
        exceeds_file_limit = data_size > limits.max_file_bytes
        exceeds_total_limit = total_bytes + data_size > limits.max_total_bytes
        if exceeds_file_limit or exceeds_total_limit:
            if requested is None:
                break
            label = _safe_file_label(reference.file)
            if exceeds_file_limit:
                raise ValueError(
                    f"SQL result exceeds {limits.max_file_bytes} bytes: {label}"
                )
            raise ValueError(f"Workspace exceeds {limits.max_total_bytes} total bytes")

        provenance = {"query": reference.query} if reference.query is not None else {}
        media_type = (
            resolved.descriptor.media_type
            if resolved.descriptor is not None
            else QUERY_RESULT_MEDIA_TYPE
        )
        try:
            selected.append(
                WorkspaceFile(
                    reference.file,
                    bytes(resolved.data),
                    media_type=media_type,
                    provenance=provenance,
                )
            )
        except (TypeError, ValueError):
            raise ValueError("SQL result has an invalid workspace filename") from None
        total_bytes += data_size

    workspace = Workspace(tuple(selected) + resolved_inputs)
    limits.validate(workspace.files)
    return workspace
