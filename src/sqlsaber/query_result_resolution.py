"""Resolve complete SQL results from message-scoped, model-facing references."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

from pydantic_ai.messages import ModelMessage

from sqlsaber.query_results import (
    QueryResultContext,
    QueryResultStore,
    QueryResultUnavailable,
    StoredQueryResult,
    build_model_projection,
    descriptor_for_data,
    logical_result_file,
    validate_loaded_query_result,
)
from sqlsaber.utils.json_utils import json_dumps


@dataclass(frozen=True, slots=True)
class QueryResultReference:
    tool_call_id: str
    file: str
    descriptor: StoredQueryResult | None
    query: str | None = None
    legacy_data: bytes | None = None


@dataclass(frozen=True, slots=True)
class ResolvedQueryResult:
    reference: QueryResultReference
    descriptor: StoredQueryResult | None
    data: bytes
    source: Literal["store", "legacy"]

    def payload(self) -> dict[str, Any]:
        try:
            value = json.loads(self.data)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise QueryResultUnavailable() from exc
        if not isinstance(value, dict):
            raise QueryResultUnavailable()
        return value


def query_result_from_metadata(metadata: object) -> StoredQueryResult | None:
    if not isinstance(metadata, dict) or "query_result" not in metadata:
        return None
    values: dict[str, object] = {str(key): value for key, value in metadata.items()}
    return StoredQueryResult.from_dict(values.get("query_result"))


def _args_as_dict(part: object) -> dict[str, Any]:
    method = getattr(part, "args_as_dict", None)
    if callable(method):
        try:
            value = method()
        except (TypeError, ValueError):
            return {}
        return value if isinstance(value, dict) else {}
    args = getattr(part, "args", None)
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            value = json.loads(args)
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}
    return {}


def _payload_as_dict(content: object) -> dict[str, Any] | None:
    if isinstance(content, dict):
        return {str(key): value for key, value in content.items()}
    if not isinstance(content, str):
        return None
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def query_result_references_from_messages(
    messages: Sequence[ModelMessage],
) -> list[QueryResultReference]:
    """Scan execute_sql returns in execution order and pair them to SQL calls."""

    queries: dict[str, str] = {}
    for message in messages:
        for part in getattr(message, "parts", ()):
            if (
                getattr(part, "part_kind", "") not in ("tool-call", "builtin-tool-call")
                or getattr(part, "tool_name", "") != "execute_sql"
            ):
                continue
            call_id = getattr(part, "tool_call_id", None)
            if not isinstance(call_id, str) or not call_id:
                continue
            query = _args_as_dict(part).get("query")
            if isinstance(query, str) and query.strip():
                queries[call_id] = query

    references: list[QueryResultReference] = []
    seen_ids: set[str] = set()
    seen_legacy_calls: set[str] = set()
    for message in messages:
        for part in getattr(message, "parts", ()):
            if (
                getattr(part, "part_kind", "")
                not in ("tool-return", "builtin-tool-return")
                or getattr(part, "tool_name", "") != "execute_sql"
            ):
                continue
            call_id = getattr(part, "tool_call_id", None)
            if not isinstance(call_id, str) or not call_id:
                continue
            metadata = getattr(part, "metadata", None)
            descriptor = query_result_from_metadata(metadata)
            if isinstance(metadata, dict) and "query_result" in metadata:
                # Descriptor-bearing metadata is authoritative. Malformed metadata is
                # deliberately not interpreted as a legacy full result.
                if descriptor is None or descriptor.id in seen_ids:
                    continue
                seen_ids.add(descriptor.id)
                references.append(
                    QueryResultReference(
                        tool_call_id=call_id,
                        file=descriptor.file,
                        descriptor=descriptor,
                        query=queries.get(call_id),
                    )
                )
                continue

            payload = _payload_as_dict(getattr(part, "content", None))
            if (
                payload is None
                or payload.get("success") is not True
                or not isinstance(payload.get("results"), list)
                or call_id in seen_legacy_calls
            ):
                continue
            file = payload.get("file")
            if not isinstance(file, str):
                file = logical_result_file(call_id, f"qr_{'0' * 32}")
            try:
                data = json_dumps(payload, ensure_ascii=False).encode("utf-8")
            except (TypeError, ValueError):
                continue
            seen_legacy_calls.add(call_id)
            references.append(
                QueryResultReference(
                    tool_call_id=call_id,
                    file=file,
                    descriptor=None,
                    query=queries.get(call_id),
                    legacy_data=data,
                )
            )
    return references


def find_query_result_reference(
    messages: Sequence[ModelMessage],
    selector: str,
) -> QueryResultReference | None:
    references = query_result_references_from_messages(messages)
    matches = [
        reference
        for reference in references
        if selector
        in {
            reference.tool_call_id,
            reference.file,
            reference.descriptor.id if reference.descriptor else "",
        }
    ]
    if not matches:
        return None
    identities = {
        reference.descriptor.id
        if reference.descriptor is not None
        else f"legacy:{reference.tool_call_id}"
        for reference in matches
    }
    if len(identities) != 1:
        raise QueryResultUnavailable()
    return matches[-1]


async def resolve_query_result(
    reference: QueryResultReference,
    *,
    store: QueryResultStore,
    context: QueryResultContext,
) -> ResolvedQueryResult:
    """Resolve and verify complete bytes without trusting historical auth context."""

    descriptor = reference.descriptor
    if descriptor is None:
        if reference.legacy_data is None:
            raise QueryResultUnavailable()
        # Re-parse to ensure compatibility data still represents a successful,
        # complete old payload.
        try:
            payload = json.loads(reference.legacy_data)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise QueryResultUnavailable() from exc
        if (
            not isinstance(payload, dict)
            or payload.get("success") is not True
            or not isinstance(payload.get("results"), list)
        ):
            raise QueryResultUnavailable()
        return ResolvedQueryResult(
            reference=reference,
            descriptor=None,
            data=bytes(reference.legacy_data),
            source="legacy",
        )

    try:
        loaded = await store.get(descriptor.id, context=context)
        validate_loaded_query_result(loaded, expected=descriptor)
    except QueryResultUnavailable:
        raise
    except Exception as exc:
        raise QueryResultUnavailable() from exc
    return ResolvedQueryResult(
        reference=reference,
        descriptor=loaded.descriptor,
        data=bytes(loaded.data),
        source="store",
    )


def query_result_context_from_run(ctx: object) -> QueryResultContext:
    metadata = getattr(ctx, "metadata", None)
    return QueryResultContext(
        run_id=getattr(ctx, "run_id", None),
        conversation_id=getattr(ctx, "conversation_id", None),
        tool_call_id=getattr(ctx, "tool_call_id", None),
        metadata=metadata if isinstance(metadata, dict) else {},
    )


async def compact_legacy_query_result_history(
    messages: list[ModelMessage],
) -> list[ModelMessage]:
    """Deterministically compact complete pre-store SQL returns for model requests."""

    compacted: list[ModelMessage] = []
    for message in messages:
        changed = False
        parts: list[Any] = []
        for part in getattr(message, "parts", ()):
            if (
                getattr(part, "part_kind", "") != "tool-return"
                or getattr(part, "tool_name", "") != "execute_sql"
                or (
                    isinstance(getattr(part, "metadata", None), dict)
                    and "query_result" in getattr(part, "metadata")
                )
            ):
                parts.append(part)
                continue
            payload = _payload_as_dict(getattr(part, "content", None))
            rows = payload.get("results") if payload else None
            if (
                payload is None
                or payload.get("success") is not True
                or not isinstance(rows, list)
            ):
                parts.append(part)
                continue
            canonical = json_dumps(payload, ensure_ascii=False).encode("utf-8")
            call_id = str(getattr(part, "tool_call_id", "legacy"))
            digest = hashlib.sha256(call_id.encode() + b"\0" + canonical).hexdigest()
            result_id = f"qr_{digest[:32]}"
            file = payload.get("file")
            if not isinstance(file, str):
                file = logical_result_file(call_id, result_id)
            columns: list[str] = []
            seen: set[str] = set()
            for row in rows:
                keys = row.keys() if isinstance(row, dict) else ("value",)
                for key in keys:
                    name = str(key)
                    if name not in seen:
                        seen.add(name)
                        columns.append(name)
            descriptor = descriptor_for_data(
                canonical,
                result_id=result_id,
                file=file,
                row_count=len(rows),
                columns=tuple(columns),
                created_at=None,
            )
            projection = build_model_projection(payload, descriptor)
            parts.append(
                replace(
                    part,
                    content=json_dumps(
                        projection,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                )
            )
            changed = True
        compacted.append(replace(message, parts=parts) if changed else message)
    return compacted
