"""Threads CLI: list, show, and resume threads (pydantic-ai message snapshots)."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import cyclopts

from sqlsaber.cli.output import fail, fail_usage, out
from sqlsaber.cli.safety import confirm_action
from sqlsaber.config.logging import get_logger
from sqlsaber.render import blocks as b
from sqlsaber.render import cli_out
from sqlsaber.render.surface import Surface

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage

    from sqlsaber.artifact_resolution import ResolvedArtifactPublication
    from sqlsaber.tools.base import Tool

logger = get_logger(__name__)


threads_app = cyclopts.App(
    name="threads",
    help="Manage SQLsaber threads",
    help_epilogue=(
        "Examples:\n\n"
        "saber threads list\n\n"
        "saber threads show THREAD_ID\n\n"
        "saber threads prune --days 30 --dry-run"
    ),
)


def _human_readable(timestamp: float | None) -> str:
    if not timestamp:
        return "-"
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))


def _render_transcript(
    surface: Surface,
    all_msgs: list[ModelMessage],
    last_n: int | None = None,
    *,
    hydrated_results: dict[str, str] | None = None,
    unavailable_results: set[str] | None = None,
    unavailable_artifacts: set[str] | None = None,
    resolved_artifacts: Mapping[str, ResolvedArtifactPublication] | None = None,
    display_registry: Mapping[str, Tool] | None = None,
) -> None:
    """Render conversation turns from ModelMessage[] as blocks."""
    from sqlsaber.tools.renderer import (
        ToolRenderContext,
        ToolRenderer,
        core_display_registry,
    )

    registry = dict(display_registry or core_display_registry())
    if resolved_artifacts:
        for tool in registry.values():
            tool.set_resolved_artifact_publications(resolved_artifacts)
    renderer = ToolRenderer(registry)
    unavailable = unavailable_artifacts or set()

    user_indices: list[int] = []
    for idx, message in enumerate(all_msgs):
        for part in getattr(message, "parts", []):
            if getattr(part, "part_kind", "") == "user-prompt":
                user_indices.append(idx)
                break

    slices: list[tuple[int, int]] = []
    if user_indices:
        for i, start_idx in enumerate(user_indices):
            end_idx = (
                user_indices[i + 1] if i + 1 < len(user_indices) else len(all_msgs)
            )
            slices.append((start_idx, end_idx))

    if last_n is not None and last_n > 0 and slices:
        slices = slices[-last_n:]

    def _render_user(message: ModelMessage) -> None:
        for part in getattr(message, "parts", []):
            if getattr(part, "part_kind", "") == "user-prompt":
                content = getattr(part, "content", None)
                text: str | None = None
                if isinstance(content, str):
                    text = content
                elif isinstance(content, list):
                    parts: list[str] = []
                    for seg in content:
                        if isinstance(seg, str):
                            parts.append(seg)
                        else:
                            try:
                                parts.append(json.dumps(seg, ensure_ascii=False))
                            except Exception:
                                parts.append(str(seg))
                    text = "\n".join([s for s in parts if s]) or None
                if text:
                    surface.emit(b.panel((b.md(text),), title="User", role="info"))
                    return
        surface.emit(b.panel((b.md("(no content)"),), title="User", role="info"))

    def _render_response(message: ModelMessage) -> None:
        for part in getattr(message, "parts", []):
            kind = getattr(part, "part_kind", "")
            if kind == "text":
                text = getattr(part, "content", "")
                if isinstance(text, str) and text.strip():
                    surface.emit(
                        b.panel((b.md(text),), title="Assistant", role="success")
                    )
            elif kind in ("tool-call", "builtin-tool-call"):
                name = getattr(part, "tool_name", "tool")
                args = getattr(part, "args", None)
                args_dict: dict = {}
                if isinstance(args, dict):
                    args_dict = args
                elif isinstance(args, str):
                    try:
                        parsed = json.loads(args)
                        if isinstance(parsed, dict):
                            args_dict = parsed
                    except Exception:
                        args_dict = {}
                executing = renderer.executing(name, args_dict)
                if executing:
                    surface.emit(*executing)
            elif kind in ("tool-return", "builtin-tool-return"):
                name = getattr(part, "tool_name", "tool")
                tool_call_id = getattr(part, "tool_call_id", None)
                content = getattr(part, "content", None)
                if hydrated_results and tool_call_id in hydrated_results:
                    content = hydrated_results[tool_call_id]
                result_blocks = renderer.result(
                    name,
                    content,
                    context=ToolRenderContext(
                        tool_call_id=tool_call_id,
                        metadata=getattr(part, "metadata", None),
                        replay_messages=all_msgs,
                        unavailable_artifacts=frozenset(unavailable),
                    ),
                )
                if result_blocks:
                    surface.emit(*result_blocks)
                if unavailable_results and tool_call_id in unavailable_results:
                    surface.emit(
                        b.warn("Complete query result unavailable; showing preview.")
                    )

    for start_idx, end_idx in slices or [(0, len(all_msgs))]:
        if start_idx < len(all_msgs):
            _render_user(all_msgs[start_idx])
        for i in range(start_idx + 1, end_idx):
            _render_response(all_msgs[i])


@threads_app.command(
    name="list",
    help_epilogue=(
        "Examples:\n\n"
        "saber threads list\n\n"
        "saber threads list --database analytics --limit 10"
    ),
)
def list_threads(
    database: Annotated[
        str | None,
        cyclopts.Parameter(["--database", "-d"], help="Filter by database name"),
    ] = None,
    limit: Annotated[
        int,
        cyclopts.Parameter(["--limit", "-n"], help="Max threads to return"),
    ] = 50,
):
    """List threads (optionally filtered by database).

    Examples:
        saber threads list
        saber threads list --database analytics --limit 10
    """
    from sqlsaber.threads import ThreadStorage

    logger.info("threads.cli.list.start", database=database, limit=limit)
    store = ThreadStorage()
    threads = asyncio.run(store.list_threads(database_name=database, limit=limit))
    if not threads:
        out(b.md("No threads found."))
        logger.info("threads.cli.list.empty")
        return
    out(
        b.table(
            [
                {
                    "id": t.id,
                    "database": t.database_name or "-",
                    "title": (t.title or "-")[:60],
                    "last_activity": _human_readable(
                        getattr(t, "last_activity_at", None)
                    ),
                    "model": t.model_name or "-",
                }
                for t in threads
            ],
            columns=(
                b.Column("id", "ID", role="info"),
                b.Column("database", "Database", role="accent"),
                b.Column("title", "Title", role="success"),
                b.Column("last_activity", "Last Activity", role="muted"),
                b.Column("model", "Model", role="warning"),
            ),
            caption="Threads",
            max_rows=1000,
        )
    )
    logger.info("threads.cli.list.complete", count=len(threads))


@threads_app.command(help_epilogue="Example:\n\nsaber threads show THREAD_ID")
def show(
    thread_id: Annotated[str, cyclopts.Parameter(help="Thread ID")],
):
    """Show thread metadata and render the full transcript.

    Example:
        saber threads show THREAD_ID
    """
    from sqlsaber.threads import ThreadStorage

    logger.info("threads.cli.show.start", thread_id=thread_id)
    store = ThreadStorage()
    thread = asyncio.run(store.get_thread(thread_id))
    if not thread:
        logger.error("threads.cli.show.not_found", thread_id=thread_id)
        fail(f"thread not found: {thread_id}\n  List threads with: saber threads list")
    msgs = asyncio.run(store.get_thread_messages(thread_id))
    from sqlsaber.cli.query_results import (
        cli_query_result_store,
        hydrate_query_result_contents,
    )

    from sqlsaber.query_result_resolution import (
        query_result_references_from_messages,
    )

    if any(
        reference.descriptor is not None
        for reference in query_result_references_from_messages(msgs)
    ):
        hydrated, unavailable = asyncio.run(
            hydrate_query_result_contents(msgs, store=cli_query_result_store())
        )
    else:
        hydrated, unavailable = {}, set()

    from sqlsaber.artifact_resolution import artifact_references_from_messages

    if artifact_references_from_messages(msgs):
        from sqlsaber.cli.artifacts import (
            cli_artifact_store,
            resolve_cli_artifact_publications,
        )

        resolved_artifacts = asyncio.run(
            resolve_cli_artifact_publications(msgs, store=cli_artifact_store())
        )
        unavailable_artifacts = {
            artifact.id
            for publication in resolved_artifacts.values()
            for artifact in publication.unavailable
        }
    else:
        resolved_artifacts = {}
        unavailable_artifacts = set()
    out(
        b.key_values(
            {
                "Thread": thread.id,
                "Database": thread.database_name,
                "Title": thread.title,
                "Last activity": _human_readable(thread.last_activity_at),
                "Model": thread.model_name,
            }
        )
    )

    if hydrated or unavailable or unavailable_artifacts or resolved_artifacts:
        _render_transcript(
            cli_out(),
            msgs,
            None,
            hydrated_results=hydrated,
            unavailable_results=unavailable,
            unavailable_artifacts=unavailable_artifacts,
            resolved_artifacts=resolved_artifacts,
        )
    else:
        _render_transcript(cli_out(), msgs, None)
    logger.info("threads.cli.show.complete", thread_id=thread_id)


@threads_app.command(
    name="artifacts",
    help_epilogue="Example:\n\nsaber threads artifacts THREAD_ID",
)
def list_artifacts(
    thread_id: Annotated[str, cyclopts.Parameter(help="Thread ID")],
):
    """List durable artifacts referenced by a retained thread.

    Example:
        saber threads artifacts THREAD_ID
    """

    from sqlsaber.threads import ThreadStorage

    store = ThreadStorage()

    async def _run() -> None:
        from sqlsaber.artifact_resolution import (
            artifact_references_from_messages,
            resolve_artifact_publication,
        )
        from sqlsaber.artifacts import ArtifactContext
        from sqlsaber.cli.artifacts import cli_artifact_store

        thread = await store.get_thread(thread_id)
        if thread is None:
            fail(
                f"thread not found: {thread_id}\n  List threads with: saber threads list"
            )
        messages = await store.get_thread_messages(thread_id)
        references = artifact_references_from_messages(messages)
        if not references:
            out(b.md("No artifacts found."))
            return

        artifact_store = cli_artifact_store()
        rows: list[tuple[str, str, str, str, str, str]] = []
        for reference in references:
            resolved = await resolve_artifact_publication(
                reference,
                store=artifact_store,
                context=ArtifactContext(),
            )
            available_ids = {loaded.descriptor.id for loaded in resolved.artifacts}
            for descriptor in reference.artifacts:
                rows.append(
                    (
                        reference.publication_id,
                        reference.publication_kind,
                        descriptor.kind,
                        descriptor.name,
                        str(descriptor.size),
                        descriptor.uri
                        if descriptor.id in available_ids
                        else f"{descriptor.uri} (unavailable)",
                    )
                )

        from sqlsaber.render.terminal import PlainSurface

        if isinstance(cli_out(), PlainSurface):
            for row in rows:
                out(b.md("\t".join(row)))
            return
        out(
            b.table(
                [
                    {
                        "publication": row[0],
                        "publication_kind": row[1],
                        "kind": row[2],
                        "name": row[3],
                        "size": row[4],
                        "uri": row[5],
                    }
                    for row in rows
                ],
                columns=(
                    b.Column("publication", "Publication"),
                    b.Column("publication_kind", "Publication kind"),
                    b.Column("kind", "Kind"),
                    b.Column("name", "Name"),
                    b.Column("size", "Size"),
                    b.Column("uri", "URI"),
                ),
                caption=f"Artifacts for {thread_id}",
                max_rows=1000,
            )
        )

    asyncio.run(_run())


@threads_app.command(
    help_epilogue=(
        "Examples:\n\n"
        "saber threads resume THREAD_ID\n\n"
        "saber threads resume THREAD_ID --database analytics"
    )
)
def resume(
    thread_id: Annotated[str, cyclopts.Parameter(help="Thread ID to resume")],
    database: Annotated[
        list[str] | None,
        cyclopts.Parameter(
            ["--database", "-d"],
            help="Database name, DSN override, or one/more CSV files via repeated -d",
        ),
    ] = None,
):
    """Render transcript, then resume thread in interactive mode.

    Examples:
        saber threads resume THREAD_ID
        saber threads resume THREAD_ID --database analytics
    """
    from sqlsaber.threads import ThreadStorage

    logger.info("threads.cli.resume.start", thread_id=thread_id, database=database)
    store = ThreadStorage()

    async def _run() -> None:
        from sqlsaber import (
            SQLSaber,
            SQLSaberOptions,
            ThreadDatabaseRequiredError,
            ThreadDatabaseUnavailableError,
            ThreadNotFoundError,
            ThreadResumeHistoryError,
            ThreadResumeMetadataError,
        )
        from sqlsaber.cli.artifacts import cli_artifact_store
        from sqlsaber.cli.interactive import InteractiveSession
        from sqlsaber.cli.query_results import cli_query_result_store
        from sqlsaber.cli.retention import run_cli_retention
        from sqlsaber.database.resolver import DatabaseResolutionError

        artifact_store = cli_artifact_store()
        query_result_store = cli_query_result_store()
        try:
            saber = await SQLSaber.resume(
                thread_id,
                options=SQLSaberOptions(
                    database=database,
                    artifact_store=artifact_store,
                    query_result_store=query_result_store,
                ),
                storage=store,
            )
        except ThreadNotFoundError:
            logger.error("threads.cli.resume.not_found", thread_id=thread_id)
            fail(
                f"thread not found: {thread_id}\n  List threads with: saber threads list"
            )
        except ThreadResumeHistoryError as exc:
            logger.error(
                "threads.cli.resume.history_invalid",
                thread_id=thread_id,
                error=exc.reason,
            )
            fail(f"thread history cannot be resumed: {exc.reason}")
        except ThreadDatabaseUnavailableError as exc:
            logger.error(
                "threads.cli.resume.database_not_configured",
                thread_id=thread_id,
                missing=exc.database_names,
            )
            fail(
                "the thread database is not configured for automatic resume.\n"
                f"  Retry with: saber threads resume {thread_id} --database DATABASE"
            )
        except ThreadDatabaseRequiredError as exc:
            if (
                exc.reason
                == "No configured database selector is stored for this thread."
            ):
                logger.error("threads.cli.resume.no_database", thread_id=thread_id)
                fail(
                    "no database is specified or stored with this thread.\n"
                    f"  Retry with: saber threads resume {thread_id} --database DATABASE"
                )
            logger.error(
                "threads.cli.resume.metadata_invalid",
                thread_id=thread_id,
                error=exc.reason,
            )
            fail(
                f"invalid thread metadata: {exc.reason}\n"
                f"  Retry with: saber threads resume {thread_id} --database DATABASE"
            )
        except ThreadResumeMetadataError as exc:
            logger.error(
                "threads.cli.resume.metadata_invalid",
                thread_id=thread_id,
                error=exc.reason,
            )
            fail(
                f"invalid thread metadata: {exc.reason}\n"
                f"  Retry with: saber threads resume {thread_id} --database DATABASE"
            )
        except DatabaseResolutionError as exc:
            logger.error(
                "threads.cli.resume.resolve_failed",
                thread_id=thread_id,
                error=str(exc),
            )
            fail(
                f"Error resolving database: {exc}\n"
                f"  Retry with: saber threads resume {thread_id} --database DATABASE"
            )

        try:
            history = await store.get_thread_messages(thread_id)
            from sqlsaber.render.terminal import TerminalSurface

            if isinstance(cli_out(), TerminalSurface):
                out(b.panel((b.md(f"Thread: {thread_id}"),), role="primary"))
            else:
                out(b.md(f"# Thread: {thread_id}"))
            from sqlsaber.cli.query_results import hydrate_query_result_contents

            from sqlsaber.query_result_resolution import (
                query_result_references_from_messages,
            )

            if any(
                reference.descriptor is not None
                for reference in query_result_references_from_messages(history)
            ):
                hydrated, unavailable = await hydrate_query_result_contents(
                    history, store=saber.query_result_store
                )
            else:
                hydrated, unavailable = {}, set()
            from sqlsaber.cli.artifacts import resolve_cli_artifact_publications

            resolved_artifacts = await resolve_cli_artifact_publications(
                history,
                store=artifact_store,
            )
            artifact_unavailable = {
                artifact.id
                for publication in resolved_artifacts.values()
                for artifact in publication.unavailable
            }
            _render_transcript(
                cli_out(),
                history,
                None,
                hydrated_results=hydrated,
                unavailable_results=unavailable,
                unavailable_artifacts=artifact_unavailable,
                resolved_artifacts=resolved_artifacts,
            )
            interactive_session = InteractiveSession(saber)
            await interactive_session.run()
        finally:
            try:
                await saber.close()
            finally:
                await run_cli_retention(store, artifact_store, query_result_store)
            out(b.success("Goodbye!"))
            logger.info("threads.cli.resume.closed")

    asyncio.run(_run())


@threads_app.command(
    help_epilogue=(
        "Examples:\n\n"
        "saber threads prune --days 30 --dry-run\n\n"
        "saber threads prune --days 30 --yes"
    )
)
def prune(
    days: Annotated[
        int,
        cyclopts.Parameter(
            ["--days", "-n"], help="Delete threads older than this many days"
        ),
    ] = 30,
    dry_run: Annotated[
        bool,
        cyclopts.Parameter(["--dry-run"], help="Show how many threads would be pruned"),
    ] = False,
    yes: Annotated[
        bool,
        cyclopts.Parameter(["--yes"], help="Skip confirmation prompt"),
    ] = False,
):
    """Prune old threads by last activity timestamp.

    Examples:
        saber threads prune --days 30 --dry-run
        saber threads prune --days 30 --yes
    """
    from sqlsaber.threads import ThreadStorage

    logger.info("threads.cli.prune.start", days=days)
    store = ThreadStorage()

    if days < 1:
        fail_usage(
            "--days must be at least 1.\n"
            "  Example: saber threads prune --days 30 --dry-run"
        )

    prunable = asyncio.run(store.count_prunable_threads(older_than_days=days))
    if dry_run:
        out(
            b.md(
                f"Dry run: {prunable} thread(s) older than {days} day(s) would be pruned.",
                role="info",
            )
        )
        logger.info("threads.cli.prune.dry_run", days=days, count=prunable)
        return
    if prunable == 0:
        out(b.success(f"No threads older than {days} day(s) to prune."))
        return
    if not confirm_action(
        yes=yes,
        prompt=f"Prune {prunable} thread(s) older than {days} day(s)?",
        non_interactive_command=f"saber threads prune --days {days} --yes",
    ):
        out(b.warn("Operation cancelled"))
        logger.info("threads.cli.prune.cancelled", days=days)
        return

    async def _run() -> None:
        deleted = await store.prune_threads(older_than_days=days)
        out(b.success(f"Pruned {deleted} thread(s)."))
        from sqlsaber.cli.artifact_gc import collect_cli_artifacts
        from sqlsaber.cli.artifacts import cli_artifact_store
        from sqlsaber.cli.query_result_gc import collect_cli_query_results
        from sqlsaber.cli.query_results import cli_query_result_store

        query_cleanup = await collect_cli_query_results(
            store, cli_query_result_store(), force=True
        )
        artifact_cleanup = await collect_cli_artifacts(
            store, cli_artifact_store(), force=True
        )
        if not query_cleanup.complete or not artifact_cleanup.complete:
            out(
                b.warn(
                    "Thread pruning succeeded, but durable output cleanup was incomplete."
                )
            )
        logger.info(
            "threads.cli.prune.complete",
            deleted=deleted,
            query_results_deleted=query_cleanup.deleted,
            artifacts_deleted=artifact_cleanup.deleted,
        )

    asyncio.run(_run())


@threads_app.command(
    help_epilogue=(
        "Examples:\n\n"
        "saber threads export THREAD_ID\n\n"
        "saber threads export THREAD_ID --output analysis.html"
    )
)
def export(
    thread_id: Annotated[str, cyclopts.Parameter(help="Thread ID")],
    output: Annotated[
        Path | None,
        cyclopts.Parameter(
            ["--output", "-o"],
            help="Output HTML file path (default: ./thread-<id>.html)",
        ),
    ] = None,
):
    """Export a thread transcript as a standalone HTML file.

    Examples:
        saber threads export THREAD_ID
        saber threads export THREAD_ID --output analysis.html
    """
    from sqlsaber.cli.html_export import render_thread_html
    from sqlsaber.threads import ThreadStorage

    logger.info(
        "threads.cli.export.start",
        thread_id=thread_id,
        output=str(output) if output else None,
    )
    store = ThreadStorage()

    async def _run() -> None:
        thread = await store.get_thread(thread_id)
        if not thread:
            logger.error("threads.cli.share.not_found", thread_id=thread_id)
            fail(
                f"thread not found: {thread_id}\n  List threads with: saber threads list"
            )

        messages = await store.get_thread_messages(thread_id)
        from sqlsaber.cli.query_results import (
            cli_query_result_store,
            hydrate_query_result_contents,
        )

        hydrated, unavailable = await hydrate_query_result_contents(
            messages, store=cli_query_result_store()
        )
        html = render_thread_html(
            thread,
            messages,
            hydrated_results=hydrated,
            unavailable_results=unavailable,
        )

        out_path = output or (Path.cwd() / f"thread-{thread.id}.html")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(html, encoding="utf-8")

        out(b.success(f"Wrote thread HTML to: {out_path}"))
        logger.info(
            "threads.cli.export.complete", thread_id=thread_id, output=str(out_path)
        )

    asyncio.run(_run())


def create_threads_app() -> cyclopts.App:
    """Return the threads sub-app (for registration)."""
    return threads_app
