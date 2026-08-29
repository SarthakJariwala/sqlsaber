"""Threads CLI: list, show, and resume threads (pydantic-ai message snapshots)."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from collections.abc import Mapping
from typing import TYPE_CHECKING, Annotated

import cyclopts
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from sqlsaber.cli.safety import confirm_action
from sqlsaber.config.logging import get_logger
from sqlsaber.theme.manager import create_console, get_theme_manager

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage

    from sqlsaber.artifact_resolution import ResolvedArtifactPublication
    from sqlsaber.tools.base import Tool

# Globals consistent with other CLI modules
console = create_console()
error_console = create_console(stderr=True)
tm = get_theme_manager()
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
    console: Console,
    all_msgs: list[ModelMessage],
    last_n: int | None = None,
    *,
    hydrated_results: dict[str, str] | None = None,
    unavailable_results: set[str] | None = None,
    unavailable_artifacts: set[str] | None = None,
    resolved_artifacts: Mapping[str, ResolvedArtifactPublication] | None = None,
    display_registry: Mapping[str, Tool] | None = None,
) -> None:
    """Render conversation turns from ModelMessage[] using DisplayManager."""
    # Lazy import to avoid pulling UI helpers at startup
    from sqlsaber.cli.display import DisplayManager

    dm = DisplayManager(console, display_registry)
    dm.set_replay_messages(all_msgs)
    dm.set_unavailable_artifacts(unavailable_artifacts or set())
    dm.set_resolved_artifact_publications(resolved_artifacts or {})
    # Check if output is being redirected (for clean markdown export)
    is_redirected = not console.is_terminal

    # Locate indices of user prompts
    user_indices: list[int] = []
    for idx, message in enumerate(all_msgs):
        for part in getattr(message, "parts", []):
            if getattr(part, "part_kind", "") == "user-prompt":
                user_indices.append(idx)
                break

    # Build turn slices as (start_idx, end_idx)
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
                elif isinstance(content, list):  # multimodal
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
                    if is_redirected:
                        console.print(f"**User:**\n\n{text}\n")
                    else:
                        console.print(
                            Panel.fit(
                                Markdown(text, code_theme=tm.pygments_style_name),
                                title="User",
                                border_style=tm.style("panel.border.user"),
                            )
                        )
                    return
        if is_redirected:
            console.print("**User:** (no content)\n")
        else:
            console.print(
                Panel.fit(
                    "(no content)",
                    title="User",
                    border_style=tm.style("panel.border.user"),
                )
            )

    def _render_response(message: ModelMessage) -> None:
        for part in getattr(message, "parts", []):
            kind = getattr(part, "part_kind", "")
            if kind == "text":
                text = getattr(part, "content", "")
                if isinstance(text, str) and text.strip():
                    if is_redirected:
                        console.print(f"**Assistant:**\n\n{text}\n")
                    else:
                        console.print(
                            Panel.fit(
                                Markdown(text, code_theme=tm.pygments_style_name),
                                title="Assistant",
                                border_style=tm.style("panel.border.assistant"),
                            )
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
                dm.show_tool_executing(name, args_dict)
            elif kind in ("tool-return", "builtin-tool-return"):
                name = getattr(part, "tool_name", "tool")
                tool_call_id = getattr(part, "tool_call_id", None)
                content = getattr(part, "content", None)
                if hydrated_results and tool_call_id in hydrated_results:
                    content = hydrated_results[tool_call_id]
                dm.show_tool_result(
                    name,
                    content,
                    tool_call_id=tool_call_id,
                    metadata=getattr(part, "metadata", None),
                )
                if unavailable_results and tool_call_id in unavailable_results:
                    console.print(
                        "[warning]Complete query result unavailable; "
                        "showing preview.[/warning]"
                    )
        # Thinking parts omitted

    for start_idx, end_idx in slices or [(0, len(all_msgs))]:
        if start_idx < len(all_msgs):
            _render_user(all_msgs[start_idx])
        for i in range(start_idx + 1, end_idx):
            _render_response(all_msgs[i])
        console.print("")


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
        console.print("No threads found.")
        logger.info("threads.cli.list.empty")
        return
    table = Table(title="Threads")
    table.add_column("ID", style=tm.style("info"), no_wrap=True, min_width=36)
    table.add_column("Database", style=tm.style("accent"))
    table.add_column("Title", style=tm.style("success"))
    table.add_column("Last Activity", style=tm.style("muted"))
    table.add_column("Model", style=tm.style("warning"))
    for t in threads:
        table.add_row(
            t.id,
            t.database_name or "-",
            (t.title or "-")[:60],
            _human_readable(getattr(t, "last_activity_at", None)),
            t.model_name or "-",
        )
    console.print(table)
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
        error_console.print(
            f"[error]Error: thread not found: {thread_id}[/error]\n"
            "  List threads with: saber threads list"
        )
        logger.error("threads.cli.show.not_found", thread_id=thread_id)
        raise SystemExit(1)
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
    console.print(f"[bold]Thread: {thread.id}[/bold]")
    console.print("")
    console.print(f"Database: {thread.database_name}")
    console.print(f"Title: {thread.title}")
    console.print(f"Last activity: {_human_readable(thread.last_activity_at)}")
    console.print(f"Model: {thread.model_name}")
    console.print("")

    if hydrated or unavailable or unavailable_artifacts or resolved_artifacts:
        _render_transcript(
            console,
            msgs,
            None,
            hydrated_results=hydrated,
            unavailable_results=unavailable,
            unavailable_artifacts=unavailable_artifacts,
            resolved_artifacts=resolved_artifacts,
        )
    else:
        _render_transcript(console, msgs, None)
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
            error_console.print(
                f"[error]Error: thread not found: {thread_id}[/error]\n"
                "  List threads with: saber threads list"
            )
            raise SystemExit(1)
        messages = await store.get_thread_messages(thread_id)
        references = artifact_references_from_messages(messages)
        if not references:
            console.print("No artifacts found.")
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

        if not console.is_terminal:
            for row in rows:
                console.print("\t".join(row))
            return
        table = Table(title=f"Artifacts for {thread_id}")
        for heading in (
            "Publication",
            "Publication kind",
            "Kind",
            "Name",
            "Size",
            "URI",
        ):
            table.add_column(heading)
        for row in rows:
            table.add_row(*row)
        console.print(table)

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
        # Lazy imports to avoid heavy modules at CLI startup
        from sqlsaber.cli.artifacts import cli_artifact_store
        from sqlsaber.cli.interactive import InteractiveSession
        from sqlsaber.cli.query_results import cli_query_result_store
        from sqlsaber.config.database import DatabaseConfigManager
        from sqlsaber.database.resolver import DatabaseResolutionError
        from sqlsaber.options import SQLSaberOptions
        from sqlsaber.session import SQLSaberSession
        from sqlsaber.threads.manager import ThreadManager
        from sqlsaber.threads.metadata import resolve_thread_database_selector

        thread = await store.get_thread(thread_id)
        if not thread:
            error_console.print(
                f"[error]Error: thread not found: {thread_id}[/error]\n"
                "  List threads with: saber threads list"
            )
            logger.error("threads.cli.resume.not_found", thread_id=thread_id)
            raise SystemExit(1)
        if database is not None:
            db_selector = database
        else:
            try:
                db_selector = resolve_thread_database_selector(
                    database_name=thread.database_name,
                    extra_metadata=thread.extra_metadata,
                )
            except ValueError as e:
                error_console.print(
                    f"[error]Error: invalid thread metadata: {e}[/error]\n"
                    f"  Retry with: saber threads resume {thread_id} --database DATABASE"
                )
                logger.error(
                    "threads.cli.resume.metadata_invalid",
                    thread_id=thread_id,
                    error=str(e),
                )
                raise SystemExit(1) from None
        if not db_selector:
            error_console.print(
                "[error]Error: no database is specified or stored with this thread.[/error]\n"
                f"  Retry with: saber threads resume {thread_id} --database DATABASE"
            )
            logger.error("threads.cli.resume.no_database", thread_id=thread_id)
            raise SystemExit(1)
        if database is None:
            config_mgr = DatabaseConfigManager()
            selectors = [db_selector] if isinstance(db_selector, str) else db_selector
            missing = [
                selector
                for selector in selectors
                if config_mgr.get_database(selector) is None
            ]
            if missing:
                error_console.print(
                    "[error]Error: the thread database is not configured for automatic "
                    "resume.[/error]\n"
                    f"  Retry with: saber threads resume {thread_id} --database DATABASE"
                )
                logger.error(
                    "threads.cli.resume.database_not_configured",
                    thread_id=thread_id,
                    missing=missing,
                )
                raise SystemExit(1)
        history = await store.get_thread_messages(thread_id)
        session_thread_manager = ThreadManager(
            initial_thread_id=thread_id, storage=store
        )

        sqlsaber_session: SQLSaberSession | None = None
        try:
            sqlsaber_session = SQLSaberSession(
                SQLSaberOptions(
                    database=db_selector,
                    thread_manager=session_thread_manager,
                    artifact_store=cli_artifact_store(),
                    query_result_store=cli_query_result_store(),
                )
            )
        except DatabaseResolutionError as e:
            error_console.print(
                f"[error]Error resolving database: {e}[/error]\n"
                f"  Retry with: saber threads resume {thread_id} --database DATABASE"
            )
            logger.error(
                "threads.cli.resume.resolve_failed", thread_id=thread_id, error=str(e)
            )
            raise SystemExit(1) from None

        try:
            if console.is_terminal:
                console.print(
                    Panel.fit(
                        f"Thread: {thread.id}",
                        border_style=tm.style("panel.border.thread"),
                    )
                )
            else:
                console.print(f"# Thread: {thread.id}\n")
            from sqlsaber.cli.query_results import hydrate_query_result_contents

            from sqlsaber.query_result_resolution import (
                query_result_references_from_messages,
            )

            if any(
                reference.descriptor is not None
                for reference in query_result_references_from_messages(history)
            ):
                result_store = getattr(
                    sqlsaber_session,
                    "query_result_store",
                    cli_query_result_store(),
                )
                hydrated, unavailable = await hydrate_query_result_contents(
                    history, store=result_store
                )
            else:
                hydrated, unavailable = {}, set()
            from sqlsaber.cli.artifacts import resolve_cli_artifact_publications

            artifact_store = getattr(sqlsaber_session, "artifact_store", None)
            resolved_artifacts = (
                await resolve_cli_artifact_publications(
                    history,
                    store=artifact_store,
                )
                if artifact_store is not None
                else {}
            )
            artifact_unavailable = {
                artifact.id
                for publication in resolved_artifacts.values()
                for artifact in publication.unavailable
            }
            _render_transcript(
                console,
                history,
                None,
                hydrated_results=hydrated,
                unavailable_results=unavailable,
                unavailable_artifacts=artifact_unavailable,
                resolved_artifacts=resolved_artifacts,
            )
            interactive_session = InteractiveSession(
                console=console,
                session=sqlsaber_session,
                initial_history=history,
            )
            await interactive_session.run()
        finally:
            await sqlsaber_session.close()
            console.print("\n[success]Goodbye![/success]")
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
        error_console.print(
            "[error]Error: --days must be at least 1.[/error]\n"
            "  Example: saber threads prune --days 30 --dry-run"
        )
        raise SystemExit(2)

    prunable = asyncio.run(store.count_prunable_threads(older_than_days=days))
    if dry_run:
        console.print(
            f"[info]Dry run: {prunable} thread(s) older than {days} day(s) would be pruned.[/info]"
        )
        logger.info("threads.cli.prune.dry_run", days=days, count=prunable)
        return
    if prunable == 0:
        console.print(
            f"[success]No threads older than {days} day(s) to prune.[/success]"
        )
        return
    if not confirm_action(
        yes=yes,
        prompt=f"Prune {prunable} thread(s) older than {days} day(s)?",
        non_interactive_command=f"saber threads prune --days {days} --yes",
    ):
        console.print("[warning]Operation cancelled[/warning]")
        logger.info("threads.cli.prune.cancelled", days=days)
        return

    async def _run() -> None:
        deleted = await store.prune_threads(older_than_days=days)
        console.print(f"[success]✓ Pruned {deleted} thread(s).[/success]")
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
            console.print(
                "[warning]Thread pruning succeeded, but durable output cleanup "
                "was incomplete.[/warning]"
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
            error_console.print(
                f"[error]Error: thread not found: {thread_id}[/error]\n"
                "  List threads with: saber threads list"
            )
            logger.error("threads.cli.share.not_found", thread_id=thread_id)
            raise SystemExit(1)

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

        console.print(f"[success]✓ Wrote thread HTML to:[/success] {out_path}")
        logger.info(
            "threads.cli.export.complete", thread_id=thread_id, output=str(out_path)
        )

    asyncio.run(_run())


def create_threads_app() -> cyclopts.App:
    """Return the threads sub-app (for registration)."""
    return threads_app
