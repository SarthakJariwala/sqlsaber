"""Threads CLI: list, show, and resume threads (pydantic-ai message snapshots)."""

import asyncio
import json
import time
from typing import Annotated

import cyclopts
from pydantic_ai.messages import ModelMessage
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from sqlsaber.threads import ThreadStorage

# Globals consistent with other CLI modules
console = Console()


threads_app = cyclopts.App(
    name="threads",
    help="Manage SQLsaber threads",
)


def _human_readable(timestamp: float | None) -> str:
    if not timestamp:
        return "-"
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))


def _render_transcript(
    console: Console, all_msgs: list[ModelMessage], last_n: int | None = None
) -> None:
    """Render conversation turns from ModelMessage[] using DisplayManager."""
    # Lazy import to avoid pulling UI helpers at startup
    from sqlsaber.cli.display import DisplayManager

    dm = DisplayManager(console)
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
                            Panel.fit(Markdown(text), title="User", border_style="cyan")
                        )
                    return
        if is_redirected:
            console.print("**User:** (no content)\n")
        else:
            console.print(Panel.fit("(no content)", title="User", border_style="cyan"))

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
                                Markdown(text), title="Assistant", border_style="green"
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
                content = getattr(part, "content", None)
                if isinstance(content, (dict, list)):
                    content_str = json.dumps(content, ensure_ascii=False)
                elif isinstance(content, str):
                    content_str = content
                else:
                    content_str = json.dumps({"return_value": str(content)})
                if name == "list_tables":
                    dm.show_table_list(content_str)
                elif name == "introspect_schema":
                    dm.show_schema_info(content_str)
                elif name == "execute_sql":
                    try:
                        data = json.loads(content_str)
                        if (
                            isinstance(data, dict)
                            and data.get("success")
                            and data.get("results")
                        ):
                            dm.show_query_results(data["results"])  # type: ignore[arg-type]
                        elif isinstance(data, dict) and "error" in data:
                            dm.show_sql_error(
                                data.get("error"), data.get("suggestions")
                            )
                        else:
                            if is_redirected:
                                console.print(
                                    f"**Tool result ({name}):**\n\n{content_str}\n"
                                )
                            else:
                                console.print(
                                    Panel.fit(
                                        content_str,
                                        title=f"Tool result: {name}",
                                        border_style="yellow",
                                    )
                                )
                    except Exception:
                        if is_redirected:
                            console.print(
                                f"**Tool result ({name}):**\n\n{content_str}\n"
                            )
                        else:
                            console.print(
                                Panel.fit(
                                    content_str,
                                    title=f"Tool result: {name}",
                                    border_style="yellow",
                                )
                            )
                else:
                    if is_redirected:
                        console.print(f"**Tool result ({name}):**\n\n{content_str}\n")
                    else:
                        console.print(
                            Panel.fit(
                                content_str,
                                title=f"Tool result: {name}",
                                border_style="yellow",
                            )
                        )
        # Thinking parts omitted

    for start_idx, end_idx in slices or [(0, len(all_msgs))]:
        if start_idx < len(all_msgs):
            _render_user(all_msgs[start_idx])
        for i in range(start_idx + 1, end_idx):
            _render_response(all_msgs[i])
        console.print("")


@threads_app.command(name="list")
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
    """List threads (optionally filtered by database)."""
    store = ThreadStorage()
    threads = asyncio.run(store.list_threads(database_name=database, limit=limit))
    if not threads:
        console.print("No threads found.")
        return
    table = Table(title="Threads")
    table.add_column("ID", style="cyan")
    table.add_column("Database", style="magenta")
    table.add_column("Title", style="green")
    table.add_column("Last Activity", style="dim")
    table.add_column("Model", style="yellow")
    for t in threads:
        table.add_row(
            t.id,
            t.database_name or "-",
            (t.title or "-")[:60],
            _human_readable(getattr(t, "last_activity_at", None)),
            t.model_name or "-",
        )
    console.print(table)


@threads_app.command
def show(
    thread_id: Annotated[str, cyclopts.Parameter(help="Thread ID")],
):
    """Show thread metadata and render the full transcript."""
    store = ThreadStorage()
    thread = asyncio.run(store.get_thread(thread_id))
    if not thread:
        console.print(f"[red]Thread not found:[/red] {thread_id}")
        return
    msgs = asyncio.run(store.get_thread_messages(thread_id))
    console.print(f"[bold]Thread: {thread.id}[/bold]")
    console.print("")
    console.print(f"Database: {thread.database_name}")
    console.print(f"Title: {thread.title}")
    console.print(f"Last activity: {_human_readable(thread.last_activity_at)}")
    console.print(f"Model: {thread.model_name}")
    console.print("")

    _render_transcript(console, msgs, None)


@threads_app.command
def resume(
    thread_id: Annotated[str, cyclopts.Parameter(help="Thread ID to resume")],
    database: Annotated[
        str | None,
        cyclopts.Parameter(["--database", "-d"], help="Database name or DSN override"),
    ] = None,
):
    """Render transcript, then resume thread in interactive mode."""
    store = ThreadStorage()

    async def _run() -> None:
        # Lazy imports to avoid heavy modules at CLI startup
        from sqlsaber.agents import SQLSaberAgent
        from sqlsaber.cli.interactive import InteractiveSession
        from sqlsaber.config.database import DatabaseConfigManager
        from sqlsaber.database import DatabaseConnection
        from sqlsaber.database.resolver import (
            DatabaseResolutionError,
            resolve_database,
        )

        thread = await store.get_thread(thread_id)
        if not thread:
            console.print(f"[red]Thread not found:[/red] {thread_id}")
            return
        db_selector = database or thread.database_name
        if not db_selector:
            console.print(
                "[red]No database specified or stored with this thread.[/red]"
            )
            return
        try:
            config_manager = DatabaseConfigManager()
            resolved = resolve_database(db_selector, config_manager)
            connection_string = resolved.connection_string
            db_name = resolved.name
        except DatabaseResolutionError as e:
            console.print(f"[red]Database resolution error:[/red] {e}")
            return

        db_conn = DatabaseConnection(connection_string)
        try:
            sqlsaber_agent = SQLSaberAgent(db_conn, db_name)
            history = await store.get_thread_messages(thread_id)
            if console.is_terminal:
                console.print(Panel.fit(f"Thread: {thread.id}", border_style="blue"))
            else:
                console.print(f"# Thread: {thread.id}\n")
            _render_transcript(console, history, None)
            session = InteractiveSession(
                console=console,
                sqlsaber_agent=sqlsaber_agent,
                db_conn=db_conn,
                database_name=db_name,
                initial_thread_id=thread_id,
                initial_history=history,
            )
            await session.run()
        finally:
            await db_conn.close()
            console.print("\n[green]Goodbye![/green]")

    asyncio.run(_run())


@threads_app.command
def prune(
    days: Annotated[
        int,
        cyclopts.Parameter(
            ["--days", "-n"], help="Delete threads older than this many days"
        ),
    ] = 30,
):
    """Prune old threads by last activity timestamp."""
    store = ThreadStorage()

    async def _run() -> None:
        deleted = await store.prune_threads(older_than_days=days)
        console.print(f"[green]✓ Pruned {deleted} thread(s).[/green]")

    asyncio.run(_run())


def create_threads_app() -> cyclopts.App:
    """Return the threads sub-app (for registration)."""
    return threads_app
