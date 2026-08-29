"""CLI command definitions and handlers."""

from sqlsaber.config.logging import setup_logging

setup_logging()

# ruff: noqa: E402

import asyncio
import sys
from typing import Annotated

import cyclopts

from sqlsaber.cli.auth import create_auth_app
from sqlsaber.cli.database import create_db_app
from sqlsaber.cli.knowledge import create_knowledge_app
from sqlsaber.cli.models import create_models_app
from sqlsaber.cli.onboarding import needs_onboarding, run_onboarding
from sqlsaber.cli.output import fail, fail_usage, out
from sqlsaber.cli.theme import create_theme_app
from sqlsaber.cli.threads import create_threads_app
from sqlsaber.cli.update_check import schedule_update_check
from sqlsaber.config.logging import get_logger
from sqlsaber.render import blocks as b

DANGEROUS_MODE_SCOPE = (
    "INSERT/UPDATE/DELETE and restricted DDL (CREATE TABLE/VIEW/INDEX, ALTER "
    "TABLE). DROP/TRUNCATE and admin/security operations stay blocked; "
    "UPDATE/DELETE require WHERE."
)

DANGEROUS_MODE_WARNING = f"The assistant can execute {DANGEROUS_MODE_SCOPE}"
DANGEROUS_MODE_HELP = f"Allow {DANGEROUS_MODE_SCOPE}"


class CLIError(Exception):
    """Exception raised for CLI errors that should result in exit."""

    def __init__(self, message: str, exit_code: int = 1):
        super().__init__(message)
        self.exit_code = exit_code


app = cyclopts.App(
    name="sqlsaber",
    help="SQLsaber - Open-source agentic SQL assistant for your database",
    help_epilogue=(
        "Examples:\n\n"
        'saber "show me all users"\n\n'
        'echo "top customers by revenue" | saber\n\n'
        'saber --thread THREAD_ID "now compare that with last quarter"'
    ),
)

app.command(create_auth_app(), name="auth")
app.command(create_db_app(), name="db")
app.command(create_knowledge_app(), name="knowledge")
app.command(create_models_app(), name="models")
app.command(create_theme_app(), name="theme")
app.command(create_threads_app(), name="threads")


@app.meta.default
def meta_handler(
    database: Annotated[
        list[str] | None,
        cyclopts.Parameter(
            ["--database", "-d"],
            help="Database connection name, file path (CSV/SQLite/DuckDB), connection string (postgresql://, mysql://, duckdb://), or one/more CSV files via repeated -d (uses default if not specified)",
        ),
    ] = None,
):
    """
    Query your database using natural language.

    Examples:
        saber                                  # Start interactive mode
        saber "show me all users"              # Run a single query with default database
        saber -d mydb "show me users"          # Run a query with specific database
        saber -d data.csv "show me users"      # Run a query with ad-hoc CSV file
        saber -d users.csv -d orders.csv "join users and orders"  # Multiple CSV files (one view per file)
        saber -d data.db "show me users"       # Run a query with ad-hoc SQLite file
        saber -d data.duckdb "show me users"   # Run a query with ad-hoc DuckDB file
        saber -d "postgresql://user:pass@host:5432/db" "show users"  # PostgreSQL connection string
        saber -d "mysql://user:pass@host:3306/db" "show users"       # MySQL connection string
        saber -d "duckdb:///data.duckdb" "show users"                 # DuckDB connection string
        saber --thread THREAD_ID "now compare that with last quarter" # Continue a saved thread
        echo "show me all users" | saber       # Read query from stdin
        cat query.txt | saber                  # Read query from file via stdin
    """


@app.default
def query(
    query_text: Annotated[
        str | None,
        cyclopts.Parameter(
            help="Question in natural language (if not provided, reads from stdin or starts interactive mode)",
        ),
    ] = None,
    database: Annotated[
        list[str] | None,
        cyclopts.Parameter(
            ["--database", "-d"],
            help="Database connection name, file path (CSV/SQLite/DuckDB), connection string (postgresql://, mysql://, duckdb://), or one/more CSV files via repeated -d (uses default if not specified)",
        ),
    ] = None,
    thinking: Annotated[
        bool | None,
        cyclopts.Parameter(
            ["--thinking", "--no-thinking"],
            help="Enable/disable extended thinking/reasoning mode",
        ),
    ] = None,
    allow_dangerous: Annotated[
        bool,
        cyclopts.Parameter(
            ["--allow-dangerous"],
            help=DANGEROUS_MODE_HELP,
        ),
    ] = False,
    system_prompt: Annotated[
        str | None,
        cyclopts.Parameter(
            ["--system-prompt"],
            help="Custom system prompt text or path to a file (overrides built-in prompt)",
        ),
    ] = None,
    thread: Annotated[
        str | None,
        cyclopts.Parameter(
            ["--thread"],
            help="Continue a saved thread non-interactively",
        ),
    ] = None,
):
    """Run a query against the database or start interactive mode.

    When called without arguments:
    - If stdin has data, reads query from stdin
    - Otherwise, starts interactive mode

    When called with a query string, executes that query and exits.

    Examples:
        saber                             # Start interactive mode
        saber "show me all users"         # Run a single query
        saber -d data.csv "show users"    # Run a query with ad-hoc CSV file
        saber -d users.csv -d orders.csv "join users and orders"  # Multiple CSV files (one view per file)
        saber -d data.db "show users"     # Run a query with ad-hoc SQLite file
        saber -d data.duckdb "show users" # Run a query with ad-hoc DuckDB file
        saber -d "postgresql://user:pass@host:5432/db" "show users"  # PostgreSQL connection string
        saber -d "mysql://user:pass@host:3306/db" "show users"       # MySQL connection string
        saber -d "duckdb:///data.duckdb" "show users"                 # DuckDB connection string
        saber --thread THREAD_ID "now compare that with last quarter" # Continue a saved thread
        echo "show me all users" | saber  # Read query from stdin
    """

    async def run_session():
        schedule_update_check()

        selected_database: str | list[str] | None = database

        log = get_logger(__name__)
        log.info(
            "cli.session.start",
            argv=sys.argv[1:],
            database=selected_database,
            has_query=query_text is not None,
            thread_id=thread,
            thinking=thinking,
            allow_dangerous=allow_dangerous,
            system_prompt_provided=system_prompt is not None,
        )
        from sqlsaber.cli.artifacts import cli_artifact_store
        from sqlsaber.cli.interactive import InteractiveSession
        from sqlsaber.cli.query_results import cli_query_result_store
        from sqlsaber.cli.stream_presenter import AgentStreamPresenter
        from sqlsaber.cli.usage import SessionUsage, request_usages_from_run_result
        from sqlsaber.cli.usage import session_summary_blocks
        from sqlsaber.config.database import DatabaseConfigManager
        from sqlsaber.database.resolver import DatabaseResolutionError
        from sqlsaber.options import SQLSaberOptions
        from sqlsaber.render import cli_out
        from sqlsaber.render.terminal import TerminalSurface
        from sqlsaber.session import SQLSaberSession
        from sqlsaber.threads import ThreadStorage
        from sqlsaber.threads.manager import ThreadManager
        from sqlsaber.threads.metadata import resolve_thread_database_selector

        actual_query = query_text
        if query_text is None and not sys.stdin.isatty():
            actual_query = sys.stdin.read().strip()
            if not actual_query:
                actual_query = None

        message_history = None
        thread_manager = ThreadManager()
        if thread is not None:
            if actual_query is None:
                raise CLIError(
                    "A query is required with --thread. Example: "
                    f'saber --thread {thread} "follow-up question"',
                    exit_code=2,
                )
            store = ThreadStorage()
            stored_thread = await store.get_thread(thread)
            if stored_thread is None:
                raise CLIError(
                    f"Thread not found: {thread}. List threads with: saber threads list"
                )
            message_history = await store.get_thread_messages(thread)
            if selected_database is None:
                try:
                    selected_database = resolve_thread_database_selector(
                        database_name=stored_thread.database_name,
                        extra_metadata=stored_thread.extra_metadata,
                    )
                except ValueError as exc:
                    raise CLIError(
                        f"Invalid thread metadata: {exc}. Retry with: "
                        f'saber --thread {thread} --database DATABASE "follow-up question"'
                    ) from None
                if not selected_database:
                    raise CLIError(
                        "No database is stored with this thread. Retry with: "
                        f'saber --thread {thread} --database DATABASE "follow-up question"'
                    )
                config_manager = DatabaseConfigManager()
                selectors = (
                    [selected_database]
                    if isinstance(selected_database, str)
                    else selected_database
                )
                missing = [
                    selector
                    for selector in selectors
                    if config_manager.get_database(selector) is None
                ]
                if missing:
                    raise CLIError(
                        "The thread database is not configured for automatic "
                        "continuation. Retry with: "
                        f'saber --thread {thread} --database DATABASE "follow-up question"'
                    )
            thread_manager = ThreadManager(initial_thread_id=thread, storage=store)

        if needs_onboarding(selected_database):
            log.debug("cli.onboarding.start")
            onboarding_success = await run_onboarding()
            if not onboarding_success:
                raise CLIError(
                    "Setup incomplete. Please configure your database and try again."
                )
            log.info("cli.onboarding.complete", success=True)
        try:
            session = SQLSaberSession(
                SQLSaberOptions(
                    database=selected_database,
                    thinking_enabled=thinking,
                    allow_dangerous=allow_dangerous,
                    system_prompt=system_prompt,
                    thread_manager=thread_manager,
                    artifact_store=cli_artifact_store(),
                    query_result_store=cli_query_result_store(),
                )
            )
            db_name = session.db_name
            log.info("db.resolve.success", name=db_name)
            log.info("db.connection.created", db_type=type(session.connection).__name__)
        except DatabaseResolutionError as e:
            log.error("db.resolve.error", error=str(e))
            raise CLIError(str(e))
        except (ValueError, OSError) as e:
            log.exception("db.connection.error", error=str(e))
            raise CLIError(f"Error creating database connection: {e}")

        surface = cli_out()
        try:
            if actual_query:
                streaming_handler = AgentStreamPresenter(
                    surface,
                    display_registry=session.agent.display_registry,
                    query_result_store=session.query_result_store,
                )
                db_type = session.agent.db_type
                model_name = session.agent.config.model.name
                out(
                    b.key_values(
                        {
                            "Connected to": f"{db_name} ({db_type})",
                            "Model": model_name,
                        }
                    )
                )
                if allow_dangerous:
                    out(
                        b.warn(
                            DANGEROUS_MODE_WARNING, label="DANGEROUS MODE ENABLED"
                        )
                    )
                log.info("query.execute.start", db_name=db_name, db_type=db_type)
                run = await streaming_handler.execute_streaming_query(
                    actual_query,
                    run_query=session.query,
                    message_history=message_history,
                )

                if run is not None:
                    session_usage = SessionUsage()
                    final_context = run.response.usage.input_tokens
                    model_id = getattr(session.agent.agent.model, "model_id", None)
                    session_usage.add_run(
                        run.usage,
                        final_context,
                        model_name=str(model_id) if model_id else model_name,
                        request_usages=request_usages_from_run_result(run),
                    )
                    if isinstance(surface, TerminalSurface):
                        summary = session_summary_blocks(session_usage)
                        if summary:
                            surface.emit(*summary)

                thread_id = thread_manager.current_thread_id
                if thread_id:
                    out(
                        b.md(
                            f'Continue non-interactively: `saber --thread {thread_id} "follow-up question"`\n'
                            f"Continue interactively: `saber threads resume {thread_id}`",
                            role="muted",
                        )
                    )
                    log.info("thread.save.success", thread_id=thread_id)
            else:
                if allow_dangerous:
                    out(
                        b.warn(
                            DANGEROUS_MODE_WARNING, label="DANGEROUS MODE ENABLED"
                        )
                    )
                interactive_session = InteractiveSession(session=session)
                await interactive_session.run()

        finally:
            await session.close()
            log.info("db.connection.closed")
            out(b.success("Goodbye!"))

    try:
        asyncio.run(run_session())
    except CLIError as e:
        get_logger(__name__).error("cli.error", error=str(e))
        if e.exit_code == 2:
            fail_usage(str(e))
        fail(str(e), code=e.exit_code)


def main():
    """Entry point for the CLI application."""
    get_logger(__name__).info("cli.start")
    app()
