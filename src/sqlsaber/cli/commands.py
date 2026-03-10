"""CLI command definitions and handlers."""

# Setup logging early, before any imports that trigger plugin discovery
from sqlsaber.config.logging import setup_logging

setup_logging()

# ruff: noqa: E402

import asyncio
import sys
from typing import Annotated

import cyclopts
from rich.panel import Panel

from sqlsaber.cli.auth import create_auth_app
from sqlsaber.cli.database import create_db_app
from sqlsaber.cli.knowledge import create_knowledge_app
from sqlsaber.cli.memory import create_memory_app
from sqlsaber.cli.models import create_models_app
from sqlsaber.cli.onboarding import needs_onboarding, run_onboarding
from sqlsaber.cli.theme import create_theme_app
from sqlsaber.cli.threads import create_threads_app
from sqlsaber.cli.update_check import schedule_update_check
from sqlsaber.config.logging import get_logger
from sqlsaber.theme.manager import create_console

DANGEROUS_MODE_WARNING = (
    "The assistant can execute INSERT/UPDATE/DELETE and restricted DDL "
    "(CREATE TABLE/VIEW/INDEX, ALTER TABLE)."
)


class CLIError(Exception):
    """Exception raised for CLI errors that should result in exit."""

    def __init__(self, message: str, exit_code: int = 1):
        super().__init__(message)
        self.exit_code = exit_code


app = cyclopts.App(
    name="sqlsaber",
    help="SQLsaber - Open-source agentic SQL assistant for your database",
)

app.command(create_auth_app(), name="auth")
app.command(create_db_app(), name="db")
app.command(create_knowledge_app(), name="knowledge")
app.command(create_memory_app(), name="memory")
app.command(create_models_app(), name="models")
app.command(create_theme_app(), name="theme")
app.command(create_threads_app(), name="threads")

console = create_console()


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
            help="Allow INSERT/UPDATE/DELETE and restricted DDL (CREATE TABLE/VIEW/INDEX, ALTER TABLE). DROP/TRUNCATE and admin/security operations stay blocked; UPDATE/DELETE require WHERE",
        ),
    ] = False,
    system_prompt: Annotated[
        str | None,
        cyclopts.Parameter(
            ["--system-prompt"],
            help="Custom system prompt text or path to a file (overrides built-in prompt)",
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
        echo "show me all users" | saber  # Read query from stdin
    """

    async def run_session():
        schedule_update_check(console)

        log = get_logger(__name__)
        log.info(
            "cli.session.start",
            argv=sys.argv[1:],
            database=database,
            has_query=query_text is not None,
            thinking=thinking,
            allow_dangerous=allow_dangerous,
            system_prompt_provided=system_prompt is not None,
        )
        # Import heavy dependencies only when actually running a query
        # This is only done to speed up startup time
        from sqlsaber.cli.display import DisplayManager
        from sqlsaber.cli.interactive import InteractiveSession
        from sqlsaber.cli.streaming import StreamingQueryHandler
        from sqlsaber.cli.usage import SessionUsage
        from sqlsaber.database.resolver import DatabaseResolutionError
        from sqlsaber.options import SQLSaberOptions
        from sqlsaber.session import SQLSaberSession
        from sqlsaber.threads.manager import ThreadManager

        # Check if query_text is None and stdin has data
        actual_query = query_text
        if query_text is None and not sys.stdin.isatty():
            # Read from stdin
            actual_query = sys.stdin.read().strip()
            if not actual_query:
                # If stdin was empty, fall back to interactive mode
                actual_query = None

        # Check if onboarding is needed (only for interactive mode or when no database is configured)
        if needs_onboarding(database):
            # Run onboarding flow
            log.debug("cli.onboarding.start")
            onboarding_success = await run_onboarding()
            if not onboarding_success:
                # User cancelled or onboarding failed
                raise CLIError(
                    "Setup incomplete. Please configure your database and try again."
                )
            log.info("cli.onboarding.complete", success=True)
        thread_manager = ThreadManager()
        try:
            session = SQLSaberSession(
                SQLSaberOptions(
                    database=database,
                    thinking_enabled=thinking,
                    allow_dangerous=allow_dangerous,
                    system_prompt=system_prompt,
                    thread_manager=thread_manager,
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

        try:
            if actual_query:
                # Single query mode with streaming
                streaming_handler = StreamingQueryHandler(console)
                db_type = session.agent.db_type
                model_name = session.agent.config.model.name
                console.print(
                    f"[primary]Connected to:[/primary] {db_name} ({db_type})\n"
                    f"[primary]Model:[/primary] {model_name}\n"
                )
                if allow_dangerous:
                    console.print(
                        Panel(
                            DANGEROUS_MODE_WARNING,
                            title=":warning: DANGEROUS MODE ENABLED",
                            style="warning",
                        )
                    )
                log.info("query.execute.start", db_name=db_name, db_type=db_type)
                run = await streaming_handler.execute_streaming_query(
                    actual_query,
                    run_query=session.query,
                )

                # Track and display session usage
                if run is not None:
                    session_usage = SessionUsage()
                    final_context = run.response.usage.input_tokens
                    session_usage.add_run(run.usage(), final_context)
                    display = DisplayManager(console)
                    display.show_session_summary(session_usage)

                thread_id = thread_manager.current_thread_id
                if thread_id:
                    console.print(
                        f"[dim]You can continue this thread using:[/dim] saber threads resume {thread_id}"
                    )
                    log.info("thread.save.success", thread_id=thread_id)
            else:
                # Interactive mode
                if allow_dangerous:
                    console.print(
                        Panel(
                            DANGEROUS_MODE_WARNING,
                            title=":warning: DANGEROUS MODE ENABLED",
                            style="warning",
                        )
                    )
                interactive_session = InteractiveSession(
                    console=console, session=session
                )
                await interactive_session.run()

        finally:
            # Clean up
            await session.close()
            log.info("db.connection.closed")
            console.print("\n[success]Goodbye![/success]")

    # Run the async function with proper error handling
    try:
        asyncio.run(run_session())
    except CLIError as e:
        get_logger(__name__).error("cli.error", error=str(e))
        console.print(f"[error]Error:[/error] {e}")
        sys.exit(e.exit_code)


def main():
    """Entry point for the CLI application."""
    get_logger(__name__).info("cli.start")
    app()
