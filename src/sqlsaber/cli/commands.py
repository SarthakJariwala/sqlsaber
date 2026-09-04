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
from sqlsaber.cli.update_check import bind_update_notice, schedule_update_check
from sqlsaber.config.logging import get_logger
from sqlsaber.render import blocks as b

DANGEROUS_MODE_SCOPE = (
    "INSERT/UPDATE/DELETE and restricted DDL (CREATE TABLE/VIEW/INDEX, ALTER "
    "TABLE). DROP/TRUNCATE and admin/security operations stay blocked; "
    "UPDATE/DELETE require WHERE."
)

DANGEROUS_MODE_WARNING = f"The assistant can execute {DANGEROUS_MODE_SCOPE}"
DANGEROUS_MODE_HELP = f"Allow {DANGEROUS_MODE_SCOPE}"
DATABASE_OPTION_HELP = (
    "Database connection name, file path (CSV/SQLite/DuckDB), or connection "
    "string (postgresql://, mysql://, duckdb://). Repeat -d for multiple saved "
    "names, files, or DSNs. Repeated CSV files merge into one session. Uses "
    "the default if omitted"
)


class CLIError(Exception):
    """Exception raised for CLI errors that should result in exit."""

    def __init__(self, message: str, exit_code: int = 1):
        super().__init__(message)
        self.exit_code = exit_code


async def _create_cli_saber(
    *,
    selected_database: str | list[str] | None,
    thinking: bool | None,
    allow_dangerous: bool,
    system_prompt: str | None,
    thread: str | None,
    log,
):
    """Construct SQLSaber and persistence handles for a CLI session."""
    from sqlsaber.cli.artifacts import cli_artifact_store
    from sqlsaber.cli.query_results import cli_query_result_store
    from sqlsaber import (
        SQLSaber,
        SQLSaberOptions,
        ThreadDatabaseRequiredError,
        ThreadDatabaseUnavailableError,
        ThreadNotFoundError,
        ThreadResumeHistoryError,
        ThreadResumeMetadataError,
    )
    from sqlsaber.database.resolver import DatabaseResolutionError
    from sqlsaber.threads import ThreadStorage
    from sqlsaber.threads.manager import ThreadManager

    storage = ThreadStorage()
    artifact_store = cli_artifact_store()
    query_result_store = cli_query_result_store()
    options = SQLSaberOptions(
        database=selected_database,
        thinking_enabled=thinking,
        allow_dangerous=allow_dangerous,
        system_prompt=system_prompt,
        thread_manager=(ThreadManager(storage=storage) if thread is None else None),
        artifact_store=artifact_store,
        query_result_store=query_result_store,
    )
    try:
        saber = (
            await SQLSaber.resume(thread, options=options, storage=storage)
            if thread is not None
            else SQLSaber(options=options)
        )
        info = saber.info
        log.info("db.resolve.success", name=info.primary_database_name)
        log.info("db.connection.created", db_type=info.primary_database_type)
    except ThreadNotFoundError:
        raise CLIError(
            f"Thread not found: {thread}. List threads with: saber threads list"
        ) from None
    except ThreadResumeHistoryError as exc:
        raise CLIError(f"Thread history cannot be resumed: {exc.reason}") from None
    except ThreadDatabaseUnavailableError:
        raise CLIError(
            "The thread database is not configured for automatic continuation. "
            "Retry with: "
            f'saber --thread {thread} --database DATABASE "follow-up question"'
        ) from None
    except ThreadDatabaseRequiredError as exc:
        if exc.reason == "No configured database selector is stored for this thread.":
            raise CLIError(
                "No database is stored with this thread. Retry with: "
                f'saber --thread {thread} --database DATABASE "follow-up question"'
            ) from None
        raise CLIError(
            f"Invalid thread metadata: {exc.reason}. Retry with: "
            f'saber --thread {thread} --database DATABASE "follow-up question"'
        ) from None
    except ThreadResumeMetadataError as exc:
        raise CLIError(
            f"Invalid thread metadata: {exc.reason}. Retry with: "
            f'saber --thread {thread} --database DATABASE "follow-up question"'
        ) from None
    except DatabaseResolutionError as exc:
        log.error("db.resolve.error", error=str(exc))
        raise CLIError(str(exc)) from None
    except (ValueError, OSError) as exc:
        log.exception("db.connection.error", error=str(exc))
        raise CLIError(f"Error creating database connection: {exc}") from None
    return saber, storage, artifact_store, query_result_store


app = cyclopts.App(
    name="sqlsaber",
    help="SQLsaber - Open-source agentic SQL assistant for your database",
    help_epilogue=(
        "Examples:\n\n"
        'saber "show me all users"\n\n'
        'echo "top customers by revenue" | saber\n\n'
        'saber -d sales -d analytics "compare revenue to sessions"\n\n'
        'saber -d users.csv -d orders.csv "join users and orders"\n\n'
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
            help=DATABASE_OPTION_HELP,
        ),
    ] = None,
):
    """
    Query your database using natural language.

    Examples:
        saber                                  # Start interactive mode
        saber "show me all users"              # Run a single query with default database
        saber -d mydb "show me users"          # Run a query with specific database
        saber -d sales -d analytics "compare revenue"  # Multiple saved connections
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
            help=DATABASE_OPTION_HELP,
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
        saber -d sales -d analytics "compare revenue"  # Multiple saved connections
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

        actual_query = query_text
        if query_text is None and not sys.stdin.isatty():
            actual_query = sys.stdin.read().strip()
            if not actual_query:
                actual_query = None

        if thread is not None and actual_query is None:
            raise CLIError(
                "A query is required with --thread. Example: "
                f'saber --thread {thread} "follow-up question"',
                exit_code=2,
            )

        if thread is None and needs_onboarding(selected_database):
            log.debug("cli.onboarding.start")
            onboarding_success = await run_onboarding()
            if not onboarding_success:
                raise CLIError(
                    "Setup incomplete. Please configure your database and try again."
                )
            log.info("cli.onboarding.complete", success=True)

        from sqlsaber.cli.retention import run_cli_retention
        from sqlsaber.render import cli_out

        saber = None
        storage = None
        artifact_store = None
        query_result_store = None
        shell = None
        try:
            if actual_query:
                bind_update_notice(out)
                from sqlsaber.cli.stream_presenter import AgentStreamPresenter
                from sqlsaber.cli.usage import UsageMeter, session_summary_blocks
                from sqlsaber.render.terminal import TerminalSurface

                (
                    saber,
                    storage,
                    artifact_store,
                    query_result_store,
                ) = await _create_cli_saber(
                    selected_database=selected_database,
                    thinking=thinking,
                    allow_dangerous=allow_dangerous,
                    system_prompt=system_prompt,
                    thread=thread,
                    log=log,
                )
                surface = cli_out()
                streaming_handler = AgentStreamPresenter(
                    surface,
                    display_registry=saber.display_registry,
                    query_result_store=saber.query_result_store,
                )
                info = saber.info
                db_name = info.primary_database_name
                db_type = info.primary_database_type
                out(
                    b.key_values(
                        {
                            "Connected to": f"{db_name} ({db_type})",
                            "Model": info.model_name,
                        }
                    )
                )
                if allow_dangerous:
                    out(b.warn(DANGEROUS_MODE_WARNING, label="DANGEROUS MODE ENABLED"))
                log.info("query.execute.start", db_name=db_name, db_type=db_type)
                meter = UsageMeter(model_id=lambda: info.model_id or info.model_name)
                await streaming_handler.execute_streaming_query(
                    actual_query,
                    run_query=meter.metered(saber.query),
                )

                if isinstance(surface, TerminalSurface):
                    if summary := session_summary_blocks(meter.session):
                        surface.emit(*summary)

                thread_id = saber.info.thread_id
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
                from sqlsaber.cli.interactive import InteractiveSession

                if allow_dangerous:
                    out(b.warn(DANGEROUS_MODE_WARNING, label="DANGEROUS MODE ENABLED"))
                shell = InteractiveSession.start_unbound_shell(
                    database=selected_database,
                    allow_dangerous=allow_dangerous,
                )
                (
                    saber,
                    storage,
                    artifact_store,
                    query_result_store,
                ) = await _create_cli_saber(
                    selected_database=selected_database,
                    thinking=thinking,
                    allow_dangerous=allow_dangerous,
                    system_prompt=system_prompt,
                    thread=thread,
                    log=log,
                )
                interactive_session = InteractiveSession(saber)
                try:
                    await interactive_session.run(shell=shell)
                finally:
                    saber = getattr(interactive_session, "saber", saber)
        except BaseException:
            if shell is not None:
                shell.stop()
            raise
        finally:
            if saber is not None:
                try:
                    await saber.close()
                finally:
                    if storage is not None:
                        await run_cli_retention(
                            storage, artifact_store, query_result_store
                        )
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
