"""``saber rpc`` — headless JSONL mode over stdin/stdout."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from typing import Annotated, Any

import cyclopts

from sqlsaber.cli.commands import (
    DANGEROUS_MODE_HELP,
    DATABASE_OPTION_HELP,
)

rpc_app = cyclopts.App(
    name="rpc",
    help="Headless JSONL mode over stdin/stdout for embedding SQLsaber",
    help_epilogue=(
        "Examples:\n\n"
        "saber rpc -d analytics\n\n"
        "saber rpc -d sales -d analytics --thinking\n\n"
        "saber rpc --thread THREAD_ID\n\n"
        "saber rpc -d ./orders.csv --no-thread\n\n"
        'echo \'{"type":"prompt","message":"how many users?"}\' | saber rpc'
    ),
)


@rpc_app.default
def rpc(
    database: Annotated[
        list[str] | None,
        cyclopts.Parameter(["--database", "-d"], help=DATABASE_OPTION_HELP),
    ] = None,
    thread: Annotated[
        str | None,
        cyclopts.Parameter(["--thread"], help="Resume a saved thread"),
    ] = None,
    no_thread: Annotated[
        bool,
        cyclopts.Parameter(
            ["--no-thread"],
            negative=(),
            help="Do not persist the conversation",
        ),
    ] = False,
    thinking: Annotated[
        bool | None,
        cyclopts.Parameter(
            ["--thinking", "--no-thinking"],
            help="Enable/disable extended thinking/reasoning mode",
        ),
    ] = None,
    allow_dangerous: Annotated[
        bool,
        cyclopts.Parameter(["--allow-dangerous"], help=DANGEROUS_MODE_HELP),
    ] = False,
    csv_tool_results: Annotated[
        bool,
        cyclopts.Parameter(
            ["--csv-tool-results"],
            help="Opt in to experimental CSV tool results for the model (default: JSON)",
        ),
    ] = False,
    system_prompt: Annotated[
        str | None,
        cyclopts.Parameter(
            ["--system-prompt"],
            help="Custom system prompt text or path to a file (overrides built-in prompt)",
        ),
    ] = None,
) -> None:
    """Speak JSONL on stdin/stdout. See docs/reference/rpc.md for the protocol.

    Exit 0 after shutdown/EOF, 1 on startup failure (one JSON line on stdout),
    2 on usage errors (this parser, stderr).
    """
    from sqlsaber.cli.output import fail_usage

    if thread and no_thread:
        fail_usage("--thread and --no-thread are mutually exclusive")

    write = claim_stdout()
    from sqlsaber.cli.commands import _ensure_logging

    log = _ensure_logging()
    import asyncio

    raise SystemExit(
        asyncio.run(
            _main(
                write=write,
                database=database,
                thread=thread,
                persist_thread=not no_thread,
                thinking=thinking,
                allow_dangerous=allow_dangerous,
                csv_tool_results=csv_tool_results,
                system_prompt=system_prompt,
                log=log,
            )
        )
    )


def claim_stdout() -> Callable[[bytes], None]:
    """Dup fd 1 as the protocol channel and point Python stdout at stderr.

    Returns:
        Write-and-flush callback for complete JSONL records on the dup'd fd.
    """
    from sqlsaber.render import reset_io

    protocol_fd = os.dup(1)
    try:
        os.dup2(sys.stderr.fileno(), 1)
    except OSError:
        pass
    sys.stdout = sys.stderr
    reset_io(stdout=sys.stderr, stderr=sys.stderr, tty=False)

    def write(record: bytes) -> None:
        os.write(protocol_fd, record)

    return write


async def _main(
    *,
    write: Callable[[bytes], None],
    database: list[str] | None,
    thread: str | None,
    persist_thread: bool,
    thinking: bool | None,
    allow_dangerous: bool,
    csv_tool_results: bool,
    system_prompt: str | None,
    log: Any,
) -> int:
    """Build the SQLSaber with CLI option/store wiring, serve, close, retain.

    Args:
        write: Protocol stdout writer.
        database: ``-d`` values, or ``None`` for the configured default.
        thread: Thread id to resume, if any.
        persist_thread: When false, do not attach a ``ThreadManager``.
        thinking: Initial thinking flag.
        allow_dangerous: Dangerous-mode flag.
        csv_tool_results: Model-facing CSV results flag.
        system_prompt: Optional prompt override.
        log: Structured logger.

    Returns:
        Process exit code (0 success, 1 startup failure).
    """
    import getpass
    from typing import TextIO

    from sqlsaber.cli.commands import CLIError, _create_cli_saber
    from sqlsaber.cli.onboarding import needs_onboarding
    from sqlsaber.rpc.protocol import Err, encode
    from sqlsaber.rpc.session import ThreadedLineReader, serve

    def _no_prompt(prompt: str = "Password: ", stream: TextIO | None = None) -> str:
        del prompt, stream
        raise EOFError("API key prompt is not available in RPC mode")

    setattr(getpass, "getpass", _no_prompt)

    if thread is None and needs_onboarding(database):
        write(
            encode(
                Err(
                    "startup",
                    None,
                    "No database connections configured. Use 'sqlsaber db add <name>' to add one.",
                )
            )
        )
        return 1

    import asyncio

    from sqlsaber.cli.retention import run_cli_retention

    saber = None
    storage = None
    artifact_store = None
    query_result_store = None
    try:
        saber, storage, artifact_store, query_result_store = await _create_cli_saber(
            selected_database=database,
            thinking=thinking,
            allow_dangerous=allow_dangerous,
            system_prompt=system_prompt,
            thread=thread,
            csv_tool_results=csv_tool_results,
            persist_thread=persist_thread,
            log=log,
        )
        loop = asyncio.get_running_loop()
        reader = ThreadedLineReader(sys.stdin.buffer, loop)
        await serve(
            saber,
            reader=reader,
            write=write,
            persist_thread=persist_thread,
        )
        return 0
    except CLIError as exc:
        write(encode(Err("startup", None, str(exc))))
        return 1
    except Exception as exc:
        write(encode(Err("startup", None, str(exc))))
        return 1
    finally:
        if saber is not None:
            try:
                await saber.close()
            finally:
                if (
                    storage is not None
                    and artifact_store is not None
                    and query_result_store is not None
                ):
                    await run_cli_retention(
                        storage, artifact_store, query_result_store
                    )
