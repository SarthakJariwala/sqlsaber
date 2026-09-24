"""Local MCP server command."""

import sys
from typing import Annotated, Literal

import cyclopts

from sqlsaber.cli.commands import DATABASE_OPTION_HELP

mcp_app = cyclopts.App(
    name="mcp",
    help="Expose read-only database tools to MCP clients (no model required)",
    help_epilogue=(
        "Examples:\n\nsaber mcp -d analytics\n\n"
        "saber mcp -d sales -d analytics\n\n"
        "saber mcp -d ./data.db --transport http --port 8000"
    ),
)


@mcp_app.default
def mcp(
    database: Annotated[
        list[str] | None,
        cyclopts.Parameter(["--database", "-d"], help=DATABASE_OPTION_HELP),
    ] = None,
    transport: Annotated[
        Literal["stdio", "http"],
        cyclopts.Parameter(help="stdio or Streamable HTTP (not legacy SSE)"),
    ] = "stdio",
    port: Annotated[int, cyclopts.Parameter(help="Local HTTP port (1-65535)")] = 8000,
) -> None:
    """Serve selected databases. HTTP listens on 127.0.0.1 at /mcp without auth."""
    if not 1 <= port <= 65535:
        print("Error: --port must be between 1 and 65535", file=sys.stderr)
        raise SystemExit(2)

    from sqlsaber.mcp import create_server

    server = create_server(database)
    try:
        if transport == "stdio":
            server.run(transport="stdio", show_banner=False)
        else:
            server.run(
                transport="http",
                host="127.0.0.1",
                port=port,
                path="/mcp",
                show_banner=False,
                host_origin_protection=True,
            )
    except Exception:
        print(
            "Error: MCP server failed. Check the selected database configuration "
            "and HTTP port; use 'saber db list' to see configured databases.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
