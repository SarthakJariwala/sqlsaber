"""FastMCP server implementation for SQLSaber."""

import json

from fastmcp import FastMCP

from sqlsaber.agents.mcp import MCPSQLAgent
from sqlsaber.config.database import DatabaseConfigManager
from sqlsaber.database import DatabaseConnection
from sqlsaber.tools import SQLTool, tool_registry
from sqlsaber.tools.instructions import InstructionBuilder

# Initialize the instruction builder
instruction_builder = InstructionBuilder(tool_registry)

# Generate dynamic instructions
DYNAMIC_INSTRUCTIONS = instruction_builder.build_mcp_instructions()

# Create the FastMCP server instance with dynamic instructions
mcp = FastMCP(name="SQL Assistant", instructions=DYNAMIC_INSTRUCTIONS)

# Initialize the database config manager
config_manager = DatabaseConfigManager()


async def _create_agent_for_database(database_name: str) -> MCPSQLAgent | None:
    """Create a MCPSQLAgent for the specified database."""
    try:
        # Look up configured database connection
        db_config = config_manager.get_database(database_name)
        if not db_config:
            return None
        connection_string = db_config.to_connection_string()

        # Create database connection
        db_conn = DatabaseConnection(connection_string)

        # Create and return the agent
        agent = MCPSQLAgent(db_conn)
        return agent

    except Exception:
        return None


@mcp.tool
def get_databases() -> dict:
    """List all configured databases with their types."""
    databases = []
    for db_config in config_manager.list_databases():
        databases.append(
            {
                "name": db_config.name,
                "type": db_config.type,
                "database": db_config.database,
                "host": db_config.host,
                "port": db_config.port,
                "is_default": db_config.name == config_manager.get_default_name(),
            }
        )

    return {"databases": databases, "count": len(databases)}


async def _execute_with_connection(tool_name: str, database: str, **kwargs) -> str:
    """Execute a SQL tool with database connection management.

    Args:
        tool_name: Name of the tool to execute
        database: Database name to connect to
        **kwargs: Tool-specific parameters

    Returns:
        JSON string with the tool's output
    """
    try:
        agent = await _create_agent_for_database(database)
        if not agent:
            return json.dumps(
                {"error": f"Database '{database}' not found or could not connect"}
            )

        # Get the tool and set up connection
        tool = tool_registry.get_tool(tool_name)
        if isinstance(tool, SQLTool):
            tool.set_connection(agent.db)

        # Execute the tool
        result = await tool.execute(**kwargs)
        await agent.db.close()
        return result

    except Exception as e:
        return json.dumps({"error": f"Error in {tool_name}: {str(e)}"})


# SQL Tool Wrappers with explicit signatures


@mcp.tool
async def list_tables(database: str) -> str:
    """Get a list of all tables in the database with row counts. Use this first to discover available tables."""
    return await _execute_with_connection("list_tables", database)


@mcp.tool
async def introspect_schema(database: str, table_pattern: str = None) -> str:
    """Introspect database schema to understand table structures."""
    kwargs = {}
    if table_pattern is not None:
        kwargs["table_pattern"] = table_pattern
    return await _execute_with_connection("introspect_schema", database, **kwargs)


@mcp.tool
async def execute_sql(database: str, query: str, limit: int = 100) -> str:
    """Execute a SQL query against the database."""
    return await _execute_with_connection(
        "execute_sql", database, query=query, limit=limit
    )


def main():
    """Entry point for the MCP server console script."""
    mcp.run()


if __name__ == "__main__":
    main()
