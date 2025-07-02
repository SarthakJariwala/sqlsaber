"""FastMCP server implementation for SQLSaber."""

import json

from fastmcp import FastMCP

from sqlsaber.agents.mcp import MCPSQLAgent
from sqlsaber.config.database import DatabaseConfigManager
from sqlsaber.database.connection import DatabaseConnection

INSTRUCTIONS = """
This server provides helpful resources and tools that will help you address users queries on their database.

- Get all databases using `get_databases()`
- Call `list_tables()` to get a list of all tables in the database with row counts. Use this first to discover available tables.
- Call `introspect_schema()` to introspect database schema to understand table structures.
- Call `execute_sql()` to execute SQL queries against the database and retrieve results.

Guidelines:
- Use list_tables first, then introspect_schema for specific tables only
- Use table patterns like 'sample%' or '%experiment%' to filter related tables
- Use proper JOIN syntax and avoid cartesian products
- Include appropriate WHERE clauses to limit results
- Handle errors gracefully and suggest fixes
"""

# Create the FastMCP server instance
mcp = FastMCP(name="SQL Assistant", instructions=INSTRUCTIONS)

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


@mcp.tool
async def list_tables(database: str) -> str:
    """
    Get a list of all tables in the database with row counts. Use this first to discover available tables.
    """
    try:
        agent = await _create_agent_for_database(database)
        if not agent:
            return json.dumps(
                {"error": f"Database '{database}' not found or could not connect"}
            )

        result = await agent.list_tables()
        await agent.db.close()
        return result

    except Exception as e:
        return json.dumps({"error": f"Error listing tables: {str(e)}"})


@mcp.tool
async def introspect_schema(database: str, table_pattern: str | None = None) -> str:
    """
    Introspect database schema to understand table structures. Use optional pattern to filter tables (e.g., 'public.users', 'user%', '%order%').
    """
    try:
        agent = await _create_agent_for_database(database)
        if not agent:
            return json.dumps(
                {"error": f"Database '{database}' not found or could not connect"}
            )

        result = await agent.introspect_schema(table_pattern)
        await agent.db.close()
        return result

    except Exception as e:
        return json.dumps({"error": f"Error introspecting schema: {str(e)}"})


@mcp.tool
async def execute_sql(database: str, query: str, limit: int | None = 100) -> str:
    """Execute a SQL query against the specified database."""
    try:
        agent = await _create_agent_for_database(database)
        if not agent:
            return json.dumps(
                {"error": f"Database '{database}' not found or could not connect"}
            )

        result = await agent.execute_sql(query, limit)
        await agent.db.close()
        return result

    except Exception as e:
        return json.dumps({"error": f"Error executing SQL: {str(e)}"})


def main():
    """Entry point for the MCP server console script."""
    mcp.run()


if __name__ == "__main__":
    main()
