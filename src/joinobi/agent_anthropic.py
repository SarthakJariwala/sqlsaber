import json
import os
from typing import Any, AsyncIterator, Dict, List, Optional

from anthropic import AsyncAnthropic

from .config import get_api_key
from .database import DatabaseConnection


class StreamEvent:
    """Event emitted during streaming processing."""

    def __init__(self, event_type: str, data: Any = None):
        self.type = (
            event_type  # 'tool_use', 'text', 'query_result', 'error', 'processing'
        )
        self.data = data


class SQLResponse:
    """Response from the SQL agent."""

    def __init__(
        self,
        query: Optional[str] = None,
        explanation: str = "",
        results: Optional[List[Dict[str, Any]]] = None,
        error: Optional[str] = None,
    ):
        self.query = query
        self.explanation = explanation
        self.results = results
        self.error = error


class AnthropicSQLAgent:
    """SQL Agent using Anthropic SDK directly."""

    def __init__(self, db_connection: DatabaseConnection, allow_write: bool = False):
        self.db = db_connection
        self.allow_write = allow_write
        self.client = AsyncAnthropic(api_key=get_api_key())
        self.model = os.getenv("JOINOBI_MODEL", "claude-sonnet-4-20250514").replace(
            "anthropic:", ""
        )
        self.conversation_history: List[Dict[str, Any]] = []

        # Define tools in Anthropic format
        self.tools = [
            {
                "name": "introspect_schema",
                "description": "Introspect database schema to understand table structures.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "table_pattern": {
                            "type": "string",
                            "description": "Optional pattern to filter tables (e.g., 'public.users')",
                        }
                    },
                    "required": [],
                },
            },
            {
                "name": "execute_sql",
                "description": "Execute a SQL query against the database.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "SQL query to execute",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of rows to return (default: 100)",
                            "default": 100,
                        },
                    },
                    "required": ["query"],
                },
            },
        ]

        self.system_prompt = """You are a helpful SQL assistant that helps users query their PostgreSQL database.

Your responsibilities:
1. Understand user's natural language requests and convert them to SQL
2. Use the database schema introspection tool to understand table structures
3. Generate appropriate SQL queries
4. Execute queries safely (only SELECT queries unless explicitly allowed)
5. Format and explain results clearly

Guidelines:
- Always check the schema before writing queries
- Use proper JOIN syntax and avoid cartesian products
- Include appropriate WHERE clauses to limit results
- Explain what the query does in simple terms
- Handle errors gracefully and suggest fixes
- Be security conscious - use parameterized queries when needed
"""

    async def introspect_schema(self, table_pattern: Optional[str] = None) -> str:
        """Introspect database schema to understand table structures."""
        try:
            schema_info = await self.db.get_schema_info()

            # Filter by pattern if provided
            if table_pattern:
                pattern = table_pattern.lower()
                filtered_schema = {
                    k: v
                    for k, v in schema_info.items()
                    if pattern in k.lower() or pattern in v["name"].lower()
                }
                schema_info = filtered_schema

            # Format the schema information
            formatted_info = {}
            for table_name, table_info in schema_info.items():
                formatted_info[table_name] = {
                    "columns": {
                        col_name: {
                            "type": col_info["data_type"],
                            "nullable": col_info["nullable"],
                            "default": col_info["default"],
                        }
                        for col_name, col_info in table_info["columns"].items()
                    },
                    "primary_keys": table_info["primary_keys"],
                    "foreign_keys": [
                        f"{fk['column']} -> {fk['references']['table']}.{fk['references']['column']}"
                        for fk in table_info["foreign_keys"]
                    ],
                }

            return json.dumps(formatted_info)
        except Exception as e:
            return json.dumps({"error": f"Error introspecting schema: {str(e)}"})

    async def execute_sql(
        self, query: str, limit: Optional[int] = 100
    ) -> Dict[str, Any]:
        """Execute a SQL query against the database."""
        try:
            # Security check - only allow SELECT queries unless write is enabled
            query_upper = query.strip().upper()

            # Check for write operations
            write_keywords = [
                "INSERT",
                "UPDATE",
                "DELETE",
                "DROP",
                "CREATE",
                "ALTER",
                "TRUNCATE",
            ]
            is_write_query = any(query_upper.startswith(kw) for kw in write_keywords)

            if is_write_query and not self.allow_write:
                return json.dumps(
                    {
                        "error": "Write operations are not allowed. Only SELECT queries are permitted.",
                        "suggestion": "If you need to perform write operations, please enable write mode.",
                    }
                )

            # Add LIMIT if not present and it's a SELECT query
            if query_upper.startswith("SELECT") and "LIMIT" not in query_upper:
                query = f"{query.rstrip(';')} LIMIT {limit};"

            # Execute the query (wrapped in a transaction for safety)
            results = await self.db.execute_query(query)

            # Format results - but also store the actual data
            self._last_results = results[:limit]
            self._last_query = query

            return json.dumps(
                {
                    "success": True,
                    "row_count": len(results),
                    "results": results[:limit],  # Extra safety for limit
                    "truncated": len(results) > limit,
                }
            )

        except Exception as e:
            error_msg = str(e)

            # Provide helpful error messages
            suggestions = []
            if "column" in error_msg.lower() and "does not exist" in error_msg.lower():
                suggestions.append(
                    "Check column names using the schema introspection tool"
                )
            elif "table" in error_msg.lower() and "does not exist" in error_msg.lower():
                suggestions.append(
                    "Check table names using the schema introspection tool"
                )
            elif "syntax error" in error_msg.lower():
                suggestions.append(
                    "Review SQL syntax, especially JOIN conditions and WHERE clauses"
                )

            return json.dumps({"error": error_msg, "suggestions": suggestions})

    async def process_tool_call(
        self, tool_name: str, tool_input: Dict[str, Any]
    ) -> str:
        """Process a tool call and return the result."""
        if tool_name == "introspect_schema":
            return await self.introspect_schema(tool_input.get("table_pattern"))
        elif tool_name == "execute_sql":
            return await self.execute_sql(
                tool_input["query"], tool_input.get("limit", 100)
            )
        else:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

    async def query_stream(
        self, user_query: str, use_history: bool = True
    ) -> AsyncIterator[StreamEvent]:
        """Process a user query and stream responses."""
        # Initialize for tracking state
        self._last_results = None
        self._last_query = None

        # Build messages with history if requested
        if use_history:
            messages = self.conversation_history + [
                {"role": "user", "content": user_query}
            ]
        else:
            messages = [{"role": "user", "content": user_query}]

        try:
            # Stream the initial message
            stream = await self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=self.system_prompt,
                messages=messages,
                tools=self.tools,
                stream=True,
            )

            collected_content = []
            content_blocks = []
            tool_use_blocks = []

            async for event in stream:
                if event.type == "content_block_start":
                    if hasattr(event.content_block, "type"):
                        if event.content_block.type == "tool_use":
                            yield StreamEvent(
                                "tool_use",
                                {"name": event.content_block.name, "status": "started"},
                            )
                            tool_use_blocks.append(
                                {
                                    "id": event.content_block.id,
                                    "name": event.content_block.name,
                                    "input": {},
                                }
                            )
                        elif event.content_block.type == "text":
                            content_blocks.append({"type": "text", "text": ""})

                elif event.type == "content_block_delta":
                    if hasattr(event.delta, "text"):
                        yield StreamEvent("text", event.delta.text)
                        if content_blocks and content_blocks[-1]["type"] == "text":
                            content_blocks[-1]["text"] += event.delta.text
                    elif hasattr(event.delta, "partial_json"):
                        # Accumulate tool input
                        if tool_use_blocks:
                            import json

                            try:
                                # Parse accumulated JSON
                                current_json = tool_use_blocks[-1].get("_partial", "")
                                current_json += event.delta.partial_json
                                tool_use_blocks[-1]["_partial"] = current_json
                                # Try to parse complete JSON
                                tool_use_blocks[-1]["input"] = json.loads(current_json)
                            except json.JSONDecodeError:
                                # Not complete yet
                                pass

                elif event.type == "message_stop":
                    break

            # Check if we need to use tools
            stop_reason = "stop"
            if tool_use_blocks:
                stop_reason = "tool_use"
                # Convert to proper format
                for block in tool_use_blocks:
                    block["type"] = "tool_use"
                    if "_partial" in block:
                        del block["_partial"]
                content_blocks.extend(tool_use_blocks)

            # Create a response-like object
            class Response:
                def __init__(self, content, stop_reason):
                    self.content = content
                    self.stop_reason = stop_reason

            response = Response(content_blocks, stop_reason)

            # Process tool calls if needed
            while response.stop_reason == "tool_use":
                # Add assistant's response to conversation
                collected_content.append(
                    {"role": "assistant", "content": response.content}
                )

                # Process each tool use
                tool_results = []
                for block in response.content:
                    if block.get("type") == "tool_use":
                        yield StreamEvent(
                            "tool_use",
                            {
                                "name": block["name"],
                                "input": block["input"],
                                "status": "executing",
                            },
                        )

                        tool_result = await self.process_tool_call(
                            block["name"], block["input"]
                        )

                        # If this was a SQL execution, yield the results
                        if block["name"] == "execute_sql" and self._last_results:
                            yield StreamEvent(
                                "query_result",
                                {
                                    "query": self._last_query,
                                    "results": self._last_results,
                                },
                            )

                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block["id"],
                                "content": tool_result,
                            }
                        )

                # Continue conversation with tool results
                collected_content.append({"role": "user", "content": tool_results})

                # Signal that we're processing the tool results
                yield StreamEvent("processing", "Analyzing results...")

                # Stream the next response
                stream = await self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=self.system_prompt,
                    messages=messages + collected_content,
                    tools=self.tools,
                    stream=True,
                )

                # Reset for next iteration
                content_blocks = []
                tool_use_blocks = []

                async for event in stream:
                    if event.type == "content_block_start":
                        if hasattr(event.content_block, "type"):
                            if event.content_block.type == "tool_use":
                                tool_use_blocks.append(
                                    {
                                        "id": event.content_block.id,
                                        "name": event.content_block.name,
                                        "input": {},
                                    }
                                )
                            elif event.content_block.type == "text":
                                content_blocks.append({"type": "text", "text": ""})

                    elif event.type == "content_block_delta":
                        if hasattr(event.delta, "text"):
                            yield StreamEvent("text", event.delta.text)
                            if content_blocks and content_blocks[-1]["type"] == "text":
                                content_blocks[-1]["text"] += event.delta.text
                        elif hasattr(event.delta, "partial_json"):
                            if tool_use_blocks:
                                import json

                                try:
                                    current_json = tool_use_blocks[-1].get(
                                        "_partial", ""
                                    )
                                    current_json += event.delta.partial_json
                                    tool_use_blocks[-1]["_partial"] = current_json
                                    tool_use_blocks[-1]["input"] = json.loads(
                                        current_json
                                    )
                                except json.JSONDecodeError:
                                    pass

                    elif event.type == "message_stop":
                        break

                # Check if we need more tools
                stop_reason = "stop"
                if tool_use_blocks:
                    stop_reason = "tool_use"
                    for block in tool_use_blocks:
                        block["type"] = "tool_use"
                        if "_partial" in block:
                            del block["_partial"]
                    content_blocks.extend(tool_use_blocks)

                response = Response(content_blocks, stop_reason)

            # Update conversation history if using history
            if use_history:
                self.conversation_history.append(
                    {"role": "user", "content": user_query}
                )
                self.conversation_history.extend(collected_content)
                # Add final assistant response
                self.conversation_history.append(
                    {"role": "assistant", "content": response.content}
                )

        except Exception as e:
            yield StreamEvent("error", str(e))

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []

    async def query(self, user_query: str) -> SQLResponse:
        """Process a user query and return the response (legacy non-streaming)."""
        messages = [{"role": "user", "content": user_query}]

        try:
            # Initial message to Claude
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=self.system_prompt,
                messages=messages,
                tools=self.tools,
            )

            # Process tool calls if needed
            while response.stop_reason == "tool_use":
                # Add assistant's response to messages
                messages.append({"role": "assistant", "content": response.content})

                # Process each tool use
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        tool_result = await self.process_tool_call(
                            block.name, block.input
                        )
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": tool_result,
                            }
                        )

                # Send tool results back
                messages.append({"role": "user", "content": tool_results})

                # Get next response
                response = await self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=self.system_prompt,
                    messages=messages,
                    tools=self.tools,
                )

            # Extract final response
            final_text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    final_text += block.text

            # Parse the response to extract query and results
            # The agent should mention the query and results in the text
            # This is a simple approach - in production, you might want a more structured response
            query_executed = None
            results = None
            error = None

            # Try to extract information from the response
            if "```sql" in final_text:
                # Extract SQL query
                sql_start = final_text.find("```sql") + 6
                sql_end = final_text.find("```", sql_start)
                if sql_end > sql_start:
                    query_executed = final_text[sql_start:sql_end].strip()

            return SQLResponse(
                query=query_executed,
                explanation=final_text,
                results=results,
                error=error,
            )

        except Exception as e:
            return SQLResponse(
                explanation=f"An error occurred while processing your request: {str(e)}",
                error=str(e),
            )


async def query_database(
    db_connection: DatabaseConnection, user_query: str, allow_write: bool = False
) -> SQLResponse:
    """
    Main function to query the database using natural language.

    Args:
        db_connection: Database connection instance
        user_query: Natural language query from the user
        allow_write: Whether to allow write operations

    Returns:
        SQLResponse with query, explanation, and results
    """
    agent = AnthropicSQLAgent(db_connection, allow_write)
    return await agent.query(user_query)
