"""Anthropic-specific SQL agent implementation using the custom client."""

import asyncio
import json
from typing import Any, AsyncIterator, Dict, List

from sqlsaber.agents.base import BaseSQLAgent
from sqlsaber.agents.streaming import (
    build_tool_result_block,
)
from sqlsaber.clients import AnthropicClient
from sqlsaber.clients.models import (
    ContentBlock,
    ContentType,
    CreateMessageRequest,
    Message,
    MessageRole,
    ToolDefinition,
)
from sqlsaber.config.settings import Config
from sqlsaber.database.connection import BaseDatabaseConnection
from sqlsaber.memory.manager import MemoryManager
from sqlsaber.models.events import StreamEvent


class AnthropicSQLAgent(BaseSQLAgent):
    """SQL Agent using the custom Anthropic client."""

    def __init__(
        self, db_connection: BaseDatabaseConnection, database_name: str | None = None
    ):
        super().__init__(db_connection)

        config = Config()
        config.validate()  # This will raise ValueError if API key is missing

        self.client = AnthropicClient(api_key=config.api_key)
        self.model = config.model_name.replace("anthropic:", "")

        self.database_name = database_name
        self.memory_manager = MemoryManager()

        # Track last query results for streaming
        self._last_results = None
        self._last_query = None

        # Define tools in the new format
        self.tools: List[ToolDefinition] = [
            ToolDefinition(
                name="list_tables",
                description="Get a list of all tables in the database with row counts. Use this first to discover available tables.",
                input_schema={
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            ),
            ToolDefinition(
                name="introspect_schema",
                description="Introspect database schema to understand table structures.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "table_pattern": {
                            "type": "string",
                            "description": "Optional pattern to filter tables (e.g., 'public.users', 'user%', '%order%')",
                        }
                    },
                    "required": [],
                },
            ),
            ToolDefinition(
                name="execute_sql",
                description="Execute a SQL query against the database.",
                input_schema={
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
            ),
            ToolDefinition(
                name="plot_data",
                description="Create a plot of query results.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "y_values": {
                            "type": "array",
                            "items": {"type": ["number", "null"]},
                            "description": "Y-axis data points (required)",
                        },
                        "x_values": {
                            "type": "array",
                            "items": {"type": ["number", "null"]},
                            "description": "X-axis data points (optional, will use indices if not provided)",
                        },
                        "plot_type": {
                            "type": "string",
                            "enum": ["line", "scatter", "histogram"],
                            "description": "Type of plot to create (default: line)",
                            "default": "line",
                        },
                        "title": {
                            "type": "string",
                            "description": "Title for the plot",
                        },
                        "x_label": {
                            "type": "string",
                            "description": "Label for X-axis",
                        },
                        "y_label": {
                            "type": "string",
                            "description": "Label for Y-axis",
                        },
                    },
                    "required": ["y_values"],
                },
            ),
        ]

        # Build system prompt with memories if available
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """Build system prompt with optional memory context."""
        db_type = self._get_database_type_name()
        base_prompt = f"""You are a helpful SQL assistant that helps users query their {db_type} database.

Your responsibilities:
1. Understand user's natural language requests, think and convert them to SQL
2. Use the provided tools efficiently to explore database schema
3. Generate appropriate SQL queries
4. Execute queries safely - queries that modify the database are not allowed
5. Format and explain results clearly
6. Create visualizations when requested or when they would be helpful

IMPORTANT - Schema Discovery Strategy:
1. ALWAYS start with 'list_tables' to see available tables and row counts
2. Based on the user's query, identify which specific tables are relevant
3. Use 'introspect_schema' with a table_pattern to get details ONLY for relevant tables
4. Timestamp columns must be converted to text when you write queries

Guidelines:
- Use list_tables first, then introspect_schema for specific tables only
- Use table patterns like 'sample%' or '%experiment%' to filter related tables
- Use proper JOIN syntax and avoid cartesian products
- Include appropriate WHERE clauses to limit results
- Explain what the query does in simple terms
- Handle errors gracefully and suggest fixes
- Be security conscious - use parameterized queries when needed
"""

        # Add memory context if database name is available
        if self.database_name:
            memory_context = self.memory_manager.format_memories_for_prompt(
                self.database_name
            )
            if memory_context.strip():
                base_prompt += memory_context

        return base_prompt

    def add_memory(self, content: str) -> str | None:
        """Add a memory for the current database."""
        if not self.database_name:
            return None

        memory = self.memory_manager.add_memory(self.database_name, content)
        # Rebuild system prompt with new memory
        self.system_prompt = self._build_system_prompt()
        return memory.id

    async def execute_sql(self, query: str, limit: int | None = None) -> str:
        """Execute a SQL query against the database with streaming support."""
        # Call parent implementation for core functionality
        result = await super().execute_sql(query, limit)

        # Parse result to extract data for streaming (AnthropicSQLAgent specific)
        try:
            result_data = json.loads(result)
            if result_data.get("success") and "results" in result_data:
                # Store results for streaming
                actual_limit = (
                    limit if limit is not None else len(result_data["results"])
                )
                self._last_results = result_data["results"][:actual_limit]
                self._last_query = query
        except (json.JSONDecodeError, KeyError):
            # If we can't parse the result, just continue without storing
            pass

        return result

    async def process_tool_call(
        self, tool_name: str, tool_input: Dict[str, Any]
    ) -> str:
        """Process a tool call and return the result."""
        # Use parent implementation for core tools
        return await super().process_tool_call(tool_name, tool_input)

    async def query_stream(
        self,
        user_query: str,
        use_history: bool = True,
        cancellation_token: asyncio.Event | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Process a user query and stream responses."""
        # Initialize for tracking state
        self._last_results = None
        self._last_query = None

        # Build messages with history if requested
        messages = []
        if use_history:
            # Convert conversation history to new format
            for msg in self.conversation_history:
                if msg["role"] == "user":
                    # User content might be a string or list of tool results
                    content = msg["content"]
                    if isinstance(content, str):
                        messages.append(Message(MessageRole.USER, content))
                    else:
                        # Handle tool results format
                        tool_result_blocks = []
                        if isinstance(content, list):
                            for item in content:
                                if (
                                    isinstance(item, dict)
                                    and item.get("type") == "tool_result"
                                ):
                                    tool_result_blocks.append(
                                        ContentBlock(ContentType.TOOL_RESULT, item)
                                    )
                        if tool_result_blocks:
                            messages.append(
                                Message(MessageRole.USER, tool_result_blocks)
                            )
                        else:
                            # Fallback to string representation
                            messages.append(Message(MessageRole.USER, str(content)))

                elif msg["role"] == "assistant":
                    # Assistant content is typically a list of content blocks
                    content = msg["content"]
                    if isinstance(content, str):
                        messages.append(Message(MessageRole.ASSISTANT, content))
                    elif isinstance(content, list):
                        # Convert to ContentBlock objects
                        content_blocks = []
                        for block in content:
                            if isinstance(block, dict):
                                if block.get("type") == "text":
                                    text_content = block.get("text", "")
                                    if text_content:  # Only add non-empty text blocks
                                        content_blocks.append(
                                            ContentBlock(ContentType.TEXT, text_content)
                                        )
                                elif block.get("type") == "tool_use":
                                    content_blocks.append(
                                        ContentBlock(
                                            ContentType.TOOL_USE,
                                            {
                                                "id": block["id"],
                                                "name": block["name"],
                                                "input": block["input"],
                                            },
                                        )
                                    )
                        if content_blocks:
                            messages.append(
                                Message(MessageRole.ASSISTANT, content_blocks)
                            )
                    else:
                        # Fallback to string representation
                        messages.append(Message(MessageRole.ASSISTANT, str(content)))

        # Add current user message
        messages.append(Message(MessageRole.USER, user_query))

        try:
            # Create request
            request = CreateMessageRequest(
                model=self.model,
                messages=messages,
                max_tokens=4096,
                system=self.system_prompt,
                tools=self.tools,
                stream=True,
            )

            # Create initial stream and get response
            response = None
            async for event in self.client.create_message_with_tools(
                request, cancellation_token
            ):
                if cancellation_token is not None and cancellation_token.is_set():
                    return

                # Handle different event types
                if hasattr(event, "type"):
                    if event.type == "content_block_start":
                        if hasattr(event.content_block, "type"):
                            if event.content_block.type == "tool_use":
                                yield StreamEvent(
                                    "tool_use",
                                    {
                                        "name": event.content_block.name,
                                        "status": "started",
                                    },
                                )
                    elif event.type == "content_block_delta":
                        if hasattr(event.delta, "text"):
                            text = event.delta.text
                            if text is not None and text:  # Only yield non-empty text
                                yield StreamEvent("text", text)
                elif isinstance(event, dict) and event.get("type") == "response_ready":
                    response = event["data"]

            collected_content = []

            # Process tool calls if needed
            while response is not None and response.stop_reason == "tool_use":
                # Check for cancellation at the start of tool cycle
                if cancellation_token is not None and cancellation_token.is_set():
                    return

                # Add assistant's response to conversation
                collected_content.append(
                    {"role": "assistant", "content": response.content}
                )

                # Process tool results
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

                        # Yield specific events based on tool type
                        if block["name"] == "execute_sql" and self._last_results:
                            yield StreamEvent(
                                "query_result",
                                {
                                    "query": self._last_query,
                                    "results": self._last_results,
                                },
                            )
                        elif block["name"] in ["list_tables", "introspect_schema"]:
                            yield StreamEvent(
                                "tool_result",
                                {
                                    "tool_name": block["name"],
                                    "result": tool_result,
                                },
                            )
                        elif block["name"] == "plot_data":
                            yield StreamEvent(
                                "plot_result",
                                {
                                    "tool_name": block["name"],
                                    "input": block["input"],
                                    "result": tool_result,
                                },
                            )

                        tool_results.append(
                            build_tool_result_block(block["id"], tool_result)
                        )

                # Continue conversation with tool results
                collected_content.append({"role": "user", "content": tool_results})
                if use_history:
                    self.conversation_history.extend(collected_content)

                # Check for cancellation AFTER tool results are complete
                if cancellation_token is not None and cancellation_token.is_set():
                    return

                # Signal that we're processing the tool results
                yield StreamEvent("processing", "Analyzing results...")

                # Convert collected content to new message format
                new_messages = messages.copy()
                for content in collected_content:
                    if content["role"] == "user":
                        # Tool results are in the format expected by the API
                        tool_result_blocks = []
                        for tool_result in content["content"]:
                            tool_result_blocks.append(
                                ContentBlock(ContentType.TOOL_RESULT, tool_result)
                            )
                        new_messages.append(
                            Message(MessageRole.USER, tool_result_blocks)
                        )
                    elif content["role"] == "assistant":
                        # Convert assistant content to ContentBlock objects
                        assistant_content = content["content"]
                        content_blocks = []
                        for block in assistant_content:
                            if block.get("type") == "text":
                                text_content = block["text"]
                                if text_content:  # Only add non-empty text blocks
                                    content_blocks.append(
                                        ContentBlock(ContentType.TEXT, text_content)
                                    )
                            elif block.get("type") == "tool_use":
                                content_blocks.append(
                                    ContentBlock(
                                        ContentType.TOOL_USE,
                                        {
                                            "id": block["id"],
                                            "name": block["name"],
                                            "input": block["input"],
                                        },
                                    )
                                )
                        new_messages.append(
                            Message(MessageRole.ASSISTANT, content_blocks)
                        )

                # Create new request with updated messages
                request = CreateMessageRequest(
                    model=self.model,
                    messages=new_messages,
                    max_tokens=4096,
                    system=self.system_prompt,
                    tools=self.tools,
                    stream=True,
                )

                # Get next response
                response = None
                async for event in self.client.create_message_with_tools(
                    request, cancellation_token
                ):
                    if cancellation_token is not None and cancellation_token.is_set():
                        return

                    # Handle events similar to above
                    if hasattr(event, "type"):
                        if event.type == "content_block_delta":
                            if hasattr(event.delta, "text"):
                                text = event.delta.text
                                if (
                                    text is not None and text
                                ):  # Only yield non-empty text
                                    yield StreamEvent("text", text)
                    elif (
                        isinstance(event, dict)
                        and event.get("type") == "response_ready"
                    ):
                        response = event["data"]

            # Update conversation history if using history
            if use_history:
                # Add final assistant response
                if response is not None:
                    self.conversation_history.append(
                        {"role": "assistant", "content": response.content}
                    )

        except asyncio.CancelledError:
            return
        except Exception as e:
            yield StreamEvent("error", str(e))

    async def close(self):
        """Close the client."""
        await self.client.close()
