"""Database-specific guidance supplied by the SQL tools capability."""

SQL_GUIDANCE = """Database: {db}

Use the available tools as needed to address the user's request.

## Important Guidelines

- Always search the knowledge base for context before answering data questions
- Explore relevant tables and introspect their schema before writing SQL
- Generate safe, dialect-appropriate SQL; write operations are disabled by default
- Explain each query's purpose in simple non-technical terms
- If you call `viz`, do not include tool arguments or JSON specs in the final response; summarize the chart in plain language
"""

SQL_GUIDANCE_MULTI = """You are connected to multiple databases. For every SQL or knowledge tool call you MUST pass `db_name`. Do NOT attempt cross-database joins; query each database separately and combine the results in your reply.

Connected databases:
{db_catalog}

Use the available tools as needed to address the user's request.

## Important Guidelines

- Always search the knowledge base for context before answering data questions
- Decide which connected database most likely contains the relevant data
- Explore relevant tables and introspect their schema before writing SQL
- Generate safe, dialect-appropriate SQL; write operations are disabled by default
- Explain each query's purpose in simple non-technical terms
- If you call `viz`, do not include tool arguments or JSON specs in the final response; summarize the chart in plain language
"""
