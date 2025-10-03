"""SQL query validation and security using sqlglot AST analysis."""

from dataclasses import dataclass
from typing import Optional

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

# Prohibited AST node types that indicate write/mutation operations
# Only include expression types that exist in sqlglot
PROHIBITED_NODES = {
    # DML operations
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Merge,
    # DDL operations
    exp.Create,
    exp.Drop,
    exp.Alter,
    exp.TruncateTable,
    exp.AlterRename,
    # MySQL specific
    exp.Replace,
    # Transaction control
    exp.Transaction,
    # Analysis and maintenance
    exp.Analyze,
    # Data loading/copying
    exp.Copy,
    exp.LoadData,
    # Session and configuration
    exp.Set,
    exp.Use,
    exp.Pragma,
    # Security
    exp.Grant,
    exp.Revoke,
    # Database operations
    exp.Attach,
    exp.Detach,
    # Locking and process control
    exp.Lock,
    exp.Kill,
    # Commands
    exp.Command,
}

try:
    # Add optional types that may not exist in all sqlglot versions
    PROHIBITED_NODES.add(exp.Vacuum)
except AttributeError:
    pass

# Dangerous functions by dialect that can read files or execute commands
DANGEROUS_FUNCTIONS_BY_DIALECT = {
    "postgres": {
        "pg_read_file",
        "pg_read_binary_file",
        "pg_ls_dir",
        "pg_stat_file",
        "pg_logdir_ls",
        "dblink",
        "dblink_exec",
    },
    "mysql": {
        "load_file",
        "sys_eval",
        "sys_exec",
    },
    "sqlite": {
        "readfile",
        "writefile",
    },
    "tsql": {
        "xp_cmdshell",
    },
}


@dataclass
class GuardResult:
    """Result of SQL query validation."""

    allowed: bool
    reason: Optional[str] = None
    is_select: bool = False


def is_select_like(stmt: exp.Expression) -> bool:
    """Check if statement is a SELECT-like query.

    Handles CTEs (WITH) and set operations (UNION/INTERSECT/EXCEPT).
    """
    root = stmt
    # WITH wraps another statement
    if isinstance(root, exp.With):
        root = root.this
    return isinstance(root, (exp.Select, exp.Union, exp.Except, exp.Intersect))


def has_prohibited_nodes(stmt: exp.Expression) -> Optional[str]:
    """Walk AST to find any prohibited operations.

    Checks for:
    - Write operations (INSERT/UPDATE/DELETE/etc)
    - DDL operations (CREATE/DROP/ALTER/etc)
    - SELECT INTO
    - Locking clauses (FOR UPDATE/FOR SHARE)
    """
    for node in stmt.walk():
        # Check prohibited node types
        if isinstance(node, tuple(PROHIBITED_NODES)):
            return f"Prohibited operation: {type(node).__name__}"

        # Block SELECT INTO (Postgres-style table creation)
        if isinstance(node, exp.Select) and node.args.get("into"):
            return "SELECT INTO is not allowed"

        # Block locking clauses (FOR UPDATE/FOR SHARE)
        if isinstance(node, exp.Select):
            locks = node.args.get("locks")
            if locks:
                return "SELECT with locking clause (FOR UPDATE/SHARE) is not allowed"

    return None


def has_dangerous_functions(stmt: exp.Expression, dialect: str) -> Optional[str]:
    """Check for dangerous functions that can read files or execute commands."""
    deny_set = DANGEROUS_FUNCTIONS_BY_DIALECT.get(dialect, set())
    if not deny_set:
        return None

    deny_lower = {f.lower() for f in deny_set}

    for fn in stmt.find_all(exp.Func):
        name = fn.name
        if name and name.lower() in deny_lower:
            return f"Use of dangerous function '{name}' is not allowed"

    return None


def validate_read_only(sql: str, dialect: str = "ansi") -> GuardResult:
    """Validate that SQL query is read-only using AST analysis.

    Args:
        sql: SQL query to validate
        dialect: SQL dialect (postgres, mysql, sqlite, tsql, etc.)

    Returns:
        GuardResult with validation outcome
    """
    try:
        statements = sqlglot.parse(sql, read=dialect)
    except ParseError as e:
        return GuardResult(False, f"Unable to parse query safely: {e}")
    except Exception as e:
        return GuardResult(False, f"Error parsing query: {e}")

    # Only allow single statements
    if len(statements) != 1:
        return GuardResult(
            False,
            f"Only single SELECT statements are allowed (got {len(statements)} statements)",
        )

    stmt = statements[0]

    # Must be a SELECT-like statement
    if not is_select_like(stmt):
        return GuardResult(False, "Only SELECT-like statements are allowed")

    # Check for prohibited operations in the AST
    reason = has_prohibited_nodes(stmt)
    if reason:
        return GuardResult(False, reason)

    # Check for dangerous functions
    reason = has_dangerous_functions(stmt, dialect)
    if reason:
        return GuardResult(False, reason)

    return GuardResult(True, None, is_select=True)


def add_limit(sql: str, dialect: str = "ansi", limit: int = 100) -> str:
    """Add LIMIT clause to query if not already present.

    Args:
        sql: SQL query
        dialect: SQL dialect for proper rendering
        limit: Maximum number of rows to return

    Returns:
        SQL with LIMIT clause added (or original if LIMIT already exists)
    """
    try:
        statements = sqlglot.parse(sql, read=dialect)
        if len(statements) != 1:
            return sql

        stmt = statements[0]

        # Check if LIMIT/TOP/FETCH already exists
        has_limit = any(
            isinstance(n, (exp.Limit, exp.Top, exp.Fetch)) for n in stmt.walk()
        )
        if has_limit:
            return stmt.sql(dialect=dialect)

        # Add LIMIT - sqlglot will render appropriately for dialect
        # (LIMIT for most, TOP for SQL Server, FETCH FIRST for Oracle)
        stmt = stmt.limit(limit)
        return stmt.sql(dialect=dialect)

    except Exception:
        # If parsing/transformation fails, fall back to simple string append
        # This maintains backward compatibility
        sql_upper = sql.strip().upper()
        if "LIMIT" not in sql_upper:
            return f"{sql.rstrip(';')} LIMIT {limit};"
        return sql
