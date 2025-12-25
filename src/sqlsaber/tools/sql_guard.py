"""SQL query validation and security using sqlglot AST analysis."""

from collections.abc import Callable
from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

# DML/DDL operations that can be unlocked in "dangerous" mode
WRITE_DML_DDL_NODES: set[type[exp.Expression]] = {
    # DML operations
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Merge,
    # MySQL specific
    exp.Replace,
    # DDL operations (non-destructive)
    exp.Create,
    exp.Alter,
    exp.AlterRename,
}

# Operations that are always prohibited, regardless of mode
ALWAYS_BLOCKED_NODES: set[type[exp.Expression]] = {
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
    # Destructive schema/data operations (no safeguards possible)
    exp.Drop,
    exp.TruncateTable,
}

try:
    vacuum_type = getattr(exp, "Vacuum", None)
    if vacuum_type is not None:
        ALWAYS_BLOCKED_NODES.add(vacuum_type)
except AttributeError:
    pass

# Dangerous functions by dialect that can read files or execute commands
DANGEROUS_FUNCTIONS_BY_DIALECT: dict[str, set[str]] = {
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
    reason: str | None = None
    is_select: bool = False
    query_type: str | None = None  # "select" | "dml" | "ddl" | "other"


def is_select_like(stmt: exp.Expression) -> bool:
    """Check if statement is a SELECT-like query.

    Handles CTEs (WITH) and set operations (UNION/INTERSECT/EXCEPT).
    """
    root = stmt
    # WITH wraps another statement
    if isinstance(root, exp.With):
        root = root.this
    return isinstance(root, (exp.Select, exp.Union, exp.Except, exp.Intersect))


def classify_statement(stmt: exp.Expression) -> str:
    """Classify statement as select/dml/ddl/other.

    Returns:
        "select" for SELECT-like queries
        "dml" for INSERT/UPDATE/DELETE/MERGE/REPLACE
        "ddl" for CREATE/DROP/ALTER/TRUNCATE
        "other" for anything else
    """
    if is_select_like(stmt):
        return "select"

    root = stmt
    if isinstance(root, exp.With):
        root = root.this

    if isinstance(root, (exp.Insert, exp.Update, exp.Delete, exp.Merge, exp.Replace)):
        return "dml"

    if isinstance(
        root,
        (exp.Create, exp.Alter, exp.AlterRename),
    ):
        return "ddl"

    # DROP and TRUNCATE are blocked, but classify them for error messages
    if isinstance(root, (exp.Drop, exp.TruncateTable)):
        return "ddl"

    return "other"


def has_unfiltered_mutation(stmt: exp.Expression) -> str | None:
    """Check for UPDATE/DELETE without WHERE clause.

    These operations are dangerous because they affect all rows in a table.
    """
    for node in stmt.walk():
        if isinstance(node, exp.Update):
            if not node.args.get("where"):
                return "UPDATE without WHERE clause is not allowed (would affect all rows)"
        if isinstance(node, exp.Delete):
            if not node.args.get("where"):
                return "DELETE without WHERE clause is not allowed (would affect all rows)"
    return None


def has_prohibited_nodes(
    stmt: exp.Expression, allow_dangerous: bool = False
) -> str | None:
    """Walk AST to find any prohibited operations.

    In read-only mode (allow_dangerous=False):
      - Block DML/DDL (WRITE_DML_DDL_NODES)
      - Block always-blocked operations (ALWAYS_BLOCKED_NODES)
      - Block SELECT INTO
      - Block locking clauses (FOR UPDATE/FOR SHARE)

    In dangerous mode (allow_dangerous=True):
      - Allow DML/DDL
      - Still block ALWAYS_BLOCKED_NODES, SELECT INTO, locking clauses
      - Block UPDATE/DELETE without WHERE clause
    """
    for node in stmt.walk():
        # Operations that are never allowed
        if isinstance(node, tuple(ALWAYS_BLOCKED_NODES)):
            return f"Prohibited operation: {type(node).__name__}"

        # DML/DDL writes are only allowed in dangerous mode
        if not allow_dangerous and isinstance(node, tuple(WRITE_DML_DDL_NODES)):
            return f"Prohibited operation: {type(node).__name__}"

        # Block SELECT INTO (Postgres-style table creation)
        if isinstance(node, exp.Select) and node.args.get("into"):
            return "SELECT INTO is not allowed"

        # Block locking clauses (FOR UPDATE/FOR SHARE)
        if isinstance(node, exp.Select):
            locks = node.args.get("locks")
            if locks:
                return "SELECT with locking clause (FOR UPDATE/SHARE) is not allowed"

    # In dangerous mode, block unfiltered mutations
    if allow_dangerous:
        reason = has_unfiltered_mutation(stmt)
        if reason:
            return reason

    return None


def has_dangerous_functions(stmt: exp.Expression, dialect: str) -> str | None:
    """Check for dangerous functions that can read files or execute commands."""
    deny_set = DANGEROUS_FUNCTIONS_BY_DIALECT.get(dialect)
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
    if stmt is None:
        return GuardResult(False, "Unable to parse query - empty statement")

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

    return GuardResult(True, None, is_select=True, query_type="select")


def validate_sql(
    sql: str, dialect: str = "ansi", allow_dangerous: bool = False
) -> GuardResult:
    """Validate SQL with optional write/DDL allowance.

    In read-only mode (default): same behavior as validate_read_only.
    In dangerous mode: allow DML/DDL, but still enforce:
      - single statement
      - parseability
      - no dangerous functions (file IO, command exec, etc.)

    Args:
        sql: SQL query to validate
        dialect: SQL dialect (postgres, mysql, sqlite, tsql, etc.)
        allow_dangerous: If True, allow DML/DDL statements

    Returns:
        GuardResult with validation outcome
    """
    if not allow_dangerous:
        return validate_read_only(sql, dialect)

    try:
        statements = sqlglot.parse(sql, read=dialect)
    except ParseError as e:
        return GuardResult(False, f"Unable to parse query safely: {e}")
    except Exception as e:
        return GuardResult(False, f"Error parsing query: {e}")

    if len(statements) != 1:
        return GuardResult(
            False,
            f"Only single statements are allowed (got {len(statements)} statements)",
        )

    stmt = statements[0]
    if stmt is None:
        return GuardResult(False, "Unable to parse query - empty statement")

    # Still enforce function-level sandbox even in dangerous mode
    reason = has_dangerous_functions(stmt, dialect)
    if reason:
        return GuardResult(False, reason)

    # Still enforce always-blocked operations (COPY, LOAD DATA, SET, PRAGMA, etc.)
    reason = has_prohibited_nodes(stmt, allow_dangerous=True)
    if reason:
        return GuardResult(False, reason)

    query_type = classify_statement(stmt)
    return GuardResult(
        True, None, is_select=(query_type == "select"), query_type=query_type
    )


def add_limit(sql: str, dialect: str = "ansi", limit: int = 100) -> str:
    """Add LIMIT clause to query if not already present.

    Args:
        sql: SQL query
        dialect: SQL dialect for proper rendering
        limit: Maximum number of rows to return

    Returns:
        SQL with LIMIT clause added (or original if LIMIT already exists)
    """
    # Strip trailing semicolon to ensure clean parsing and modification
    # This handles cases where models generate SQL with a trailing semicolon
    sql = sql.strip().rstrip(";")

    try:
        statements = sqlglot.parse(sql, read=dialect)
        if len(statements) != 1:
            return sql

        stmt = statements[0]
        if stmt is None:
            return sql

        # Check if LIMIT/FETCH already exists
        has_limit = any(isinstance(n, (exp.Limit, exp.Fetch)) for n in stmt.walk())
        if has_limit:
            return stmt.sql(dialect=dialect)

        # Add LIMIT - sqlglot will render appropriately for dialect
        # (LIMIT for most, TOP for SQL Server, FETCH FIRST for Oracle)
        limit_method: Callable[[int], exp.Expression] | None = getattr(
            stmt, "limit", None
        )
        if limit_method is not None:
            limited_stmt = limit_method(limit)
            return limited_stmt.sql(dialect=dialect)
        return stmt.sql(dialect=dialect)

    except Exception:
        # If parsing/transformation fails, fall back to simple string append
        # This maintains backward compatibility
        sql_upper = sql.strip().upper()
        if "LIMIT" not in sql_upper:
            return f"{sql.rstrip(';')} LIMIT {limit};"
        return sql
