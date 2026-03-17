"""SQL query validation and security using sqlglot AST analysis."""

import re
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

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

# Dangerous functions by dialect that can read files, execute commands,
# alter session/server state, or otherwise introduce side effects.
DANGEROUS_FUNCTIONS_BY_DIALECT: dict[str, set[str]] = {
    "postgres": {
        # File/system access
        "pg_read_file",
        "pg_read_binary_file",
        "pg_ls_dir",
        "pg_stat_file",
        "pg_logdir_ls",
        "pg_ls_logdir",
        "pg_ls_waldir",
        "pg_ls_archive_statusdir",
        "pg_write_file",
        "pg_append_file",
        "lo_import",
        "lo_export",
        # External execution / remote calls
        "dblink",
        "dblink_exec",
        # Process/server/session side effects
        "pg_terminate_backend",
        "pg_cancel_backend",
        "pg_reload_conf",
        "pg_rotate_logfile",
        "pg_notify",
        "set_config",
        # Advisory locks
        "pg_advisory_lock",
        "pg_try_advisory_lock",
        "pg_advisory_xact_lock",
        "pg_advisory_lock_shared",
        "pg_try_advisory_lock_shared",
    },
    "mysql": {
        # File/system access
        "load_file",
        "sys_eval",
        "sys_exec",
        # Resource/session/locking side effects
        "sleep",
        "benchmark",
        "get_lock",
        "release_lock",
    },
    "sqlite": {
        # File access
        "readfile",
        "writefile",
        # Extension loading
        "load_extension",
    },
    "duckdb": {
        # File-reading table functions
        "read_csv_auto",
        "read_csv",
        "read_json_auto",
        "read_json",
        "read_parquet",
        "parquet_scan",
        # Text/binary file reading
        "read_text",
        "read_blob",
        # NDJSON/JSONL readers
        "read_ndjson",
        "read_ndjson_auto",
        "read_ndjson_objects",
        # Filesystem enumeration
        "glob",
        # External database access
        "sqlite_scan",
        "postgres_scan",
        "mysql_scan",
        "postgres_query",
        "mysql_query",
        # Extension management
        "load_extension",
        "install_extension",
        # Additional format readers
        "iceberg_scan",
        "delta_scan",
        "excel_scan",
        "st_read",
    },
    "tsql": {
        "xp_cmdshell",
    },
}

# In dangerous mode, we run fail-closed and only allow these root statement types.
DANGEROUS_ALLOWED_ROOT_NODES: tuple[type[exp.Expression], ...] = (
    exp.Select,
    exp.Union,
    exp.Except,
    exp.Intersect,
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Create,
    exp.Alter,
    exp.AlterRename,
)

# In dangerous mode, CREATE is further constrained by explicit kind allowlist.
ALLOWED_DANGEROUS_CREATE_KINDS: set[str] = {
    "TABLE",
    "VIEW",
    "INDEX",
}


NUMERIC_TRUTHY_PREDICATE_DIALECTS: set[str] = {
    "mysql",
    "sqlite",
}


_UNKNOWN_SCALAR_VALUE = object()
_SQL_NULL_SCALAR_VALUE = object()


_MYSQL_VERSION_COMMENT_RE = re.compile(r"/\*!")


def _has_mysql_version_comments(sql: str) -> bool:
    """Detect MySQL version comments that create parser divergence."""
    return bool(_MYSQL_VERSION_COMMENT_RE.search(sql))


@dataclass
class GuardResult:
    """Result of SQL query validation."""

    allowed: bool
    reason: str | None = None
    is_select: bool = False
    query_type: str | None = None  # "select" | "dml" | "ddl" | "other"
    has_limit: bool = False


def _unwrap_root(stmt: exp.Expression) -> exp.Expression:
    """Return the effective statement root (unwrap WITH)."""
    root = stmt
    if isinstance(root, exp.With):
        inner = root.this
        if inner is not None:
            root = inner
    return root


def is_select_like(stmt: exp.Expression) -> bool:
    """Check if statement is a SELECT-like query.

    Handles CTEs (WITH) and set operations (UNION/INTERSECT/EXCEPT).
    """
    root = _unwrap_root(stmt)
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

    root = _unwrap_root(stmt)

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


def _unwrap_parens(expr: exp.Expression | None) -> exp.Expression | None:
    """Unwrap nested parentheses from an expression."""
    while isinstance(expr, exp.Paren):
        inner = expr.this
        if inner is None:
            break
        expr = inner
    return expr


def _numeric_literal_value(expr: exp.Expression | None, dialect: str) -> Decimal | None:
    """Return Decimal value for numeric/hex literals, with unary minus support."""
    expr = _unwrap_parens(expr)
    if expr is None:
        return None

    sign = 1
    while isinstance(expr, exp.Neg):
        sign *= -1
        expr = _unwrap_parens(expr.this)
        if expr is None:
            return None

    value: Decimal | None = None

    if isinstance(expr, exp.Literal):
        if expr.is_string:
            return None
        try:
            value = Decimal(str(expr.this))
        except (InvalidOperation, TypeError, ValueError):
            return None

    if isinstance(expr, exp.HexString):
        if dialect not in NUMERIC_TRUTHY_PREDICATE_DIALECTS:
            return None
        try:
            value = Decimal(int(str(expr.this), 16))
        except (TypeError, ValueError):
            return None

    if value is None:
        return None

    if sign < 0:
        value = -value

    return value


def _numeric_string_value(value: str) -> Decimal | None:
    """Parse a numeric-looking string literal safely."""
    stripped = value.strip()
    if not stripped:
        return None

    try:
        return Decimal(stripped)
    except (InvalidOperation, ValueError):
        return None


def _constant_scalar_value(
    expr: exp.Expression | None,
    dialect: str,
) -> Decimal | str | bool | object:
    """Evaluate expression to a constant scalar when safely possible."""
    expr = _unwrap_parens(expr)
    if expr is None:
        return _UNKNOWN_SCALAR_VALUE

    if isinstance(expr, exp.Boolean):
        return bool(expr.this)

    if isinstance(expr, exp.Null):
        return _SQL_NULL_SCALAR_VALUE

    numeric_value = _numeric_literal_value(expr, dialect)
    if numeric_value is not None:
        return numeric_value

    if isinstance(expr, exp.Literal) and expr.is_string:
        return str(expr.this)

    if isinstance(expr, exp.Abs):
        inner = _constant_scalar_value(expr.this, dialect)
        if isinstance(inner, Decimal):
            return abs(inner)
        return _UNKNOWN_SCALAR_VALUE

    if isinstance(expr, exp.Coalesce):
        args: list[exp.Expression] = []
        first = expr.this
        if isinstance(first, exp.Expression):
            args.append(first)
        args.extend(expr.expressions)

        for arg in args:
            value = _constant_scalar_value(arg, dialect)
            if value is _UNKNOWN_SCALAR_VALUE:
                return _UNKNOWN_SCALAR_VALUE
            if value is not _SQL_NULL_SCALAR_VALUE:
                return value

        return _SQL_NULL_SCALAR_VALUE

    return _UNKNOWN_SCALAR_VALUE


def _predicate_truthiness_from_scalar(
    value: Decimal | str | bool | object,
    dialect: str,
) -> bool | None:
    """Interpret scalar values as predicate truthiness when semantics are clear."""
    if value is _UNKNOWN_SCALAR_VALUE:
        return None

    if value is _SQL_NULL_SCALAR_VALUE:
        return False

    if isinstance(value, bool):
        return value

    if isinstance(value, Decimal):
        if dialect not in NUMERIC_TRUTHY_PREDICATE_DIALECTS:
            return None
        return value != 0

    if isinstance(value, str):
        if dialect not in NUMERIC_TRUTHY_PREDICATE_DIALECTS:
            return None
        numeric_value = _numeric_string_value(value)
        if numeric_value is None:
            return None
        return numeric_value != 0

    return None


def _constant_scalar_equals(
    left: Decimal | str | bool | object,
    right: Decimal | str | bool | object,
) -> bool | None:
    """Compare constant scalar values when type-compatible."""
    if (
        left is _UNKNOWN_SCALAR_VALUE
        or right is _UNKNOWN_SCALAR_VALUE
        or left is _SQL_NULL_SCALAR_VALUE
        or right is _SQL_NULL_SCALAR_VALUE
    ):
        return None

    if isinstance(left, Decimal) and isinstance(right, Decimal):
        return left == right

    if isinstance(left, bool) and isinstance(right, bool):
        return left == right

    if isinstance(left, str) and isinstance(right, str):
        return left == right

    return None


def _constant_in_predicate_truthiness(expr: exp.In, dialect: str) -> bool | None:
    """Evaluate constant IN predicates when possible."""
    left_value = _constant_scalar_value(expr.this, dialect)
    if left_value in {_UNKNOWN_SCALAR_VALUE, _SQL_NULL_SCALAR_VALUE}:
        return None

    if expr.args.get("query") is not None:
        return None

    saw_unknown = False
    for option in expr.expressions:
        option_value = _constant_scalar_value(option, dialect)
        equals = _constant_scalar_equals(left_value, option_value)

        if equals is True:
            return True
        if equals is None:
            saw_unknown = True

    if saw_unknown:
        return None

    return False


def _constant_exists_predicate_truthiness(
    expr: exp.Exists, dialect: str
) -> bool | None:
    """Evaluate simple row-invariant EXISTS predicates safely."""
    subquery = expr.this
    if isinstance(subquery, exp.Subquery):
        subquery = subquery.this

    if not isinstance(subquery, exp.Select):
        return None

    if subquery.args.get("from") is not None:
        return None

    where_clause = subquery.args.get("where")
    if isinstance(where_clause, exp.Where):
        where_truth = _constant_predicate_truthiness(where_clause.this, dialect)
        if where_truth is False:
            return False
        if where_truth is None:
            return None

    limit_clause = subquery.args.get("limit")
    if isinstance(limit_clause, exp.Limit):
        limit_expression = limit_clause.expression or limit_clause.this
        limit_value = _constant_scalar_value(limit_expression, dialect)
        if limit_value is _UNKNOWN_SCALAR_VALUE:
            return None
        if isinstance(limit_value, Decimal) and limit_value <= 0:
            return False

    return True


def _constant_predicate_truthiness(
    expr: exp.Expression | None,
    dialect: str,
) -> bool | None:
    """Evaluate obvious constant truthiness in boolean predicate position."""
    expr = _unwrap_parens(expr)

    if expr is None:
        return True

    if isinstance(expr, exp.Not):
        inner = _constant_predicate_truthiness(expr.this, dialect)
        if inner is None:
            return None
        return not inner

    if isinstance(expr, exp.And):
        left = _constant_predicate_truthiness(expr.this, dialect)
        right = _constant_predicate_truthiness(expr.expression, dialect)

        if left is False or right is False:
            return False
        if left is True and right is True:
            return True
        if left is True:
            return right
        if right is True:
            return left
        return None

    if isinstance(expr, exp.Or):
        left = _constant_predicate_truthiness(expr.this, dialect)
        right = _constant_predicate_truthiness(expr.expression, dialect)

        if left is True or right is True:
            return True
        if left is False and right is False:
            return False
        if left is False:
            return right
        if right is False:
            return left
        return None

    if isinstance(expr, exp.In):
        in_truth = _constant_in_predicate_truthiness(expr, dialect)
        if in_truth is not None:
            return in_truth

    if isinstance(expr, exp.Exists):
        exists_truth = _constant_exists_predicate_truthiness(expr, dialect)
        if exists_truth is not None:
            return exists_truth

    scalar_value = _constant_scalar_value(expr, dialect)
    return _predicate_truthiness_from_scalar(scalar_value, dialect)


def _is_tautological_where(where: exp.Where, dialect: str = "ansi") -> bool:
    """Return True when a WHERE predicate is effectively always true."""
    predicate = where.this
    if predicate is None:
        return True

    candidates: list[exp.Expression] = [predicate]

    try:
        from sqlglot.optimizer.simplify import simplify

        candidates.insert(0, simplify(predicate.copy()))
    except Exception:
        pass

    return any(
        _constant_predicate_truthiness(candidate, dialect) is True
        for candidate in candidates
    )


def has_unfiltered_mutation(stmt: exp.Expression, dialect: str = "ansi") -> str | None:
    """Check for UPDATE/DELETE without restrictive WHERE clause.

    These operations are dangerous because they can affect all rows in a table.
    """
    for node in stmt.walk():
        if isinstance(node, exp.Update):
            where = node.args.get("where")
            if not where:
                return (
                    "UPDATE without WHERE clause is not allowed (would affect all rows)"
                )
            if isinstance(where, exp.Where) and _is_tautological_where(where, dialect):
                return (
                    "UPDATE with tautological WHERE clause is not allowed "
                    "(would affect all rows)"
                )

        if isinstance(node, exp.Delete):
            where = node.args.get("where")
            if not where:
                return (
                    "DELETE without WHERE clause is not allowed (would affect all rows)"
                )
            if isinstance(where, exp.Where) and _is_tautological_where(where, dialect):
                return (
                    "DELETE with tautological WHERE clause is not allowed "
                    "(would affect all rows)"
                )

    return None


def has_prohibited_nodes(
    stmt: exp.Expression,
    allow_dangerous: bool = False,
    dialect: str = "ansi",
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
      - Block UPDATE/DELETE without restrictive WHERE clause
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
        reason = has_unfiltered_mutation(stmt, dialect)
        if reason:
            return reason

    return None


def _normalize_symbol(name: str) -> str:
    """Normalize SQL identifiers for resilient matching.

    Normalization is intentionally conservative to reduce false positives.
    """
    return name.strip().strip('"`[]').lower()


def _compact_symbol(name: str) -> str:
    """Compacted normalization used only as a fallback for AST key matching.

    This bridges representations like ``read_parquet`` and ``readparquet``.
    """
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _function_name_tokens(fn: exp.Func) -> list[tuple[str, str]]:
    """Collect normalized tokens and their source for a function node."""
    tokens: list[tuple[str, str]] = []

    if fn.name:
        tokens.append(("name", _normalize_symbol(fn.name)))

    sql_name = ""
    try:
        sql_name = fn.sql_name() or ""
    except (AttributeError, TypeError, ValueError):
        sql_name = ""
    if sql_name:
        tokens.append(("sql_name", _normalize_symbol(sql_name)))

    key = getattr(fn, "key", "") or ""
    if key:
        tokens.append(("key", _normalize_symbol(key)))

    return tokens


def has_dangerous_functions(stmt: exp.Expression, dialect: str) -> str | None:
    """Check for dangerous functions that can read files or execute commands."""
    deny_set = DANGEROUS_FUNCTIONS_BY_DIALECT.get(dialect)
    if not deny_set:
        return None

    deny_exact = {_normalize_symbol(name) for name in deny_set}
    deny_compact = {_compact_symbol(name) for name in deny_set}

    for fn in stmt.find_all(exp.Func):
        tokens = _function_name_tokens(fn)
        exact_tokens = {value for _, value in tokens if value}

        if exact_tokens & deny_exact:
            display_name = fn.name or ""
            if not display_name:
                try:
                    display_name = fn.sql_name() or ""
                except (AttributeError, TypeError, ValueError):
                    display_name = ""
            if not display_name:
                display_name = getattr(fn, "key", "unknown_function")
            return f"Use of dangerous function '{display_name}' is not allowed"

        # Fallback: sqlglot key names can be compact (e.g. readparquet).
        key_tokens = [value for source, value in tokens if source == "key" and value]
        if any(_compact_symbol(value) in deny_compact for value in key_tokens):
            display_name = fn.name or ""
            if not display_name:
                try:
                    display_name = fn.sql_name() or ""
                except (AttributeError, TypeError, ValueError):
                    display_name = ""
            if not display_name:
                display_name = getattr(fn, "key", "unknown_function")
            return f"Use of dangerous function '{display_name}' is not allowed"

    return None


def has_disallowed_dangerous_mode_statement(stmt: exp.Expression) -> str | None:
    """Fail-closed statement allowlist checks for dangerous mode."""
    root = _unwrap_root(stmt)

    if not isinstance(root, DANGEROUS_ALLOWED_ROOT_NODES):
        return (
            "Only SELECT, INSERT, UPDATE, DELETE, and restricted CREATE/ALTER "
            "statements are allowed in dangerous mode"
        )

    if isinstance(root, exp.Create):
        kind = str(root.args.get("kind") or "").upper()
        if kind not in ALLOWED_DANGEROUS_CREATE_KINDS:
            return f"CREATE {kind or '<unknown>'} is not allowed in dangerous mode"

        target = root.args.get("this")
        expression = root.args.get("expression")

        # Additional defensive checks to avoid dialect/parser gaps.
        if kind == "TABLE":
            if target is not None and not isinstance(target, (exp.Table, exp.Schema)):
                return "Only CREATE TABLE statements are allowed in dangerous mode"
        elif kind == "VIEW":
            if target is not None and not isinstance(target, exp.Table):
                return "Only CREATE VIEW statements are allowed in dangerous mode"
            if expression is not None and not is_select_like(expression):
                return "CREATE VIEW must be based on a SELECT-like expression"
        elif kind == "INDEX":
            if target is not None and not isinstance(target, exp.Index):
                return "Only CREATE INDEX statements are allowed in dangerous mode"

        if isinstance(target, exp.UserDefinedFunction):
            return "CREATE FUNCTION-like statements are not allowed in dangerous mode"

    if isinstance(root, exp.Alter):
        kind = str(root.args.get("kind") or "TABLE").upper()
        if kind != "TABLE":
            return f"ALTER {kind or '<unknown>'} is not allowed in dangerous mode"

    if isinstance(root, exp.AlterRename):
        target = root.args.get("this")
        if target is not None and not isinstance(target, exp.Table):
            return "Only ALTER TABLE style rename statements are allowed"

    return None


def has_limit_clause(stmt: exp.Expression) -> bool:
    """Check if a statement already includes a LIMIT/TOP/FETCH clause."""
    limit_types: list[type[exp.Expression]] = [exp.Limit, exp.Fetch]
    top_type = getattr(exp, "Top", None)
    if isinstance(top_type, type):
        limit_types.append(top_type)
    return any(isinstance(node, tuple(limit_types)) for node in stmt.walk())


def validate_read_only(sql: str, dialect: str = "ansi") -> GuardResult:
    """Validate that SQL query is read-only using AST analysis.

    Args:
        sql: SQL query to validate
        dialect: SQL dialect (postgres, mysql, sqlite, tsql, etc.)

    Returns:
        GuardResult with validation outcome
    """
    if dialect == "mysql" and _has_mysql_version_comments(sql):
        return GuardResult(
            False,
            "MySQL version comments (/*!...*/) are not allowed (parser divergence risk)",
        )

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
    reason = has_prohibited_nodes(stmt, dialect=dialect)
    if reason:
        return GuardResult(False, reason)

    # Check for dangerous functions
    reason = has_dangerous_functions(stmt, dialect)
    if reason:
        return GuardResult(False, reason)

    return GuardResult(
        True,
        None,
        is_select=True,
        query_type="select",
        has_limit=has_limit_clause(stmt),
    )


def validate_sql(
    sql: str, dialect: str = "ansi", allow_dangerous: bool = False
) -> GuardResult:
    """Validate SQL with optional write/DDL allowance.

    In read-only mode (default): same behavior as validate_read_only.
    In dangerous mode: fail-closed allowlist + additional guardrails:
      - single statement
      - parseability
      - no dangerous functions (file IO, command exec, etc.)
      - only allowlisted statement classes/kinds
      - no always-blocked nodes

    Args:
        sql: SQL query to validate
        dialect: SQL dialect (postgres, mysql, sqlite, tsql, etc.)
        allow_dangerous: If True, allow selected DML/DDL statements

    Returns:
        GuardResult with validation outcome
    """
    if not allow_dangerous:
        return validate_read_only(sql, dialect)

    if dialect == "mysql" and _has_mysql_version_comments(sql):
        return GuardResult(
            False,
            "MySQL version comments (/*!...*/) are not allowed (parser divergence risk)",
        )

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

    # Enforce function-level sandbox in dangerous mode too
    reason = has_dangerous_functions(stmt, dialect)
    if reason:
        return GuardResult(False, reason)

    # Enforce always-blocked operations and lock/SELECT INTO checks
    reason = has_prohibited_nodes(stmt, allow_dangerous=True, dialect=dialect)
    if reason:
        return GuardResult(False, reason)

    # Strict fail-closed statement policy in dangerous mode
    reason = has_disallowed_dangerous_mode_statement(stmt)
    if reason:
        return GuardResult(False, reason)

    query_type = classify_statement(stmt)
    return GuardResult(
        True,
        None,
        is_select=(query_type == "select"),
        query_type=query_type,
        has_limit=has_limit_clause(stmt),
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

        # Check if LIMIT/TOP/FETCH already exists
        if has_limit_clause(stmt):
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
