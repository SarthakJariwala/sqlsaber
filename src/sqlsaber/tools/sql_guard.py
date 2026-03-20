"""SQL query validation and security using sqlglot AST analysis."""

import re
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
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
    "duckdb",
    "mysql",
    "sqlite",
}

# Dialects that coerce strings to numbers via leading numeric prefix
# (e.g., '1abc' -> 1, 'abc' -> 0) in numeric/boolean contexts.
NUMERIC_PREFIX_STRING_COERCION_DIALECTS: set[str] = {
    "mysql",
    "sqlite",
}

# Dialects where constant numeric/string equality can be resolved via
# deterministic coercion rules.
NUMERIC_STRING_CROSS_TYPE_COMPARISON_DIALECTS: set[str] = {
    "duckdb",
    "mysql",
}

# Dialects where bool↔numeric coercion is deterministic for constant
# comparisons (e.g., TRUE == 1, FALSE == 0).
BOOL_NUMERIC_CROSS_TYPE_COMPARISON_DIALECTS: set[str] = {
    "duckdb",
    "mysql",
    "sqlite",
}

# Dialects where bool↔string coercion is deterministic for constant
# comparisons.
BOOL_STRING_CROSS_TYPE_COMPARISON_DIALECTS: set[str] = {
    "duckdb",
}

# Dialects where bool↔string equality is resolved via numeric coercion
# (e.g., MySQL TRUE = '1', FALSE = '0').
BOOL_STRING_NUMERIC_COERCION_DIALECTS: set[str] = {
    "mysql",
}

# Dialects where hex literal syntax in predicate position is numeric.
HEX_NUMERIC_LITERAL_DIALECTS: set[str] = {
    "mysql",
    "sqlite",
}

# DuckDB coerces a limited set of string literals to BOOLEAN in predicate
# position.
DUCKDB_TRUTHY_STRING_PREDICATES: set[str] = {
    "1",
    "true",
    "t",
    "yes",
    "y",
}
DUCKDB_FALSEY_STRING_PREDICATES: set[str] = {
    "0",
    "false",
    "f",
    "no",
    "n",
}

# PostgreSQL accepts these string forms for boolean casts.
POSTGRES_TRUTHY_BOOLEAN_CAST_STRINGS: set[str] = {
    "1",
    "true",
    "t",
    "yes",
    "y",
    "on",
}
POSTGRES_FALSEY_BOOLEAN_CAST_STRINGS: set[str] = {
    "0",
    "false",
    "f",
    "no",
    "n",
    "off",
}

# Dialects where LIMIT NULL behaves as "no limit".
LIMIT_NULL_UNBOUNDED_DIALECTS: set[str] = {
    "duckdb",
    "postgres",
}

# Dialects where LIMIT ALL behaves as "no limit".
LIMIT_ALL_UNBOUNDED_DIALECTS: set[str] = {
    "duckdb",
    "postgres",
}

# Known from-less projection nodes/functions that can yield multiple rows.
SET_RETURNING_PROJECTION_NODE_TYPES: tuple[type[exp.Expression], ...] = (
    exp.Explode,
    exp.ExplodeOuter,
    exp.Unnest,
    exp.ExplodingGenerateSeries,
)
POSTGRES_SET_RETURNING_ANONYMOUS_FUNCTIONS: set[str] = {
    "generate_series",
    "generate_subscripts",
    "json_array_elements",
    "json_array_elements_text",
    "jsonb_array_elements",
    "jsonb_array_elements_text",
    "json_each",
    "json_each_text",
    "jsonb_each",
    "jsonb_each_text",
    "regexp_matches",
    "unnest",
}

# Simplify should only run on manageable predicate shapes.
PREDICATE_SIMPLIFY_MAX_NODES = 300
PREDICATE_SIMPLIFY_MAX_DEPTH = 36
PREDICATE_SIMPLIFY_MAX_BOOLEAN_OPERATORS = 180

# Per-validation guardrail against CPU amplification on adversarial predicates.
ANALYSIS_BUDGET_MAX_STEPS = 50_000


class _AnalysisBudgetExceeded(RecursionError):
    """Raised when validation exceeds allowed analysis step budget."""


@dataclass
class _AnalysisContext:
    """Per-validation mutable analysis state (budget + memoization caches)."""

    remaining_steps: int
    predicate_truthiness_cache: dict[tuple[int, str, int], set[bool | None]]
    simplify_gate_cache: dict[int, bool]


_ANALYSIS_CONTEXT: ContextVar[_AnalysisContext | None] = ContextVar(
    "sql_guard_analysis_context",
    default=None,
)


_UNKNOWN_SCALAR_VALUE = object()
_SQL_NULL_SCALAR_VALUE = object()


_MYSQL_VERSION_COMMENT_RE = re.compile(r"/\*!")
_NUMERIC_PREFIX_RE = re.compile(r"[+-]?(?:(?:\d+\.\d*|\d+|\.\d+)(?:[eE][+-]?\d+)?)")


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


def _expression_source_text(
    expr: exp.Expression,
    source_sql: str | None,
) -> str | None:
    """Return raw SQL source text for an expression when positions are available."""
    if source_sql is None:
        return None

    start = expr.meta.get("start")
    end = expr.meta.get("end")

    if not isinstance(start, int) or not isinstance(end, int):
        return None

    if start < 0 or end < start:
        return None

    if end >= len(source_sql):
        return None

    return source_sql[start : end + 1]


def _expressions_have_matching_sql(
    left: exp.Expression | None,
    right: exp.Expression | None,
) -> bool:
    """Return True when expressions serialize to matching SQL."""
    if not isinstance(left, exp.Expression) or not isinstance(right, exp.Expression):
        return False

    try:
        return left.sql() == right.sql()
    except Exception:
        return False


def _is_row_stable_expression(expr: exp.Expression | None) -> bool:
    """Return True when expression value is row-stable (no funcs/subqueries)."""
    if not isinstance(expr, exp.Expression):
        return False

    for node in expr.walk():
        if isinstance(
            node,
            (exp.Func, exp.Subquery, exp.Select, exp.Union, exp.Except, exp.Intersect),
        ):
            return False

    return True


def _consume_analysis_budget(steps: int = 1) -> None:
    """Consume analysis steps; raise when validation budget is exhausted."""
    context = _ANALYSIS_CONTEXT.get()
    if context is None:
        return

    context.remaining_steps -= steps
    if context.remaining_steps < 0:
        raise _AnalysisBudgetExceeded()


@contextmanager
def _analysis_session(max_steps: int | None = None):
    """Run validation under a bounded analysis budget and per-call caches."""
    if max_steps is None:
        max_steps = ANALYSIS_BUDGET_MAX_STEPS

    token = _ANALYSIS_CONTEXT.set(
        _AnalysisContext(
            remaining_steps=max_steps,
            predicate_truthiness_cache={},
            simplify_gate_cache={},
        )
    )
    try:
        yield
    finally:
        _ANALYSIS_CONTEXT.reset(token)


def _predicate_expression_complexity(
    expression: exp.Expression,
) -> tuple[int, int, int]:
    """Return (node_count, max_depth, boolean_operator_count)."""
    node_count = 0
    max_depth = 0
    boolean_operator_count = 0
    stack: list[tuple[exp.Expression, int]] = [(expression, 1)]

    while stack:
        node, depth = stack.pop()
        _consume_analysis_budget()

        node_count += 1
        if depth > max_depth:
            max_depth = depth

        if isinstance(node, (exp.And, exp.Or, exp.Not, exp.If, exp.Case, exp.Coalesce)):
            boolean_operator_count += 1

        child_depth = depth + 1
        for argument in node.args.values():
            if isinstance(argument, exp.Expression):
                stack.append((argument, child_depth))
            elif isinstance(argument, list):
                for list_item in argument:
                    if isinstance(list_item, exp.Expression):
                        stack.append((list_item, child_depth))

        if (
            node_count > PREDICATE_SIMPLIFY_MAX_NODES
            or max_depth > PREDICATE_SIMPLIFY_MAX_DEPTH
            or boolean_operator_count > PREDICATE_SIMPLIFY_MAX_BOOLEAN_OPERATORS
        ):
            return node_count, max_depth, boolean_operator_count

    return node_count, max_depth, boolean_operator_count


def _should_attempt_predicate_simplify(expression: exp.Expression) -> bool:
    """Return True when predicate is small enough for sqlglot simplify."""
    context = _ANALYSIS_CONTEXT.get()
    cache_key = id(expression)

    if context is not None and cache_key in context.simplify_gate_cache:
        return context.simplify_gate_cache[cache_key]

    node_count, max_depth, boolean_operator_count = _predicate_expression_complexity(
        expression
    )
    should_simplify = (
        node_count <= PREDICATE_SIMPLIFY_MAX_NODES
        and max_depth <= PREDICATE_SIMPLIFY_MAX_DEPTH
        and boolean_operator_count <= PREDICATE_SIMPLIFY_MAX_BOOLEAN_OPERATORS
    )

    if context is not None:
        context.simplify_gate_cache[cache_key] = should_simplify

    return should_simplify


def _numeric_literal_value(
    expr: exp.Expression | None,
    dialect: str,
    source_sql: str | None = None,
) -> Decimal | None:
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
        if dialect not in HEX_NUMERIC_LITERAL_DIALECTS:
            return None

        if dialect == "sqlite":
            raw_literal = _expression_source_text(expr, source_sql)
            if raw_literal is None:
                return None

            normalized_raw = raw_literal.strip().lower()
            # In SQLite, x'..' is a BLOB literal and should not be treated as a
            # numeric hex constant in predicate truthiness evaluation.
            if normalized_raw.startswith("x'"):
                return None
            if not normalized_raw.startswith("0x"):
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


def _numeric_string_value(value: str, dialect: str) -> Decimal | None:
    """Parse string literals to numeric values using dialect coercion rules."""
    stripped = value.strip()

    if dialect in NUMERIC_PREFIX_STRING_COERCION_DIALECTS:
        # MySQL/SQLite numeric coercion uses a leading numeric prefix.
        # If no numeric prefix exists, the coerced numeric value is 0.
        if not stripped:
            return Decimal(0)

        prefix_match = _NUMERIC_PREFIX_RE.match(stripped)
        if prefix_match is None:
            return Decimal(0)

        prefix = prefix_match.group(0)
        try:
            return Decimal(prefix)
        except (InvalidOperation, ValueError):
            return None

    if not stripped:
        return None

    try:
        return Decimal(stripped)
    except (InvalidOperation, ValueError):
        return None


def _duckdb_string_predicate_truthiness(value: str) -> bool | None:
    """Evaluate DuckDB string literals in predicate context when deterministic."""
    lowered = value.lower()
    if lowered in DUCKDB_TRUTHY_STRING_PREDICATES:
        return True
    if lowered in DUCKDB_FALSEY_STRING_PREDICATES:
        return False
    return None


def _postgres_string_boolean_cast_value(value: str) -> bool | None:
    """Evaluate PostgreSQL string-to-boolean casts when deterministic."""
    lowered = value.strip().lower()
    if lowered in POSTGRES_TRUTHY_BOOLEAN_CAST_STRINGS:
        return True
    if lowered in POSTGRES_FALSEY_BOOLEAN_CAST_STRINGS:
        return False
    return None


def _string_cast_scalar_value(
    inner_value: Decimal | str | bool | object,
    dialect: str,
) -> str | None:
    """Evaluate constant casts to string-like targets when deterministic."""
    if isinstance(inner_value, str):
        return inner_value

    if dialect not in {"duckdb", "mysql", "sqlite"}:
        return None

    if isinstance(inner_value, Decimal):
        return str(inner_value)

    if isinstance(inner_value, bool):
        if dialect == "duckdb":
            return "true" if inner_value else "false"

        return "1" if inner_value else "0"

    return None


def _cast_scalar_value(
    cast_expression: exp.Cast,
    dialect: str,
    source_sql: str | None = None,
) -> Decimal | str | bool | object:
    """Evaluate CAST/TRY_CAST for constants when semantics are explicit."""
    target_type = cast_expression.args.get("to")
    if not isinstance(target_type, exp.DataType):
        return _UNKNOWN_SCALAR_VALUE

    inner_value = _constant_scalar_value(cast_expression.this, dialect, source_sql)
    if inner_value is _UNKNOWN_SCALAR_VALUE:
        return _UNKNOWN_SCALAR_VALUE

    if inner_value is _SQL_NULL_SCALAR_VALUE:
        return _SQL_NULL_SCALAR_VALUE

    is_try_cast = isinstance(cast_expression, exp.TryCast) or bool(
        cast_expression.args.get("safe")
    )

    def cast_failed() -> object:
        if is_try_cast:
            return _SQL_NULL_SCALAR_VALUE
        return _UNKNOWN_SCALAR_VALUE

    is_numeric_target = target_type.is_type(
        "tinyint",
        "smallint",
        "int",
        "integer",
        "bigint",
        "ubigint",
        "decimal",
        "numeric",
        "float",
        "double",
        "real",
    )
    is_integer_target = target_type.is_type(
        "tinyint",
        "smallint",
        "int",
        "integer",
        "bigint",
        "ubigint",
    )

    if target_type.is_type("boolean", "bool"):
        if isinstance(inner_value, bool):
            return inner_value

        if isinstance(inner_value, Decimal):
            if dialect in NUMERIC_TRUTHY_PREDICATE_DIALECTS:
                return inner_value != 0
            return cast_failed()

        if isinstance(inner_value, str):
            if dialect == "duckdb":
                truthy = _duckdb_string_predicate_truthiness(inner_value)
                if truthy is not None:
                    return truthy
                return cast_failed()

            if dialect in {"mysql", "sqlite"}:
                numeric_value = _numeric_string_value(inner_value, dialect)
                if numeric_value is not None:
                    return numeric_value != 0

            if dialect == "postgres":
                boolean_value = _postgres_string_boolean_cast_value(inner_value)
                if boolean_value is not None:
                    return boolean_value

            return cast_failed()

        return cast_failed()

    if is_numeric_target:
        if isinstance(inner_value, Decimal):
            if is_integer_target and inner_value != inner_value.to_integral_value():
                return cast_failed()
            return inner_value

        if isinstance(inner_value, bool):
            if dialect in BOOL_NUMERIC_CROSS_TYPE_COMPARISON_DIALECTS:
                return Decimal(int(inner_value))
            return cast_failed()

        if isinstance(inner_value, str):
            numeric_value = _numeric_string_value(inner_value, dialect)
            if numeric_value is None:
                return cast_failed()
            if is_integer_target and numeric_value != numeric_value.to_integral_value():
                return cast_failed()
            return numeric_value

        return cast_failed()

    if target_type.is_type(
        "char",
        "nchar",
        "varchar",
        "nvarchar",
        "text",
        "string",
    ):
        string_value = _string_cast_scalar_value(inner_value, dialect)
        if string_value is not None:
            return string_value

        return cast_failed()

    return _UNKNOWN_SCALAR_VALUE


def _constant_scalar_value(
    expr: exp.Expression | None,
    dialect: str,
    source_sql: str | None = None,
) -> Decimal | str | bool | object:
    """Evaluate expression to a constant scalar when safely possible."""
    expr = _unwrap_parens(expr)
    if expr is None:
        return _UNKNOWN_SCALAR_VALUE

    if isinstance(expr, exp.Boolean):
        return bool(expr.this)

    if isinstance(expr, exp.Null):
        return _SQL_NULL_SCALAR_VALUE

    numeric_value = _numeric_literal_value(expr, dialect, source_sql)
    if numeric_value is not None:
        return numeric_value

    if isinstance(expr, exp.Literal) and expr.is_string:
        return str(expr.this)

    if isinstance(expr, (exp.Subquery, exp.Select)):
        return _constant_scalar_subquery_value(expr, dialect, source_sql)

    if isinstance(expr, exp.Cast):
        return _cast_scalar_value(expr, dialect, source_sql)

    if isinstance(expr, exp.Abs):
        inner = _constant_scalar_value(expr.this, dialect, source_sql)
        if isinstance(inner, Decimal):
            return abs(inner)
        return _UNKNOWN_SCALAR_VALUE

    if isinstance(expr, exp.Nullif):
        left_expr = expr.this
        right_expr = expr.expression

        if not isinstance(left_expr, exp.Expression) or not isinstance(
            right_expr,
            exp.Expression,
        ):
            return _UNKNOWN_SCALAR_VALUE

        left_value = _constant_scalar_value(left_expr, dialect, source_sql)
        right_value = _constant_scalar_value(right_expr, dialect, source_sql)

        if (
            left_value is not _UNKNOWN_SCALAR_VALUE
            and right_value is not _UNKNOWN_SCALAR_VALUE
        ):
            if right_value is _SQL_NULL_SCALAR_VALUE:
                return left_value

            if left_value is _SQL_NULL_SCALAR_VALUE:
                return _SQL_NULL_SCALAR_VALUE

            equals = _constant_scalar_equals(left_value, right_value, dialect)
            if equals is True:
                return _SQL_NULL_SCALAR_VALUE
            if equals is False:
                return left_value
            return _UNKNOWN_SCALAR_VALUE

        if right_value is not _UNKNOWN_SCALAR_VALUE:
            possible_values: set[Decimal | str | bool | object] = set()
            left_outcomes = _predicate_truthiness_possibilities(
                left_expr,
                dialect,
                source_sql,
            )

            for outcome in left_outcomes:
                if outcome is None:
                    possible_values.add(_SQL_NULL_SCALAR_VALUE)
                    continue

                if right_value is _SQL_NULL_SCALAR_VALUE:
                    possible_values.add(outcome)
                    continue

                equals = _constant_scalar_equals(outcome, right_value, dialect)
                if equals is True:
                    possible_values.add(_SQL_NULL_SCALAR_VALUE)
                elif equals is False:
                    possible_values.add(outcome)
                else:
                    possible_values.add(_UNKNOWN_SCALAR_VALUE)

            if _UNKNOWN_SCALAR_VALUE in possible_values:
                return _UNKNOWN_SCALAR_VALUE

            if len(possible_values) == 1:
                return next(iter(possible_values))

        if (
            _expressions_have_matching_sql(left_expr, right_expr)
            and _is_row_stable_expression(left_expr)
            and _is_row_stable_expression(right_expr)
        ):
            return _SQL_NULL_SCALAR_VALUE

        return _UNKNOWN_SCALAR_VALUE

    if isinstance(expr, exp.Coalesce):
        args: list[exp.Expression] = []
        first = expr.this
        if isinstance(first, exp.Expression):
            args.append(first)
        args.extend(expr.expressions)

        for arg in args:
            value = _constant_scalar_value(arg, dialect, source_sql)
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

    # Preserve SQL three-valued logic: NULL in predicate position is UNKNOWN,
    # not FALSE. This ensures NOT NULL remains UNKNOWN (non-tautological).
    if value is _SQL_NULL_SCALAR_VALUE:
        return None

    if isinstance(value, bool):
        return value

    if isinstance(value, Decimal):
        if dialect not in NUMERIC_TRUTHY_PREDICATE_DIALECTS:
            return None
        return value != 0

    if isinstance(value, str):
        if dialect == "duckdb":
            return _duckdb_string_predicate_truthiness(value)

        if dialect not in NUMERIC_TRUTHY_PREDICATE_DIALECTS:
            return None

        numeric_value = _numeric_string_value(value, dialect)
        if numeric_value is None:
            return None
        return numeric_value != 0

    return None


def _constant_scalar_equals(
    left: Decimal | str | bool | object,
    right: Decimal | str | bool | object,
    dialect: str,
) -> bool | None:
    """Compare constant scalar values with dialect-aware coercion where safe."""
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

    if dialect in BOOL_NUMERIC_CROSS_TYPE_COMPARISON_DIALECTS:
        if isinstance(left, bool) and isinstance(right, Decimal):
            return Decimal(int(left)) == right

        if isinstance(left, Decimal) and isinstance(right, bool):
            return left == Decimal(int(right))

    if dialect in BOOL_STRING_CROSS_TYPE_COMPARISON_DIALECTS:
        if isinstance(left, bool) and isinstance(right, str):
            right_bool = _duckdb_string_predicate_truthiness(right)
            if right_bool is not None:
                return left == right_bool
            return None

        if isinstance(left, str) and isinstance(right, bool):
            left_bool = _duckdb_string_predicate_truthiness(left)
            if left_bool is not None:
                return left_bool == right
            return None

    if dialect in BOOL_STRING_NUMERIC_COERCION_DIALECTS:
        if isinstance(left, bool) and isinstance(right, str):
            right_numeric = _numeric_string_value(right, dialect)
            if right_numeric is None:
                return None
            return Decimal(int(left)) == right_numeric

        if isinstance(left, str) and isinstance(right, bool):
            left_numeric = _numeric_string_value(left, dialect)
            if left_numeric is None:
                return None
            return left_numeric == Decimal(int(right))

    if dialect in NUMERIC_STRING_CROSS_TYPE_COMPARISON_DIALECTS:
        if isinstance(left, Decimal) and isinstance(right, str):
            right_numeric = _numeric_string_value(right, dialect)
            if right_numeric is None:
                return None
            return left == right_numeric

        if isinstance(left, str) and isinstance(right, Decimal):
            left_numeric = _numeric_string_value(left, dialect)
            if left_numeric is None:
                return None
            return left_numeric == right

    return None


def _comparison_decimal_operands(
    left: Decimal | str | bool | object,
    right: Decimal | str | bool | object,
    dialect: str,
) -> tuple[Decimal, Decimal] | None:
    """Coerce comparison operands to Decimals when semantics are clear."""

    def to_decimal(value: Decimal | str | bool | object) -> Decimal | None:
        if isinstance(value, Decimal):
            return value

        if dialect in BOOL_NUMERIC_CROSS_TYPE_COMPARISON_DIALECTS and isinstance(
            value, bool
        ):
            return Decimal(int(value))

        if dialect in NUMERIC_STRING_CROSS_TYPE_COMPARISON_DIALECTS and isinstance(
            value, str
        ):
            return _numeric_string_value(value, dialect)

        return None

    left_decimal = to_decimal(left)
    right_decimal = to_decimal(right)

    if left_decimal is None or right_decimal is None:
        return None

    return left_decimal, right_decimal


def _constant_comparison_predicate_truthiness(
    expr: exp.Expression,
    dialect: str,
    source_sql: str | None = None,
) -> bool | None:
    """Evaluate constant comparison predicates without relying on simplify()."""
    if not isinstance(
        expr,
        (
            exp.EQ,
            exp.NEQ,
            exp.NullSafeEQ,
            exp.NullSafeNEQ,
            exp.GT,
            exp.GTE,
            exp.LT,
            exp.LTE,
            exp.Is,
        ),
    ):
        return None

    if isinstance(expr, exp.Is):
        right_value = _constant_scalar_value(expr.expression, dialect, source_sql)
        if right_value is _UNKNOWN_SCALAR_VALUE:
            return None

        left_value = _constant_scalar_value(expr.this, dialect, source_sql)

        if right_value is _SQL_NULL_SCALAR_VALUE:
            if left_value is not _UNKNOWN_SCALAR_VALUE:
                return left_value is _SQL_NULL_SCALAR_VALUE

            left_truth = _constant_predicate_truthiness(expr.this, dialect, source_sql)
            if left_truth is None:
                return None
            return False

        if isinstance(right_value, bool):
            left_truth = _constant_predicate_truthiness(expr.this, dialect, source_sql)
            if left_truth is not None:
                return left_truth == right_value

            if left_value is _SQL_NULL_SCALAR_VALUE:
                return False
            if left_value is _UNKNOWN_SCALAR_VALUE:
                return None
            if isinstance(left_value, bool):
                return left_value == right_value

            scalar_truth = _predicate_truthiness_from_scalar(left_value, dialect)
            if scalar_truth is None:
                return None
            return scalar_truth == right_value

        return None

    left_value = _constant_scalar_value(expr.this, dialect, source_sql)
    right_value = _constant_scalar_value(expr.expression, dialect, source_sql)

    if left_value is _UNKNOWN_SCALAR_VALUE or right_value is _UNKNOWN_SCALAR_VALUE:
        return None

    if isinstance(expr, exp.NullSafeEQ):
        if (
            left_value is _SQL_NULL_SCALAR_VALUE
            and right_value is _SQL_NULL_SCALAR_VALUE
        ):
            return True
        if (
            left_value is _SQL_NULL_SCALAR_VALUE
            or right_value is _SQL_NULL_SCALAR_VALUE
        ):
            return False

        return _constant_scalar_equals(left_value, right_value, dialect)

    if isinstance(expr, exp.NullSafeNEQ):
        if (
            left_value is _SQL_NULL_SCALAR_VALUE
            and right_value is _SQL_NULL_SCALAR_VALUE
        ):
            return False
        if (
            left_value is _SQL_NULL_SCALAR_VALUE
            or right_value is _SQL_NULL_SCALAR_VALUE
        ):
            return True

        equals = _constant_scalar_equals(left_value, right_value, dialect)
        if equals is None:
            return None
        return not equals

    if left_value is _SQL_NULL_SCALAR_VALUE or right_value is _SQL_NULL_SCALAR_VALUE:
        return None

    if isinstance(expr, exp.EQ):
        return _constant_scalar_equals(left_value, right_value, dialect)

    if isinstance(expr, exp.NEQ):
        equals = _constant_scalar_equals(left_value, right_value, dialect)
        if equals is None:
            return None
        return not equals

    decimal_operands = _comparison_decimal_operands(left_value, right_value, dialect)
    if decimal_operands is None:
        return None

    left_decimal, right_decimal = decimal_operands

    if isinstance(expr, exp.GT):
        return left_decimal > right_decimal

    if isinstance(expr, exp.GTE):
        return left_decimal >= right_decimal

    if isinstance(expr, exp.LT):
        return left_decimal < right_decimal

    if isinstance(expr, exp.LTE):
        return left_decimal <= right_decimal

    return None


def _duckdb_bool_coercion_value(value: Decimal | str | bool | object) -> bool | None:
    """Coerce scalar values to DuckDB BOOLEAN where semantics are explicit."""
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return _duckdb_string_predicate_truthiness(value)

    return None


def _duckdb_boolean_common_type_in_truthiness(
    left_value: Decimal | str | bool | object,
    option_values: list[Decimal | str | bool | object],
) -> bool | None:
    """Evaluate DuckDB IN when mixed bool/string literals resolve via BOOLEAN."""
    if not any(isinstance(option, bool) for option in option_values):
        return None

    values = [left_value, *option_values]
    if any(not isinstance(value, (bool, str)) for value in values):
        return None

    coerced_values: list[bool] = []
    for value in values:
        coerced_value = _duckdb_bool_coercion_value(value)
        if coerced_value is None:
            return None
        coerced_values.append(coerced_value)

    return coerced_values[0] in coerced_values[1:]


def _duckdb_in_has_mixed_scalar_types(
    left_value: Decimal | str | bool | object,
    option_values: list[Decimal | str | bool | object],
) -> bool:
    """Return True when DuckDB IN operands mix scalar domains."""

    def scalar_type_tag(value: Decimal | str | bool | object) -> str:
        if isinstance(value, bool):
            return "bool"
        if isinstance(value, Decimal):
            return "decimal"
        if isinstance(value, str):
            return "string"
        return "other"

    type_tags = {scalar_type_tag(left_value)}
    type_tags.update(scalar_type_tag(option) for option in option_values)
    return len(type_tags) > 1


def _constant_in_subquery_option_values(
    query_expression: exp.Expression,
    dialect: str,
    source_sql: str | None = None,
) -> list[Decimal | str | bool | object] | None:
    """Return deterministic IN-subquery option values when safely foldable."""
    subquery = _subquery_select_expression(query_expression)
    if subquery is None:
        return None

    if len(subquery.expressions) != 1:
        return None

    row_count = _constant_select_without_from_row_count(
        subquery,
        dialect,
        source_sql,
    )
    if row_count is None:
        return None

    if row_count == 0:
        return []

    option_value = _constant_scalar_value(subquery.expressions[0], dialect, source_sql)
    if option_value is _UNKNOWN_SCALAR_VALUE:
        return None

    return [option_value]


def _constant_in_predicate_truthiness(
    expr: exp.In,
    dialect: str,
    source_sql: str | None = None,
) -> bool | None:
    """Evaluate constant IN predicates when possible."""
    left_value = _constant_scalar_value(expr.this, dialect, source_sql)
    if left_value in {_UNKNOWN_SCALAR_VALUE, _SQL_NULL_SCALAR_VALUE}:
        return None

    query_expression = expr.args.get("query")
    if isinstance(query_expression, exp.Expression):
        option_values = _constant_in_subquery_option_values(
            query_expression,
            dialect,
            source_sql,
        )
        if option_values is None:
            return None
    else:
        option_values = [
            _constant_scalar_value(option, dialect, source_sql)
            for option in expr.expressions
        ]

    saw_unknown = False

    for option_value in option_values:
        equals = _constant_scalar_equals(left_value, option_value, dialect)

        if equals is True:
            return True
        if equals is None:
            saw_unknown = True

    if dialect == "duckdb":
        duckdb_bool_truth = _duckdb_boolean_common_type_in_truthiness(
            left_value,
            option_values,
        )
        if duckdb_bool_truth is True:
            return True

    if saw_unknown:
        return None

    # DuckDB performs list-level coercion for mixed-type IN predicates. If we don't
    # have a positive match above, fail closed for mixed scalar domains instead of
    # returning False and risking tautology false negatives.
    if dialect == "duckdb" and _duckdb_in_has_mixed_scalar_types(
        left_value,
        option_values,
    ):
        return None

    return False


def _is_unbounded_limit_all(
    limit_expression: exp.Expression,
    dialect: str,
) -> bool:
    """Return True when LIMIT ALL is explicitly represented and unbounded."""
    if dialect not in LIMIT_ALL_UNBOUNDED_DIALECTS:
        return False

    expression = _unwrap_parens(limit_expression)

    if isinstance(expression, exp.Column):
        identifier = expression.this
        if not isinstance(identifier, exp.Identifier):
            return False
        if expression.table:
            return False
        return str(identifier.this).upper() == "ALL"

    if isinstance(expression, exp.Identifier):
        return str(expression.this).upper() == "ALL"

    if isinstance(expression, exp.Var):
        return str(expression.this).upper() == "ALL"

    return False


def _fetch_count_expression(fetch_clause: exp.Fetch) -> exp.Expression | None:
    """Return FETCH row-count expression when it is explicit or implicit."""
    fetch_count = fetch_clause.args.get("count")
    if isinstance(fetch_count, exp.Expression):
        normalized = _unwrap_parens(fetch_count)
        if isinstance(normalized, exp.Identifier):
            name = str(normalized.this).upper()
            if name in {"ROW", "ROWS"}:
                return exp.Literal.number("1")
        if isinstance(normalized, exp.Var):
            name = str(normalized.this).upper()
            if name in {"ROW", "ROWS"}:
                return exp.Literal.number("1")
        return fetch_count

    if fetch_count is None:
        return exp.Literal.number("1")

    return None


def _subquery_select_expression(expression: exp.Expression | None) -> exp.Select | None:
    """Return SELECT body from a scalar subquery expression when present."""
    expression = _unwrap_parens(expression)

    if isinstance(expression, exp.Subquery):
        expression = expression.this

    if isinstance(expression, exp.Select):
        return expression

    return None


def _has_set_returning_projection(
    select: exp.Select,
    dialect: str,
) -> bool:
    """Return True when a FROM-less SELECT projection may emit multiple rows."""
    for projection in select.expressions:
        _consume_analysis_budget()

        if not isinstance(projection, exp.Expression):
            continue

        for node in projection.walk():
            _consume_analysis_budget()

            if isinstance(node, SET_RETURNING_PROJECTION_NODE_TYPES):
                return True

            if dialect == "postgres" and isinstance(node, exp.Anonymous):
                function_name = _normalize_symbol(node.name or "")
                if function_name in POSTGRES_SET_RETURNING_ANONYMOUS_FUNCTIONS:
                    return True

    return False


def _has_global_aggregate_without_group(select: exp.Select) -> bool:
    """Return True when SELECT is a no-GROUP-BY aggregate query."""
    if select.args.get("group") is not None:
        return False

    def should_prune(node: exp.Expression) -> bool:
        return node is not select and isinstance(
            node,
            (exp.Subquery, exp.Select, exp.Union, exp.Except, exp.Intersect),
        )

    for node in select.walk(prune=should_prune):
        _consume_analysis_budget()

        if not isinstance(node, exp.AggFunc):
            continue

        # Windowed aggregates do not collapse row count into a single global row.
        parent = node.parent
        inside_window = False
        while isinstance(parent, exp.Expression) and parent is not select:
            _consume_analysis_budget()
            if isinstance(parent, exp.Window):
                inside_window = True
                break
            parent = parent.parent

        if not inside_window:
            return True

    return False


def _constant_select_without_from_row_count(
    select: exp.Select,
    dialect: str,
    source_sql: str | None = None,
) -> int | None:
    """Return deterministic row count (0 or 1) for simple SELECT without FROM."""
    _consume_analysis_budget()

    if select.args.get("from_") is not None or select.args.get("from") is not None:
        return None

    has_global_aggregate = _has_global_aggregate_without_group(select)

    where_clause = select.args.get("where")
    if isinstance(where_clause, exp.Where):
        where_truth = _constant_predicate_truthiness(
            where_clause.this,
            dialect,
            source_sql,
        )
        if where_truth is False and not has_global_aggregate:
            return 0
        if where_truth is None and not has_global_aggregate:
            return None

    having_clause = select.args.get("having")
    if isinstance(having_clause, exp.Having):
        having_truth = _constant_predicate_truthiness(
            having_clause.this,
            dialect,
            source_sql,
        )
        if having_truth is False:
            return 0
        if having_truth is None:
            return None

    has_set_returning_projection = _has_set_returning_projection(select, dialect)

    rows_eliminated = False
    has_uncertain_row_bound = False

    limit_clause = select.args.get("limit")
    limit_expression: exp.Expression | None = None
    if isinstance(limit_clause, exp.Limit):
        limit_expression = limit_clause.expression or limit_clause.this
    elif isinstance(limit_clause, exp.Fetch):
        limit_expression = _fetch_count_expression(limit_clause)

    if limit_expression is not None:
        if not _is_unbounded_limit_all(limit_expression, dialect):
            limit_value = _constant_scalar_value(limit_expression, dialect, source_sql)

            if isinstance(limit_value, Decimal):
                if limit_value != limit_value.to_integral_value():
                    has_uncertain_row_bound = True
                elif limit_value == 0:
                    rows_eliminated = True
                elif limit_value < 0 and dialect != "sqlite":
                    has_uncertain_row_bound = True
            elif limit_value is _SQL_NULL_SCALAR_VALUE:
                if dialect not in LIMIT_NULL_UNBOUNDED_DIALECTS:
                    has_uncertain_row_bound = True
            else:
                has_uncertain_row_bound = True
    elif limit_clause is not None:
        has_uncertain_row_bound = True

    offset_clause = select.args.get("offset")
    if isinstance(offset_clause, exp.Offset):
        offset_expression = offset_clause.expression or offset_clause.this
        offset_value = _constant_scalar_value(offset_expression, dialect, source_sql)

        if isinstance(offset_value, Decimal):
            if offset_value != offset_value.to_integral_value():
                has_uncertain_row_bound = True
            elif offset_value < 0:
                has_uncertain_row_bound = True
            elif offset_value > 0:
                if has_set_returning_projection:
                    has_uncertain_row_bound = True
                else:
                    rows_eliminated = True
        else:
            has_uncertain_row_bound = True

    if rows_eliminated:
        return 0

    if has_set_returning_projection:
        return None

    if has_uncertain_row_bound:
        return None

    return 1


def _constant_scalar_subquery_value(
    expression: exp.Expression,
    dialect: str,
    source_sql: str | None = None,
) -> Decimal | str | bool | object:
    """Evaluate constant scalar subqueries when result is deterministic."""
    subquery = _subquery_select_expression(expression)
    if subquery is None:
        return _UNKNOWN_SCALAR_VALUE

    if len(subquery.expressions) != 1:
        return _UNKNOWN_SCALAR_VALUE

    row_count = _constant_select_without_from_row_count(
        subquery,
        dialect,
        source_sql,
    )
    if row_count is None:
        return _UNKNOWN_SCALAR_VALUE

    if row_count == 0:
        return _SQL_NULL_SCALAR_VALUE

    return _constant_scalar_value(subquery.expressions[0], dialect, source_sql)


def _constant_exists_predicate_truthiness(
    expr: exp.Exists,
    dialect: str,
    source_sql: str | None = None,
) -> bool | None:
    """Evaluate simple row-invariant EXISTS predicates safely."""
    subquery = _subquery_select_expression(expr.this)
    if subquery is None:
        return None

    row_count = _constant_select_without_from_row_count(
        subquery,
        dialect,
        source_sql,
    )
    if row_count is None:
        return None

    return row_count > 0


def _negated_predicate_truthiness(value: bool | None) -> bool | None:
    """Return SQL three-valued NOT truthiness."""
    if value is None:
        return None
    return not value


def _and_predicate_truthiness(
    left: bool | None,
    right: bool | None,
) -> bool | None:
    """Return SQL three-valued AND truthiness."""
    if left is False or right is False:
        return False
    if left is True and right is True:
        return True
    return None


def _or_predicate_truthiness(
    left: bool | None,
    right: bool | None,
) -> bool | None:
    """Return SQL three-valued OR truthiness."""
    if left is True or right is True:
        return True
    if left is False and right is False:
        return False
    return None


def _self_comparison_predicate_truthiness_possibilities(
    expr: exp.Expression,
) -> set[bool | None] | None:
    """Return truthiness outcomes for self-comparisons (e.g. x = x)."""
    if not isinstance(
        expr,
        (
            exp.EQ,
            exp.NEQ,
            exp.NullSafeEQ,
            exp.NullSafeNEQ,
            exp.GT,
            exp.GTE,
            exp.LT,
            exp.LTE,
        ),
    ):
        return None

    left_expr = _unwrap_parens(expr.this)
    right_expr = _unwrap_parens(expr.expression)
    if left_expr is None or right_expr is None:
        return None

    try:
        if left_expr.sql() != right_expr.sql():
            return None
    except Exception:
        return None

    if not (
        _is_row_stable_expression(left_expr) and _is_row_stable_expression(right_expr)
    ):
        return None

    if isinstance(expr, exp.NullSafeEQ):
        return {True}

    if isinstance(expr, exp.NullSafeNEQ):
        return {False}

    if isinstance(expr, (exp.EQ, exp.GTE, exp.LTE)):
        return {True, None}

    if isinstance(expr, (exp.NEQ, exp.GT, exp.LT)):
        return {False, None}

    return None


def _self_comparison_nullable_true_operand(
    expr: exp.Expression | None,
) -> exp.Expression | None:
    """Return operand when expression can be TRUE/UNKNOWN via self-comparison."""
    if not isinstance(expr, (exp.EQ, exp.GTE, exp.LTE)):
        return None

    left_expr = _unwrap_parens(expr.this)
    right_expr = _unwrap_parens(expr.expression)
    if not isinstance(left_expr, exp.Expression) or not isinstance(
        right_expr,
        exp.Expression,
    ):
        return None

    if not _expressions_have_matching_sql(left_expr, right_expr):
        return None

    if not (
        _is_row_stable_expression(left_expr) and _is_row_stable_expression(right_expr)
    ):
        return None

    return left_expr


def _is_null_check_operand(expr: exp.Expression | None) -> exp.Expression | None:
    """Return checked operand for IS NULL expressions."""
    expr = _unwrap_parens(expr)
    if not isinstance(expr, exp.Is):
        return None

    right_expr = _unwrap_parens(expr.expression)
    if not isinstance(right_expr, exp.Null):
        return None

    operand = _unwrap_parens(expr.this)
    if isinstance(operand, exp.Expression):
        return operand

    return None


def _is_or_self_comparison_null_tautology(expr: exp.Or) -> bool:
    """Detect tautologies like x = x OR x IS NULL."""
    expression_pairs = (
        (expr.this, expr.expression),
        (expr.expression, expr.this),
    )

    for comparison_side, null_side in expression_pairs:
        comparison_operand = _self_comparison_nullable_true_operand(comparison_side)
        null_check_operand = _is_null_check_operand(null_side)

        if (
            comparison_operand is not None
            and null_check_operand is not None
            and _expressions_have_matching_sql(comparison_operand, null_check_operand)
        ):
            return True

    return False


def _is_boolean_is_operand(
    expr: exp.Expression | None,
    expected_boolean: bool,
    dialect: str,
    source_sql: str | None = None,
) -> exp.Expression | None:
    """Return IS operand for expressions like <expr> IS TRUE/FALSE."""
    expr = _unwrap_parens(expr)
    if not isinstance(expr, exp.Is):
        return None

    right_value = _constant_scalar_value(expr.expression, dialect, source_sql)
    if right_value is not expected_boolean:
        return None

    operand = _unwrap_parens(expr.this)
    if isinstance(operand, exp.Expression):
        return operand

    return None


def _is_boolean_is_not_operand(
    expr: exp.Expression | None,
    expected_boolean: bool,
    dialect: str,
    source_sql: str | None = None,
) -> exp.Expression | None:
    """Return IS operand for expressions like <expr> IS NOT TRUE/FALSE."""
    expr = _unwrap_parens(expr)
    if not isinstance(expr, exp.Not):
        return None

    inner = _unwrap_parens(expr.this)
    if not isinstance(inner, exp.Is):
        return None

    right_value = _constant_scalar_value(inner.expression, dialect, source_sql)
    if right_value is not expected_boolean:
        return None

    operand = _unwrap_parens(inner.this)
    if isinstance(operand, exp.Expression):
        return operand

    return None


def _expressions_are_logical_negations(
    left: exp.Expression | None,
    right: exp.Expression | None,
) -> bool:
    """Return True when expressions are syntactic negations of each other."""
    left = _unwrap_parens(left)
    right = _unwrap_parens(right)

    if not isinstance(left, exp.Expression) or not isinstance(right, exp.Expression):
        return False

    if isinstance(left, exp.Not):
        left_inner = _unwrap_parens(left.this)
        if _expressions_have_matching_sql(left_inner, right):
            return True

    if isinstance(right, exp.Not):
        right_inner = _unwrap_parens(right.this)
        if _expressions_have_matching_sql(right_inner, left):
            return True

    return False


def _null_safe_comparisons_are_complements(
    distinct_expression: exp.Expression | None,
    not_distinct_expression: exp.Expression | None,
) -> bool:
    """Return True for ``x IS DISTINCT FROM y`` OR ``x IS NOT DISTINCT FROM y``."""
    distinct_expression = _unwrap_parens(distinct_expression)
    not_distinct_expression = _unwrap_parens(not_distinct_expression)

    if not isinstance(distinct_expression, exp.NullSafeNEQ) or not isinstance(
        not_distinct_expression,
        exp.NullSafeEQ,
    ):
        return False

    distinct_left = _unwrap_parens(distinct_expression.this)
    distinct_right = _unwrap_parens(distinct_expression.expression)
    not_distinct_left = _unwrap_parens(not_distinct_expression.this)
    not_distinct_right = _unwrap_parens(not_distinct_expression.expression)

    if not isinstance(distinct_left, exp.Expression) or not isinstance(
        distinct_right,
        exp.Expression,
    ):
        return False

    if not isinstance(not_distinct_left, exp.Expression) or not isinstance(
        not_distinct_right,
        exp.Expression,
    ):
        return False

    same_order_match = _expressions_have_matching_sql(
        distinct_left,
        not_distinct_left,
    ) and _expressions_have_matching_sql(
        distinct_right,
        not_distinct_right,
    )

    swapped_order_match = _expressions_have_matching_sql(
        distinct_left,
        not_distinct_right,
    ) and _expressions_have_matching_sql(
        distinct_right,
        not_distinct_left,
    )

    return same_order_match or swapped_order_match


def _is_or_boolean_partition_tautology(
    expr: exp.Or,
    dialect: str,
    source_sql: str | None = None,
) -> bool:
    """Detect OR tautologies built from boolean partitions of same predicate."""
    left_expr = expr.this
    right_expr = expr.expression

    if _null_safe_comparisons_are_complements(
        left_expr,
        right_expr,
    ) or _null_safe_comparisons_are_complements(
        right_expr,
        left_expr,
    ):
        # p IS DISTINCT FROM q OR p IS NOT DISTINCT FROM q
        return True

    def is_non_nullable_boolean(predicate: exp.Expression) -> bool:
        outcomes = _predicate_truthiness_possibilities(predicate, dialect, source_sql)
        return None not in outcomes

    if _expressions_are_logical_negations(left_expr, right_expr):
        left_normalized = _unwrap_parens(left_expr)
        right_normalized = _unwrap_parens(right_expr)

        negated_operand: exp.Expression | None = None
        if isinstance(left_normalized, exp.Not):
            negated_operand = _unwrap_parens(left_normalized.this)
        elif isinstance(right_normalized, exp.Not):
            negated_operand = _unwrap_parens(right_normalized.this)

        if isinstance(negated_operand, exp.Expression) and is_non_nullable_boolean(
            negated_operand,
        ):
            # p OR NOT p  => tautological when p cannot be NULL.
            return True

    expression_pairs = (
        (left_expr, right_expr),
        (right_expr, left_expr),
    )

    for predicate_side, is_false_side in expression_pairs:
        false_operand = _is_boolean_is_operand(
            is_false_side,
            False,
            dialect,
            source_sql,
        )
        if false_operand is None:
            continue

        if _expressions_have_matching_sql(
            predicate_side, false_operand
        ) and is_non_nullable_boolean(false_operand):
            # p OR p IS FALSE  => tautological when p cannot be NULL.
            return True

    left_true_operand = _is_boolean_is_operand(
        left_expr,
        True,
        dialect,
        source_sql,
    )
    right_false_operand = _is_boolean_is_operand(
        right_expr,
        False,
        dialect,
        source_sql,
    )
    if (
        left_true_operand is not None
        and right_false_operand is not None
        and _expressions_have_matching_sql(left_true_operand, right_false_operand)
        and is_non_nullable_boolean(left_true_operand)
    ):
        # p IS TRUE OR p IS FALSE
        return True

    left_false_operand = _is_boolean_is_operand(
        left_expr,
        False,
        dialect,
        source_sql,
    )
    right_true_operand = _is_boolean_is_operand(
        right_expr,
        True,
        dialect,
        source_sql,
    )
    if (
        left_false_operand is not None
        and right_true_operand is not None
        and _expressions_have_matching_sql(left_false_operand, right_true_operand)
        and is_non_nullable_boolean(left_false_operand)
    ):
        return True

    for expected_boolean in (False, True):
        left_is_not_operand = _is_boolean_is_not_operand(
            left_expr,
            expected_boolean,
            dialect,
            source_sql,
        )
        right_is_not_operand = _is_boolean_is_not_operand(
            right_expr,
            expected_boolean,
            dialect,
            source_sql,
        )

        if (
            left_is_not_operand is not None
            and right_is_not_operand is not None
            and _expressions_are_logical_negations(
                left_is_not_operand,
                right_is_not_operand,
            )
        ):
            # p IS NOT b OR (NOT p) IS NOT b is always TRUE.
            return True

    return False


def _is_predicate_truthiness_possibilities(
    expr: exp.Expression,
    dialect: str,
    source_sql: str | None = None,
) -> set[bool | None] | None:
    """Return truthiness outcomes for IS predicates when determinable."""
    if not isinstance(expr, exp.Is):
        return None

    right_value = _constant_scalar_value(expr.expression, dialect, source_sql)
    if right_value is _UNKNOWN_SCALAR_VALUE:
        return None

    left_value = _constant_scalar_value(expr.this, dialect, source_sql)

    if right_value is _SQL_NULL_SCALAR_VALUE:
        if left_value is not _UNKNOWN_SCALAR_VALUE:
            return {left_value is _SQL_NULL_SCALAR_VALUE}

        left_outcomes = _predicate_truthiness_possibilities(
            expr.this,
            dialect,
            source_sql,
        )
        outcomes: set[bool | None] = set()
        for outcome in left_outcomes:
            outcomes.add(outcome is None)

        if not outcomes:
            fallback_outcomes: set[bool | None] = {True, False}
            return fallback_outcomes

        return outcomes

    if isinstance(right_value, bool):
        if left_value is not _UNKNOWN_SCALAR_VALUE:
            if left_value is _SQL_NULL_SCALAR_VALUE:
                return {False}

            if isinstance(left_value, bool):
                return {left_value == right_value}

            scalar_truth = _predicate_truthiness_from_scalar(left_value, dialect)
            if scalar_truth is None:
                return {True, False}

            return {scalar_truth == right_value}

        left_outcomes = _predicate_truthiness_possibilities(
            expr.this,
            dialect,
            source_sql,
        )
        outcomes: set[bool | None] = set()
        for outcome in left_outcomes:
            if outcome is None:
                outcomes.add(False)
            else:
                outcomes.add(outcome == right_value)

        if not outcomes:
            fallback_outcomes: set[bool | None] = {True, False}
            return fallback_outcomes

        return outcomes

    return None


def _if_predicate_truthiness_possibilities(
    expr: exp.If,
    dialect: str,
    source_sql: str | None = None,
) -> set[bool | None]:
    """Evaluate IF(condition, true_value, false_value) in predicate context."""
    condition_outcomes = _predicate_truthiness_possibilities(
        expr.this,
        dialect,
        source_sql,
    )

    true_expression = expr.args.get("true")
    if isinstance(true_expression, exp.Expression):
        true_outcomes = _predicate_truthiness_possibilities(
            true_expression,
            dialect,
            source_sql,
        )
    else:
        true_outcomes = {None}

    false_expression = expr.args.get("false")
    if isinstance(false_expression, exp.Expression):
        false_outcomes = _predicate_truthiness_possibilities(
            false_expression,
            dialect,
            source_sql,
        )
    else:
        false_outcomes = {None}

    outcomes: set[bool | None] = set()
    if True in condition_outcomes:
        outcomes.update(true_outcomes)

    # MySQL IF() treats NULL conditions as false branch.
    if False in condition_outcomes or None in condition_outcomes:
        outcomes.update(false_outcomes)

    if not outcomes:
        outcomes.add(None)

    return outcomes


def _case_predicate_truthiness_possibilities(
    expr: exp.Case,
    dialect: str,
    source_sql: str | None = None,
) -> set[bool | None]:
    """Evaluate CASE outcomes in predicate context when safely possible."""
    outcomes: set[bool | None] = set()
    can_fallthrough = True

    case_operand = expr.args.get("this")
    if_clauses = [
        if_clause
        for if_clause in (expr.args.get("ifs") or [])
        if isinstance(if_clause, exp.If)
    ]

    for if_clause in if_clauses:
        if not can_fallthrough:
            break

        condition_expression = if_clause.this
        if isinstance(case_operand, exp.Expression) and isinstance(
            condition_expression,
            exp.Expression,
        ):
            condition_expression = exp.EQ(
                this=case_operand.copy(),
                expression=condition_expression.copy(),
            )

        if isinstance(condition_expression, exp.Expression):
            condition_outcomes = _predicate_truthiness_possibilities(
                condition_expression,
                dialect,
                source_sql,
            )
        else:
            condition_outcomes = {True, False, None}

        result_expression = if_clause.args.get("true")
        if isinstance(result_expression, exp.Expression):
            result_outcomes = _predicate_truthiness_possibilities(
                result_expression,
                dialect,
                source_sql,
            )
        else:
            result_outcomes = {None}

        if True in condition_outcomes:
            outcomes.update(result_outcomes)

        can_fallthrough = can_fallthrough and (
            False in condition_outcomes or None in condition_outcomes
        )

    default_expression = expr.args.get("default")
    if can_fallthrough:
        if isinstance(default_expression, exp.Expression):
            outcomes.update(
                _predicate_truthiness_possibilities(
                    default_expression,
                    dialect,
                    source_sql,
                )
            )
        else:
            outcomes.add(None)

    if not outcomes:
        outcomes.add(None)

    return outcomes


def _coalesce_arguments(expr: exp.Coalesce) -> list[exp.Expression]:
    """Return COALESCE arguments as normalized expression list."""
    args: list[exp.Expression] = []

    first = expr.this
    if isinstance(first, exp.Expression):
        args.append(first)

    args.extend(
        expression
        for expression in expr.expressions
        if isinstance(expression, exp.Expression)
    )

    return args


def _coalesce_predicate_truthiness_possibilities(
    expr: exp.Coalesce,
    dialect: str,
    source_sql: str | None = None,
) -> set[bool | None]:
    """Evaluate COALESCE outcomes in predicate context when safely possible."""
    args = _coalesce_arguments(expr)

    if not args:
        return {None}

    outcomes: set[bool | None] = set()
    can_fallthrough = True

    for arg in args:
        if not can_fallthrough:
            break

        arg_outcomes = _predicate_truthiness_possibilities(arg, dialect, source_sql)
        outcomes.update(value for value in arg_outcomes if value is not None)

        can_fallthrough = can_fallthrough and (None in arg_outcomes)

    if can_fallthrough:
        outcomes.add(None)

    if not outcomes:
        outcomes.add(None)

    return outcomes


def _predicate_truthiness_possibilities_uncached(
    expr: exp.Expression | None,
    dialect: str,
    source_sql: str | None = None,
) -> set[bool | None]:
    """Return possible truthiness outcomes in predicate position (uncached)."""
    if expr is None:
        return {True}

    if isinstance(expr, exp.Not):
        return {
            _negated_predicate_truthiness(value)
            for value in _predicate_truthiness_possibilities(
                expr.this,
                dialect,
                source_sql,
            )
        }

    if isinstance(expr, exp.And):
        left_outcomes = _predicate_truthiness_possibilities(
            expr.this,
            dialect,
            source_sql,
        )
        right_outcomes = _predicate_truthiness_possibilities(
            expr.expression,
            dialect,
            source_sql,
        )
        return {
            _and_predicate_truthiness(left, right)
            for left in left_outcomes
            for right in right_outcomes
        }

    if isinstance(expr, exp.Or):
        if _is_or_self_comparison_null_tautology(expr):
            return {True}

        if _is_or_boolean_partition_tautology(expr, dialect, source_sql):
            return {True}

        left_outcomes = _predicate_truthiness_possibilities(
            expr.this,
            dialect,
            source_sql,
        )
        right_outcomes = _predicate_truthiness_possibilities(
            expr.expression,
            dialect,
            source_sql,
        )
        return {
            _or_predicate_truthiness(left, right)
            for left in left_outcomes
            for right in right_outcomes
        }

    if isinstance(expr, exp.If):
        return _if_predicate_truthiness_possibilities(expr, dialect, source_sql)

    if isinstance(expr, exp.Case):
        return _case_predicate_truthiness_possibilities(expr, dialect, source_sql)

    if isinstance(expr, exp.Coalesce):
        return _coalesce_predicate_truthiness_possibilities(expr, dialect, source_sql)

    comparison_truth = _constant_comparison_predicate_truthiness(
        expr,
        dialect,
        source_sql,
    )
    if comparison_truth is not None:
        return {comparison_truth}

    is_predicate_outcomes = _is_predicate_truthiness_possibilities(
        expr,
        dialect,
        source_sql,
    )
    if is_predicate_outcomes is not None:
        return is_predicate_outcomes

    self_comparison_outcomes = _self_comparison_predicate_truthiness_possibilities(expr)
    if self_comparison_outcomes is not None:
        return self_comparison_outcomes

    if isinstance(expr, exp.In):
        in_truth = _constant_in_predicate_truthiness(expr, dialect, source_sql)
        if in_truth is not None:
            return {in_truth}

    if isinstance(expr, exp.Exists):
        exists_truth = _constant_exists_predicate_truthiness(expr, dialect, source_sql)
        if exists_truth is not None:
            return {exists_truth}

        # EXISTS/NOT EXISTS are SQL booleans and never evaluate to NULL.
        return {True, False}

    scalar_value = _constant_scalar_value(expr, dialect, source_sql)
    if scalar_value is _SQL_NULL_SCALAR_VALUE:
        return {None}

    scalar_truth = _predicate_truthiness_from_scalar(scalar_value, dialect)
    if scalar_truth is not None:
        return {scalar_truth}

    return {True, False, None}


def _predicate_truthiness_possibilities(
    expr: exp.Expression | None,
    dialect: str,
    source_sql: str | None = None,
) -> set[bool | None]:
    """Return possible truthiness outcomes in predicate position."""
    _consume_analysis_budget()

    expr = _unwrap_parens(expr)
    if expr is None:
        return {True}

    context = _ANALYSIS_CONTEXT.get()
    source_key = id(source_sql) if source_sql is not None else 0
    cache_key = (id(expr), dialect, source_key)

    if context is not None:
        cached = context.predicate_truthiness_cache.get(cache_key)
        if cached is not None:
            return cached

    outcomes = _predicate_truthiness_possibilities_uncached(expr, dialect, source_sql)

    if context is not None:
        context.predicate_truthiness_cache[cache_key] = outcomes

    return outcomes


def _constant_predicate_truthiness(
    expr: exp.Expression | None,
    dialect: str,
    source_sql: str | None = None,
) -> bool | None:
    """Evaluate obvious constant truthiness in boolean predicate position."""
    outcomes = _predicate_truthiness_possibilities(expr, dialect, source_sql)

    if outcomes == {True}:
        return True

    if outcomes == {False}:
        return False

    return None


def _is_tautological_where(
    where: exp.Where,
    dialect: str = "ansi",
    source_sql: str | None = None,
) -> bool:
    """Return True when a WHERE predicate is effectively always true."""
    predicate = _unwrap_parens(where.this)
    if predicate is None:
        return True

    direct_truth = _constant_predicate_truthiness(predicate, dialect, source_sql)
    if direct_truth is True:
        return True
    if direct_truth is False:
        return False

    simplified_predicate = _simplified_predicate_expression(predicate)
    if simplified_predicate is predicate:
        return False

    return (
        _constant_predicate_truthiness(
            simplified_predicate,
            dialect,
            source_sql,
        )
        is True
    )


def _mutation_target_symbols(node: exp.Expression) -> set[str]:
    """Collect normalized symbols that can reference the mutation target table.

    Intentionally conservative: for MySQL DELETE ... USING forms, aliases in
    ``node.args['using']`` are not promoted to target symbols here.
    This may reject some restrictive queries that correlate through USING-only
    aliases, but keeps fail-closed behavior until alias-to-target mapping for
    multi-table delete forms is modeled explicitly.
    """
    target = node.args.get("this")
    if not isinstance(target, exp.Table):
        return set()

    symbols: set[str] = set()

    alias_or_name = target.alias_or_name
    if alias_or_name:
        symbols.add(_normalize_symbol(alias_or_name))

    table_name = target.name
    if table_name:
        symbols.add(_normalize_symbol(table_name))

    return symbols


def _local_symbols_in_select(select: exp.Select) -> set[str]:
    """Collect local source symbols visible in a SELECT scope."""
    symbols: set[str] = set()

    def add_source_symbol(source: exp.Expression | None) -> None:
        if source is None:
            return

        alias_or_name = getattr(source, "alias_or_name", "") or ""
        if alias_or_name:
            symbols.add(_normalize_symbol(alias_or_name))

        if isinstance(source, exp.Table):
            table_name = source.name
            if table_name:
                symbols.add(_normalize_symbol(table_name))

    from_clause = select.args.get("from_")
    if isinstance(from_clause, exp.From):
        add_source_symbol(from_clause.this)

    joins = select.args.get("joins") or []
    for join in joins:
        if isinstance(join, exp.Join):
            add_source_symbol(join.this)

    return symbols


def _expression_references_target_symbols(
    expression: exp.Expression | None,
    target_symbols: set[str],
    local_symbols: set[str],
    allow_unqualified_outer: bool = False,
) -> bool:
    """Return True when an expression references outer mutation target symbols."""
    if expression is None or not target_symbols:
        return False

    def should_prune(node: exp.Expression) -> bool:
        return isinstance(
            node,
            (exp.Subquery, exp.Select, exp.Union, exp.Except, exp.Intersect),
        )

    for node in expression.walk(prune=should_prune):
        _consume_analysis_budget()

        if not isinstance(node, exp.Column):
            continue

        table_symbol = _normalize_symbol(node.table)
        if not table_symbol:
            if allow_unqualified_outer:
                return True
            continue

        if table_symbol in local_symbols:
            continue

        if table_symbol in target_symbols:
            return True

    return False


def _expression_references_target_symbols_reachable(
    expression: exp.Expression | None,
    target_symbols: set[str],
    local_symbols: set[str],
    dialect: str,
    source_sql: str | None,
    allow_unqualified_outer: bool,
) -> bool:
    """Return True when reachable evaluation paths reference target symbols."""
    _consume_analysis_budget()

    expression = _unwrap_parens(expression)
    if expression is None:
        return False

    if isinstance(expression, exp.If):
        case_parent = (
            expression.parent if isinstance(expression.parent, exp.Case) else None
        )
        if isinstance(case_parent, exp.Case) and expression.arg_key == "ifs":
            condition_outcomes = _case_when_condition_outcomes(
                case_parent,
                expression,
                dialect,
                source_sql,
            )
        else:
            condition_outcomes = _predicate_truthiness_possibilities(
                expression.this,
                dialect,
                source_sql,
            )

        if _expression_references_target_symbols_reachable(
            expression.this,
            target_symbols,
            local_symbols,
            dialect,
            source_sql,
            allow_unqualified_outer,
        ):
            return True

        true_branch = expression.args.get("true")
        if (
            True in condition_outcomes
            and _expression_references_target_symbols_reachable(
                true_branch,
                target_symbols,
                local_symbols,
                dialect,
                source_sql,
                allow_unqualified_outer,
            )
        ):
            return True

        false_branch = expression.args.get("false")
        if (
            False in condition_outcomes or None in condition_outcomes
        ) and _expression_references_target_symbols_reachable(
            false_branch,
            target_symbols,
            local_symbols,
            dialect,
            source_sql,
            allow_unqualified_outer,
        ):
            return True

        return False

    if isinstance(expression, exp.Case):
        case_operand = expression.args.get("this")
        if _expression_references_target_symbols_reachable(
            case_operand,
            target_symbols,
            local_symbols,
            dialect,
            source_sql,
            allow_unqualified_outer,
        ):
            return True

        can_fallthrough = True
        for if_clause in _case_if_clauses(expression):
            if not can_fallthrough:
                break

            condition_expression = _case_when_condition_expression(
                expression,
                if_clause,
            )
            if _expression_references_target_symbols_reachable(
                condition_expression,
                target_symbols,
                local_symbols,
                dialect,
                source_sql,
                allow_unqualified_outer,
            ):
                return True

            condition_outcomes = _case_when_condition_outcomes(
                expression,
                if_clause,
                dialect,
                source_sql,
            )
            result_expression = if_clause.args.get("true")
            if (
                True in condition_outcomes
                and _expression_references_target_symbols_reachable(
                    result_expression,
                    target_symbols,
                    local_symbols,
                    dialect,
                    source_sql,
                    allow_unqualified_outer,
                )
            ):
                return True

            can_fallthrough = can_fallthrough and (
                False in condition_outcomes or None in condition_outcomes
            )

        if can_fallthrough and _expression_references_target_symbols_reachable(
            expression.args.get("default"),
            target_symbols,
            local_symbols,
            dialect,
            source_sql,
            allow_unqualified_outer,
        ):
            return True

        return False

    if isinstance(expression, exp.Coalesce):
        can_fallthrough = True
        for argument in _coalesce_arguments(expression):
            if not can_fallthrough:
                break

            if _expression_references_target_symbols_reachable(
                argument,
                target_symbols,
                local_symbols,
                dialect,
                source_sql,
                allow_unqualified_outer,
            ):
                return True

            argument_outcomes = _predicate_truthiness_possibilities(
                argument,
                dialect,
                source_sql,
            )
            can_fallthrough = None in argument_outcomes

        return False

    return _expression_references_target_symbols(
        expression,
        target_symbols,
        local_symbols,
        allow_unqualified_outer,
    )


def _simplified_predicate_expression(expression: exp.Expression) -> exp.Expression:
    """Return simplified predicate when sqlglot optimizer is available and safe."""
    normalized = _unwrap_parens(expression)
    if isinstance(normalized, exp.Expression):
        expression = normalized

    if not _should_attempt_predicate_simplify(expression):
        return expression

    try:
        from sqlglot.optimizer.simplify import simplify

        simplified = _unwrap_parens(simplify(expression.copy()))
        if isinstance(simplified, exp.Expression):
            return simplified
    except Exception:
        pass

    return expression


def _predicate_effectively_references_target_symbols_impl(
    expression: exp.Expression | None,
    target_symbols: set[str],
    local_symbols: set[str],
    dialect: str,
    source_sql: str | None,
    allow_unqualified_outer: bool,
) -> bool:
    """Return True when target refs constrain predicate truthiness."""
    _consume_analysis_budget()

    expression = _unwrap_parens(expression)
    if expression is None:
        return False

    truthiness_outcomes = _predicate_truthiness_possibilities(
        expression,
        dialect,
        source_sql,
    )

    if truthiness_outcomes == {True} or truthiness_outcomes == {False}:
        return False

    if isinstance(expression, exp.Not):
        return _predicate_effectively_references_target_symbols_impl(
            expression.this,
            target_symbols,
            local_symbols,
            dialect,
            source_sql,
            allow_unqualified_outer,
        )

    if isinstance(expression, exp.And):
        left_expr = expression.this
        right_expr = expression.expression

        left_truth = _constant_predicate_truthiness(left_expr, dialect, source_sql)
        right_truth = _constant_predicate_truthiness(right_expr, dialect, source_sql)

        left_effective = _predicate_effectively_references_target_symbols_impl(
            left_expr,
            target_symbols,
            local_symbols,
            dialect,
            source_sql,
            allow_unqualified_outer,
        )
        right_effective = _predicate_effectively_references_target_symbols_impl(
            right_expr,
            target_symbols,
            local_symbols,
            dialect,
            source_sql,
            allow_unqualified_outer,
        )

        if left_truth is False or right_truth is False:
            return False
        if left_truth is True:
            return right_effective
        if right_truth is True:
            return left_effective

        return left_effective or right_effective

    if isinstance(expression, exp.Or):
        left_expr = expression.this
        right_expr = expression.expression

        left_truth = _constant_predicate_truthiness(left_expr, dialect, source_sql)
        right_truth = _constant_predicate_truthiness(right_expr, dialect, source_sql)

        left_effective = _predicate_effectively_references_target_symbols_impl(
            left_expr,
            target_symbols,
            local_symbols,
            dialect,
            source_sql,
            allow_unqualified_outer,
        )
        right_effective = _predicate_effectively_references_target_symbols_impl(
            right_expr,
            target_symbols,
            local_symbols,
            dialect,
            source_sql,
            allow_unqualified_outer,
        )

        if left_truth is True or right_truth is True:
            return False
        if left_truth is False:
            return right_effective
        if right_truth is False:
            return left_effective

        # OR is row-restrictive only when both sides are row-restrictive.
        return left_effective and right_effective

    # Fallback correlation requires potential row filtering behavior:
    # the predicate must be able to evaluate TRUE for some rows and non-TRUE
    # (FALSE/UNKNOWN) for others.
    if True not in truthiness_outcomes:
        return False

    if False not in truthiness_outcomes and None not in truthiness_outcomes:
        return False

    return _expression_references_target_symbols_reachable(
        expression,
        target_symbols,
        local_symbols,
        dialect,
        source_sql,
        allow_unqualified_outer,
    )


def _predicate_effectively_references_target_symbols(
    expression: exp.Expression | None,
    target_symbols: set[str],
    local_symbols: set[str],
    dialect: str,
    source_sql: str | None,
    allow_unqualified_outer: bool = False,
) -> bool:
    """Return True when target refs effectively constrain predicate truthiness."""
    expression = _unwrap_parens(expression)
    if expression is None:
        return False

    direct_effective = _predicate_effectively_references_target_symbols_impl(
        expression,
        target_symbols,
        local_symbols,
        dialect,
        source_sql,
        allow_unqualified_outer,
    )

    if not direct_effective:
        return False

    # Normalize only when needed and only for manageable predicates. This keeps
    # tautology-neutralization behavior while avoiding unconditional simplify cost.
    normalized_expression = _simplified_predicate_expression(expression)
    if normalized_expression is expression:
        return True

    return _predicate_effectively_references_target_symbols_impl(
        normalized_expression,
        target_symbols,
        local_symbols,
        dialect,
        source_sql,
        allow_unqualified_outer,
    )


def _is_exists_subquery_correlated_to_target(
    subquery: exp.Select,
    target_symbols: set[str],
    dialect: str,
    source_sql: str | None,
) -> bool:
    """Return True when EXISTS depends on target row filters (not projection)."""
    if not target_symbols:
        return False

    local_symbols = _local_symbols_in_select(subquery)

    has_from_clause = (
        subquery.args.get("from_") is not None or subquery.args.get("from") is not None
    )
    has_join_sources = any(
        isinstance(join, exp.Join) for join in (subquery.args.get("joins") or [])
    )
    allow_unqualified_outer = not has_from_clause and not has_join_sources

    where_clause = subquery.args.get("where")
    if isinstance(
        where_clause, exp.Where
    ) and _predicate_effectively_references_target_symbols(
        where_clause.this,
        target_symbols,
        local_symbols,
        dialect,
        source_sql,
        allow_unqualified_outer,
    ):
        return True

    having_clause = subquery.args.get("having")
    if isinstance(
        having_clause, exp.Having
    ) and _predicate_effectively_references_target_symbols(
        having_clause.this,
        target_symbols,
        local_symbols,
        dialect,
        source_sql,
        allow_unqualified_outer,
    ):
        return True

    joins = subquery.args.get("joins") or []
    for join in joins:
        if not isinstance(join, exp.Join):
            continue

        on_expression = join.args.get("on")
        if isinstance(
            on_expression, exp.Expression
        ) and _predicate_effectively_references_target_symbols(
            on_expression,
            target_symbols,
            local_symbols,
            dialect,
            source_sql,
            allow_unqualified_outer,
        ):
            return True

    return False


def _case_if_clauses(case_expression: exp.Case) -> list[exp.If]:
    """Return CASE ``WHEN`` clauses normalized as sqlglot ``exp.If`` nodes."""
    return [
        if_clause
        for if_clause in (case_expression.args.get("ifs") or [])
        if isinstance(if_clause, exp.If)
    ]


def _case_when_condition_expression(
    case_expression: exp.Case,
    if_clause: exp.If,
) -> exp.Expression | None:
    """Build CASE ``WHEN`` condition as a boolean predicate expression."""
    when_expression = if_clause.this
    if not isinstance(when_expression, exp.Expression):
        return None

    case_operand = case_expression.args.get("this")
    if isinstance(case_operand, exp.Expression):
        # Simple CASE: ``CASE x WHEN y THEN ...`` maps to ``x = y``.
        return exp.EQ(this=case_operand.copy(), expression=when_expression.copy())

    # Searched CASE: ``CASE WHEN predicate THEN ...``.
    return when_expression


def _case_when_condition_outcomes(
    case_expression: exp.Case,
    if_clause: exp.If,
    dialect: str,
    source_sql: str | None,
) -> set[bool | None]:
    """Return possible truthiness outcomes for a CASE ``WHEN`` condition."""
    condition_expression = _case_when_condition_expression(case_expression, if_clause)
    if not isinstance(condition_expression, exp.Expression):
        return {True, False, None}

    return _predicate_truthiness_possibilities(
        condition_expression,
        dialect,
        source_sql,
    )


def _has_definitely_true_prior_case_clause(
    case_expression: exp.Case,
    stop_before: exp.If | None,
    dialect: str,
    source_sql: str | None,
) -> bool:
    """Return True when an earlier CASE ``WHEN`` is guaranteed to match."""
    for if_clause in _case_if_clauses(case_expression):
        if stop_before is not None and if_clause is stop_before:
            break

        if _case_when_condition_outcomes(
            case_expression,
            if_clause,
            dialect,
            source_sql,
        ) == {True}:
            return True

    return False


def _is_reachable_case_child(
    case_expression: exp.Case,
    child: exp.Expression,
    dialect: str,
    source_sql: str | None,
) -> bool:
    """Return True when a direct CASE child can execute."""
    case_operand = case_expression.args.get("this")
    if child is case_operand:
        # Simple CASE operand must always be evaluated first.
        return True

    for if_clause in _case_if_clauses(case_expression):
        if child is if_clause:
            return not _has_definitely_true_prior_case_clause(
                case_expression,
                if_clause,
                dialect,
                source_sql,
            )

    default_expression = case_expression.args.get("default")
    if child is default_expression:
        return not _has_definitely_true_prior_case_clause(
            case_expression,
            stop_before=None,
            dialect=dialect,
            source_sql=source_sql,
        )

    return True


def _is_reachable_predicate_branch(
    expression: exp.Expression,
    root_expression: exp.Expression,
    dialect: str,
    source_sql: str | None,
) -> bool:
    """Return True when expression branch is reachable from boolean context."""
    node: exp.Expression = expression

    while True:
        _consume_analysis_budget()

        parent = node.parent
        if not isinstance(parent, exp.Expression):
            return True

        if isinstance(parent, exp.And):
            sibling: exp.Expression | None = None
            if node is parent.this:
                sibling = _unwrap_parens(parent.expression)
            elif node is parent.expression:
                sibling = _unwrap_parens(parent.this)

            sibling_truth = _constant_predicate_truthiness(sibling, dialect, source_sql)
            if sibling_truth is False:
                return False

        elif isinstance(parent, exp.Or):
            sibling = None
            if node is parent.this:
                sibling = _unwrap_parens(parent.expression)
            elif node is parent.expression:
                sibling = _unwrap_parens(parent.this)

            sibling_truth = _constant_predicate_truthiness(sibling, dialect, source_sql)
            if sibling_truth is True:
                return False

        elif isinstance(parent, exp.If):
            case_parent = parent.parent if isinstance(parent.parent, exp.Case) else None
            if isinstance(case_parent, exp.Case) and parent.arg_key == "ifs":
                condition_outcomes = _case_when_condition_outcomes(
                    case_parent,
                    parent,
                    dialect,
                    source_sql,
                )
            else:
                condition_outcomes = _predicate_truthiness_possibilities(
                    parent.this,
                    dialect,
                    source_sql,
                )

            true_branch = parent.args.get("true")
            false_branch = parent.args.get("false")

            if node is true_branch and True not in condition_outcomes:
                return False

            if node is false_branch and (
                False not in condition_outcomes and None not in condition_outcomes
            ):
                return False

        elif isinstance(parent, exp.Coalesce):
            node_index: int | None = None
            coalesce_arguments = _coalesce_arguments(parent)
            for index, argument in enumerate(coalesce_arguments):
                if node is argument:
                    node_index = index
                    break

            if node_index is not None:
                for previous_argument in coalesce_arguments[:node_index]:
                    previous_outcomes = _predicate_truthiness_possibilities(
                        previous_argument,
                        dialect,
                        source_sql,
                    )
                    if None not in previous_outcomes:
                        return False

        elif isinstance(parent, exp.Case) and not _is_reachable_case_child(
            parent,
            node,
            dialect,
            source_sql,
        ):
            return False

        if node is root_expression:
            return True

        node = parent


def _has_uncorrelated_exists_subquery(
    where: exp.Where,
    target_symbols: set[str],
    dialect: str,
    source_sql: str | None,
) -> bool:
    """Check for EXISTS/NOT EXISTS subqueries that are global to all target rows.

    Note: This is intentionally conservative. Any reachable uncorrelated
    EXISTS/NOT EXISTS subquery is rejected, including FROM-less SELECT forms,
    even when other sibling predicates may be row-restrictive. Non-SELECT EXISTS
    bodies (VALUES, set operations, etc.) are rejected because row-correlation
    is not analyzed for those forms here. Traversal is intentionally global
    across nested subquery contexts (fail closed).
    """
    root_expression = _unwrap_parens(where.this)
    if not isinstance(root_expression, exp.Expression):
        return False

    for exists_predicate in where.find_all(exp.Exists):
        _consume_analysis_budget()

        if not _is_reachable_predicate_branch(
            exists_predicate,
            root_expression,
            dialect,
            source_sql,
        ):
            continue

        subquery = exists_predicate.this
        if isinstance(subquery, exp.Subquery):
            subquery = subquery.this

        # Conservative handling: non-SELECT EXISTS bodies (VALUES, set operations,
        # etc.) are not analyzed for row-correlation here and are rejected.
        if not isinstance(subquery, exp.Select):
            return True

        exists_truth = _constant_exists_predicate_truthiness(
            exists_predicate,
            dialect,
            source_sql,
        )

        # Keep behavior predictable with the tautology checker:
        # - EXISTS known FALSE is non-mutating and should not be rejected here.
        # - EXISTS known TRUE is handled by tautological WHERE detection, which
        #   still blocks full-table mutations while allowing restrictive AND forms.
        if exists_truth is False or exists_truth is True:
            continue

        if not _is_exists_subquery_correlated_to_target(
            subquery,
            target_symbols,
            dialect,
            source_sql,
        ):
            # Policy note: stay fail-closed and reject any reachable uncorrelated
            # EXISTS/NOT EXISTS, even when sibling predicates may be restrictive.
            return True

    return False


def has_unfiltered_mutation(
    stmt: exp.Expression,
    dialect: str = "ansi",
    source_sql: str | None = None,
) -> str | None:
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

            target_symbols = _mutation_target_symbols(node)

            if isinstance(where, exp.Where) and _is_tautological_where(
                where,
                dialect,
                source_sql,
            ):
                return (
                    "UPDATE with tautological WHERE clause is not allowed "
                    "(would affect all rows)"
                )

            if isinstance(where, exp.Where) and _has_uncorrelated_exists_subquery(
                where,
                target_symbols,
                dialect,
                source_sql,
            ):
                return (
                    "UPDATE with uncorrelated EXISTS subquery is not allowed "
                    "(predicate is not row-restrictive)"
                )

        if isinstance(node, exp.Delete):
            where = node.args.get("where")
            if not where:
                return (
                    "DELETE without WHERE clause is not allowed (would affect all rows)"
                )

            target_symbols = _mutation_target_symbols(node)

            if isinstance(where, exp.Where) and _is_tautological_where(
                where,
                dialect,
                source_sql,
            ):
                return (
                    "DELETE with tautological WHERE clause is not allowed "
                    "(would affect all rows)"
                )

            if isinstance(where, exp.Where) and _has_uncorrelated_exists_subquery(
                where,
                target_symbols,
                dialect,
                source_sql,
            ):
                return (
                    "DELETE with uncorrelated EXISTS subquery is not allowed "
                    "(predicate is not row-restrictive)"
                )

    return None


def has_prohibited_nodes(
    stmt: exp.Expression,
    allow_dangerous: bool = False,
    dialect: str = "ansi",
    source_sql: str | None = None,
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
        reason = has_unfiltered_mutation(stmt, dialect, source_sql)
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
    reason = has_prohibited_nodes(stmt, dialect=dialect, source_sql=sql)
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

    try:
        with _analysis_session():
            # Enforce function-level sandbox in dangerous mode too
            reason = has_dangerous_functions(stmt, dialect)
            if reason:
                return GuardResult(False, reason)

            # Enforce always-blocked operations and lock/SELECT INTO checks
            reason = has_prohibited_nodes(
                stmt,
                allow_dangerous=True,
                dialect=dialect,
                source_sql=sql,
            )
            if reason:
                return GuardResult(False, reason)

            # Strict fail-closed statement policy in dangerous mode
            reason = has_disallowed_dangerous_mode_statement(stmt)
            if reason:
                return GuardResult(False, reason)
    except _AnalysisBudgetExceeded:
        return GuardResult(
            False,
            "Query is too complex to validate safely (analysis budget exceeded)",
        )
    except RecursionError:
        return GuardResult(
            False,
            "Query is too complex to validate safely "
            "(analysis recursion limit reached)",
        )

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
