"""SQL query validation and security using sqlglot AST analysis."""

import re
import sys as _sys
from collections.abc import Callable as Callable
from contextlib import contextmanager as contextmanager
from contextvars import ContextVar as ContextVar
from dataclasses import dataclass
from decimal import (
    ROUND_DOWN as ROUND_DOWN,
    ROUND_HALF_UP as ROUND_HALF_UP,
    Decimal as Decimal,
    InvalidOperation as InvalidOperation,
)
from types import ModuleType as _ModuleType
from typing import cast

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from ._limits import add_limit, has_limit_clause
from ._mutation_analysis import (
    ANALYSIS_BUDGET_MAX_STEPS,
    BOOL_NUMERIC_CROSS_TYPE_COMPARISON_DIALECTS,
    BOOL_STRING_CROSS_TYPE_COMPARISON_DIALECTS,
    BOOL_STRING_NUMERIC_COERCION_DIALECTS,
    DUCKDB_FALSEY_STRING_PREDICATES,
    DUCKDB_TRUTHY_STRING_PREDICATES,
    HEX_NUMERIC_LITERAL_DIALECTS,
    INTEGER_CAST_HALF_AWAY_FROM_ZERO_DIALECTS,
    INTEGER_CAST_TRUNCATE_DIALECTS,
    LIMIT_ALL_UNBOUNDED_DIALECTS,
    LIMIT_NULL_UNBOUNDED_DIALECTS,
    NUMERIC_PREFIX_STRING_COERCION_DIALECTS,
    NUMERIC_STRING_CROSS_TYPE_COMPARISON_DIALECTS,
    NUMERIC_TRUTHY_PREDICATE_DIALECTS,
    POSTGRES_FALSEY_BOOLEAN_CAST_STRINGS,
    POSTGRES_SET_RETURNING_ANONYMOUS_FUNCTIONS,
    POSTGRES_TRUTHY_BOOLEAN_CAST_STRINGS,
    PREDICATE_SIMPLIFY_MAX_BOOLEAN_OPERATORS,
    PREDICATE_SIMPLIFY_MAX_DEPTH,
    PREDICATE_SIMPLIFY_MAX_NODES,
    SET_RETURNING_PROJECTION_NODE_TYPES,
    _AnalysisBudgetExceeded,
    _ANALYSIS_CONTEXT as _ANALYSIS_CONTEXT,
    _analysis_session,
    _predicate_truthiness_possibilities as _predicate_truthiness_possibilities,
    _should_attempt_predicate_simplify as _should_attempt_predicate_simplify,
    has_unfiltered_mutation,
)
from ._resource_policy import (
    DANGEROUS_FUNCTION_PREFIXES_BY_DIALECT,
    DANGEROUS_FUNCTIONS_BY_DIALECT,
    FILE_PATH_DATA_EXTENSIONS,
    FILE_PATH_TARGET_DIALECTS,
    has_dangerous_functions,
    has_file_path_target,
)
from ._statement_policy import (
    ALLOWED_DANGEROUS_CREATE_KINDS,
    ALWAYS_BLOCKED_NODES,
    DANGEROUS_ALLOWED_ROOT_NODES,
    WRITE_DML_DDL_NODES,
    classify_statement,
    has_disallowed_dangerous_mode_statement,
    has_prohibited_nodes,
    is_select_like,
    vacuum_type,
)
from . import (
    _limits as _limits_module,
    _mutation_analysis as _mutation_analysis_module,
    _resource_policy as _resource_policy_module,
    _statement_policy as _statement_policy_module,
)

__all__ = [
    "ALLOWED_DANGEROUS_CREATE_KINDS",
    "ALWAYS_BLOCKED_NODES",
    "ANALYSIS_BUDGET_MAX_STEPS",
    "BOOL_NUMERIC_CROSS_TYPE_COMPARISON_DIALECTS",
    "BOOL_STRING_CROSS_TYPE_COMPARISON_DIALECTS",
    "BOOL_STRING_NUMERIC_COERCION_DIALECTS",
    "Callable",
    "ContextVar",
    "DANGEROUS_ALLOWED_ROOT_NODES",
    "DANGEROUS_FUNCTION_PREFIXES_BY_DIALECT",
    "DANGEROUS_FUNCTIONS_BY_DIALECT",
    "DUCKDB_FALSEY_STRING_PREDICATES",
    "DUCKDB_TRUTHY_STRING_PREDICATES",
    "Decimal",
    "FILE_PATH_DATA_EXTENSIONS",
    "FILE_PATH_TARGET_DIALECTS",
    "GuardResult",
    "HEX_NUMERIC_LITERAL_DIALECTS",
    "INTEGER_CAST_HALF_AWAY_FROM_ZERO_DIALECTS",
    "INTEGER_CAST_TRUNCATE_DIALECTS",
    "InvalidOperation",
    "LIMIT_ALL_UNBOUNDED_DIALECTS",
    "LIMIT_NULL_UNBOUNDED_DIALECTS",
    "NUMERIC_PREFIX_STRING_COERCION_DIALECTS",
    "NUMERIC_STRING_CROSS_TYPE_COMPARISON_DIALECTS",
    "NUMERIC_TRUTHY_PREDICATE_DIALECTS",
    "POSTGRES_FALSEY_BOOLEAN_CAST_STRINGS",
    "POSTGRES_SET_RETURNING_ANONYMOUS_FUNCTIONS",
    "POSTGRES_TRUTHY_BOOLEAN_CAST_STRINGS",
    "PREDICATE_SIMPLIFY_MAX_BOOLEAN_OPERATORS",
    "PREDICATE_SIMPLIFY_MAX_DEPTH",
    "PREDICATE_SIMPLIFY_MAX_NODES",
    "ParseError",
    "ROUND_DOWN",
    "ROUND_HALF_UP",
    "SET_RETURNING_PROJECTION_NODE_TYPES",
    "WRITE_DML_DDL_NODES",
    "add_limit",
    "cast",
    "classify_statement",
    "contextmanager",
    "dataclass",
    "exp",
    "has_dangerous_functions",
    "has_disallowed_dangerous_mode_statement",
    "has_file_path_target",
    "has_limit_clause",
    "has_prohibited_nodes",
    "has_unfiltered_mutation",
    "is_select_like",
    "re",
    "sqlglot",
    "validate_read_only",
    "validate_sql",
    "vacuum_type",
]

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

    raw_stmt = statements[0]
    if raw_stmt is None:
        return GuardResult(False, "Unable to parse query - empty statement")
    stmt = cast(exp.Expression, raw_stmt)

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

    # Block file-path/URL/glob table references (DuckDB replacement scans)
    reason = has_file_path_target(stmt, dialect)
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

    raw_stmt = statements[0]
    if raw_stmt is None:
        return GuardResult(False, "Unable to parse query - empty statement")
    stmt = cast(exp.Expression, raw_stmt)

    try:
        with _analysis_session():
            # Enforce function-level sandbox in dangerous mode too
            reason = has_dangerous_functions(stmt, dialect)
            if reason:
                return GuardResult(False, reason)

            # Block file-path/URL/glob table references (DuckDB replacement scans)
            reason = has_file_path_target(stmt, dialect)
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


_OWNER_MODULES: tuple[_ModuleType, ...] = (
    _limits_module,
    _mutation_analysis_module,
    _resource_policy_module,
    _statement_policy_module,
)
_COMPATIBILITY_NAMES = frozenset(__all__) | {
    "_AnalysisBudgetExceeded",
    "_ANALYSIS_CONTEXT",
    "_analysis_session",
    "_predicate_truthiness_possibilities",
    "_should_attempt_predicate_simplify",
    "id",
}
_REBINDING_TARGETS: dict[str, tuple[_ModuleType, ...]] = {
    name: tuple(module for module in _OWNER_MODULES if name in vars(module))
    for name in _COMPATIBILITY_NAMES
}
_REBINDING_TARGETS["id"] = (_mutation_analysis_module,)


class _SqlGuardFacade(_ModuleType):
    """Keep facade rebindings authoritative across split owner modules."""

    def __setattr__(self, name: str, value: object) -> None:
        super().__setattr__(name, value)
        for owner in _REBINDING_TARGETS.get(name, ()):
            setattr(owner, name, value)

    def __delattr__(self, name: str) -> None:
        super().__delattr__(name)
        for owner in _REBINDING_TARGETS.get(name, ()):
            if name in vars(owner):
                delattr(owner, name)


setattr(_sys.modules[__name__], "__class__", _SqlGuardFacade)
