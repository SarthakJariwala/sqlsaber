"""Compatibility contract for the SQL guard package boundary."""

import inspect
from dataclasses import MISSING, fields

import pytest
import sqlglot

import sqlsaber.tools.sql_guard as sql_guard
import sqlsaber.tools.sql_guard._limits as limits
import sqlsaber.tools.sql_guard._mutation_analysis as mutation_analysis
import sqlsaber.tools.sql_guard._resource_policy as resource_policy
import sqlsaber.tools.sql_guard._statement_policy as statement_policy
from sqlsaber.tools.sql_guard import (
    GuardResult,
    _analysis_session,
    _ANALYSIS_CONTEXT,
    _predicate_truthiness_possibilities,
    _should_attempt_predicate_simplify,
    add_limit,
    has_prohibited_nodes,
    validate_read_only,
    validate_sql,
)

PUBLIC_SYMBOLS = {
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
}

PRIVATE_COMPATIBILITY_SYMBOLS = {
    "_AnalysisBudgetExceeded",
    "_ANALYSIS_CONTEXT",
    "_analysis_session",
    "_predicate_truthiness_possibilities",
    "_should_attempt_predicate_simplify",
}

OWNER_MODULES = (
    limits,
    mutation_analysis,
    resource_policy,
    statement_policy,
)


def test_existing_public_imports_remain_available():
    assert len(PUBLIC_SYMBOLS) == 55
    assert PUBLIC_SYMBOLS == set(sql_guard.__all__)
    assert not (PUBLIC_SYMBOLS - set(vars(sql_guard)))


def test_historic_wildcard_import_set_remains_exact():
    namespace: dict[str, object] = {}
    exec("from sqlsaber.tools.sql_guard import *", namespace)
    namespace.pop("__builtins__", None)

    assert set(namespace) == PUBLIC_SYMBOLS
    assert namespace["vacuum_type"] is statement_policy.vacuum_type


@pytest.mark.parametrize("name", sorted(PUBLIC_SYMBOLS | PRIVATE_COMPATIBILITY_SYMBOLS))
def test_facade_rebinding_propagates_to_every_owner_and_restores(name, monkeypatch):
    owners = tuple(module for module in OWNER_MODULES if name in vars(module))
    original_facade_value = getattr(sql_guard, name)
    original_owner_values = {module: getattr(module, name) for module in owners}
    replacement = object()

    with monkeypatch.context() as patch:
        patch.setattr(sql_guard, name, replacement)

        assert getattr(sql_guard, name) is replacement
        assert all(getattr(module, name) is replacement for module in owners)

    assert getattr(sql_guard, name) is original_facade_value
    assert all(
        getattr(module, name) is original_value
        for module, original_value in original_owner_values.items()
    )


def test_injected_id_propagates_and_is_removed_from_facade_and_owner(monkeypatch):
    assert "id" not in vars(sql_guard)
    assert "id" not in vars(mutation_analysis)

    def replacement(_value: object) -> int:
        return 1

    with monkeypatch.context() as patch:
        patch.setattr(sql_guard, "id", replacement, raising=False)

        assert sql_guard.id is replacement
        assert mutation_analysis.id is replacement

    assert "id" not in vars(sql_guard)
    assert "id" not in vars(mutation_analysis)


def test_facade_rebinding_changes_preimported_mutation_policy_behavior(monkeypatch):
    statement = sqlglot.parse_one("DELETE FROM users WHERE id = 1", read="postgres")

    def replacement(*_args: object) -> str:
        return "facade mutation replacement"

    with monkeypatch.context() as patch:
        patch.setattr(sql_guard, "has_unfiltered_mutation", replacement)

        assert (
            has_prohibited_nodes(statement, allow_dangerous=True)
            == "facade mutation replacement"
        )


def test_facade_rebinding_changes_preimported_add_limit_behavior(monkeypatch):
    sql = "SELECT * FROM users"
    assert "LIMIT 7" in add_limit(sql, "postgres", limit=7)

    with monkeypatch.context() as patch:
        patch.setattr(sql_guard, "has_limit_clause", lambda _statement: True)

        assert add_limit(sql, "postgres", limit=7) == sql


def test_test_used_private_imports_remain_aliases_of_their_owner():
    assert _ANALYSIS_CONTEXT is mutation_analysis._ANALYSIS_CONTEXT
    assert _analysis_session is mutation_analysis._analysis_session
    assert (
        _predicate_truthiness_possibilities
        is mutation_analysis._predicate_truthiness_possibilities
    )
    assert (
        _should_attempt_predicate_simplify
        is mutation_analysis._should_attempt_predicate_simplify
    )


def test_guard_result_field_order_and_defaults_remain_stable():
    assert [(field.name, field.default) for field in fields(GuardResult)] == [
        ("allowed", MISSING),
        ("reason", None),
        ("is_select", False),
        ("query_type", None),
        ("has_limit", False),
    ]
    assert GuardResult(True) == GuardResult(True, None, False, None, False)


def test_core_function_signatures_remain_stable():
    assert str(inspect.signature(validate_read_only)) == (
        "(sql: str, dialect: str = 'ansi') -> sqlsaber.tools.sql_guard.GuardResult"
    )
    assert str(inspect.signature(validate_sql)) == (
        "(sql: str, dialect: str = 'ansi', allow_dangerous: bool = False) "
        "-> sqlsaber.tools.sql_guard.GuardResult"
    )
    assert str(inspect.signature(add_limit)) == (
        "(sql: str, dialect: str = 'ansi', limit: int = 100) -> str"
    )
    assert validate_read_only.__module__ == "sqlsaber.tools.sql_guard"
    assert validate_sql.__module__ == "sqlsaber.tools.sql_guard"
