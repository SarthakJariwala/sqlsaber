"""Unfiltered-mutation analysis tests."""

import sys

import sqlglot
from sqlglot import exp

import sqlsaber.tools.sql_guard as sql_guard
import sqlsaber.tools.sql_guard._mutation_analysis as mutation_analysis_module
from sqlsaber.tools.sql_guard import _should_attempt_predicate_simplify, validate_sql


class TestUnfilteredMutationsInDangerousMode:
    def test_update_without_where_blocked_in_dangerous_mode(self):
        """UPDATE without WHERE should be blocked in dangerous mode."""
        result = validate_sql(
            "UPDATE users SET active = false", "postgres", allow_dangerous=True
        )
        assert not result.allowed
        assert result.reason
        assert "UPDATE without WHERE" in result.reason

    def test_delete_without_where_blocked_in_dangerous_mode(self):
        """DELETE without WHERE should be blocked in dangerous mode."""
        result = validate_sql("DELETE FROM users", "postgres", allow_dangerous=True)
        assert not result.allowed
        assert result.reason
        assert "DELETE without WHERE" in result.reason

    def test_update_with_where_true_blocked_in_dangerous_mode(self):
        """UPDATE with tautological WHERE TRUE should be blocked in dangerous mode."""
        result = validate_sql(
            "UPDATE users SET active = false WHERE TRUE",
            "postgres",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_update_with_where_one_equals_one_blocked_in_dangerous_mode(self):
        """UPDATE with tautological WHERE 1=1 should be blocked in dangerous mode."""
        result = validate_sql(
            "UPDATE users SET active = false WHERE 1 = 1",
            "postgres",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_delete_with_where_true_blocked_in_dangerous_mode(self):
        """DELETE with tautological WHERE TRUE should be blocked in dangerous mode."""
        result = validate_sql(
            "DELETE FROM users WHERE TRUE", "postgres", allow_dangerous=True
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_delete_with_where_one_equals_one_blocked_in_dangerous_mode(self):
        """DELETE with tautological WHERE 1=1 should be blocked in dangerous mode."""
        result = validate_sql(
            "DELETE FROM users WHERE 1 = 1", "postgres", allow_dangerous=True
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_update_with_nested_parenthesized_true_blocked_in_dangerous_mode(self):
        """Deeply parenthesized tautologies should still be blocked."""
        result = validate_sql(
            "UPDATE users SET active = false WHERE (((TRUE)))",
            "postgres",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_delete_with_nested_parenthesized_one_equals_one_blocked_in_dangerous_mode(
        self,
    ):
        """Deeply parenthesized 1=1 should still be blocked."""
        result = validate_sql(
            "DELETE FROM users WHERE ((((1 = 1))))",
            "postgres",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_update_with_tautological_or_clause_blocked_in_dangerous_mode(self):
        """Tautological OR should be treated as unfiltered mutation."""
        result = validate_sql(
            "UPDATE users SET active = false WHERE (1 = 1) OR id > 0",
            "postgres",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_update_with_tautological_and_filter_allowed_in_dangerous_mode(self):
        """A tautological AND with a real filter should remain allowed."""
        result = validate_sql(
            "UPDATE users SET active = false WHERE (1 = 1) AND id > 0",
            "postgres",
            allow_dangerous=True,
        )
        assert result.allowed
        assert result.query_type == "dml"

    def test_mysql_update_with_where_one_blocked_in_dangerous_mode(self):
        """MySQL truthy numeric predicates should be blocked."""
        result = validate_sql(
            "UPDATE users SET active = 0 WHERE 1", "mysql", allow_dangerous=True
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_mysql_delete_with_parenthesized_where_one_blocked_in_dangerous_mode(self):
        """MySQL parenthesized numeric truthy predicates should be blocked."""
        result = validate_sql(
            "DELETE FROM users WHERE (1)", "mysql", allow_dangerous=True
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_sqlite_delete_with_where_not_zero_blocked_in_dangerous_mode(self):
        """SQLite numeric boolean syntax like NOT 0 should be blocked."""
        result = validate_sql(
            "DELETE FROM users WHERE NOT 0", "sqlite", allow_dangerous=True
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_update_with_where_allowed_in_dangerous_mode(self):
        """UPDATE with WHERE should be allowed in dangerous mode."""
        result = validate_sql(
            "UPDATE users SET active = false WHERE id = 1",
            "postgres",
            allow_dangerous=True,
        )
        assert result.allowed
        assert result.query_type == "dml"

    def test_delete_with_where_allowed_in_dangerous_mode(self):
        """DELETE with WHERE should be allowed in dangerous mode."""
        result = validate_sql(
            "DELETE FROM users WHERE id = 1", "postgres", allow_dangerous=True
        )
        assert result.allowed
        assert result.query_type == "dml"


class TestDangerousModeTautologyHardening:
    """Additional tautology hardening tests for dangerous mode mutations.

    Regression note: keep EXISTS checks strictly fail-closed for any reachable
    uncorrelated EXISTS/NOT EXISTS, including nested helper EXISTS.
    """

    def test_update_with_constant_in_predicate_blocked(self):
        """Constant IN predicate evaluating to true should be blocked."""
        result = validate_sql(
            "UPDATE users SET active = false WHERE 1 IN (1)",
            "postgres",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_delete_with_constant_in_is_true_predicate_blocked(self):
        """IS TRUE wrappers over constant IN predicates should be blocked."""
        for dialect in ("postgres", "duckdb"):
            result = validate_sql(
                "DELETE FROM users WHERE (1 IN (1)) IS TRUE",
                dialect,
                allow_dangerous=True,
            )
            assert not result.allowed
            assert result.reason
            assert "tautological WHERE" in result.reason

    def test_delete_with_constant_exists_is_true_predicate_blocked(self):
        """IS TRUE wrappers over constant EXISTS predicates should be blocked."""
        for dialect in ("postgres", "duckdb"):
            result = validate_sql(
                "DELETE FROM users WHERE EXISTS (SELECT 1) IS TRUE",
                dialect,
                allow_dangerous=True,
            )
            assert not result.allowed
            assert result.reason
            assert "tautological WHERE" in result.reason

    def test_delete_with_constant_scalar_subquery_predicate_blocked(self):
        """Constant scalar subquery predicates should be blocked as tautological."""
        for dialect in ("postgres", "duckdb"):
            result = validate_sql(
                "DELETE FROM users WHERE (SELECT TRUE)",
                dialect,
                allow_dangerous=True,
            )
            assert not result.allowed
            assert result.reason
            assert "tautological WHERE" in result.reason

    def test_delete_with_constant_in_subquery_predicate_blocked(self):
        """Constant IN-subquery predicates should be blocked as tautological."""
        for dialect in ("postgres", "duckdb"):
            result = validate_sql(
                "DELETE FROM users WHERE 1 IN (SELECT 1)",
                dialect,
                allow_dangerous=True,
            )
            assert not result.allowed
            assert result.reason
            assert "tautological WHERE" in result.reason

    def test_mysql_delete_with_constant_in_cross_type_predicate_blocked(self):
        """MySQL numeric/string coercion in IN should be treated as tautological."""
        result = validate_sql(
            "DELETE FROM users WHERE 1 IN ('1')",
            "mysql",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_duckdb_delete_with_constant_in_cross_type_predicate_blocked(self):
        """DuckDB numeric/string coercion in IN should be treated as tautological."""
        result = validate_sql(
            "DELETE FROM users WHERE 1 IN ('1')",
            "duckdb",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_delete_with_constant_exists_predicate_blocked(self):
        """Constant EXISTS predicate evaluating to true should be blocked."""
        result = validate_sql(
            "DELETE FROM users WHERE EXISTS (SELECT 1)",
            "postgres",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_delete_with_tautological_case_predicate_blocked(self):
        """CASE predicates that are always TRUE should be blocked."""
        result = validate_sql(
            "DELETE FROM users u WHERE CASE WHEN u.id IS NULL THEN TRUE ELSE TRUE END",
            "postgres",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_delete_with_row_restrictive_case_predicate_allowed(self):
        """CASE predicates that can filter rows should remain allowed."""
        result = validate_sql(
            "DELETE FROM users u WHERE CASE WHEN u.id IS NULL THEN TRUE ELSE FALSE END",
            "postgres",
            allow_dangerous=True,
        )
        assert result.allowed
        assert result.query_type == "dml"

    def test_delete_with_tautological_coalesce_self_comparison_blocked(self):
        """COALESCE wrappers over x=x with TRUE fallback should be blocked."""
        result = validate_sql(
            "DELETE FROM users u WHERE COALESCE(u.id = u.id, TRUE)",
            "postgres",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_delete_with_tautological_nullif_is_null_blocked(self):
        """NULLIF(x, x) IS NULL should be blocked as tautological."""
        result = validate_sql(
            "DELETE FROM users u WHERE NULLIF(u.id, u.id) IS NULL",
            "postgres",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_delete_with_tautological_coalesce_nullif_predicate_blocked(self):
        """COALESCE(NULLIF(x, x) IS NULL, TRUE) should be blocked."""
        result = validate_sql(
            "DELETE FROM users u WHERE COALESCE(NULLIF(u.id, u.id) IS NULL, TRUE)",
            "postgres",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_delete_with_row_restrictive_nullif_predicate_allowed(self):
        """NULLIF predicates that can filter rows should remain allowed."""
        result = validate_sql(
            "DELETE FROM users u WHERE NULLIF(u.id, 0) IS NULL",
            "postgres",
            allow_dangerous=True,
        )
        assert result.allowed
        assert result.query_type == "dml"

    def test_delete_with_tautological_nullif_self_comparison_predicate_blocked(self):
        """NULLIF(x=x, TRUE) IS NULL should be blocked as tautological."""
        result = validate_sql(
            "DELETE FROM users u WHERE NULLIF(u.id = u.id, TRUE) IS NULL",
            "postgres",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_delete_with_row_restrictive_nullif_self_comparison_allowed(self):
        """NULLIF(x=x, TRUE) IS NOT NULL can still filter rows and should be allowed."""
        result = validate_sql(
            "DELETE FROM users u WHERE NULLIF(u.id = u.id, TRUE) IS NOT NULL",
            "postgres",
            allow_dangerous=True,
        )
        assert result.allowed
        assert result.query_type == "dml"

    def test_delete_with_volatile_self_comparison_nullif_allowed(self):
        """Volatile self-comparisons must not be treated as deterministic tautologies."""
        result = validate_sql(
            "DELETE FROM users WHERE NULLIF(RANDOM() = RANDOM(), TRUE) IS NULL",
            "postgres",
            allow_dangerous=True,
        )
        assert result.allowed
        assert result.query_type == "dml"

    def test_delete_with_self_equality_or_null_check_blocked(self):
        """x = x OR x IS NULL should be blocked as tautological."""
        result = validate_sql(
            "DELETE FROM users u WHERE u.id = u.id OR u.id IS NULL",
            "postgres",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_delete_with_self_equality_and_not_null_allowed(self):
        """x = x AND x IS NOT NULL still filters NULL rows and should be allowed."""
        result = validate_sql(
            "DELETE FROM users u WHERE u.id = u.id AND u.id IS NOT NULL",
            "postgres",
            allow_dangerous=True,
        )
        assert result.allowed
        assert result.query_type == "dml"

    def test_delete_with_self_equality_is_not_false_blocked(self):
        """(x = x) IS NOT FALSE should be blocked as tautological."""
        result = validate_sql(
            "DELETE FROM users u WHERE (u.id = u.id) IS NOT FALSE",
            "postgres",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_delete_with_self_inequality_is_not_true_blocked(self):
        """(x <> x) IS NOT TRUE should be blocked as tautological."""
        result = validate_sql(
            "DELETE FROM users u WHERE (u.id <> u.id) IS NOT TRUE",
            "postgres",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_delete_with_self_equality_is_true_allowed(self):
        """(x = x) IS TRUE can still filter NULL rows and should remain allowed."""
        result = validate_sql(
            "DELETE FROM users u WHERE (u.id = u.id) IS TRUE",
            "postgres",
            allow_dangerous=True,
        )
        assert result.allowed
        assert result.query_type == "dml"

    def test_delete_with_nonnullable_partition_or_predicate_blocked(self):
        """p OR p IS FALSE should be blocked when p is non-null boolean."""
        result = validate_sql(
            "DELETE FROM users u WHERE (u.id IS NULL) OR ((u.id IS NULL) IS FALSE)",
            "postgres",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_delete_with_nullable_partition_or_predicate_allowed(self):
        """p OR p IS FALSE should stay allowed when p can be UNKNOWN."""
        result = validate_sql(
            "DELETE FROM users u WHERE (u.id = u.id) OR ((u.id = u.id) IS FALSE)",
            "postgres",
            allow_dangerous=True,
        )
        assert result.allowed
        assert result.query_type == "dml"

    def test_delete_with_correlated_exists_or_not_exists_blocked(self):
        """EXISTS OR NOT EXISTS partitions should be blocked as tautological."""
        result = validate_sql(
            (
                "DELETE FROM users u WHERE "
                "EXISTS (SELECT 1 FROM audit a WHERE a.user_id = u.id) "
                "OR NOT EXISTS (SELECT 1 FROM audit a WHERE a.user_id = u.id)"
            ),
            "postgres",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_delete_with_correlated_exists_or_exists_is_false_blocked(self):
        """EXISTS OR EXISTS IS FALSE should be blocked as tautological."""
        result = validate_sql(
            (
                "DELETE FROM users u WHERE "
                "EXISTS (SELECT 1 FROM audit a WHERE a.user_id = u.id) "
                "OR (EXISTS (SELECT 1 FROM audit a WHERE a.user_id = u.id) IS FALSE)"
            ),
            "postgres",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_delete_with_values_exists_predicate_blocked(self):
        """EXISTS over VALUES should be blocked as non-row-restrictive."""
        result = validate_sql(
            "DELETE FROM users WHERE EXISTS (VALUES (1))",
            "postgres",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "uncorrelated EXISTS subquery" in result.reason

    def test_delete_with_union_exists_predicate_blocked(self):
        """EXISTS over set-operation subqueries should be blocked conservatively."""
        result = validate_sql(
            "DELETE FROM users WHERE EXISTS ((SELECT 1) UNION SELECT 1)",
            "postgres",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "uncorrelated EXISTS subquery" in result.reason

    def test_delete_with_exists_from_subquery_blocked(self):
        """Uncorrelated EXISTS FROM subqueries should be rejected as global."""
        result = validate_sql(
            "DELETE FROM users WHERE EXISTS (SELECT 1 FROM audit)",
            "postgres",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "uncorrelated EXISTS subquery" in result.reason

    def test_delete_with_fromless_exists_nested_global_subquery_blocked(self):
        """FROM-less EXISTS with global nested subquery predicates should be blocked."""
        result = validate_sql(
            "DELETE FROM users WHERE EXISTS (SELECT 1 WHERE (SELECT COUNT(*) FROM audit) > 0)",
            "postgres",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "uncorrelated EXISTS subquery" in result.reason

    def test_delete_with_fromless_correlated_exists_predicate_allowed(self):
        """FROM-less EXISTS should remain allowed when WHERE is row-correlated."""
        result = validate_sql(
            "DELETE FROM users u WHERE EXISTS (SELECT 1 WHERE u.id > 0)",
            "postgres",
            allow_dangerous=True,
        )
        assert result.allowed
        assert result.query_type == "dml"

    def test_delete_with_fromless_unqualified_correlated_exists_predicate_allowed(self):
        """FROM-less EXISTS should accept unqualified outer-column predicates."""
        result = validate_sql(
            "DELETE FROM users WHERE EXISTS (SELECT 1 WHERE id > 0)",
            "postgres",
            allow_dangerous=True,
        )
        assert result.allowed
        assert result.query_type == "dml"

    def test_update_with_fromless_unqualified_correlated_exists_predicate_allowed(self):
        """UPDATE should also accept unqualified outer refs in FROM-less EXISTS."""
        result = validate_sql(
            "UPDATE users SET active = false WHERE EXISTS (SELECT 1 WHERE id > 0)",
            "postgres",
            allow_dangerous=True,
        )
        assert result.allowed
        assert result.query_type == "dml"

    def test_delete_with_false_and_uncorrelated_exists_allowed(self):
        """Dead EXISTS branches under FALSE AND should not trigger uncorrelated errors."""
        result = validate_sql(
            "DELETE FROM users WHERE FALSE AND EXISTS (SELECT 1 FROM audit)",
            "postgres",
            allow_dangerous=True,
        )
        assert result.allowed
        assert result.query_type == "dml"

    def test_update_with_false_and_uncorrelated_exists_allowed(self):
        """Dead EXISTS branches under FALSE AND should not trigger for UPDATE."""
        result = validate_sql(
            "UPDATE users SET active = false WHERE FALSE AND EXISTS (SELECT 1 FROM audit)",
            "postgres",
            allow_dangerous=True,
        )
        assert result.allowed
        assert result.query_type == "dml"

    def test_delete_with_true_and_uncorrelated_exists_blocked(self):
        """Reachable uncorrelated EXISTS under TRUE AND should still be blocked."""
        result = validate_sql(
            "DELETE FROM users WHERE TRUE AND EXISTS (SELECT 1 FROM audit)",
            "postgres",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "uncorrelated EXISTS subquery" in result.reason

    def test_delete_with_restrictive_and_uncorrelated_exists_blocked(self):
        """Policy: any reachable uncorrelated EXISTS remains blocked."""
        result = validate_sql(
            "DELETE FROM users WHERE id = 1 AND EXISTS (SELECT 1 FROM audit)",
            "postgres",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "uncorrelated EXISTS subquery" in result.reason

    def test_delete_with_restrictive_and_not_exists_blocked(self):
        """NOT EXISTS is also blocked when the subquery is uncorrelated."""
        result = validate_sql(
            "DELETE FROM users WHERE NOT EXISTS (SELECT 1 FROM audit) AND id = 1",
            "postgres",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "uncorrelated EXISTS subquery" in result.reason

    def test_update_with_restrictive_and_uncorrelated_exists_blocked(self):
        """UPDATE follows the same conservative uncorrelated EXISTS policy."""
        result = validate_sql(
            "UPDATE users SET active = false WHERE id = 1 AND EXISTS (SELECT 1 FROM audit)",
            "postgres",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "uncorrelated EXISTS subquery" in result.reason

    def test_delete_with_nonconstant_case_then_uncorrelated_exists_blocked(self):
        """CASE THEN branches with unknown truthiness must remain reachable."""
        result = validate_sql(
            (
                "DELETE FROM users u WHERE CASE "
                "WHEN u.id > 0 THEN EXISTS (SELECT 1 FROM audit) "
                "ELSE FALSE END"
            ),
            "postgres",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "uncorrelated EXISTS subquery" in result.reason

    def test_delete_with_simple_case_then_uncorrelated_exists_blocked(self):
        """Simple CASE WHEN matches should keep THEN EXISTS reachable."""
        result = validate_sql(
            (
                "DELETE FROM users WHERE CASE FALSE "
                "WHEN FALSE THEN EXISTS (SELECT 1 FROM audit) "
                "ELSE FALSE END"
            ),
            "postgres",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "uncorrelated EXISTS subquery" in result.reason

    def test_delete_with_dead_case_else_exists_allowed(self):
        """Dead CASE ELSE branches should not trigger uncorrelated EXISTS checks."""
        result = validate_sql(
            (
                "DELETE FROM users u WHERE CASE "
                "WHEN TRUE THEN u.id = 1 "
                "ELSE EXISTS (SELECT 1 FROM audit) END"
            ),
            "postgres",
            allow_dangerous=True,
        )
        assert result.allowed
        assert result.query_type == "dml"

    def test_delete_with_dead_case_when_exists_allowed(self):
        """WHEN clauses after a constant TRUE branch should be unreachable."""
        result = validate_sql(
            (
                "DELETE FROM users u WHERE CASE "
                "WHEN TRUE THEN u.id = 1 "
                "WHEN EXISTS (SELECT 1 FROM audit) THEN TRUE "
                "ELSE FALSE END"
            ),
            "postgres",
            allow_dangerous=True,
        )
        assert result.allowed
        assert result.query_type == "dml"

    def test_mysql_delete_with_nonconstant_if_then_uncorrelated_exists_blocked(self):
        """MySQL IF true branches with unknown conditions must stay reachable."""
        result = validate_sql(
            "DELETE FROM users u WHERE IF(u.id > 0, EXISTS (SELECT 1 FROM audit), FALSE)",
            "mysql",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "uncorrelated EXISTS subquery" in result.reason

    def test_mysql_delete_with_simple_case_then_uncorrelated_exists_blocked(self):
        """MySQL simple CASE matching should not hide uncorrelated EXISTS."""
        result = validate_sql(
            (
                "DELETE FROM users WHERE CASE FALSE "
                "WHEN FALSE THEN EXISTS (SELECT 1 FROM audit) "
                "ELSE FALSE END"
            ),
            "mysql",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "uncorrelated EXISTS subquery" in result.reason

    def test_delete_with_dead_coalesce_exists_allowed(self):
        """Unreachable EXISTS in COALESCE should not trigger false blocking."""
        result = validate_sql(
            "DELETE FROM users WHERE COALESCE(FALSE, EXISTS (SELECT 1 FROM audit))",
            "postgres",
            allow_dangerous=True,
        )
        assert result.allowed
        assert result.query_type == "dml"

    def test_delete_with_coalesce_null_fallthrough_exists_blocked(self):
        """COALESCE fallthrough to EXISTS should remain protected."""
        result = validate_sql(
            "DELETE FROM users WHERE COALESCE(NULL, EXISTS (SELECT 1 FROM audit), FALSE)",
            "postgres",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "uncorrelated EXISTS subquery" in result.reason

    def test_duckdb_delete_with_exists_from_subquery_blocked(self):
        """DuckDB uncorrelated EXISTS FROM subqueries should also be blocked."""
        result = validate_sql(
            "DELETE FROM users WHERE EXISTS (SELECT 1 FROM audit)",
            "duckdb",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "uncorrelated EXISTS subquery" in result.reason

    def test_delete_with_projection_only_exists_reference_blocked(self):
        """Projection-only outer refs should not count as row correlation."""
        result = validate_sql(
            "DELETE FROM users u WHERE EXISTS (SELECT u.id FROM audit a)",
            "postgres",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "uncorrelated EXISTS subquery" in result.reason

    def test_delete_with_simple_case_dead_outer_reference_exists_blocked(self):
        """Dead outer refs in simple CASE branches must not imply correlation."""
        result = validate_sql(
            (
                "DELETE FROM users u WHERE EXISTS "
                "(SELECT 1 FROM audit a "
                "WHERE CASE FALSE WHEN FALSE THEN a.user_id > 0 ELSE u.id = 1 END)"
            ),
            "postgres",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "uncorrelated EXISTS subquery" in result.reason

    def test_delete_with_simple_case_live_outer_reference_exists_allowed(self):
        """Simple CASE with reachable outer-ref branch should remain correlated."""
        result = validate_sql(
            (
                "DELETE FROM users u WHERE EXISTS "
                "(SELECT 1 FROM audit a "
                "WHERE CASE FALSE WHEN TRUE THEN a.user_id > 0 ELSE u.id = 1 END)"
            ),
            "postgres",
            allow_dangerous=True,
        )
        assert result.allowed
        assert result.query_type == "dml"

    def test_mysql_delete_with_simple_case_dead_outer_reference_exists_blocked(self):
        """MySQL simple CASE dead branches must not create fake correlation."""
        result = validate_sql(
            (
                "DELETE FROM users u WHERE EXISTS "
                "(SELECT 1 FROM audit a "
                "WHERE CASE FALSE WHEN FALSE THEN a.user_id > 0 ELSE u.id = 1 END)"
            ),
            "mysql",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "uncorrelated EXISTS subquery" in result.reason

    def test_delete_with_tautological_or_exists_reference_blocked(self):
        """Outer refs neutralized by OR TRUE should not count as correlation."""
        result = validate_sql(
            "DELETE FROM users u WHERE EXISTS (SELECT 1 FROM audit a WHERE a.user_id = u.id OR TRUE)",
            "postgres",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "uncorrelated EXISTS subquery" in result.reason

    def test_delete_with_null_and_correlated_exists_predicate_blocked(self):
        """NULL AND correlated predicates must not create effective correlation."""
        predicates = (
            "NULL AND a.user_id = u.id",
            "a.user_id = u.id AND NULL",
        )
        for predicate in predicates:
            result = validate_sql(
                (
                    "DELETE FROM users u WHERE NOT EXISTS "
                    f"(SELECT 1 FROM audit a WHERE {predicate})"
                ),
                "postgres",
                allow_dangerous=True,
            )
            assert not result.allowed
            assert result.reason
            assert "uncorrelated EXISTS subquery" in result.reason

    def test_delete_with_correlated_or_null_exists_predicate_allowed(self):
        """Correlated predicates OR NULL should remain row-restrictive."""
        predicates = (
            "a.user_id = u.id OR NULL",
            "NULL OR a.user_id = u.id",
        )
        for predicate in predicates:
            result = validate_sql(
                (
                    "DELETE FROM users u WHERE EXISTS "
                    f"(SELECT 1 FROM audit a WHERE {predicate})"
                ),
                "postgres",
                allow_dangerous=True,
            )
            assert result.allowed
            assert result.query_type == "dml"

    def test_delete_with_correlated_and_true_exists_predicate_allowed(self):
        """p AND TRUE should preserve correlated row-restrictive behavior."""
        predicates = (
            "a.user_id = u.id AND TRUE",
            "TRUE AND a.user_id = u.id",
        )
        for predicate in predicates:
            result = validate_sql(
                (
                    "DELETE FROM users u WHERE EXISTS "
                    f"(SELECT 1 FROM audit a WHERE {predicate})"
                ),
                "postgres",
                allow_dangerous=True,
            )
            assert result.allowed
            assert result.query_type == "dml"

    def test_delete_with_correlated_or_false_exists_predicate_allowed(self):
        """p OR FALSE should preserve correlated row-restrictive behavior."""
        predicates = (
            "a.user_id = u.id OR FALSE",
            "FALSE OR a.user_id = u.id",
        )
        for predicate in predicates:
            result = validate_sql(
                (
                    "DELETE FROM users u WHERE EXISTS "
                    f"(SELECT 1 FROM audit a WHERE {predicate})"
                ),
                "postgres",
                allow_dangerous=True,
            )
            assert result.allowed
            assert result.query_type == "dml"

    def test_delete_with_tautological_outer_reference_exists_predicate_blocked(self):
        """Outer-reference tautologies should not count as effective correlation."""
        result = validate_sql(
            "DELETE FROM users u WHERE EXISTS (SELECT 1 FROM audit a WHERE u.id IS NULL OR u.id IS NOT NULL)",
            "postgres",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "uncorrelated EXISTS subquery" in result.reason

    def test_delete_with_is_distinct_partition_outer_reference_exists_blocked(self):
        """DISTINCT partition tautologies over outer refs must not correlate."""
        result = validate_sql(
            (
                "DELETE FROM users u WHERE EXISTS "
                "(SELECT 1 FROM audit a WHERE "
                "u.id IS DISTINCT FROM NULL OR u.id IS NOT DISTINCT FROM NULL)"
            ),
            "postgres",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "uncorrelated EXISTS subquery" in result.reason

    def test_delete_with_self_equality_or_null_exists_predicate_blocked(self):
        """x = x OR x IS NULL wrappers must not count as correlation."""
        result = validate_sql(
            "DELETE FROM users u WHERE EXISTS (SELECT 1 FROM audit a WHERE u.id = u.id OR u.id IS NULL)",
            "postgres",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "uncorrelated EXISTS subquery" in result.reason

    def test_delete_with_tautological_coalesce_outer_reference_exists_blocked(self):
        """Tautological COALESCE wrappers over outer refs must not count as correlation."""
        result = validate_sql(
            "DELETE FROM users u WHERE EXISTS (SELECT 1 FROM audit a WHERE COALESCE(u.id = u.id, TRUE))",
            "postgres",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "uncorrelated EXISTS subquery" in result.reason

    def test_delete_with_tautological_is_not_false_outer_reference_exists_blocked(self):
        """IS NOT FALSE wrappers over outer-ref tautologies must not correlate."""
        result = validate_sql(
            "DELETE FROM users u WHERE EXISTS (SELECT 1 FROM audit a WHERE (u.id = u.id) IS NOT FALSE)",
            "postgres",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "uncorrelated EXISTS subquery" in result.reason

    def test_delete_with_nonnullable_partition_outer_reference_exists_blocked(self):
        """p OR p IS FALSE tautologies over outer refs must not correlate."""
        result = validate_sql(
            "DELETE FROM users u WHERE EXISTS (SELECT 1 FROM audit a WHERE (u.id IS NULL) OR ((u.id IS NULL) IS FALSE))",
            "postgres",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "uncorrelated EXISTS subquery" in result.reason

    def test_delete_with_correlated_partition_exists_predicate_allowed(self):
        """Correlated p OR p IS FALSE predicates that can filter rows should be allowed."""
        result = validate_sql(
            "DELETE FROM users u WHERE EXISTS (SELECT 1 FROM audit a WHERE (a.user_id = u.id) OR ((a.user_id = u.id) IS FALSE))",
            "postgres",
            allow_dangerous=True,
        )
        assert result.allowed
        assert result.query_type == "dml"

    def test_analysis_cache_object_id_collisions_do_not_change_result(
        self, monkeypatch
    ):
        """Cache entries must verify AST identity when object IDs are recycled."""
        assert "id" not in vars(sql_guard)
        assert "id" not in vars(mutation_analysis_module)

        def replacement(_value: object) -> int:
            return 1

        with monkeypatch.context() as patch:
            patch.setattr(sql_guard, "id", replacement, raising=False)

            assert sql_guard.id is replacement
            assert mutation_analysis_module.id is replacement

            result = validate_sql(
                "DELETE FROM users u WHERE u.id = 1 OR TRUE",
                "postgres",
                allow_dangerous=True,
            )

            assert not result.allowed
            assert result.reason
            assert "tautological WHERE" in result.reason

        assert "id" not in vars(sql_guard)
        assert "id" not in vars(mutation_analysis_module)

    def test_analysis_cache_does_not_retain_simple_case_copies(self):
        """Temporary CASE comparison trees should bypass identity caches."""
        when_clauses = " ".join(f"WHEN {value} THEN FALSE" for value in range(50))
        sql = f"DELETE FROM users WHERE CASE id {when_clauses} ELSE TRUE END"
        statement = sqlglot.parse_one(sql, read="postgres")
        where = statement.args.get("where")
        assert isinstance(where, exp.Where)
        original_expressions = tuple(where.this.walk())

        with mutation_analysis_module._analysis_session():
            mutation_analysis_module._predicate_truthiness_possibilities(
                where.this,
                "postgres",
                sql,
            )
            context = mutation_analysis_module._ANALYSIS_CONTEXT.get()
            assert context is not None
            cached_expressions = tuple(
                entry[0] for entry in context.predicate_truthiness_cache.values()
            )

        assert cached_expressions
        assert all(
            any(cached is original for original in original_expressions)
            for cached in cached_expressions
        )

    def test_delete_with_correlated_is_not_false_exists_predicate_allowed(self):
        """IS NOT FALSE wrappers that can filter correlated rows should be allowed."""
        result = validate_sql(
            "DELETE FROM users u WHERE EXISTS (SELECT 1 FROM audit a WHERE (a.user_id = u.id) IS NOT FALSE)",
            "postgres",
            allow_dangerous=True,
        )
        assert result.allowed
        assert result.query_type == "dml"

    def test_delete_with_tautological_nullif_outer_reference_exists_blocked(self):
        """NULLIF tautologies over outer refs must not count as correlation."""
        result = validate_sql(
            "DELETE FROM users u WHERE EXISTS (SELECT 1 FROM audit a WHERE NULLIF(u.id, u.id) IS NULL)",
            "postgres",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "uncorrelated EXISTS subquery" in result.reason

    def test_delete_with_tautological_nullif_self_comparison_exists_blocked(self):
        """NULLIF(x=x, TRUE) tautologies over outer refs must not correlate."""
        result = validate_sql(
            "DELETE FROM users u WHERE EXISTS (SELECT 1 FROM audit a WHERE NULLIF(u.id = u.id, TRUE) IS NULL)",
            "postgres",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "uncorrelated EXISTS subquery" in result.reason

    def test_delete_with_correlated_nullif_self_comparison_exists_allowed(self):
        """Correlated NULLIF(x=y, TRUE) predicates that can filter should be allowed."""
        result = validate_sql(
            "DELETE FROM users u WHERE EXISTS (SELECT 1 FROM audit a WHERE NULLIF(a.user_id = u.id, TRUE) IS NULL)",
            "postgres",
            allow_dangerous=True,
        )
        assert result.allowed
        assert result.query_type == "dml"

    def test_delete_with_correlated_nullif_exists_predicate_allowed(self):
        """Correlated NULLIF predicates that can filter rows should remain allowed."""
        result = validate_sql(
            "DELETE FROM users u WHERE EXISTS (SELECT 1 FROM audit a WHERE NULLIF(a.user_id, u.id) IS NULL)",
            "postgres",
            allow_dangerous=True,
        )
        assert result.allowed
        assert result.query_type == "dml"

    def test_delete_with_correlated_coalesce_exists_predicate_allowed(self):
        """Correlated COALESCE predicates that can filter rows should remain allowed."""
        result = validate_sql(
            "DELETE FROM users u WHERE EXISTS (SELECT 1 FROM audit a WHERE COALESCE(a.user_id = u.id, FALSE))",
            "postgres",
            allow_dangerous=True,
        )
        assert result.allowed
        assert result.query_type == "dml"

    def test_mysql_delete_with_tautological_if_outer_reference_exists_blocked(self):
        """MySQL IF tautologies over outer refs must not count as correlation."""
        result = validate_sql(
            "DELETE FROM users u WHERE EXISTS (SELECT 1 FROM audit a WHERE IF(u.id IS NULL, TRUE, TRUE))",
            "mysql",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "uncorrelated EXISTS subquery" in result.reason

    def test_mysql_delete_with_correlated_if_exists_predicate_allowed(self):
        """MySQL IF predicates that can filter correlated rows should remain allowed."""
        result = validate_sql(
            "DELETE FROM users u WHERE EXISTS (SELECT 1 FROM audit a WHERE IF(a.user_id = u.id, TRUE, FALSE))",
            "mysql",
            allow_dangerous=True,
        )
        assert result.allowed
        assert result.query_type == "dml"

    def test_delete_with_shadowed_alias_exists_reference_blocked(self):
        """Inner aliases shadowing target alias must not be treated as outer refs."""
        result = validate_sql(
            "DELETE FROM users u WHERE EXISTS (SELECT 1 FROM audit u WHERE u.id > 0)",
            "postgres",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "uncorrelated EXISTS subquery" in result.reason

    def test_delete_with_shadowed_table_name_exists_reference_blocked(self):
        """Inner table names shadowing target names must not imply correlation."""
        result = validate_sql(
            "DELETE FROM users WHERE EXISTS (SELECT 1 FROM users WHERE users.id > 0)",
            "postgres",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "uncorrelated EXISTS subquery" in result.reason

    def test_delete_with_self_table_exists_subquery_blocked(self):
        """Self-table uncorrelated EXISTS can still become full-table delete."""
        result = validate_sql(
            "DELETE FROM audit WHERE EXISTS (SELECT 1 FROM audit)",
            "postgres",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "uncorrelated EXISTS subquery" in result.reason

    def test_delete_with_correlated_exists_subquery_allowed(self):
        """Correlated EXISTS subqueries should remain allowed."""
        result = validate_sql(
            "DELETE FROM users u WHERE EXISTS (SELECT 1 FROM audit a WHERE a.user_id = u.id)",
            "postgres",
            allow_dangerous=True,
        )
        assert result.allowed
        assert result.query_type == "dml"

    def test_delete_with_nested_exists_inside_correlated_exists_blocked(self):
        """Policy: nested helper EXISTS is evaluated fail-closed and blocked."""
        result = validate_sql(
            (
                "DELETE FROM users u WHERE EXISTS "
                "(SELECT 1 FROM audit a "
                "WHERE a.user_id = u.id AND EXISTS (SELECT 1 FROM flags))"
            ),
            "postgres",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "uncorrelated EXISTS subquery" in result.reason

    def test_deep_exists_boolean_predicate_fails_closed_instead_of_crashing(self):
        """Deep boolean chains should fail closed instead of raising RecursionError."""
        original_recursion_limit = sys.getrecursionlimit()

        try:
            # Keep the regression deterministic and lightweight while still
            # exercising deep-predicate recursion handling in guard analysis.
            sys.setrecursionlimit(400)

            disjuncts = [f"a.user_id = u.id + {index}" for index in range(400)]
            predicate = " OR ".join(disjuncts)
            result = validate_sql(
                (
                    "DELETE FROM users u WHERE EXISTS "
                    f"(SELECT 1 FROM audit a WHERE {predicate})"
                ),
                "postgres",
                allow_dangerous=True,
            )
        finally:
            sys.setrecursionlimit(original_recursion_limit)

        assert not result.allowed
        assert result.reason
        assert "too complex to validate safely" in result.reason

    def test_predicate_simplify_gate_skips_large_boolean_chains(self):
        """Large predicates should skip simplify via structural complexity gating."""
        predicate = " OR ".join(f"id = {index}" for index in range(400))
        statement = sqlglot.parse_one(
            f"DELETE FROM users WHERE {predicate}",
            read="postgres",
        )
        where = statement.args.get("where")
        assert isinstance(where, exp.Where)
        assert not _should_attempt_predicate_simplify(where.this)

    def test_large_or_not_partition_tautology_blocked_when_simplify_skipped(self):
        """Large p OR NOT p chains must stay blocked even without simplify()."""
        disjunct = "(u.id IS NULL OR NOT (u.id IS NULL))"
        predicate = " OR ".join(disjunct for _ in range(40))
        statement = sqlglot.parse_one(
            f"DELETE FROM users u WHERE {predicate}",
            read="postgres",
        )
        where = statement.args.get("where")
        assert isinstance(where, exp.Where)
        assert not _should_attempt_predicate_simplify(where.this)

        result = validate_sql(
            f"DELETE FROM users u WHERE {predicate}",
            "postgres",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_predicate_simplify_gate_allows_small_predicates(self):
        """Small predicates should remain eligible for simplify."""
        statement = sqlglot.parse_one(
            "DELETE FROM users WHERE id = 1 OR id = 2",
            read="postgres",
        )
        where = statement.args.get("where")
        assert isinstance(where, exp.Where)
        assert _should_attempt_predicate_simplify(where.this)

    def test_validation_analysis_budget_exceeded_fails_closed(self, monkeypatch):
        """Validation should fail closed when the per-query analysis budget is exhausted."""
        monkeypatch.setattr(sql_guard, "ANALYSIS_BUDGET_MAX_STEPS", 25)
        assert mutation_analysis_module.ANALYSIS_BUDGET_MAX_STEPS == 25

        predicate = " OR ".join(f"id = {index}" for index in range(200))
        result = validate_sql(
            f"DELETE FROM users WHERE {predicate}",
            "postgres",
            allow_dangerous=True,
        )

        assert not result.allowed
        assert result.reason
        assert "analysis budget exceeded" in result.reason

    def test_mysql_delete_using_alias_exists_reference_blocked_conservatively(self):
        """USING-only aliases are intentionally treated as non-correlating (fail closed)."""
        result = validate_sql(
            "DELETE FROM users USING users u WHERE EXISTS (SELECT 1 FROM audit a WHERE a.user_id = u.id)",
            "mysql",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "uncorrelated EXISTS subquery" in result.reason

    def test_update_with_uncorrelated_exists_from_subquery_blocked(self):
        """UPDATE with global uncorrelated EXISTS FROM should be blocked."""
        result = validate_sql(
            "UPDATE users SET active = false WHERE EXISTS (SELECT 1 FROM audit)",
            "postgres",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "uncorrelated EXISTS subquery" in result.reason

    def test_delete_with_null_and_uncorrelated_exists_predicate_allowed(self):
        """NULL AND should make uncorrelated EXISTS branches unreachable."""
        result = validate_sql(
            "DELETE FROM users WHERE NULL AND EXISTS (SELECT 1 FROM audit)",
            "postgres",
            allow_dangerous=True,
        )
        assert result.allowed
        assert result.query_type == "dml"

    def test_delete_with_null_or_uncorrelated_exists_predicate_blocked(self):
        """NULL OR should still keep uncorrelated EXISTS reachable and blocked."""
        result = validate_sql(
            "DELETE FROM users WHERE NULL OR EXISTS (SELECT 1 FROM audit)",
            "postgres",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "uncorrelated EXISTS subquery" in result.reason

    def test_update_with_correlated_exists_from_subquery_allowed(self):
        """Correlated EXISTS FROM in UPDATE should remain allowed."""
        result = validate_sql(
            "UPDATE users u SET active = false WHERE EXISTS (SELECT 1 FROM audit a WHERE a.user_id = u.id)",
            "postgres",
            allow_dangerous=True,
        )
        assert result.allowed
        assert result.query_type == "dml"

    def test_delete_with_not_null_predicate_allowed(self):
        """NOT NULL should remain UNKNOWN and not be treated as tautological."""
        result = validate_sql(
            "DELETE FROM users WHERE NOT NULL",
            "postgres",
            allow_dangerous=True,
        )
        assert result.allowed
        assert result.query_type == "dml"

    def test_delete_with_constant_exists_offset_allowed(self):
        """Constant EXISTS should account for OFFSET row elimination."""
        result = validate_sql(
            "DELETE FROM users WHERE EXISTS (SELECT 1 OFFSET 1)",
            "postgres",
            allow_dangerous=True,
        )
        assert result.allowed
        assert result.query_type == "dml"

    def test_delete_with_aggregate_exists_where_false_blocked(self):
        """No-FROM aggregate EXISTS with WHERE FALSE still yields one row."""
        result = validate_sql(
            "DELETE FROM users WHERE EXISTS (SELECT COUNT(*) WHERE FALSE)",
            "postgres",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_update_with_aggregate_exists_where_false_and_filter_allowed(self):
        """Tautological aggregate EXISTS in AND should preserve row filtering."""
        result = validate_sql(
            (
                "UPDATE users SET active = false "
                "WHERE EXISTS (SELECT COUNT(*) WHERE FALSE) AND id = 1"
            ),
            "postgres",
            allow_dangerous=True,
        )
        assert result.allowed
        assert result.query_type == "dml"

    def test_delete_with_aggregate_exists_where_false_having_false_allowed(self):
        """HAVING FALSE should still collapse aggregate EXISTS to false."""
        result = validate_sql(
            "DELETE FROM users WHERE EXISTS (SELECT COUNT(*) WHERE FALSE HAVING FALSE)",
            "postgres",
            allow_dangerous=True,
        )
        assert result.allowed
        assert result.query_type == "dml"

    def test_postgres_delete_with_set_returning_exists_offset_blocked(self):
        """FROM-less SRF EXISTS with OFFSET must not be folded to false."""
        result = validate_sql(
            "DELETE FROM users WHERE EXISTS (SELECT generate_series(1, 2) OFFSET 1)",
            "postgres",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "uncorrelated EXISTS subquery" in result.reason

    def test_postgres_delete_with_unnest_exists_offset_blocked(self):
        """UNNEST projections with OFFSET should also stay non-constant."""
        result = validate_sql(
            "DELETE FROM users WHERE EXISTS (SELECT unnest(ARRAY[1,2]) OFFSET 1)",
            "postgres",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "uncorrelated EXISTS subquery" in result.reason

    def test_delete_with_constant_exists_having_false_allowed(self):
        """Constant EXISTS should account for HAVING row elimination."""
        result = validate_sql(
            "DELETE FROM users WHERE EXISTS (SELECT 1 HAVING FALSE)",
            "postgres",
            allow_dangerous=True,
        )
        assert result.allowed
        assert result.query_type == "dml"

    def test_delete_with_constant_exists_fetch_zero_allowed(self):
        """FETCH FIRST 0 should be treated as an empty EXISTS subquery."""
        result = validate_sql(
            "DELETE FROM users WHERE EXISTS (SELECT 1 FETCH FIRST 0 ROWS ONLY)",
            "postgres",
            allow_dangerous=True,
        )
        assert result.allowed
        assert result.query_type == "dml"

    def test_postgres_delete_with_constant_exists_fetch_first_row_only_blocked(self):
        """Implicit FETCH FIRST ROW ONLY should still be tautological here."""
        result = validate_sql(
            "DELETE FROM users WHERE EXISTS (SELECT 1 FETCH FIRST ROW ONLY)",
            "postgres",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_duckdb_delete_with_constant_exists_fetch_first_row_only_blocked(self):
        """DuckDB FETCH FIRST ROW ONLY should be treated as one-row fetch."""
        result = validate_sql(
            "DELETE FROM users WHERE EXISTS (SELECT 1 FETCH FIRST ROW ONLY)",
            "duckdb",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_sqlite_delete_with_constant_exists_limit_negative_blocked(self):
        """SQLite LIMIT -1 means unlimited rows, so EXISTS is tautological."""
        result = validate_sql(
            "DELETE FROM users WHERE EXISTS (SELECT 1 LIMIT -1)",
            "sqlite",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_postgres_delete_with_constant_exists_limit_null_blocked(self):
        """Postgres LIMIT NULL behaves as no limit, so EXISTS is tautological."""
        result = validate_sql(
            "DELETE FROM users WHERE EXISTS (SELECT 1 LIMIT NULL)",
            "postgres",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_duckdb_delete_with_constant_exists_limit_null_blocked(self):
        """DuckDB LIMIT NULL behaves as no limit, so EXISTS is tautological."""
        result = validate_sql(
            "DELETE FROM users WHERE EXISTS (SELECT 1 LIMIT NULL)",
            "duckdb",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_postgres_delete_with_constant_exists_limit_all_blocked(self):
        """Postgres LIMIT ALL is unbounded, so EXISTS is tautological."""
        result = validate_sql(
            "DELETE FROM users WHERE EXISTS (SELECT 1 LIMIT ALL)",
            "postgres",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_duckdb_delete_with_constant_exists_limit_all_blocked(self):
        """DuckDB LIMIT ALL is unbounded, so EXISTS is tautological."""
        result = validate_sql(
            "DELETE FROM users WHERE EXISTS (SELECT 1 LIMIT ALL)",
            "duckdb",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_mysql_update_with_abs_constant_predicate_blocked(self):
        """Deterministic constant function predicates should be blocked."""
        result = validate_sql(
            "UPDATE users SET active = 0 WHERE ABS(1)",
            "mysql",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_mysql_delete_with_coalesce_constant_predicate_blocked(self):
        """Constant COALESCE predicates should be blocked when truthy."""
        result = validate_sql(
            "DELETE FROM users WHERE COALESCE(NULL, 1)",
            "mysql",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_postgres_delete_with_cast_true_boolean_blocked(self):
        """Constant boolean CAST wrappers should be blocked as tautological."""
        result = validate_sql(
            "DELETE FROM users WHERE CAST(TRUE AS BOOLEAN)",
            "postgres",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_sqlite_delete_with_cast_truthy_string_boolean_blocked(self):
        """SQLite truthy string->BOOLEAN casts should be blocked as tautological."""
        result = validate_sql(
            "DELETE FROM users WHERE CAST('1' AS BOOLEAN)",
            "sqlite",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_sqlite_delete_with_cast_falsey_string_boolean_allowed(self):
        """SQLite falsey string->BOOLEAN casts should remain allowed."""
        result = validate_sql(
            "DELETE FROM users WHERE CAST('0' AS BOOLEAN)",
            "sqlite",
            allow_dangerous=True,
        )
        assert result.allowed
        assert result.query_type == "dml"

    def test_sqlite_delete_with_cast_fractional_numeric_int_truthy_blocked(self):
        """SQLite fractional numeric->INT casts should be folded deterministically."""
        result = validate_sql(
            "DELETE FROM users WHERE CAST(1.2 AS INT)",
            "sqlite",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_sqlite_delete_with_cast_fractional_string_int_truthy_blocked(self):
        """SQLite fractional string->INT casts should be folded deterministically."""
        result = validate_sql(
            "DELETE FROM users WHERE CAST('1.2' AS INT)",
            "sqlite",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_sqlite_delete_with_cast_half_numeric_int_falsey_allowed(self):
        """SQLite INT casts should truncate toward zero for fractional halves."""
        result = validate_sql(
            "DELETE FROM users WHERE CAST(0.5 AS INT)",
            "sqlite",
            allow_dangerous=True,
        )
        assert result.allowed
        assert result.query_type == "dml"

    def test_sqlite_delete_with_cast_negative_half_numeric_int_falsey_allowed(self):
        """SQLite INT casts should keep negative half fractions falsey via truncation."""
        result = validate_sql(
            "DELETE FROM users WHERE CAST(-0.5 AS INT)",
            "sqlite",
            allow_dangerous=True,
        )
        assert result.allowed
        assert result.query_type == "dml"

    def test_sqlite_delete_with_cast_numeric_text_truthy_blocked(self):
        """SQLite CAST(... AS TEXT) truthy constants should be blocked."""
        result = validate_sql(
            "DELETE FROM users WHERE CAST(1 AS TEXT)",
            "sqlite",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_sqlite_delete_with_cast_numeric_text_falsey_allowed(self):
        """SQLite CAST(... AS TEXT) falsey constants should remain allowed."""
        result = validate_sql(
            "DELETE FROM users WHERE CAST(0 AS TEXT)",
            "sqlite",
            allow_dangerous=True,
        )
        assert result.allowed
        assert result.query_type == "dml"

    def test_postgres_delete_with_cast_truthy_string_boolean_blocked(self):
        """Postgres truthy string->BOOLEAN casts should be blocked."""
        result = validate_sql(
            "DELETE FROM users WHERE CAST('true' AS BOOLEAN)",
            "postgres",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_postgres_delete_with_cast_falsey_string_boolean_allowed(self):
        """Postgres falsey string->BOOLEAN casts should remain allowed."""
        result = validate_sql(
            "DELETE FROM users WHERE CAST('false' AS BOOLEAN)",
            "postgres",
            allow_dangerous=True,
        )
        assert result.allowed
        assert result.query_type == "dml"

    def test_mysql_delete_with_cast_truthy_string_boolean_blocked(self):
        """MySQL truthy string->BOOLEAN casts should be blocked."""
        result = validate_sql(
            "DELETE FROM users WHERE CAST('1' AS BOOLEAN)",
            "mysql",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_mysql_delete_with_cast_falsey_string_boolean_allowed(self):
        """MySQL falsey string->BOOLEAN casts should remain allowed."""
        result = validate_sql(
            "DELETE FROM users WHERE CAST('0' AS BOOLEAN)",
            "mysql",
            allow_dangerous=True,
        )
        assert result.allowed
        assert result.query_type == "dml"

    def test_mysql_delete_with_cast_numeric_truthy_blocked(self):
        """MySQL truthy numeric CAST wrappers should be blocked as tautological."""
        result = validate_sql(
            "DELETE FROM users WHERE CAST(1 AS SIGNED)",
            "mysql",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_mysql_delete_with_cast_numeric_falsey_allowed(self):
        """MySQL falsey numeric CAST wrappers should remain allowed."""
        result = validate_sql(
            "DELETE FROM users WHERE CAST(0 AS SIGNED)",
            "mysql",
            allow_dangerous=True,
        )
        assert result.allowed
        assert result.query_type == "dml"

    def test_duckdb_delete_with_cast_numeric_varchar_truthy_blocked(self):
        """DuckDB CAST(... AS VARCHAR) truthy constants should be blocked."""
        result = validate_sql(
            "DELETE FROM users WHERE CAST(1 AS VARCHAR)",
            "duckdb",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_duckdb_delete_with_cast_numeric_varchar_falsey_allowed(self):
        """DuckDB CAST(... AS VARCHAR) falsey constants should remain allowed."""
        result = validate_sql(
            "DELETE FROM users WHERE CAST(0 AS VARCHAR)",
            "duckdb",
            allow_dangerous=True,
        )
        assert result.allowed
        assert result.query_type == "dml"

    def test_duckdb_delete_with_cast_fractional_numeric_int_truthy_blocked(self):
        """DuckDB fractional numeric->INT casts should be folded deterministically."""
        result = validate_sql(
            "DELETE FROM users WHERE CAST(1.2 AS INT)",
            "duckdb",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_duckdb_delete_with_cast_fractional_string_int_truthy_blocked(self):
        """DuckDB fractional string->INT casts should be folded deterministically."""
        result = validate_sql(
            "DELETE FROM users WHERE CAST('1.2' AS INT)",
            "duckdb",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_duckdb_delete_with_cast_half_numeric_int_truthy_blocked(self):
        """DuckDB INT casts should round halves away from zero in predicates."""
        result = validate_sql(
            "DELETE FROM users WHERE CAST(0.5 AS INT)",
            "duckdb",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_duckdb_delete_with_cast_negative_half_numeric_int_truthy_blocked(self):
        """DuckDB negative half INT casts should round away from zero as truthy."""
        result = validate_sql(
            "DELETE FROM users WHERE CAST(-0.5 AS INT)",
            "duckdb",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_sqlite_delete_with_try_cast_truthy_string_boolean_blocked(self):
        """SQLite TRY_CAST truthy string->BOOLEAN should be blocked."""
        result = validate_sql(
            "DELETE FROM users WHERE TRY_CAST('1' AS BOOLEAN)",
            "sqlite",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_sqlite_delete_with_try_cast_falsey_string_boolean_allowed(self):
        """SQLite TRY_CAST falsey string->BOOLEAN should remain allowed."""
        result = validate_sql(
            "DELETE FROM users WHERE TRY_CAST('0' AS BOOLEAN)",
            "sqlite",
            allow_dangerous=True,
        )
        assert result.allowed
        assert result.query_type == "dml"

    def test_postgres_delete_with_try_cast_truthy_string_boolean_blocked(self):
        """Postgres TRY_CAST truthy string->BOOLEAN should be blocked."""
        result = validate_sql(
            "DELETE FROM users WHERE TRY_CAST('true' AS BOOLEAN)",
            "postgres",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_postgres_delete_with_try_cast_falsey_string_boolean_allowed(self):
        """Postgres TRY_CAST falsey string->BOOLEAN should remain allowed."""
        result = validate_sql(
            "DELETE FROM users WHERE TRY_CAST('false' AS BOOLEAN)",
            "postgres",
            allow_dangerous=True,
        )
        assert result.allowed
        assert result.query_type == "dml"

    def test_mysql_delete_with_try_cast_truthy_string_boolean_blocked(self):
        """MySQL TRY_CAST truthy string->BOOLEAN should be blocked."""
        result = validate_sql(
            "DELETE FROM users WHERE TRY_CAST('1' AS BOOLEAN)",
            "mysql",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_mysql_delete_with_try_cast_falsey_string_boolean_allowed(self):
        """MySQL TRY_CAST falsey string->BOOLEAN should remain allowed."""
        result = validate_sql(
            "DELETE FROM users WHERE TRY_CAST('0' AS BOOLEAN)",
            "mysql",
            allow_dangerous=True,
        )
        assert result.allowed
        assert result.query_type == "dml"

    def test_duckdb_delete_with_try_cast_numeric_varchar_truthy_blocked(self):
        """DuckDB TRY_CAST(... AS VARCHAR) truthy constants should be blocked."""
        result = validate_sql(
            "DELETE FROM users WHERE TRY_CAST(1 AS VARCHAR)",
            "duckdb",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_duckdb_delete_with_try_cast_numeric_varchar_falsey_allowed(self):
        """DuckDB TRY_CAST(... AS VARCHAR) falsey constants should remain allowed."""
        result = validate_sql(
            "DELETE FROM users WHERE TRY_CAST(0 AS VARCHAR)",
            "duckdb",
            allow_dangerous=True,
        )
        assert result.allowed
        assert result.query_type == "dml"

    def test_duckdb_delete_with_try_cast_invalid_boolean_string_allowed(self):
        """DuckDB invalid TRY_CAST string->BOOLEAN should fold to NULL and stay allowed."""
        result = validate_sql(
            "DELETE FROM users WHERE TRY_CAST('abc' AS BOOLEAN)",
            "duckdb",
            allow_dangerous=True,
        )
        assert result.allowed
        assert result.query_type == "dml"

    def test_mysql_delete_with_tautological_if_predicate_blocked(self):
        """MySQL IF wrappers that are always TRUE should be blocked."""
        result = validate_sql(
            "DELETE FROM users u WHERE IF(u.id IS NULL, TRUE, TRUE)",
            "mysql",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_mysql_delete_with_row_restrictive_if_predicate_allowed(self):
        """MySQL IF predicates that can filter rows should remain allowed."""
        result = validate_sql(
            "DELETE FROM users u WHERE IF(u.id = u.id, TRUE, FALSE)",
            "mysql",
            allow_dangerous=True,
        )
        assert result.allowed
        assert result.query_type == "dml"

    def test_mysql_delete_with_truthy_string_literal_blocked(self):
        """MySQL truthy numeric strings in WHERE should be blocked."""
        result = validate_sql(
            "DELETE FROM users WHERE '1'",
            "mysql",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_mysql_delete_with_bool_numeric_equality_blocked(self):
        """MySQL TRUE = 1 should be folded as a tautological predicate."""
        result = validate_sql(
            "DELETE FROM users WHERE TRUE = 1",
            "mysql",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_mysql_delete_with_bool_numeric_in_predicate_blocked(self):
        """MySQL bool↔numeric IN coercion should be recognized as tautological."""
        result = validate_sql(
            "DELETE FROM users WHERE FALSE IN (0)",
            "mysql",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_mysql_delete_with_bool_numeric_comparison_blocked(self):
        """MySQL TRUE > 0 should be treated as tautological in dangerous mode."""
        result = validate_sql(
            "DELETE FROM users WHERE TRUE > 0",
            "mysql",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_mysql_delete_with_bool_string_equality_blocked(self):
        """MySQL TRUE = '1' should be recognized as tautological."""
        result = validate_sql(
            "DELETE FROM users WHERE TRUE = '1'",
            "mysql",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_mysql_delete_with_string_bool_equality_blocked(self):
        """MySQL '1' = TRUE should be recognized as tautological."""
        result = validate_sql(
            "DELETE FROM users WHERE '1' = TRUE",
            "mysql",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_mysql_delete_with_zero_string_false_equality_blocked(self):
        """MySQL '0' = FALSE should be recognized as tautological."""
        result = validate_sql(
            "DELETE FROM users WHERE '0' = FALSE",
            "mysql",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_mysql_delete_with_bool_string_in_predicate_blocked(self):
        """MySQL TRUE IN ('1') should be recognized as tautological."""
        result = validate_sql(
            "DELETE FROM users WHERE TRUE IN ('1')",
            "mysql",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_mysql_delete_with_false_bool_string_in_predicate_blocked(self):
        """MySQL FALSE IN ('0') should be recognized as tautological."""
        result = validate_sql(
            "DELETE FROM users WHERE FALSE IN ('0')",
            "mysql",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_mysql_delete_with_nonmatching_bool_string_in_allowed(self):
        """Non-matching MySQL bool↔string IN predicates should remain allowed."""
        result = validate_sql(
            "DELETE FROM users WHERE TRUE IN ('0')",
            "mysql",
            allow_dangerous=True,
        )
        assert result.allowed
        assert result.query_type == "dml"

    def test_mysql_delete_with_nonmatching_bool_numeric_in_allowed(self):
        """MySQL non-matching bool↔numeric IN should remain allowed."""
        result = validate_sql(
            "DELETE FROM users WHERE TRUE IN (0)",
            "mysql",
            allow_dangerous=True,
        )
        assert result.allowed
        assert result.query_type == "dml"

    def test_delete_with_constant_equality_true_blocked(self):
        """Constant TRUE = TRUE predicates should be blocked as tautological."""
        result = validate_sql(
            "DELETE FROM users WHERE TRUE = TRUE",
            "postgres",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_delete_with_constant_exists_boolean_equality_blocked(self):
        """Boolean EXISTS comparisons should fold to constant TRUE and be blocked."""
        result = validate_sql(
            "DELETE FROM users WHERE (EXISTS (SELECT 1)) = TRUE",
            "postgres",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_delete_with_constant_in_boolean_inequality_blocked(self):
        """Boolean IN comparisons should fold through <> FALSE tautology checks."""
        result = validate_sql(
            "DELETE FROM users WHERE (TRUE IN (TRUE)) <> FALSE",
            "postgres",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_delete_with_constant_is_distinct_from_true_blocked(self):
        """Constant IS DISTINCT FROM true predicates should be blocked."""
        result = validate_sql(
            "DELETE FROM users WHERE 1 IS DISTINCT FROM 2",
            "postgres",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_delete_with_is_distinct_or_not_distinct_partition_blocked(self):
        """IS DISTINCT / IS NOT DISTINCT partitions should be tautological."""
        result = validate_sql(
            (
                "DELETE FROM users u WHERE "
                "u.id IS DISTINCT FROM NULL OR u.id IS NOT DISTINCT FROM NULL"
            ),
            "postgres",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_delete_with_is_distinct_or_not_distinct_swapped_operands_blocked(self):
        """Operand order swaps should still detect DISTINCT partition tautologies."""
        result = validate_sql(
            (
                "DELETE FROM users u WHERE "
                "NULL IS DISTINCT FROM u.id OR u.id IS NOT DISTINCT FROM NULL"
            ),
            "postgres",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_delete_with_constant_is_true_blocked(self):
        """Constant IS predicates evaluating true should be blocked."""
        result = validate_sql(
            "DELETE FROM users WHERE TRUE IS TRUE",
            "postgres",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_delete_with_constant_is_distinct_from_false_allowed(self):
        """False constant IS DISTINCT FROM predicates should remain allowed."""
        result = validate_sql(
            "DELETE FROM users WHERE 1 IS DISTINCT FROM 1",
            "postgres",
            allow_dangerous=True,
        )
        assert result.allowed
        assert result.query_type == "dml"

    def test_sqlite_delete_with_string_bool_inequality_blocked(self):
        """SQLite string<>bool constants should be folded and blocked when true."""
        result = validate_sql(
            "DELETE FROM users WHERE '0' <> FALSE",
            "sqlite",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_sqlite_delete_with_string_is_not_distinct_from_true_blocked(self):
        """SQLite string IS NOT DISTINCT FROM TRUE should use boolean IS semantics."""
        result = validate_sql(
            "DELETE FROM users WHERE '1' IS NOT DISTINCT FROM TRUE",
            "sqlite",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_sqlite_delete_with_true_is_not_distinct_from_string_allowed(self):
        """SQLite TRUE IS NOT DISTINCT FROM '1' should stay non-tautological."""
        result = validate_sql(
            "DELETE FROM users WHERE TRUE IS NOT DISTINCT FROM '1'",
            "sqlite",
            allow_dangerous=True,
        )
        assert result.allowed
        assert result.query_type == "dml"

    def test_sqlite_delete_with_bool_numeric_in_predicate_blocked(self):
        """SQLite bool↔numeric IN coercion should be recognized as tautological."""
        result = validate_sql(
            "DELETE FROM users WHERE TRUE IN (1)",
            "sqlite",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_duckdb_delete_with_false_bool_numeric_in_predicate_blocked(self):
        """DuckDB bool↔numeric IN coercion should be recognized as tautological."""
        result = validate_sql(
            "DELETE FROM users WHERE FALSE IN (0)",
            "duckdb",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_duckdb_delete_with_bool_string_in_predicate_blocked(self):
        """DuckDB bool↔string IN coercion should be recognized as tautological."""
        result = validate_sql(
            "DELETE FROM users WHERE TRUE IN ('1')",
            "duckdb",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_duckdb_delete_with_mixed_bool_string_in_predicate_blocked(self):
        """DuckDB mixed bool/string IN coercion should detect tautological TRUE."""
        result = validate_sql(
            "DELETE FROM users WHERE '0' IN ('false', TRUE)",
            "duckdb",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_duckdb_delete_with_mixed_bool_string_in_predicate_ordered_blocked(self):
        """DuckDB list-level coercion should be order-insensitive for tautologies."""
        result = validate_sql(
            "DELETE FROM users WHERE '0' IN (TRUE, 'false')",
            "duckdb",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_duckdb_delete_with_nonmatching_bool_numeric_in_allowed(self):
        """Non-matching bool↔numeric IN predicates should remain allowed."""
        result = validate_sql(
            "DELETE FROM users WHERE TRUE IN (0)",
            "duckdb",
            allow_dangerous=True,
        )
        assert result.allowed
        assert result.query_type == "dml"

    def test_duckdb_delete_with_truthy_numeric_literal_blocked(self):
        """DuckDB truthy numeric predicates in WHERE should be blocked."""
        result = validate_sql(
            "DELETE FROM users WHERE 1",
            "duckdb",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_duckdb_delete_with_truthy_string_literal_blocked(self):
        """DuckDB truthy numeric strings in WHERE should be blocked."""
        result = validate_sql(
            "DELETE FROM users WHERE '1'",
            "duckdb",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_duckdb_delete_with_truthy_boolean_string_literal_blocked(self):
        """DuckDB boolean-like truthy strings should be blocked as tautological."""
        result = validate_sql(
            "DELETE FROM users WHERE 'true'",
            "duckdb",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_duckdb_update_with_truthy_yes_string_literal_blocked(self):
        """DuckDB aliases like 'yes' should also be blocked when tautological."""
        result = validate_sql(
            "UPDATE users SET active = false WHERE 'yes'",
            "duckdb",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_duckdb_delete_with_falsey_boolean_string_literal_allowed(self):
        """DuckDB falsey boolean-like strings should remain non-tautological."""
        result = validate_sql(
            "DELETE FROM users WHERE 'false'",
            "duckdb",
            allow_dangerous=True,
        )
        assert result.allowed
        assert result.query_type == "dml"

    def test_mysql_delete_with_truthy_prefixed_string_literal_blocked(self):
        """MySQL numeric-prefix strings should be treated as truthy constants."""
        result = validate_sql(
            "DELETE FROM users WHERE '1abc'",
            "mysql",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_sqlite_delete_with_truthy_prefixed_string_literal_blocked(self):
        """SQLite numeric-prefix strings should be treated as truthy constants."""
        result = validate_sql(
            "DELETE FROM users WHERE '+1foo'",
            "sqlite",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_sqlite_delete_with_truthy_hex_literal_blocked(self):
        """SQLite truthy hex literals in WHERE should be blocked."""
        result = validate_sql(
            "DELETE FROM users WHERE 0x1",
            "sqlite",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_sqlite_delete_with_blob_hex_literal_allowed(self):
        """SQLite blob literals should not be treated as numeric tautologies."""
        result = validate_sql(
            "DELETE FROM users WHERE x'41'",
            "sqlite",
            allow_dangerous=True,
        )
        assert result.allowed
        assert result.query_type == "dml"

    def test_sqlite_delete_with_zero_blob_hex_literal_allowed(self):
        """SQLite zero blob literals should remain non-tautological."""
        result = validate_sql(
            "DELETE FROM users WHERE x'00'",
            "sqlite",
            allow_dangerous=True,
        )
        assert result.allowed
        assert result.query_type == "dml"

    def test_update_with_constant_in_false_allowed(self):
        """Constant IN predicate evaluating to false should remain allowed."""
        result = validate_sql(
            "UPDATE users SET active = false WHERE 1 IN (2)",
            "postgres",
            allow_dangerous=True,
        )
        assert result.allowed
        assert result.query_type == "dml"

    def test_delete_with_constant_exists_false_allowed(self):
        """EXISTS with constant false subquery should remain allowed."""
        result = validate_sql(
            "DELETE FROM users WHERE EXISTS (SELECT 1 WHERE FALSE)",
            "postgres",
            allow_dangerous=True,
        )
        assert result.allowed
        assert result.query_type == "dml"

    def test_mysql_update_with_falsey_coalesce_allowed(self):
        """Falsey constant COALESCE predicate should remain allowed."""
        result = validate_sql(
            "UPDATE users SET active = 0 WHERE COALESCE(NULL, 0)",
            "mysql",
            allow_dangerous=True,
        )
        assert result.allowed
        assert result.query_type == "dml"

    def test_mysql_delete_with_falsey_string_literal_allowed(self):
        """MySQL falsey numeric strings in WHERE should remain allowed."""
        result = validate_sql(
            "DELETE FROM users WHERE '0'",
            "mysql",
            allow_dangerous=True,
        )
        assert result.allowed
        assert result.query_type == "dml"

    def test_mysql_delete_with_falsey_prefixed_string_literal_allowed(self):
        """MySQL non-numeric suffixes after zero-prefix should stay falsey."""
        result = validate_sql(
            "DELETE FROM users WHERE '0abc'",
            "mysql",
            allow_dangerous=True,
        )
        assert result.allowed
        assert result.query_type == "dml"

    def test_mysql_update_with_dynamic_abs_predicate_allowed(self):
        """Dynamic function predicates should remain allowed."""
        result = validate_sql(
            "UPDATE users SET active = 0 WHERE ABS(id)",
            "mysql",
            allow_dangerous=True,
        )
        assert result.allowed
        assert result.query_type == "dml"

    def test_update_with_exists_or_clause_blocked(self):
        """Tautological EXISTS in OR should be blocked as unfiltered."""
        result = validate_sql(
            "UPDATE users SET active = false WHERE EXISTS (SELECT 1) OR id = 1",
            "postgres",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "tautological WHERE" in result.reason

    def test_update_with_exists_and_filter_allowed(self):
        """Tautological EXISTS in AND should preserve filtering behavior."""
        result = validate_sql(
            "UPDATE users SET active = false WHERE EXISTS (SELECT 1) AND id = 1",
            "postgres",
            allow_dangerous=True,
        )
        assert result.allowed
        assert result.query_type == "dml"
