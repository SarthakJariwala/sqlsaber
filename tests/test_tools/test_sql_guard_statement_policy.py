"""SQL statement policy tests."""

import pytest
import sqlglot
from sqlglot import exp

from sqlsaber.tools.sql_guard import (
    classify_statement,
    has_disallowed_dangerous_mode_statement,
    validate_sql,
)


class TestClassifyStatement:
    """Tests for classify_statement function."""

    def test_select_classified_as_select(self):
        """SELECT statements should be classified as 'select'."""
        stmt = sqlglot.parse("SELECT * FROM users")[0]
        assert stmt
        assert classify_statement(stmt) == "select"

    def test_select_with_cte_classified_as_select(self):
        """SELECT with CTE should be classified as 'select'."""
        stmt = sqlglot.parse("WITH cte AS (SELECT 1) SELECT * FROM cte")[0]
        assert stmt
        assert classify_statement(stmt) == "select"

    def test_union_classified_as_select(self):
        """UNION queries should be classified as 'select'."""
        stmt = sqlglot.parse("SELECT 1 UNION SELECT 2")[0]
        assert stmt
        assert classify_statement(stmt) == "select"

    def test_insert_classified_as_dml(self):
        """INSERT statements should be classified as 'dml'."""
        stmt = sqlglot.parse("INSERT INTO users (name) VALUES ('test')")[0]
        assert stmt
        assert classify_statement(stmt) == "dml"

    def test_update_classified_as_dml(self):
        """UPDATE statements should be classified as 'dml'."""
        stmt = sqlglot.parse("UPDATE users SET name = 'test' WHERE id = 1")[0]
        assert stmt
        assert classify_statement(stmt) == "dml"

    def test_delete_classified_as_dml(self):
        """DELETE statements should be classified as 'dml'."""
        stmt = sqlglot.parse("DELETE FROM users WHERE id = 1")[0]
        assert stmt
        assert classify_statement(stmt) == "dml"

    def test_create_table_classified_as_ddl(self):
        """CREATE TABLE statements should be classified as 'ddl'."""
        stmt = sqlglot.parse("CREATE TABLE users (id INT)")[0]
        assert stmt
        assert classify_statement(stmt) == "ddl"

    def test_drop_table_classified_as_ddl(self):
        """DROP TABLE statements should be classified as 'ddl'."""
        stmt = sqlglot.parse("DROP TABLE users")[0]
        assert stmt
        assert classify_statement(stmt) == "ddl"

    def test_alter_table_classified_as_ddl(self):
        """ALTER TABLE statements should be classified as 'ddl'."""
        stmt = sqlglot.parse("ALTER TABLE users ADD COLUMN email VARCHAR(100)")[0]
        assert stmt
        assert classify_statement(stmt) == "ddl"

    def test_truncate_classified_as_ddl(self):
        """TRUNCATE statements should be classified as 'ddl'."""
        stmt = sqlglot.parse("TRUNCATE TABLE users")[0]
        assert stmt
        assert classify_statement(stmt) == "ddl"


class TestAlwaysBlockedInDangerousMode:
    """Tests for operations that remain blocked in dangerous mode."""

    def test_copy_blocked_in_dangerous_mode(self):
        """COPY should be blocked even in dangerous mode."""
        result = validate_sql(
            "COPY users TO '/tmp/users.csv'", "postgres", allow_dangerous=True
        )
        assert not result.allowed
        assert result.reason
        assert "Prohibited operation" in result.reason

    def test_set_blocked_in_dangerous_mode(self):
        """SET should be blocked even in dangerous mode."""
        result = validate_sql(
            "SET search_path TO myschema", "postgres", allow_dangerous=True
        )
        assert not result.allowed
        assert result.reason
        assert "Prohibited operation" in result.reason

    def test_pragma_blocked_in_dangerous_mode(self):
        """PRAGMA should be blocked even in dangerous mode."""
        result = validate_sql("PRAGMA journal_mode=WAL", "sqlite", allow_dangerous=True)
        assert not result.allowed
        assert result.reason
        assert "Prohibited operation" in result.reason

    def test_attach_blocked_in_dangerous_mode(self):
        """ATTACH should be blocked even in dangerous mode."""
        result = validate_sql(
            "ATTACH DATABASE 'file.db' AS other", "sqlite", allow_dangerous=True
        )
        assert not result.allowed
        assert result.reason
        assert "Prohibited operation" in result.reason

    def test_detach_blocked_in_dangerous_mode(self):
        """DETACH should be blocked even in dangerous mode."""
        result = validate_sql("DETACH DATABASE other", "sqlite", allow_dangerous=True)
        assert not result.allowed
        assert result.reason
        assert "Prohibited operation" in result.reason

    def test_grant_blocked_in_dangerous_mode(self):
        """GRANT should be blocked even in dangerous mode."""
        result = validate_sql(
            "GRANT SELECT ON users TO public", "postgres", allow_dangerous=True
        )
        assert not result.allowed
        assert result.reason
        assert "Prohibited operation" in result.reason

    def test_revoke_blocked_in_dangerous_mode(self):
        """REVOKE should be blocked even in dangerous mode."""
        result = validate_sql(
            "REVOKE SELECT ON users FROM public", "postgres", allow_dangerous=True
        )
        assert not result.allowed
        assert result.reason
        assert "Prohibited operation" in result.reason

    def test_select_for_update_blocked_in_dangerous_mode(self):
        """SELECT FOR UPDATE should be blocked even in dangerous mode."""
        result = validate_sql(
            "SELECT * FROM users WHERE id = 1 FOR UPDATE",
            "postgres",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "locking clause" in result.reason

    def test_select_into_blocked_in_dangerous_mode(self):
        """SELECT INTO should be blocked even in dangerous mode."""
        result = validate_sql(
            "SELECT * INTO new_table FROM users", "postgres", allow_dangerous=True
        )
        assert not result.allowed
        assert result.reason
        assert "SELECT INTO" in result.reason


class TestAlterSubOperations:
    """Tests for ALTER sub-operations (DROP COLUMN, etc.)."""

    def test_alter_add_column_allowed(self):
        """ALTER TABLE ADD COLUMN should be allowed in dangerous mode."""
        result = validate_sql(
            "ALTER TABLE users ADD COLUMN email VARCHAR(100)",
            "postgres",
            allow_dangerous=True,
        )
        assert result.allowed
        assert result.query_type == "ddl"

    def test_alter_drop_column_blocked(self):
        """ALTER TABLE DROP COLUMN should be blocked (contains exp.Drop)."""
        result = validate_sql(
            "ALTER TABLE users DROP COLUMN email",
            "postgres",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "Prohibited operation" in result.reason

    def test_alter_rename_table_allowed(self):
        """ALTER TABLE RENAME should be allowed (non-destructive)."""
        result = validate_sql(
            "ALTER TABLE users RENAME TO old_users",
            "postgres",
            allow_dangerous=True,
        )
        assert result.allowed
        assert result.query_type == "ddl"

    def test_alter_rename_column_allowed(self):
        """ALTER TABLE RENAME COLUMN should be allowed (non-destructive)."""
        result = validate_sql(
            "ALTER TABLE users RENAME COLUMN name TO full_name",
            "postgres",
            allow_dangerous=True,
        )
        assert result.allowed
        assert result.query_type == "ddl"

    def test_alter_drop_constraint_blocked(self):
        """ALTER TABLE DROP CONSTRAINT should be blocked (contains exp.Drop)."""
        result = validate_sql(
            "ALTER TABLE users DROP CONSTRAINT fk_orders",
            "postgres",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason
        assert "Prohibited operation" in result.reason

    def test_alter_set_default_allowed(self):
        """ALTER TABLE SET DEFAULT should be allowed (non-destructive)."""
        result = validate_sql(
            "ALTER TABLE users ALTER COLUMN age SET DEFAULT 0",
            "postgres",
            allow_dangerous=True,
        )
        assert result.allowed
        assert result.query_type == "ddl"


class TestDangerousModeCreateValidation:
    """Tests for CREATE statement subtype hardening in dangerous mode."""

    def test_create_view_requires_select_expression(self):
        """CREATE VIEW should only allow SELECT-like expressions."""
        table = exp.Table(this=exp.to_identifier("v"))
        insert_expr = sqlglot.parse_one("INSERT INTO t VALUES (1)", read="postgres")
        assert insert_expr

        stmt = exp.Create(kind="VIEW", this=table, expression=insert_expr)
        reason = has_disallowed_dangerous_mode_statement(stmt)

        assert reason
        assert "CREATE VIEW must be based on a SELECT-like expression" in reason

    def test_create_index_requires_index_target(self):
        """CREATE INDEX should reject non-index AST targets."""
        stmt = exp.Create(kind="INDEX", this=exp.Table(this=exp.to_identifier("idx")))

        reason = has_disallowed_dangerous_mode_statement(stmt)

        assert reason
        assert "Only CREATE INDEX statements" in reason

    def test_create_table_requires_table_or_schema_target(self):
        """CREATE TABLE should reject malformed AST targets."""
        stmt = exp.Create(kind="TABLE", this=exp.Literal.string("bad_target"))

        reason = has_disallowed_dangerous_mode_statement(stmt)

        assert reason
        assert "Only CREATE TABLE statements" in reason


STRICT_DANGEROUS_MODE_BLOCK_CASES = [
    # Unknown/unclassified command should not fail-open
    ("postgres", "FOO BAR"),
    # DML not in dangerous-mode allowlist
    (
        "postgres",
        "MERGE INTO target t USING source s ON t.id = s.id "
        "WHEN MATCHED THEN UPDATE SET t.value = s.value "
        "WHEN NOT MATCHED THEN INSERT (id, value) VALUES (s.id, s.value)",
    ),
    ("mysql", "REPLACE INTO users(id, name) VALUES (1, 'x')"),
    # PostgreSQL statements that should not be in dangerous allowlist
    ("postgres", "CHECKPOINT"),
    ("postgres", "LISTEN chan"),
    ("postgres", "DISCARD ALL"),
    ("postgres", "COMMENT ON TABLE users IS 'x'"),
    ("postgres", "CREATE DATABASE scratch_db"),
    (
        "postgres",
        "CREATE OR REPLACE FUNCTION f() RETURNS int LANGUAGE sql AS $$ SELECT 1 $$",
    ),
    # MySQL statements that should not be in dangerous allowlist
    ("mysql", "FLUSH PRIVILEGES"),
    ("mysql", "RESET MASTER"),
    ("mysql", "CREATE FUNCTION myudf RETURNS STRING SONAME 'udf.so'"),
    # DuckDB statements that should not be in dangerous allowlist
    ("duckdb", "INSTALL httpfs"),
    ("duckdb", "CHECKPOINT"),
    ("duckdb", "COMMENT ON TABLE t IS 'x'"),
    # SQLite statements that should not be in dangerous allowlist
    ("sqlite", "REINDEX"),
]


@pytest.mark.parametrize(("dialect", "query"), STRICT_DANGEROUS_MODE_BLOCK_CASES)
def test_dangerous_mode_uses_strict_allowlist_no_fail_open(
    dialect: str,
    query: str,
):
    """allow_dangerous=True must not allow unknown/admin/executable statements."""
    result = validate_sql(query, dialect, allow_dangerous=True)

    assert not result.allowed
    assert result.reason
