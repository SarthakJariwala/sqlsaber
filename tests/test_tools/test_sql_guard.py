"""Tests for SQL query validation and security."""

import sys

import sqlglot
from sqlglot import exp

from sqlsaber.tools.sql_guard import (
    add_limit,
    classify_statement,
    has_disallowed_dangerous_mode_statement,
    validate_read_only,
    validate_sql,
)


class TestValidateReadOnly:
    """Tests for read-only query validation."""

    def test_simple_select_allowed(self):
        """Simple SELECT queries should be allowed."""
        result = validate_read_only("SELECT * FROM users", "postgres")
        assert result.allowed
        assert result.is_select

    def test_select_with_where_allowed(self):
        """SELECT with WHERE clause should be allowed."""
        result = validate_read_only(
            "SELECT id, name FROM users WHERE age > 18", "postgres"
        )
        assert result.allowed
        assert result.is_select

    def test_select_with_joins_allowed(self):
        """SELECT with JOINs should be allowed."""
        query = """
        SELECT u.id, u.name, o.total
        FROM users u
        JOIN orders o ON u.id = o.user_id
        WHERE o.status = 'completed'
        """
        result = validate_read_only(query, "postgres")
        assert result.allowed
        assert result.is_select

    def test_select_with_subquery_allowed(self):
        """SELECT with subqueries should be allowed."""
        query = """
        SELECT * FROM users
        WHERE id IN (SELECT user_id FROM orders WHERE total > 100)
        """
        result = validate_read_only(query, "postgres")
        assert result.allowed
        assert result.is_select

    def test_select_with_cte_allowed(self):
        """SELECT with CTEs should be allowed."""
        query = """
        WITH high_value_users AS (
            SELECT user_id FROM orders GROUP BY user_id HAVING SUM(total) > 1000
        )
        SELECT * FROM users WHERE id IN (SELECT user_id FROM high_value_users)
        """
        result = validate_read_only(query, "postgres")
        assert result.allowed
        assert result.is_select

    def test_union_queries_allowed(self):
        """UNION queries should be allowed."""
        query = """
        SELECT name FROM users WHERE active = true
        UNION
        SELECT name FROM archived_users WHERE archived_date > '2024-01-01'
        """
        result = validate_read_only(query, "postgres")
        assert result.allowed

    def test_insert_blocked(self):
        """INSERT queries should be blocked."""
        result = validate_read_only(
            "INSERT INTO users (name, email) VALUES ('John', 'john@example.com')",
            "postgres",
        )
        assert not result.allowed
        assert "Only SELECT" in result.reason

    def test_update_blocked(self):
        """UPDATE queries should be blocked."""
        result = validate_read_only(
            "UPDATE users SET name = 'Jane' WHERE id = 1", "postgres"
        )
        assert not result.allowed
        assert "Only SELECT" in result.reason

    def test_delete_blocked(self):
        """DELETE queries should be blocked."""
        result = validate_read_only("DELETE FROM users WHERE id = 1", "postgres")
        assert not result.allowed
        assert "Only SELECT" in result.reason

    def test_drop_blocked(self):
        """DROP queries should be blocked."""
        result = validate_read_only("DROP TABLE users", "postgres")
        assert not result.allowed
        assert "Only SELECT" in result.reason

    def test_create_table_blocked(self):
        """CREATE TABLE queries should be blocked."""
        result = validate_read_only(
            "CREATE TABLE new_users (id INT, name VARCHAR(100))", "postgres"
        )
        assert not result.allowed
        assert "Only SELECT" in result.reason

    def test_alter_table_blocked(self):
        """ALTER TABLE queries should be blocked."""
        result = validate_read_only(
            "ALTER TABLE users ADD COLUMN phone VARCHAR(20)", "postgres"
        )
        assert not result.allowed
        assert "Only SELECT" in result.reason

    def test_truncate_blocked(self):
        """TRUNCATE queries should be blocked."""
        result = validate_read_only("TRUNCATE TABLE users", "postgres")
        assert not result.allowed
        assert "Only SELECT" in result.reason

    def test_cte_with_insert_blocked(self):
        """CTEs with INSERT should be blocked."""
        query = """
        WITH new_users AS (
            INSERT INTO users (name) VALUES ('John') RETURNING id
        )
        SELECT * FROM new_users
        """
        result = validate_read_only(query, "postgres")
        assert not result.allowed
        assert "Prohibited operation" in result.reason

    def test_cte_with_update_blocked(self):
        """CTEs with UPDATE should be blocked."""
        query = """
        WITH updated AS (
            UPDATE users SET active = false WHERE id = 1 RETURNING id
        )
        SELECT * FROM updated
        """
        result = validate_read_only(query, "postgres")
        assert not result.allowed
        assert "Prohibited operation" in result.reason

    def test_cte_with_delete_blocked(self):
        """CTEs with DELETE should be blocked."""
        query = """
        WITH deleted AS (
            DELETE FROM users WHERE id = 1 RETURNING id
        )
        SELECT * FROM deleted
        """
        result = validate_read_only(query, "postgres")
        assert not result.allowed
        assert "Prohibited operation" in result.reason

    def test_select_into_blocked(self):
        """SELECT INTO should be blocked (Postgres)."""
        result = validate_read_only("SELECT * INTO new_table FROM users", "postgres")
        assert not result.allowed
        assert "SELECT INTO" in result.reason

    def test_select_for_update_blocked(self):
        """SELECT FOR UPDATE should be blocked."""
        result = validate_read_only(
            "SELECT * FROM users WHERE id = 1 FOR UPDATE", "postgres"
        )
        assert not result.allowed
        assert "locking clause" in result.reason

    def test_select_for_share_blocked(self):
        """SELECT FOR SHARE should be blocked."""
        result = validate_read_only(
            "SELECT * FROM users WHERE id = 1 FOR SHARE", "postgres"
        )
        assert not result.allowed
        assert "locking clause" in result.reason

    def test_multi_statement_blocked(self):
        """Multiple statements should be blocked."""
        result = validate_read_only(
            "SELECT * FROM users; SELECT * FROM orders;", "postgres"
        )
        assert not result.allowed
        assert "single SELECT" in result.reason

    def test_multi_statement_with_drop_blocked(self):
        """Multiple statements with DROP should be blocked."""
        result = validate_read_only(
            "SELECT * FROM users; DROP TABLE users;", "postgres"
        )
        assert not result.allowed
        assert "single SELECT" in result.reason

    def test_copy_blocked_postgres(self):
        """COPY should be blocked (Postgres)."""
        result = validate_read_only("COPY users TO '/tmp/users.csv'", "postgres")
        assert not result.allowed
        assert "Only SELECT" in result.reason

    def test_explain_blocked(self):
        """EXPLAIN should be blocked for simplicity."""
        result = validate_read_only("EXPLAIN SELECT * FROM users", "postgres")
        assert not result.allowed
        assert "Only SELECT" in result.reason

    def test_postgres_dangerous_function_pg_read_file(self):
        """Postgres dangerous functions should be blocked."""
        result = validate_read_only("SELECT pg_read_file('/etc/passwd')", "postgres")
        assert not result.allowed
        assert "dangerous function" in result.reason.lower()

    def test_mysql_dangerous_function_load_file(self):
        """MySQL dangerous functions should be blocked."""
        result = validate_read_only("SELECT LOAD_FILE('/etc/passwd')", "mysql")
        assert not result.allowed
        assert "dangerous function" in result.reason.lower()

    def test_sqlite_dangerous_function_readfile(self):
        """SQLite dangerous functions should be blocked."""
        result = validate_read_only("SELECT readfile('/etc/passwd')", "sqlite")
        assert not result.allowed
        assert "dangerous function" in result.reason.lower()

    def test_parse_error_blocked(self):
        """Unparseable queries should be blocked."""
        result = validate_read_only("SELECT FROM WHERE", "postgres")
        assert not result.allowed
        assert "parse" in result.reason.lower()

    def test_create_table_as_select_blocked(self):
        """CREATE TABLE AS SELECT should be blocked."""
        result = validate_read_only(
            "CREATE TABLE new_users AS SELECT * FROM users", "postgres"
        )
        assert not result.allowed
        assert "Only SELECT" in result.reason

    def test_insert_into_select_blocked(self):
        """INSERT INTO ... SELECT should be blocked."""
        result = validate_read_only(
            "INSERT INTO backup_users SELECT * FROM users", "postgres"
        )
        assert not result.allowed
        assert "Only SELECT" in result.reason

    def test_merge_blocked(self):
        """MERGE should be blocked."""
        query = """
        MERGE INTO target t
        USING source s ON t.id = s.id
        WHEN MATCHED THEN UPDATE SET t.value = s.value
        WHEN NOT MATCHED THEN INSERT VALUES (s.id, s.value)
        """
        result = validate_read_only(query, "postgres")
        assert not result.allowed
        assert "Only SELECT" in result.reason


class TestAddLimit:
    """Tests for adding LIMIT clauses."""

    def test_add_limit_to_simple_select(self):
        """Should add LIMIT to simple SELECT."""
        query = "SELECT * FROM users"
        result = add_limit(query, "postgres", 100)
        assert "LIMIT" in result.upper()
        assert "100" in result

    def test_preserve_existing_limit(self):
        """Should preserve existing LIMIT."""
        query = "SELECT * FROM users LIMIT 50"
        result = add_limit(query, "postgres", 100)
        assert "50" in result
        assert "100" not in result

    def test_add_limit_to_query_with_where(self):
        """Should add LIMIT to query with WHERE."""
        query = "SELECT * FROM users WHERE age > 18"
        result = add_limit(query, "postgres", 100)
        assert "LIMIT" in result.upper()
        assert "WHERE age > 18" in result

    def test_add_limit_to_union(self):
        """Should add LIMIT to UNION queries."""
        query = "SELECT name FROM users UNION SELECT name FROM archived_users"
        result = add_limit(query, "postgres", 100)
        assert "LIMIT" in result.upper()

    def test_add_limit_with_existing_offset(self):
        """Should work with existing OFFSET."""
        query = "SELECT * FROM users OFFSET 10"
        result = add_limit(query, "postgres", 100)
        # Should add LIMIT
        assert "LIMIT" in result.upper()

    def test_mysql_limit_syntax(self):
        """MySQL should use LIMIT syntax."""
        query = "SELECT * FROM users"
        result = add_limit(query, "mysql", 100)
        assert "LIMIT" in result.upper()
        assert "100" in result

    def test_sqlite_limit_syntax(self):
        """SQLite should use LIMIT syntax."""
        query = "SELECT * FROM users"
        result = add_limit(query, "sqlite", 100)
        assert "LIMIT" in result.upper()
        assert "100" in result

    def test_fallback_on_parse_error(self):
        """Should fall back to simple append on parse errors."""
        # Even invalid SQL should get LIMIT appended as a fallback
        query = "SELECT FROM WHERE"
        result = add_limit(query, "postgres", 100)
        # Fallback should still try to add LIMIT
        assert "LIMIT" in result.upper()

    def test_strips_trailing_semicolon(self):
        """Should strip trailing semicolon before adding LIMIT."""
        query = "SELECT * FROM users;"
        result = add_limit(query, "postgres", 100)
        # Should not end with ;
        assert result.strip().endswith("LIMIT 100")
        assert ";" not in result[-5:]  # Ensure no semicolon at the very end


class TestValidateSql:
    """Tests for validate_sql with allow_dangerous mode."""

    def test_delegates_to_read_only_by_default(self):
        """When allow_dangerous=False, should behave like validate_read_only."""
        result = validate_sql("SELECT * FROM users", "postgres", allow_dangerous=False)
        assert result.allowed
        assert result.is_select

        result = validate_sql(
            "INSERT INTO users (name) VALUES ('test')",
            "postgres",
            allow_dangerous=False,
        )
        assert not result.allowed

    def test_insert_allowed_in_dangerous_mode(self):
        """INSERT should be allowed when allow_dangerous=True."""
        result = validate_sql(
            "INSERT INTO users (name, email) VALUES ('John', 'john@example.com')",
            "postgres",
            allow_dangerous=True,
        )
        assert result.allowed
        assert not result.is_select

    def test_update_allowed_in_dangerous_mode(self):
        """UPDATE should be allowed when allow_dangerous=True."""
        result = validate_sql(
            "UPDATE users SET name = 'Jane' WHERE id = 1",
            "postgres",
            allow_dangerous=True,
        )
        assert result.allowed
        assert not result.is_select

    def test_delete_allowed_in_dangerous_mode(self):
        """DELETE should be allowed when allow_dangerous=True."""
        result = validate_sql(
            "DELETE FROM users WHERE id = 1", "postgres", allow_dangerous=True
        )
        assert result.allowed
        assert not result.is_select

    def test_merge_blocked_in_dangerous_mode(self):
        """MERGE should be blocked in dangerous mode (not in allowlist)."""
        result = validate_sql(
            """
            MERGE INTO target t
            USING source s ON t.id = s.id
            WHEN MATCHED THEN UPDATE SET t.value = s.value
            WHEN NOT MATCHED THEN INSERT (id, value) VALUES (s.id, s.value)
            """,
            "postgres",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason

    def test_replace_blocked_in_dangerous_mode(self):
        """REPLACE should be blocked in dangerous mode (not in allowlist)."""
        result = validate_sql(
            "REPLACE INTO users(id, name) VALUES (1, 'x')",
            "mysql",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert result.reason

    def test_create_table_allowed_in_dangerous_mode(self):
        """CREATE TABLE should be allowed when allow_dangerous=True."""
        result = validate_sql(
            "CREATE TABLE new_users (id INT, name VARCHAR(100))",
            "postgres",
            allow_dangerous=True,
        )
        assert result.allowed
        assert not result.is_select

    def test_drop_table_blocked_in_dangerous_mode(self):
        """DROP TABLE should be blocked even in dangerous mode."""
        result = validate_sql("DROP TABLE users", "postgres", allow_dangerous=True)
        assert not result.allowed
        assert result.reason
        assert "Prohibited operation" in result.reason

    def test_alter_table_allowed_in_dangerous_mode(self):
        """ALTER TABLE should be allowed when allow_dangerous=True."""
        result = validate_sql(
            "ALTER TABLE users ADD COLUMN phone VARCHAR(20)",
            "postgres",
            allow_dangerous=True,
        )
        assert result.allowed
        assert not result.is_select

    def test_truncate_blocked_in_dangerous_mode(self):
        """TRUNCATE should be blocked even in dangerous mode."""
        result = validate_sql("TRUNCATE TABLE users", "postgres", allow_dangerous=True)
        assert not result.allowed
        assert result.reason
        assert "Prohibited operation" in result.reason

    def test_select_still_works_in_dangerous_mode(self):
        """SELECT should still work and be marked as is_select in dangerous mode."""
        result = validate_sql(
            "SELECT * FROM users WHERE id = 1", "postgres", allow_dangerous=True
        )
        assert result.allowed
        assert result.is_select

    def test_multi_statement_blocked_in_dangerous_mode(self):
        """Multiple statements should still be blocked in dangerous mode."""
        result = validate_sql(
            "SELECT * FROM users; DROP TABLE users;",
            "postgres",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert "single statements" in result.reason

    def test_dangerous_functions_blocked_in_dangerous_mode(self):
        """Dangerous functions should still be blocked even in dangerous mode."""
        result = validate_sql(
            "SELECT pg_read_file('/etc/passwd')", "postgres", allow_dangerous=True
        )
        assert not result.allowed
        assert "dangerous function" in result.reason.lower()

    def test_dangerous_functions_in_insert_blocked(self):
        """Dangerous functions in INSERT should be blocked in dangerous mode."""
        result = validate_sql(
            "INSERT INTO files (content) VALUES (pg_read_file('/etc/passwd'))",
            "postgres",
            allow_dangerous=True,
        )
        assert not result.allowed
        assert "dangerous function" in result.reason.lower()

    def test_mysql_load_file_blocked_in_dangerous_mode(self):
        """MySQL LOAD_FILE should be blocked even in dangerous mode."""
        result = validate_sql(
            "SELECT LOAD_FILE('/etc/passwd')", "mysql", allow_dangerous=True
        )
        assert not result.allowed
        assert "dangerous function" in result.reason.lower()

    def test_sqlite_readfile_blocked_in_dangerous_mode(self):
        """SQLite readfile should be blocked even in dangerous mode."""
        result = validate_sql(
            "SELECT readfile('/etc/passwd')", "sqlite", allow_dangerous=True
        )
        assert not result.allowed
        assert "dangerous function" in result.reason.lower()

    def test_parse_error_blocked_in_dangerous_mode(self):
        """Unparseable queries should be blocked in dangerous mode."""
        result = validate_sql("SELECT FROM WHERE", "postgres", allow_dangerous=True)
        assert not result.allowed
        assert "parse" in result.reason.lower()


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


class TestQueryTypeInGuardResult:
    """Tests for query_type field in GuardResult."""

    def test_select_query_type_in_read_only(self):
        """validate_read_only should set query_type='select'."""
        result = validate_read_only("SELECT * FROM users", "postgres")
        assert result.allowed
        assert result.query_type == "select"

    def test_select_query_type_in_dangerous_mode(self):
        """SELECT in dangerous mode should have query_type='select'."""
        result = validate_sql("SELECT * FROM users", "postgres", allow_dangerous=True)
        assert result.allowed
        assert result.query_type == "select"

    def test_insert_query_type(self):
        """INSERT should have query_type='dml'."""
        result = validate_sql(
            "INSERT INTO users (name) VALUES ('test')",
            "postgres",
            allow_dangerous=True,
        )
        assert result.allowed
        assert result.query_type == "dml"

    def test_update_query_type(self):
        """UPDATE should have query_type='dml'."""
        result = validate_sql(
            "UPDATE users SET name = 'test' WHERE id = 1",
            "postgres",
            allow_dangerous=True,
        )
        assert result.allowed
        assert result.query_type == "dml"

    def test_delete_query_type(self):
        """DELETE should have query_type='dml'."""
        result = validate_sql(
            "DELETE FROM users WHERE id = 1",
            "postgres",
            allow_dangerous=True,
        )
        assert result.allowed
        assert result.query_type == "dml"

    def test_create_table_query_type(self):
        """CREATE TABLE should have query_type='ddl'."""
        result = validate_sql(
            "CREATE TABLE users (id INT)",
            "postgres",
            allow_dangerous=True,
        )
        assert result.allowed
        assert result.query_type == "ddl"

    def test_alter_table_query_type(self):
        """ALTER TABLE should have query_type='ddl'."""
        result = validate_sql(
            "ALTER TABLE users ADD COLUMN email VARCHAR(100)",
            "postgres",
            allow_dangerous=True,
        )
        assert result.allowed
        assert result.query_type == "ddl"


class TestAlwaysBlockedInDangerousMode:
    """Tests for operations that remain blocked even in dangerous mode."""

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
