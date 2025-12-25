"""Tests for SQL query validation and security."""

import sqlglot

from sqlsaber.tools.sql_guard import (
    add_limit,
    classify_statement,
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
