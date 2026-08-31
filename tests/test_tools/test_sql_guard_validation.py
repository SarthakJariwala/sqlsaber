"""Root SQL guard validation tests."""

from sqlsaber.tools.sql_guard import validate_read_only, validate_sql


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
