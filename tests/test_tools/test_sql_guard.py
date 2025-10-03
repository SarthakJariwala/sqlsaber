"""Tests for SQL query validation and security."""

from sqlsaber.tools.sql_guard import add_limit, validate_read_only


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
