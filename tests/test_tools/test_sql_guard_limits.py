"""SQL limit handling tests."""

from sqlsaber.tools.sql_guard import add_limit


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
