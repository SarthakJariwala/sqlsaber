"""Security regression tests for SQL guard hardening.

These tests intentionally codify stricter security expectations:
- Safe mode must be side-effect free.
- Dangerous mode must be strict allowlist/fail-closed.
"""

import pytest

from sqlsaber.tools.sql_guard import validate_sql

SAFE_MODE_SIDE_EFFECT_CASES = [
    # PostgreSQL side-effectful/admin/file/process functions
    ("postgres", "SELECT pg_terminate_backend(123)"),
    ("postgres", "SELECT pg_cancel_backend(123)"),
    ("postgres", "SELECT pg_reload_conf()"),
    ("postgres", "SELECT pg_rotate_logfile()"),
    ("postgres", "SELECT pg_notify('chan', 'msg')"),
    ("postgres", "SELECT lo_import('/etc/passwd')"),
    ("postgres", "SELECT lo_export(123, '/tmp/out.bin')"),
    ("postgres", "SELECT pg_write_file('/tmp/x', 'abc')"),
    ("postgres", "SELECT pg_append_file('/tmp/x', 'abc')"),
    ("postgres", "SELECT pg_advisory_lock(1)"),
    # PostgreSQL resource exhaustion and metadata leak functions
    ("postgres", "SELECT pg_sleep(30)"),
    ("postgres", "SELECT pg_sleep_for('30 seconds')"),
    ("postgres", "SELECT pg_sleep_until('2099-01-01 00:00:00+00')"),
    ("postgres", "SELECT inet_server_addr()"),
    ("postgres", "SELECT inet_server_port()"),
    ("postgres", "SELECT pg_ls_tmpdir()"),
    ("postgres", "SELECT current_setting('data_directory')"),
    # PostgreSQL family-level hardening (prefix denylist catches variants)
    ("postgres", "SELECT pg_ls_logicalmapdir()"),
    ("postgres", "SELECT pg_ls_logicalsnapdir()"),
    ("postgres", "SELECT pg_ls_replslotdir()"),
    ("postgres", "SELECT pg_try_advisory_xact_lock(1)"),
    ("postgres", "SELECT pg_try_advisory_xact_lock_shared(1)"),
    ("postgres", "SELECT pg_advisory_unlock_all()"),
    ("postgres", "SELECT inet_client_addr()"),
    ("postgres", "SELECT inet_client_port()"),
    ("postgres", "SELECT pg_relation_filepath('pg_class')"),
    ("postgres", "SELECT pg_tablespace_location(1663)"),
    ("postgres", "SELECT pg_current_logfile()"),
    ("postgres", "SELECT pg_stat_reset()"),
    ("postgres", "SELECT pg_stat_reset_subscription_stats(1)"),
    ("postgres", "SELECT dblink_connect('myconn', 'dbname=postgres')"),
    ("postgres", "SELECT dblink_disconnect('myconn')"),
    ("postgres", "SELECT pg_switch_wal()"),
    ("postgres", "SELECT pg_create_restore_point('sweep')"),
    ("postgres", "SELECT pg_promote()"),
    # MySQL side-effectful/resource/locking functions
    ("mysql", "SELECT SLEEP(10)"),
    ("mysql", "SELECT BENCHMARK(1000000, SHA2('x', 512))"),
    ("mysql", "SELECT GET_LOCK('lockname', 10)"),
    ("mysql", "SELECT RELEASE_LOCK('lockname')"),
    # SQLite extension loading
    ("sqlite", "SELECT load_extension('/tmp/ext.so')"),
    # DuckDB file-read table/functions (including typed sqlglot nodes)
    ("duckdb", "SELECT * FROM read_csv_auto('/etc/passwd')"),
    ("duckdb", "SELECT * FROM read_json_auto('/etc/passwd')"),
    ("duckdb", "SELECT * FROM parquet_scan('/etc/passwd')"),
    ("duckdb", "SELECT * FROM read_parquet('/etc/passwd')"),
    # MySQL version comment parser divergence — sqlglot strips these as comments
    # but MySQL server executes the content inside /*!NNNNN ... */
    ("mysql", "SELECT 1, /*!50000 LOAD_FILE('/etc/passwd') */ FROM dual LIMIT 10"),
    ("mysql", "SELECT /*!50000 1, LOAD_FILE('/etc/passwd'), */ 2 FROM dual"),
    ("mysql", "SELECT 1 /*!50000 , sys_eval('id') */ FROM dual"),
    ("mysql", "SELECT /*!50000 SLEEP(10) */ FROM dual"),
    ("mysql", "SELECT /*!50000 BENCHMARK(1000000, SHA2('x', 512)) */ FROM dual"),
    # DuckDB — missing file/network/extension functions
    ("duckdb", "SELECT * FROM read_text('/etc/passwd')"),
    ("duckdb", "SELECT * FROM read_blob('/etc/shadow')"),
    ("duckdb", "SELECT * FROM read_ndjson('file.ndjson')"),
    ("duckdb", "SELECT * FROM read_ndjson_auto('file.ndjson')"),
    ("duckdb", "SELECT * FROM read_ndjson_objects('file.ndjson')"),
    ("duckdb", "SELECT * FROM glob('/etc/*')"),
    ("duckdb", "SELECT * FROM sqlite_scan('db.sqlite', 'table1')"),
    ("duckdb", "SELECT * FROM postgres_scan('connstr', 'public', 'table')"),
    ("duckdb", "SELECT * FROM mysql_scan('connstr', 'schema', 'table')"),
    ("duckdb", "SELECT * FROM postgres_query('connstr', 'SELECT 1')"),
    ("duckdb", "SELECT * FROM mysql_query('connstr', 'SELECT 1')"),
    ("duckdb", "SELECT load_extension('httpfs')"),
    ("duckdb", "SELECT install_extension('httpfs')"),
    ("duckdb", "SELECT * FROM iceberg_scan('path')"),
    ("duckdb", "SELECT * FROM delta_scan('path')"),
    ("duckdb", "SELECT * FROM excel_scan('file.xlsx')"),
    ("duckdb", "SELECT * FROM st_read('file.geojson')"),
]


@pytest.mark.parametrize(("dialect", "query"), SAFE_MODE_SIDE_EFFECT_CASES)
def test_safe_mode_blocks_side_effectful_queries(dialect: str, query: str):
    """allow_dangerous=False must be truly side-effect free."""
    result = validate_sql(query, dialect, allow_dangerous=False)

    assert not result.allowed
    assert result.reason


@pytest.mark.parametrize(("dialect", "query"), SAFE_MODE_SIDE_EFFECT_CASES)
def test_dangerous_mode_still_blocks_absolute_no_no_side_effects(
    dialect: str,
    query: str,
):
    """Absolute no-no side-effectful queries must also be blocked in dangerous mode."""
    result = validate_sql(query, dialect, allow_dangerous=True)

    assert not result.allowed
    assert result.reason


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
