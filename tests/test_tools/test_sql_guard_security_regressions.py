"""Security regression tests for SQL guard hardening.

These tests intentionally codify stricter security expectations:
- Safe mode must be side-effect free.
- Dangerous mode must be strict allowlist/fail-closed.
"""

import pytest

from sqlsaber.tools.sql_guard import validate_read_only, validate_sql

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
    ("mysql", "SELECT RELEASE_ALL_LOCKS()"),
    ("mysql", "SELECT MASTER_POS_WAIT('mysql-bin.000001', 4)"),
    ("mysql", "SELECT SOURCE_POS_WAIT('mysql-bin.000001', 4)"),
    ("mysql", "SELECT WAIT_FOR_EXECUTED_GTID_SET('uuid:1-10', 5)"),
    ("mysql", "SELECT WAIT_UNTIL_SQL_THREAD_AFTER_GTIDS('uuid:1-10', 5)"),
    # SQLite extension loading
    ("sqlite", "SELECT load_extension('/tmp/ext.so')"),
    # DuckDB file-read table/functions (including typed sqlglot nodes)
    ("duckdb", "SELECT * FROM read_csv_auto('/etc/passwd')"),
    ("duckdb", "SELECT * FROM read_json_auto('/etc/passwd')"),
    ("duckdb", "SELECT * FROM parquet_scan('/etc/passwd')"),
    ("duckdb", "SELECT * FROM read_parquet('/etc/passwd')"),
    ("duckdb", "SELECT * FROM parquet_metadata('/etc/passwd')"),
    ("duckdb", "SELECT * FROM parquet_schema('/etc/passwd')"),
    ("duckdb", "SELECT * FROM parquet_file_metadata('/etc/passwd')"),
    ("duckdb", "SELECT * FROM parquet_kv_metadata('/etc/passwd')"),
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
    ("duckdb", "SELECT * FROM read_json_objects('file.json')"),
    ("duckdb", "SELECT * FROM read_xml('file.xml')"),
    ("duckdb", "SELECT * FROM read_avro('file.avro')"),
    ("duckdb", "SELECT * FROM read_ipc('file.arrow')"),
    ("duckdb", "SELECT * FROM read_feather('file.feather')"),
    ("duckdb", "SELECT * FROM read_ndjson('file.ndjson')"),
    ("duckdb", "SELECT * FROM read_ndjson_auto('file.ndjson')"),
    ("duckdb", "SELECT * FROM read_ndjson_objects('file.ndjson')"),
    ("duckdb", "SELECT * FROM read_csv_objects('file.csv')"),
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
    ("duckdb", "SELECT * FROM read_xlsx('file.xlsx')"),
    ("duckdb", "SELECT * FROM st_read('file.geojson')"),
    # DuckDB replacement scans: a bare quoted path/URL/glob in FROM reads files
    # with no function call (parses as Table -> Identifier).
    ("duckdb", "SELECT * FROM '/etc/passwd'"),
    ("duckdb", "SELECT * FROM 'secrets.parquet'"),
    ("duckdb", "SELECT * FROM '/data/*.csv'"),
    ("duckdb", "SELECT * FROM 'https://evil.example.com/x.csv'"),
    ("duckdb", "SELECT a.x FROM '/etc/passwd' AS a"),
    (
        "duckdb",
        "SELECT * FROM read_csv_auto('/etc/passwd') UNION SELECT * FROM '/etc/passwd'",
    ),
    # DuckDB readers/secrets/session functions missing from the denylist.
    ("duckdb", "SELECT * FROM sniff_csv('/etc/passwd')"),
    ("duckdb", "SELECT * FROM duckdb_secrets()"),
    ("duckdb", "SELECT which_secret('s3://bucket/key', 's3')"),
    ("duckdb", "SELECT getvariable('my_secret')"),
    # PostgreSQL: functions that execute arbitrary SQL strings or read large objects.
    ("postgres", "SELECT query_to_xml('DELETE FROM users', true, true, '')"),
    ("postgres", "SELECT query_to_xmlschema('SELECT 1', true, true, '')"),
    ("postgres", "SELECT query_to_xml_and_xmlschema('SELECT 1', true, true, '')"),
    ("postgres", "SELECT table_to_xml('users', true, true, '')"),
    ("postgres", "SELECT cursor_to_xml('mycursor', 0, true, true, '')"),
    ("postgres", "SELECT lo_get(1234)"),
    ("postgres", "SELECT loread(0, 100)"),
    ("postgres", "SELECT lo_put(1234, 0, 'abc')"),
    ("postgres", "SELECT pg_logical_emit_message(true, 'p', 'msg')"),
    # SQLite FTS3 tokenizer pointer primitive.
    ("sqlite", "SELECT fts3_tokenizer('x', x'0000000000000000')"),
    # Sequence side effects bypass SET TRANSACTION READ ONLY in Postgres.
    ("postgres", "SELECT nextval('my_seq')"),
    ("postgres", "SELECT setval('my_seq', 1)"),
    ("postgres", "SELECT currval('my_seq')"),
    ("postgres", "SELECT lastval()"),
    ("duckdb", "SELECT nextval('my_seq')"),
    ("duckdb", "SELECT currval('my_seq')"),
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


UNKNOWN_DIALECT_FAIL_CLOSED_CASES = [
    # Dangerous functions from any known dialect must be blocked even when the
    # dialect is unrecognized (denylist must fail closed, not fail open).
    ("ansi", "SELECT pg_sleep(30)"),
    ("ansi", "SELECT load_file('/etc/passwd')"),
    ("totally-made-up", "SELECT pg_read_file('/etc/passwd')"),
    ("totally-made-up", "SELECT sys_eval('id')"),
]


@pytest.mark.parametrize(("dialect", "query"), UNKNOWN_DIALECT_FAIL_CLOSED_CASES)
def test_unknown_dialect_fails_closed_on_dangerous_functions(
    dialect: str,
    query: str,
):
    """Unrecognized dialects must still block known dangerous functions."""
    result = validate_read_only(query, dialect)

    assert not result.allowed
    assert result.reason


@pytest.mark.parametrize(("dialect", "query"), STRICT_DANGEROUS_MODE_BLOCK_CASES)
def test_dangerous_mode_uses_strict_allowlist_no_fail_open(
    dialect: str,
    query: str,
):
    """allow_dangerous=True must not allow unknown/admin/executable statements."""
    result = validate_sql(query, dialect, allow_dangerous=True)

    assert not result.allowed
    assert result.reason
