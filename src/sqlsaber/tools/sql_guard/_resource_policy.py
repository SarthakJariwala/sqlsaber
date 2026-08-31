"""Dangerous function and file-target policy."""

from sqlglot import exp

# alter session/server state, or otherwise introduce side effects.
DANGEROUS_FUNCTIONS_BY_DIALECT: dict[str, set[str]] = {
    "postgres": {
        # File/system access and filesystem enumeration
        "pg_read_file",
        "pg_read_binary_file",
        "pg_ls_dir",
        "pg_ls_tmpdir",
        "pg_ls_logdir",
        "pg_ls_waldir",
        "pg_ls_archive_statusdir",
        "pg_ls_logicalmapdir",
        "pg_ls_logicalsnapdir",
        "pg_ls_replslotdir",
        "pg_stat_file",
        "pg_logdir_ls",
        "pg_write_file",
        "pg_append_file",
        "pg_relation_filepath",
        "pg_tablespace_location",
        "pg_current_logfile",
        "lo_import",
        "lo_export",
        "lo_get",
        "lo_put",
        "loread",
        "lowrite",
        # Arbitrary SQL execution via XML mapping functions (the query string is
        # opaque to AST analysis, so these must be blocked outright)
        "query_to_xml",
        "query_to_xmlschema",
        "query_to_xml_and_xmlschema",
        "table_to_xml",
        "table_to_xmlschema",
        "table_to_xml_and_xmlschema",
        "cursor_to_xml",
        "cursor_to_xmlschema",
        # Sequence side effects (exempt from SET TRANSACTION READ ONLY)
        "nextval",
        "setval",
        "currval",
        "lastval",
        # Logical replication message emission (side effect)
        "pg_logical_emit_message",
        # External execution / remote calls
        "dblink",
        "dblink_exec",
        "dblink_connect",
        "dblink_connect_u",
        # Resource exhaustion / server sleep primitives
        "pg_sleep",
        "pg_sleep_for",
        "pg_sleep_until",
        # Process/server/session side effects
        "pg_terminate_backend",
        "pg_cancel_backend",
        "pg_reload_conf",
        "pg_rotate_logfile",
        "pg_notify",
        "set_config",
        "pg_log_backend_memory_contexts",
        "pg_backup_start",
        "pg_backup_stop",
        "pg_start_backup",
        "pg_stop_backup",
        "pg_switch_wal",
        "pg_create_restore_point",
        "pg_promote",
        "pg_wal_replay_pause",
        "pg_wal_replay_resume",
        # Server metadata leakage
        "inet_server_addr",
        "inet_server_port",
        "inet_client_addr",
        "inet_client_port",
        "current_setting",
        # Statistics reset side effects
        "pg_stat_reset",
        "pg_stat_reset_shared",
        "pg_stat_reset_single_table_counters",
        "pg_stat_reset_slru",
        "pg_stat_reset_replication_slot",
        "pg_stat_clear_snapshot",
        # Advisory locks
        "pg_advisory_lock",
        "pg_try_advisory_lock",
        "pg_advisory_xact_lock",
        "pg_advisory_lock_shared",
        "pg_try_advisory_lock_shared",
        "pg_try_advisory_xact_lock",
        "pg_try_advisory_xact_lock_shared",
        "pg_advisory_unlock",
        "pg_advisory_unlock_shared",
        "pg_advisory_unlock_all",
    },
    "mysql": {
        # File/system access
        "load_file",
        "sys_eval",
        "sys_exec",
        # Resource/session/locking side effects
        "sleep",
        "benchmark",
        "get_lock",
        "release_lock",
        "release_all_locks",
        # Replication wait primitives (can stall sessions/worker threads)
        "master_pos_wait",
        "source_pos_wait",
        "wait_for_executed_gtid_set",
        "wait_until_sql_thread_after_gtids",
    },
    "sqlite": {
        # File access
        "readfile",
        "writefile",
        # Extension loading
        "load_extension",
        # FTS3 tokenizer pointer primitive (memory corruption / code execution)
        "fts3_tokenizer",
    },
    "duckdb": {
        # File-reading table functions
        "read_csv_auto",
        "read_csv",
        "read_json_auto",
        "read_json",
        "read_json_objects",
        "read_parquet",
        "parquet_scan",
        "parquet_metadata",
        "parquet_schema",
        "parquet_file_metadata",
        "parquet_kv_metadata",
        # Text/binary file reading
        "read_text",
        "read_blob",
        "read_xml",
        "read_avro",
        "read_ipc",
        "read_feather",
        # NDJSON/JSONL readers
        "read_ndjson",
        "read_ndjson_auto",
        "read_ndjson_objects",
        # Filesystem enumeration
        "glob",
        # CSV sniffing (reads files)
        "sniff_csv",
        # External database access
        "sqlite_scan",
        "postgres_scan",
        "mysql_scan",
        "postgres_query",
        "mysql_query",
        # Secret / session-variable disclosure
        "duckdb_secrets",
        "which_secret",
        "getvariable",
        # Sequence side effects
        "nextval",
        "currval",
        # Extension management
        "load_extension",
        "install_extension",
        # Additional format readers
        "iceberg_scan",
        "delta_scan",
        "excel_scan",
        "read_xlsx",
        "st_read",
    },
    "tsql": {
        "xp_cmdshell",
    },
}

# Function-family deny prefixes (dialect scoped).
# Used to fail closed on dangerous function families where all variants are unsafe.
DANGEROUS_FUNCTION_PREFIXES_BY_DIALECT: dict[str, set[str]] = {
    "postgres": {
        "dblink",
        "pg_advisory_",
        "pg_ls_",
        "pg_sleep",
        "pg_stat_reset",
        "inet_server_",
        "inet_client_",
    },
    "duckdb": {
        "parquet_",
        "read_avro",
        "read_blob",
        "read_csv",
        "read_feather",
        "read_ipc",
        "read_json",
        "read_ndjson",
        "read_parquet",
        "read_text",
        "read_xml",
    },
}


def _normalize_symbol(name: str) -> str:
    """Normalize SQL identifiers for resilient matching.

    Normalization is intentionally conservative to reduce false positives.
    """
    return name.strip().strip('"`[]').lower()


def _compact_symbol(name: str) -> str:
    """Compacted normalization used only as a fallback for AST key matching.

    This bridges representations like ``read_parquet`` and ``readparquet``.
    """
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _function_name_tokens(fn: exp.Func) -> list[tuple[str, str]]:
    """Collect normalized tokens and their source for a function node."""
    tokens: list[tuple[str, str]] = []

    if fn.name:
        tokens.append(("name", _normalize_symbol(fn.name)))

    sql_name = ""
    try:
        sql_name = fn.sql_name() or ""
    except (AttributeError, TypeError, ValueError):
        sql_name = ""
    if sql_name:
        tokens.append(("sql_name", _normalize_symbol(sql_name)))

    key = getattr(fn, "key", "") or ""
    if key:
        tokens.append(("key", _normalize_symbol(key)))

    return tokens


def _display_function_name(fn: exp.Func) -> str:
    """Best-effort display name for error messages."""
    display_name = fn.name or ""
    if not display_name:
        try:
            display_name = fn.sql_name() or ""
        except (AttributeError, TypeError, ValueError):
            display_name = ""
    if not display_name:
        display_name = str(getattr(fn, "key", "unknown_function"))
    return display_name


def has_dangerous_functions(stmt: exp.Expression, dialect: str) -> str | None:
    """Check for dangerous functions that can read files or execute commands."""
    known_dialect = (
        dialect in DANGEROUS_FUNCTIONS_BY_DIALECT
        or dialect in DANGEROUS_FUNCTION_PREFIXES_BY_DIALECT
    )
    if known_dialect:
        deny_set = DANGEROUS_FUNCTIONS_BY_DIALECT.get(dialect, set())
        deny_prefix_set = DANGEROUS_FUNCTION_PREFIXES_BY_DIALECT.get(dialect, set())
    else:
        # Unknown dialect: fail closed against the union of every known denylist
        # rather than allowing all functions through.
        deny_set = set().union(*DANGEROUS_FUNCTIONS_BY_DIALECT.values())
        deny_prefix_set = set().union(*DANGEROUS_FUNCTION_PREFIXES_BY_DIALECT.values())

    if not deny_set and not deny_prefix_set:
        return None

    deny_exact = {_normalize_symbol(name) for name in deny_set}
    deny_compact = {_compact_symbol(name) for name in deny_set}
    deny_prefixes = tuple(_normalize_symbol(prefix) for prefix in deny_prefix_set)

    for fn in stmt.find_all(exp.Func):
        tokens = _function_name_tokens(fn)
        exact_tokens = {value for _, value in tokens if value}

        if exact_tokens & deny_exact:
            return (
                f"Use of dangerous function '{_display_function_name(fn)}' "
                "is not allowed"
            )

        if deny_prefixes and any(
            token.startswith(prefix)
            for token in exact_tokens
            for prefix in deny_prefixes
        ):
            return (
                f"Use of dangerous function '{_display_function_name(fn)}' "
                "is not allowed"
            )

        # Fallback: sqlglot key names can be compact (e.g. readparquet).
        key_tokens = [value for source, value in tokens if source == "key" and value]
        if any(_compact_symbol(value) in deny_compact for value in key_tokens):
            return (
                f"Use of dangerous function '{_display_function_name(fn)}' "
                "is not allowed"
            )

    return None


# Dialects whose engines treat a bare quoted string in FROM as a file path
# (DuckDB "replacement scans"). For these, a path/URL/glob table reference reads
# arbitrary files with no function call and must be rejected.
FILE_PATH_TARGET_DIALECTS: set[str] = {"duckdb"}

# Data-file extensions that, as a table reference suffix, indicate a file read.
FILE_PATH_DATA_EXTENSIONS: tuple[str, ...] = (
    ".csv",
    ".tsv",
    ".txt",
    ".parquet",
    ".json",
    ".jsonl",
    ".ndjson",
    ".arrow",
    ".feather",
    ".ipc",
    ".avro",
    ".xml",
    ".xlsx",
    ".xls",
    ".gz",
    ".zst",
    ".db",
    ".sqlite",
    ".duckdb",
)


def _looks_like_file_path(name: str) -> bool:
    """Heuristically detect a file path, URL, or glob used as a table reference."""
    lowered = name.strip().lower()
    if not lowered:
        return False
    if "://" in lowered:  # URLs: http(s), s3, gcs, azure, etc.
        return True
    if "/" in lowered or "\\" in lowered:  # path separators
        return True
    if "*" in lowered or "?" in lowered:  # globs
        return True
    return lowered.endswith(FILE_PATH_DATA_EXTENSIONS)


def has_file_path_target(stmt: exp.Expression, dialect: str) -> str | None:
    """Reject file-path/URL/glob table references for replacement-scan dialects."""
    if dialect not in FILE_PATH_TARGET_DIALECTS:
        return None

    for table in stmt.find_all(exp.Table):
        ident = table.this
        if not isinstance(ident, exp.Identifier):
            continue
        name = ident.name or ""
        if _looks_like_file_path(name):
            return (
                "Reading from a file path or URL is not allowed "
                f"(table reference: {name!r})"
            )
    return None
