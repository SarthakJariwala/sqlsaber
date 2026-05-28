"""Database connection resolution from CLI input."""

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode, urlparse

from sqlsaber.config.database import DatabaseConfig, DatabaseConfigManager


class DatabaseResolutionError(Exception):
    """Exception raised when database resolution fails."""


@dataclass
class ResolvedDatabase:
    """Result of database resolution containing canonical connection info."""

    name: str  # Human-readable name for display/logging
    connection_string: str  # Canonical connection string for DatabaseConnection factory
    excluded_schemas: list[str]
    description: str | None = None


SUPPORTED_SCHEMES = {"postgresql", "mysql", "sqlite", "duckdb", "csv", "csvs"}


def _is_connection_string(s: str) -> bool:
    """Check if string looks like a connection string with supported scheme."""
    try:
        scheme = urlparse(s).scheme
        return scheme in SUPPORTED_SCHEMES
    except Exception:
        return False


def _csv_stem_from_connection_string(connection_string: str) -> str:
    parsed = urlparse(connection_string)
    stem = Path(parsed.path).stem
    return stem or "csv"


def _resolve_multiple_csvs(specs: list[str]) -> ResolvedDatabase:
    csv_specs: list[str] = []
    stems: list[str] = []

    for spec in specs:
        if _is_connection_string(spec):
            scheme = urlparse(spec).scheme
            if scheme != "csv":
                raise DatabaseResolutionError(
                    "Multiple database arguments are only supported for CSV files. "
                    f"Got connection string with scheme '{scheme}': {spec}"
                )
            csv_specs.append(spec)
            stems.append(_csv_stem_from_connection_string(spec))
            continue

        path = Path(spec).expanduser().resolve()
        if path.suffix.lower() != ".csv":
            raise DatabaseResolutionError(
                "Multiple database arguments are only supported for CSV files. "
                f"Got non-CSV path: {spec}"
            )
        if not path.exists():
            raise DatabaseResolutionError(f"CSV file '{spec}' not found.")
        csv_specs.append(f"csv:///{path}")
        stems.append(path.stem or "csv")

    if not csv_specs:
        raise DatabaseResolutionError(
            "Multiple database arguments were provided, but no CSV files were found."
        )

    query = urlencode({"spec": csv_specs}, doseq=True)

    if len(stems) <= 3:
        name = " + ".join(stems)
    else:
        name = f"{stems[0]} + {len(stems) - 1} more"

    return ResolvedDatabase(
        name=name, connection_string=f"csvs:///?{query}", excluded_schemas=[]
    )


def _all_csv_specs(specs: list[str]) -> bool:
    """Return True iff every spec resolves to a CSV file (path or csv:// URL)."""
    for spec in specs:
        if _is_connection_string(spec):
            if urlparse(spec).scheme != "csv":
                return False
        else:
            if Path(spec).suffix.lower() != ".csv":
                return False
    return True


def resolve_databases(
    spec: str | list[str] | None,
    config_mgr: DatabaseConfigManager | None = None,
) -> list[ResolvedDatabase]:
    """Resolve a CLI spec into one or more `ResolvedDatabase` entries.

    - `None`, single string, or single-element list → returns 1 entry (matches
      `resolve_database` semantics).
    - All-CSV list → returns 1 entry (CSVs merger, today's behavior).
    - Mixed/non-CSV list with N > 1 → returns N entries, each resolved
      independently. Duplicate names raise `DatabaseResolutionError`.
    """
    if spec is None or isinstance(spec, str):
        return [resolve_database(spec, config_mgr)]

    if not isinstance(spec, list):
        raise DatabaseResolutionError(f"Unsupported database spec: {spec!r}")

    if len(spec) == 0:
        raise DatabaseResolutionError("Empty database argument list.")

    if len(spec) == 1:
        return [resolve_database(spec[0], config_mgr)]

    if _all_csv_specs(spec):
        return [_resolve_multiple_csvs(spec)]

    resolved = [resolve_database(item, config_mgr) for item in spec]

    seen: set[str] = set()
    for entry in resolved:
        if entry.name in seen:
            raise DatabaseResolutionError(
                f"Cannot use duplicate database name '{entry.name}' in a "
                "multi-database session."
            )
        seen.add(entry.name)

    return resolved


def resolve_database(
    spec: str | list[str] | None, config_mgr: DatabaseConfigManager | None = None
) -> ResolvedDatabase:
    """Turn user CLI input into resolved database connection info.

    Args:
        spec: User input - None (default), configured name, connection string, file path,
            or a list of CSV file paths/CSV connection strings.
        config_mgr: Optional database configuration manager for looking up configured
            connections. If omitted, one is created only when needed for configured
            name/default lookup.

    Returns:
        ResolvedDatabase with name and canonical connection string

    Raises:
        DatabaseResolutionError: If the spec cannot be resolved to a valid database connection
    """
    if spec is None:
        config_mgr = config_mgr or DatabaseConfigManager()
        db_cfg = config_mgr.get_default_database()
        if not db_cfg:
            raise DatabaseResolutionError(
                "No database connections configured. "
                "Use 'sqlsaber db add <name>' to add one."
            )
        return ResolvedDatabase(
            name=db_cfg.name,
            connection_string=db_cfg.to_connection_string(),
            excluded_schemas=list(db_cfg.exclude_schemas),
            description=db_cfg.description,
        )

    if isinstance(spec, list):
        if len(spec) == 1:
            return resolve_database(spec[0], config_mgr)
        if len(spec) > 1:
            return _resolve_multiple_csvs(spec)
        raise DatabaseResolutionError("Empty database argument list.")

    # 1. Connection string?
    if _is_connection_string(spec):
        scheme = urlparse(spec).scheme
        if scheme in {"postgresql", "mysql"}:
            db_name = urlparse(spec).path.lstrip("/") or "database"
        elif scheme in {"sqlite", "duckdb", "csv", "csvs"}:
            db_name = Path(urlparse(spec).path).stem or "database"
        else:  # should not happen because of SUPPORTED_SCHEMES
            db_name = "database"
        return ResolvedDatabase(
            name=db_name, connection_string=spec, excluded_schemas=[]
        )

    # 2. Raw file path?
    path = Path(spec).expanduser().resolve()
    if path.suffix.lower() == ".csv":
        if not path.exists():
            raise DatabaseResolutionError(f"CSV file '{spec}' not found.")
        return ResolvedDatabase(
            name=path.stem, connection_string=f"csv:///{path}", excluded_schemas=[]
        )
    if path.suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
        if not path.exists():
            raise DatabaseResolutionError(f"SQLite file '{spec}' not found.")
        return ResolvedDatabase(
            name=path.stem, connection_string=f"sqlite:///{path}", excluded_schemas=[]
        )
    if path.suffix.lower() in {".duckdb", ".ddb"}:
        if not path.exists():
            raise DatabaseResolutionError(f"DuckDB file '{spec}' not found.")
        return ResolvedDatabase(
            name=path.stem, connection_string=f"duckdb:///{path}", excluded_schemas=[]
        )

    # 3. Must be a configured name
    config_mgr = config_mgr or DatabaseConfigManager()
    db_cfg: DatabaseConfig | None = config_mgr.get_database(spec)
    if not db_cfg:
        raise DatabaseResolutionError(
            f"Database connection '{spec}' not found. "
            "Use 'sqlsaber db list' to see available connections."
        )
    return ResolvedDatabase(
        name=db_cfg.name,
        connection_string=db_cfg.to_connection_string(),
        excluded_schemas=list(db_cfg.exclude_schemas),
        description=db_cfg.description,
    )
