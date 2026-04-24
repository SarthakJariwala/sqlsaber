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
    type: str
    description: str | None = None
    id: str | None = None


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


def _is_csv_spec(spec: str) -> bool:
    if _is_connection_string(spec):
        return urlparse(spec).scheme == "csv"
    return Path(spec).suffix.lower() == ".csv"


def _config_description(db_cfg: DatabaseConfig) -> str | None:
    description = vars(db_cfg).get("description")
    if isinstance(description, str) or description is None:
        return description
    return None


def _assign_database_ids(databases: list[ResolvedDatabase]) -> list[ResolvedDatabase]:
    seen: dict[str, int] = {}
    for database in databases:
        base_id = database.name or "database"
        count = seen.get(base_id, 0) + 1
        seen[base_id] = count
        database.id = base_id if count == 1 else f"{base_id}_{count}"
    return databases


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
        name=name,
        connection_string=f"csvs:///?{query}",
        excluded_schemas=[],
        type="csvs",
    )


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
            type=db_cfg.type,
            description=_config_description(db_cfg),
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
            name=db_name, connection_string=spec, excluded_schemas=[], type=scheme
        )

    # 2. Raw file path?
    path = Path(spec).expanduser().resolve()
    if path.suffix.lower() == ".csv":
        if not path.exists():
            raise DatabaseResolutionError(f"CSV file '{spec}' not found.")
        return ResolvedDatabase(
            name=path.stem,
            connection_string=f"csv:///{path}",
            excluded_schemas=[],
            type="csv",
        )
    if path.suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
        if not path.exists():
            raise DatabaseResolutionError(f"SQLite file '{spec}' not found.")
        return ResolvedDatabase(
            name=path.stem,
            connection_string=f"sqlite:///{path}",
            excluded_schemas=[],
            type="sqlite",
        )
    if path.suffix.lower() in {".duckdb", ".ddb"}:
        if not path.exists():
            raise DatabaseResolutionError(f"DuckDB file '{spec}' not found.")
        return ResolvedDatabase(
            name=path.stem,
            connection_string=f"duckdb:///{path}",
            excluded_schemas=[],
            type="duckdb",
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
        type=db_cfg.type,
        description=_config_description(db_cfg),
    )


def resolve_databases(
    spec: str | list[str] | None, config_mgr: DatabaseConfigManager | None = None
) -> list[ResolvedDatabase]:
    """Turn user CLI input into one or more resolved database connections."""
    if not isinstance(spec, list):
        return _assign_database_ids([resolve_database(spec, config_mgr)])

    if len(spec) == 0:
        resolve_database(spec, config_mgr)

    if len(spec) == 1:
        return _assign_database_ids([resolve_database(spec[0], config_mgr)])

    if all(_is_csv_spec(item) for item in spec):
        return _assign_database_ids([_resolve_multiple_csvs(spec)])

    return _assign_database_ids(
        [resolve_database(database_spec, config_mgr) for database_spec in spec]
    )
