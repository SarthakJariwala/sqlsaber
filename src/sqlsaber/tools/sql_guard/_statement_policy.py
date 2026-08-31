"""Statement classification and block policy."""

from sqlglot import exp

from ._mutation_analysis import has_unfiltered_mutation

# DML/DDL operations that can be unlocked in "dangerous" mode
WRITE_DML_DDL_NODES: set[type[exp.Expression]] = {
    # DML operations
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Merge,
    # MySQL specific
    exp.Replace,
    # DDL operations (non-destructive)
    exp.Create,
    exp.Alter,
    exp.AlterRename,
}

# Operations that are always prohibited, regardless of mode
ALWAYS_BLOCKED_NODES: set[type[exp.Expression]] = {
    # Transaction control
    exp.Transaction,
    # Analysis and maintenance
    exp.Analyze,
    # Data loading/copying
    exp.Copy,
    exp.LoadData,
    # Session and configuration
    exp.Set,
    exp.Use,
    exp.Pragma,
    # Security
    exp.Grant,
    exp.Revoke,
    # Database operations
    exp.Attach,
    exp.Detach,
    # Locking and process control
    exp.Lock,
    exp.Kill,
    # Commands
    exp.Command,
    # Destructive schema/data operations (no safeguards possible)
    exp.Drop,
    exp.TruncateTable,
}

try:
    vacuum_type = getattr(exp, "Vacuum", None)
    if vacuum_type is not None:
        ALWAYS_BLOCKED_NODES.add(vacuum_type)
except AttributeError:
    pass

# In dangerous mode, we run fail-closed and only allow these root statement types.
DANGEROUS_ALLOWED_ROOT_NODES: tuple[type[exp.Expression], ...] = (
    exp.Select,
    exp.Union,
    exp.Except,
    exp.Intersect,
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Create,
    exp.Alter,
    exp.AlterRename,
)

# In dangerous mode, CREATE is further constrained by explicit kind allowlist.
ALLOWED_DANGEROUS_CREATE_KINDS: set[str] = {
    "TABLE",
    "VIEW",
    "INDEX",
}


def _unwrap_root(stmt: exp.Expression) -> exp.Expression:
    """Return the effective statement root (unwrap WITH)."""
    root = stmt
    if isinstance(root, exp.With):
        inner = root.this
        if inner is not None:
            root = inner
    return root


def is_select_like(stmt: exp.Expression) -> bool:
    """Check if statement is a SELECT-like query.

    Handles CTEs (WITH) and set operations (UNION/INTERSECT/EXCEPT).
    """
    root = _unwrap_root(stmt)
    return isinstance(root, (exp.Select, exp.Union, exp.Except, exp.Intersect))


def classify_statement(stmt: exp.Expression) -> str:
    """Classify statement as select/dml/ddl/other.

    Returns:
        "select" for SELECT-like queries
        "dml" for INSERT/UPDATE/DELETE/MERGE/REPLACE
        "ddl" for CREATE/DROP/ALTER/TRUNCATE
        "other" for anything else
    """
    if is_select_like(stmt):
        return "select"

    root = _unwrap_root(stmt)

    if isinstance(root, (exp.Insert, exp.Update, exp.Delete, exp.Merge, exp.Replace)):
        return "dml"

    if isinstance(
        root,
        (exp.Create, exp.Alter, exp.AlterRename),
    ):
        return "ddl"

    # DROP and TRUNCATE are blocked, but classify them for error messages
    if isinstance(root, (exp.Drop, exp.TruncateTable)):
        return "ddl"

    return "other"


def has_prohibited_nodes(
    stmt: exp.Expression,
    allow_dangerous: bool = False,
    dialect: str = "ansi",
    source_sql: str | None = None,
) -> str | None:
    """Walk AST to find any prohibited operations.

    In read-only mode (allow_dangerous=False):
      - Block DML/DDL (WRITE_DML_DDL_NODES)
      - Block always-blocked operations (ALWAYS_BLOCKED_NODES)
      - Block SELECT INTO
      - Block locking clauses (FOR UPDATE/FOR SHARE)

    In dangerous mode (allow_dangerous=True):
      - Allow DML/DDL
      - Still block ALWAYS_BLOCKED_NODES, SELECT INTO, locking clauses
      - Block UPDATE/DELETE without restrictive WHERE clause
    """
    for node in stmt.walk():
        # Operations that are never allowed
        if isinstance(node, tuple(ALWAYS_BLOCKED_NODES)):
            return f"Prohibited operation: {type(node).__name__}"

        # DML/DDL writes are only allowed in dangerous mode
        if not allow_dangerous and isinstance(node, tuple(WRITE_DML_DDL_NODES)):
            return f"Prohibited operation: {type(node).__name__}"

        # Block SELECT INTO (Postgres-style table creation)
        if isinstance(node, exp.Select) and node.args.get("into"):
            return "SELECT INTO is not allowed"

        # Block locking clauses (FOR UPDATE/FOR SHARE)
        if isinstance(node, exp.Select):
            locks = node.args.get("locks")
            if locks:
                return "SELECT with locking clause (FOR UPDATE/SHARE) is not allowed"

    # In dangerous mode, block unfiltered mutations
    if allow_dangerous:
        reason = has_unfiltered_mutation(stmt, dialect, source_sql)
        if reason:
            return reason

    return None


def has_disallowed_dangerous_mode_statement(stmt: exp.Expression) -> str | None:
    """Fail-closed statement allowlist checks for dangerous mode."""
    root = _unwrap_root(stmt)

    if not isinstance(root, DANGEROUS_ALLOWED_ROOT_NODES):
        return (
            "Only SELECT, INSERT, UPDATE, DELETE, and restricted CREATE/ALTER "
            "statements are allowed in dangerous mode"
        )

    if isinstance(root, exp.Create):
        kind = str(root.args.get("kind") or "").upper()
        if kind not in ALLOWED_DANGEROUS_CREATE_KINDS:
            return f"CREATE {kind or '<unknown>'} is not allowed in dangerous mode"

        target = root.args.get("this")
        expression = root.args.get("expression")

        # Additional defensive checks to avoid dialect/parser gaps.
        if kind == "TABLE":
            if target is not None and not isinstance(target, (exp.Table, exp.Schema)):
                return "Only CREATE TABLE statements are allowed in dangerous mode"
        elif kind == "VIEW":
            if target is not None and not isinstance(target, exp.Table):
                return "Only CREATE VIEW statements are allowed in dangerous mode"
            if expression is not None and not is_select_like(expression):
                return "CREATE VIEW must be based on a SELECT-like expression"
        elif (
            kind == "INDEX" and target is not None and not isinstance(target, exp.Index)
        ):
            return "Only CREATE INDEX statements are allowed in dangerous mode"

        if isinstance(target, exp.UserDefinedFunction):
            return "CREATE FUNCTION-like statements are not allowed in dangerous mode"

    if isinstance(root, exp.Alter):
        kind = str(root.args.get("kind") or "TABLE").upper()
        if kind != "TABLE":
            return f"ALTER {kind or '<unknown>'} is not allowed in dangerous mode"

    if isinstance(root, exp.AlterRename):
        target = root.args.get("this")
        if target is not None and not isinstance(target, exp.Table):
            return "Only ALTER TABLE style rename statements are allowed"

    return None
