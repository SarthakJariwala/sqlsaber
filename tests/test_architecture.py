from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

PACKAGE_ROOT = Path(__file__).parents[1] / "src" / "sqlsaber"


@dataclass(frozen=True, order=True)
class ImportEdge:
    source: str
    target: str


@dataclass(frozen=True, order=True)
class CoordinationDebt:
    source: str
    line: int
    pattern: str


def _module_name(path: Path, package_root: Path = PACKAGE_ROOT) -> str:
    relative = path.relative_to(package_root).with_suffix("")
    parts = [package_root.name, *relative.parts]
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _resolve_import(
    source: str,
    node: ast.ImportFrom,
    *,
    source_is_package: bool,
) -> str:
    if node.level == 0:
        return node.module or ""

    package = source.split(".") if source_is_package else source.split(".")[:-1]
    parents = node.level - 1
    prefix = package[: max(len(package) - parents, 0)]
    if node.module:
        prefix.extend(node.module.split("."))
    return ".".join(prefix)


def _import_from_targets(
    source: str,
    node: ast.ImportFrom,
    *,
    source_is_package: bool,
    modules: set[str],
) -> set[str]:
    target = _resolve_import(
        source,
        node,
        source_is_package=source_is_package,
    )
    targets = {target} if target else set()
    for alias in node.names:
        if alias.name == "*":
            continue
        member = f"{target}.{alias.name}" if target else alias.name
        if member in modules:
            targets.add(member)
    return targets


def _is_within(module: str, package: str) -> bool:
    return module == package or module.startswith(f"{package}.")


def _import_edges(package_root: Path = PACKAGE_ROOT) -> set[ImportEdge]:
    paths = list(package_root.rglob("*.py"))
    modules = {_module_name(path, package_root) for path in paths}
    edges: set[ImportEdge] = set()
    for path in paths:
        source = _module_name(path, package_root)
        source_is_package = path.name == "__init__.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                targets = {alias.name for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                targets = _import_from_targets(
                    source,
                    node,
                    source_is_package=source_is_package,
                    modules=modules,
                )
            else:
                continue
            edges.update(
                ImportEdge(source=source, target=target)
                for target in targets
                if _is_within(target, package_root.name)
            )
    return edges


def _cli_coordination_debt(
    package_root: Path = PACKAGE_ROOT,
) -> set[CoordinationDebt]:
    debt: set[CoordinationDebt] = set()
    for path in (package_root / "cli").rglob("*.py"):
        source = _module_name(path, package_root)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "run_result":
                debt.add(CoordinationDebt(source, node.lineno, ".run_result"))
            elif (
                isinstance(node, ast.Attribute)
                and node.attr == "agent"
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "agent"
            ):
                debt.add(CoordinationDebt(source, node.lineno, ".agent.agent"))
            elif isinstance(node, ast.Call) and any(
                keyword.arg == "message_history" for keyword in node.keywords
            ):
                debt.add(CoordinationDebt(source, node.lineno, "message_history="))
    return debt


def test_package_init_relative_imports_resolve_from_the_package(tmp_path: Path) -> None:
    package_root = tmp_path / "sqlsaber"
    cli_package = package_root / "cli"
    cli_package.mkdir(parents=True)
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    (package_root / "_runtime.py").write_text("", encoding="utf-8")
    (cli_package / "__init__.py").write_text(
        "from .. import _runtime\n", encoding="utf-8"
    )

    assert ImportEdge("sqlsaber.cli", "sqlsaber._runtime") in _import_edges(
        package_root
    )


def test_imported_member_names_can_identify_modules(tmp_path: Path) -> None:
    package_root = tmp_path / "sqlsaber"
    cli_package = package_root / "cli"
    cli_package.mkdir(parents=True)
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    (package_root / "_runtime.py").write_text("", encoding="utf-8")
    (cli_package / "__init__.py").write_text("", encoding="utf-8")
    (cli_package / "client.py").write_text(
        "from sqlsaber import _runtime\n", encoding="utf-8"
    )

    assert ImportEdge("sqlsaber.cli.client", "sqlsaber._runtime") in _import_edges(
        package_root
    )


def test_cli_uses_only_the_public_sdk_lifecycle() -> None:
    bypasses = {
        edge
        for edge in _import_edges()
        if _is_within(edge.source, "sqlsaber.cli")
        and any(
            _is_within(edge.target, private_module)
            for private_module in (
                "sqlsaber.agents",
                "sqlsaber.session",
                "sqlsaber._runtime",
            )
        )
    }
    assert bypasses == set()


def test_core_does_not_import_cli_clients() -> None:
    reverse_dependencies = {
        edge
        for edge in _import_edges()
        if not _is_within(edge.source, "sqlsaber.cli")
        and edge.source != "sqlsaber.__main__"
        and _is_within(edge.target, "sqlsaber.cli")
    }
    assert reverse_dependencies == set()


def test_cli_has_no_legacy_query_coordination() -> None:
    assert _cli_coordination_debt() == set()
