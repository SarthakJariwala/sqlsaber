from __future__ import annotations

import re

import pytest

from sqlsaber.cli import knowledge as knowledge_cli


def _extract_entry_id(output: str) -> str:
    match = re.search(r"ID:\s*([0-9a-fA-F-]{36})", output)
    assert match is not None
    return match.group(1)


def _run_cli(app, args: list[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        app(args)
    assert exc_info.value.code == 0


@pytest.fixture
def knowledge_app(temp_dir, monkeypatch):
    monkeypatch.setattr(
        "platformdirs.user_data_dir",
        lambda *args, **kwargs: str(temp_dir / "data"),
    )
    monkeypatch.setattr(knowledge_cli, "_knowledge_manager", None)
    monkeypatch.setattr(knowledge_cli, "_get_database_name", lambda database=None: "db")
    return knowledge_cli.create_knowledge_app()


def test_cli_add_show_search_remove_and_list(knowledge_app, capsys):
    _run_cli(
        knowledge_app,
        ["add", "Revenue KPI", "Monthly revenue rollup", "--source", "wiki"],
    )
    add_output = capsys.readouterr().out
    assert "Knowledge entry added" in add_output
    entry_id = _extract_entry_id(add_output)

    _run_cli(knowledge_app, ["show", entry_id])
    show_output = capsys.readouterr().out
    assert "Revenue KPI" in show_output
    assert "Monthly revenue rollup" in show_output

    _run_cli(knowledge_app, ["search", "revenue"])
    search_output = capsys.readouterr().out
    assert "Revenue KPI" in search_output

    _run_cli(knowledge_app, ["remove", entry_id])
    remove_output = capsys.readouterr().out
    assert "removed" in remove_output.lower()

    _run_cli(knowledge_app, ["list"])
    list_output = capsys.readouterr().out
    assert "No knowledge entries found" in list_output


def test_cli_clear_force(knowledge_app, capsys):
    _run_cli(knowledge_app, ["add", "One", "Description one"])
    _ = capsys.readouterr()
    _run_cli(knowledge_app, ["add", "Two", "Description two"])
    _ = capsys.readouterr()

    _run_cli(knowledge_app, ["clear", "--force"])
    clear_output = capsys.readouterr().out
    assert "Cleared 2 knowledge entries" in clear_output
