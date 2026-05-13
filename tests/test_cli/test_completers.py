import pytest

from sqlsaber.cli.completers import SQLSaberAutocompleteProvider


def _sync_suggestions(result):
    assert not hasattr(result, "__await__")
    return result


def test_slash_command_prefix_does_not_use_editor_autocomplete() -> None:
    provider = SQLSaberAutocompleteProvider()

    suggestions = _sync_suggestions(provider.get_suggestions(["/thi"], 0, 4))

    assert suggestions is None


def test_table_suggestions_use_cached_tables_and_at_prefix() -> None:
    provider = SQLSaberAutocompleteProvider()
    provider.update_table_cache(
        [
            ("public.users", "application users"),
            ("analytics.orders", "purchase orders"),
        ]
    )

    suggestions = _sync_suggestions(
        provider.get_suggestions(["select * from @us"], 0, len("select * from @us"))
    )

    assert suggestions is not None
    assert suggestions.prefix == "@us"
    assert suggestions.items[0].value == "@public.users"
    assert suggestions.items[0].label == "public.users"
    assert suggestions.items[0].description == "application users"


def test_table_suggestions_are_ignored_inside_quotes() -> None:
    provider = SQLSaberAutocompleteProvider()
    provider.update_table_cache([("public.users", "")])

    suggestions = _sync_suggestions(
        provider.get_suggestions(["select '@us'"], 0, len("select '@us"))
    )

    assert suggestions is None


def test_apply_table_completion_replaces_at_token() -> None:
    provider = SQLSaberAutocompleteProvider()
    provider.update_table_cache([("public.users", "")])
    suggestions = _sync_suggestions(
        provider.get_suggestions(["select * from @us"], 0, len("select * from @us"))
    )
    assert suggestions is not None

    result = provider.apply_completion(
        ["select * from @us"],
        0,
        len("select * from @us"),
        suggestions.items[0],
        suggestions.prefix,
    )

    assert result.lines == ["select * from @public.users"]
    assert result.cursor_col == len("select * from @public.users")


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("/thi", False),
        ("/handoff ./notes", True),
        ("select * from ./queries", True),
    ],
)
def test_file_completion_trigger_skips_bare_slash_commands(
    line: str, expected: bool
) -> None:
    provider = SQLSaberAutocompleteProvider()

    assert provider.should_trigger_file_completion([line], 0, len(line)) is expected
