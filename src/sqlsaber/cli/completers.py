"""Autocomplete providers for the interactive CLI."""

from saber_tui import AutocompleteItem, AutocompleteSuggestions, CompletionResult


class SQLSaberAutocompleteProvider:
    """saber-tui autocomplete provider for table names."""

    def __init__(self) -> None:
        self._table_cache: list[tuple[str, str]] = []

    def update_table_cache(self, tables_data: list[tuple[str, str]]) -> None:
        """Update the table completion cache with fresh schema data."""
        self._table_cache = tables_data

    def get_suggestions(
        self,
        lines: list[str],
        cursor_line: int,
        cursor_col: int,
        *,
        force: bool = False,
        signal: object | None = None,
    ) -> AutocompleteSuggestions | None:
        """Return completion suggestions for the current editor state."""
        if bool(getattr(signal, "aborted", False)):
            return None

        current_line = lines[cursor_line] if 0 <= cursor_line < len(lines) else ""
        text_before_cursor = current_line[:cursor_col]

        table_suggestions = self._table_suggestions(
            current_line, cursor_col, text_before_cursor
        )
        if table_suggestions is not None:
            return table_suggestions

        return None

    def apply_completion(
        self,
        lines: list[str],
        cursor_line: int,
        cursor_col: int,
        item: AutocompleteItem,
        prefix: str,
    ) -> CompletionResult:
        """Apply a selected completion to the editor buffer."""
        new_lines = list(lines) or [""]
        cursor_line = max(0, min(cursor_line, len(new_lines) - 1))
        current_line = new_lines[cursor_line]
        before_prefix = current_line[: max(0, cursor_col - len(prefix))]
        after_cursor = current_line[cursor_col:]

        replacement = item.value

        new_line = f"{before_prefix}{replacement}{after_cursor}"
        new_lines[cursor_line] = new_line
        return CompletionResult(
            new_lines,
            cursor_line,
            len(before_prefix) + len(replacement),
        )

    def should_trigger_file_completion(
        self, lines: list[str], cursor_line: int, cursor_col: int
    ) -> bool:
        """Return whether Tab should force path-style completion.

        SQLsaber owns bare slash commands. Other contexts can use Tab without
        reopening slash-command suggestions.
        """
        current_line = lines[cursor_line] if 0 <= cursor_line < len(lines) else ""
        text_before_cursor = current_line[:cursor_col]
        stripped = text_before_cursor.strip()
        return not (stripped.startswith("/") and " " not in stripped)

    def _table_suggestions(
        self,
        current_line: str,
        cursor_col: int,
        text_before_cursor: str,
    ) -> AutocompleteSuggestions | None:
        at_pos = text_before_cursor.rfind("@")
        if at_pos < 0:
            return None
        if not self._is_valid_table_context(current_line, at_pos, cursor_col):
            return None

        partial_table = text_before_cursor[at_pos + 1 :].lower()
        matches: list[tuple[int, str, str]] = []
        for table_name, description in self._table_cache:
            table_lower = table_name.lower()
            score = self._calculate_match_score(partial_table, table_name, table_lower)
            if score > 0:
                matches.append((score, table_name, description))

        matches.sort(key=lambda item: item[0], reverse=True)
        if not matches:
            return None

        items = [
            AutocompleteItem(f"@{table_name}", table_name, description or None)
            for _, table_name, description in matches
        ]
        return AutocompleteSuggestions(items, text_before_cursor[at_pos:])

    def _is_valid_table_context(self, text: str, at_pos: int, cursor_pos: int) -> bool:
        """Check if the @ is in a valid context for table completion."""
        single_quotes = text[:at_pos].count("'") - text[:at_pos].count("\\'")
        double_quotes = text[:at_pos].count('"') - text[:at_pos].count('\\"')
        if single_quotes % 2 == 1 or double_quotes % 2 == 1:
            return False

        if cursor_pos < len(text):
            next_char = text[cursor_pos]
            if next_char.isalnum() or next_char == "_":
                partial = (
                    text[at_pos + 1 :].split()[0] if text[at_pos + 1 :].split() else ""
                )
                if not any(char in partial for char in [".", "_"]):
                    return False

        return True

    def _calculate_match_score(
        self, partial: str, table_name: str, table_lower: str
    ) -> int:
        """Calculate fuzzy table match score. Higher is better."""
        if not partial:
            return 1

        if table_lower.startswith(partial):
            return 100

        if "." in table_name:
            table_part = table_name.split(".")[-1].lower()
            if table_part.startswith(partial):
                return 90
            if table_part == partial:
                return 80
            if table_part.startswith(partial + "_") or table_part.startswith(
                partial + "-"
            ):
                return 70
            if partial in table_part:
                return 50

        if partial in table_lower:
            return 30

        return 0
