"""Command line completers for the CLI interface."""

from prompt_toolkit.completion import Completer, Completion


class SlashCommandCompleter(Completer):
    """Custom completer for slash commands."""

    def get_completions(self, document, complete_event):
        """Get completions for slash commands."""
        # Only provide completions if the line starts with "/"
        text = document.text
        if text.startswith("/"):
            # Get the partial command after the slash
            partial_cmd = text[1:]

            # Define available commands with descriptions
            commands = [
                ("clear", "Clear conversation history"),
                ("exit", "Exit the interactive session"),
                ("quit", "Exit the interactive session"),
            ]

            # Yield completions that match the partial command
            for cmd, description in commands:
                if cmd.startswith(partial_cmd):
                    yield Completion(
                        cmd,
                        start_position=-len(partial_cmd),
                        display_meta=description,
                    )


class TableNameCompleter(Completer):
    """Custom completer for table names."""

    def __init__(self):
        self._table_cache: list[tuple[str, str]] = []

    def update_cache(self, tables_data: list[tuple[str, str]]):
        """Update the cache with fresh table data."""
        self._table_cache = tables_data

    def _get_table_names(self) -> list[tuple[str, str]]:
        """Get table names from cache."""
        return self._table_cache

    def get_completions(self, document, complete_event):
        """Get completions for table names with fuzzy matching."""
        text = document.text
        cursor_position = document.cursor_position

        # Find the last "@" before the cursor position
        at_pos = text.rfind("@", 0, cursor_position)

        if at_pos >= 0:
            # Extract text after the "@" up to the cursor
            partial_table = text[at_pos + 1 : cursor_position].lower()

            # Check if this looks like a valid table reference context
            # (not inside quotes, and followed by word characters or end of input)
            if self._is_valid_table_context(text, at_pos, cursor_position):
                # Get table names
                tables = self._get_table_names()

                # Collect matches with scores for ranking
                matches = []

                for table_name, description in tables:
                    table_lower = table_name.lower()
                    score = self._calculate_match_score(
                        partial_table, table_name, table_lower
                    )

                    if score > 0:
                        matches.append((score, table_name, description))

                # Sort by score (higher is better) and yield completions
                matches.sort(key=lambda x: x[0], reverse=True)

                for score, table_name, description in matches:
                    yield Completion(
                        table_name,
                        start_position=at_pos
                        + 1
                        - cursor_position,  # Start from after the @
                        display_meta=description if description else None,
                    )

    def _is_valid_table_context(self, text: str, at_pos: int, cursor_pos: int) -> bool:
        """Check if the @ is in a valid context for table completion."""
        # Simple heuristic: avoid completion inside quoted strings

        # Count quotes before the @ position
        single_quotes = text[:at_pos].count("'") - text[:at_pos].count("\\'")
        double_quotes = text[:at_pos].count('"') - text[:at_pos].count('\\"')

        # If we're inside quotes, don't complete
        if single_quotes % 2 == 1 or double_quotes % 2 == 1:
            return False

        # Check if the character after the cursor (if any) is part of a word
        # This helps avoid breaking existing words
        if cursor_pos < len(text):
            next_char = text[cursor_pos]
            if next_char.isalnum() or next_char == "_":
                # We're in the middle of a word, check if it looks like a table name
                partial = (
                    text[at_pos + 1 :].split()[0] if text[at_pos + 1 :].split() else ""
                )
                if not any(c in partial for c in [".", "_"]):
                    return False

        return True

    def _calculate_match_score(
        self, partial: str, table_name: str, table_lower: str
    ) -> int:
        """Calculate match score for fuzzy matching (higher is better)."""
        if not partial:
            return 1  # Empty search matches everything with low score

        # Score 100: Exact full name prefix match
        if table_lower.startswith(partial):
            return 100

        # Score 90: Table name (after schema) prefix match
        if "." in table_name:
            table_part = table_name.split(".")[-1].lower()
            if table_part.startswith(partial):
                return 90

        # Score 80: Exact table name match (for short names)
        if "." in table_name:
            table_part = table_name.split(".")[-1].lower()
            if table_part == partial:
                return 80

        # Score 70: Word boundary matches (e.g., "user" matches "user_accounts")
        if "." in table_name:
            table_part = table_name.split(".")[-1].lower()
            if table_part.startswith(partial + "_") or table_part.startswith(
                partial + "-"
            ):
                return 70

        # Score 50: Substring match in table name part
        if "." in table_name:
            table_part = table_name.split(".")[-1].lower()
            if partial in table_part:
                return 50

        # Score 30: Substring match in full name
        if partial in table_lower:
            return 30

        # Score 0: No match
        return 0


class CompositeCompleter(Completer):
    """Combines multiple completers."""

    def __init__(self, *completers: Completer):
        self.completers = completers

    def get_completions(self, document, complete_event):
        """Get completions from all registered completers."""
        for completer in self.completers:
            yield from completer.get_completions(document, complete_event)
