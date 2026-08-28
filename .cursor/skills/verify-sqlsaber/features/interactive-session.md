# Interactive session

Interactive session is the full-screen chat TUI started by `saber` with no query. A user sees the SQLsaber banner, types questions or slash commands, and exits back to the shell.

## Sub-features

- `tui-open` shows the banner, slash-command instructions, and a `DB: verify-sqlite (SQLite)` footer.
- `tui-slash-clear` runs `/clear` and reports that history was cleared.
- `tui-palette` opens the command palette from `/` on an empty editor.
- `tui-exit` leaves the session via `/exit` or empty-editor Ctrl+D.

## How to get to it (user POV)

- Run `saber` or `saber -d verify-sqlite` with no question.
- Type `/` on an empty prompt to open the command palette.
- Type `/clear`, `/thinking`, `/exit`, or `/quit`.
- Press Ctrl+D on an empty editor to exit.

## Driving it with control-sqlsaber

Preconditions:

- `control-sqlsaber doctor` is clean.
- `tmux` is on `PATH`.
- No tmux session named `sqlsaber-verify` exists (`doctor` reports `tui  stopped`).
- Opening the TUI does not require a provider key. Submitting a natural-language question does. This recipe does not submit a question.

- **Start session.** Run `control-sqlsaber tui start`. It returns `tui ready` when the pane contains `slash commands`.
- **Capture identity.** Run `control-sqlsaber tui capture --path artifacts/interactive-session/ready.txt`. The pane contains `slash commands`, `table name completions`, and `DB: verify-sqlite`.
- **Open palette.** Run `control-sqlsaber tui send --literal /` then `control-sqlsaber tui capture --path artifacts/interactive-session/palette.txt`. The pane contains `Thinking mode` and `Clear conversation`.
- **Dismiss palette.** Run `control-sqlsaber tui send Escape`. Capture if needed; the editor is focused again and the palette labels are gone.
- **Clear via slash command.** Run `control-sqlsaber tui send --literal '/clear'` then `control-sqlsaber tui send Enter` then capture `--path artifacts/interactive-session/clear.txt`. The pane contains `Conversation history cleared.`
- **Exit.** Run `control-sqlsaber tui send --literal '/exit'` then `control-sqlsaber tui send Enter`. If the session is still present after a second, run `control-sqlsaber tui stop`. The helper reports `tui stopped`.
- **Shell goodbye.** If `/exit` returned to the wrapper, the tmux pane or helper output contains `Goodbye!` before the session ends.
- **Proof.** Keep `ready.txt`, `palette.txt`, and `clear.txt`. Ready must identify SQLsaber and `verify-sqlite`. Clear must show the slash-command result. Do not claim chat-query verification from this recipe.

## Gotchas

- `tui start` already passes `-d verify-sqlite`. Starting a second session against the same HOME is refused.
- `/` only opens the palette when the editor is empty. If text is present, `/` is a character.
- `control-sqlsaber tui send` without `--literal` lets tmux interpret key names (`Enter`, `Escape`, `C-d`). Use `--literal` for the slash-command text.
- Submitting a question without a provider key hangs on auth. Stop the session and skip; do not keep sending keys.
- `control-sqlsaber cleanup` kills session `sqlsaber-verify` if you forget `tui stop`.
- The ASCII banner uses box-drawing characters. Assert on `slash commands` and `DB: verify-sqlite`, which survive `tmux capture-pane`.
