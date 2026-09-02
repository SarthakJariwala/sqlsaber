# Interactive session

The interactive session is the full-screen chat started by `saber` with no question. It shows database and model identity, accepts questions and local slash commands, and exits back to the shell.

## Sub-features

- `interactive-open` shows the chat editor, slash-command hint, and database footer.
- `interactive-palette` opens settings with `/` on an empty editor.
- `interactive-clear-thinking` clears history and changes session-only thinking state.
- `interactive-handoff` drafts a goal that can start a new thread with current context.
- `interactive-exit` accepts `/exit`, `/quit`, bare `exit` or `quit`, and Ctrl+D on an empty editor.
- `interactive-interrupt` uses Ctrl+C to cancel a running query.

## How to get to it (user POV)

- Run `saber -d FILE` or `saber -d SAVED_NAME` with no question.
- Type `/` on an empty prompt to open the palette. The first page lists session commands (`Thinking mode`, `Handoff thread`, `Clear conversation`, `Exit`, `Command help`) and then management commands such as `/auth setup`.
- Type `/clear`, `/thinking`, `/handoff GOAL`, `/help`, `/exit`, or `/quit`. Management families from the CLI (`/auth`, `/db`, `/knowledge`, `/models`, `/theme`, `/threads`) work as slash commands too.
- Press Ctrl+C during a query or Ctrl+D on an empty editor.

## Driving it with verify-sqlsaber

Preconditions:

- Doctor reports `HEALTHY`.
- `FIXTURE=$("$VERIFY_SQLSABER" path "$RUN_ID" fixture)` is available.
- TUI startup needs a key matching the configured model. A harmless placeholder environment value is sufficient only for local controls that never call the provider.

- **Open, palette, clear, and exit.** Start `uv run saber -d "$FIXTURE"` through `drive` with a placeholder `ANTHROPIC_API_KEY` when the configured model is Anthropic. Use `--timeout 40 --input-sequence '[[8, "/"], [11, "\u001b[B\u001b[B\r"], [15, "\u0004"]]'`. Two down arrows select `Clear conversation` (the third palette row). Require `slash commands`, `table name completions`, `DB: verification (SQLite)`, palette labels including `Thinking mode` and `Command help`, `Conversation history cleared.`, and `Goodbye!`.
- **Thinking.** In a fresh drive, open the palette and change `Thinking mode`. Do not paste `/thinking off` as one line; a leading `/` still opens the palette and leftover text can submit as a query.
- **Exit aliases.** In separate fresh drives where needed, submit bare `exit` or `quit`, or send Ctrl+D on an empty editor after the editor is ready. Require `Goodbye!`.
- **Handoff and interruption.** These paths call or interrupt a real model. Set `saber models set openai:gpt-5 --thinking-level off` first when only `OPENAI_API_KEY` is present. Capture the original thread ID, handoff goal, new thread ID, and the visible cancellation state.

## Gotchas

- A credential for the wrong provider does not satisfy TUI startup. The default model is `anthropic:claude-opus-4-5` even when `OPENAI_API_KEY` is set.
- `/` opens the palette only when the editor is empty. Bytewise automated typing that starts with `/` also opens it. Use palette keys for deterministic harness proof.
- Delays of 2/4/6 seconds fire before `uv run saber` shows the editor. Use 8/11/15 seconds from process start.
- Fixed input delays are safe for local controls, not for model responses.
- `/handoff` invokes the model. `/clear`, thinking changes, palette open, and exit do not.
- An ad-hoc `-d "$FIXTURE"` footer uses the file stem (`DB: verification (SQLite)`). A saved name prints that name.
- The helper strips ANSI. Assert visible labels and messages, not screen coordinates or colors.
