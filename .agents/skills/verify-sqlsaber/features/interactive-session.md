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
- Type `/` on an empty prompt to open the palette.
- Type `/clear`, `/thinking`, `/handoff GOAL`, `/exit`, or `/quit`.
- Press Ctrl+C during a query or Ctrl+D on an empty editor.

## Driving it with verify-sqlsaber

Preconditions:

- Doctor reports `HEALTHY`.
- `FIXTURE=$("$VERIFY_SQLSABER" path "$RUN_ID" fixture)` is available.
- TUI startup needs a key matching the configured model. A harmless placeholder environment value is sufficient only for local controls that never call the provider.

- **Open, palette, clear, and exit.** Start `env ANTHROPIC_API_KEY=verification-placeholder uv run saber -d "$FIXTURE"` through `drive`. Use `--input-sequence` to send `/`, two down arrows plus Enter to activate `Clear conversation`, then Ctrl+D. Require the slash-command hint, database footer, palette labels, `Conversation history cleared.`, and clean exit.
- **Thinking.** In a fresh drive, open the palette and change `Thinking mode`, or submit `/thinking off` as one pasted line. Require the visible status before exit.
- **Exit aliases.** In separate fresh drives where needed, submit bare `exit` or `quit`, or send Ctrl+D on an empty editor. Require clean exit.
- **Handoff and interruption.** These paths call or interrupt a real model. Exercise them only with a working provider credential. Capture the original thread ID, handoff goal, new thread ID, and the visible cancellation state.

## Gotchas

- A credential for the wrong provider does not satisfy TUI startup. Match the current model.
- `/` opens the palette only when the editor is empty. Bytewise automated typing that starts with `/` also opens it; use palette keys for deterministic harness proof.
- Fixed input delays are safe for local controls, not for model responses.
- `/handoff` invokes the model. `/clear`, thinking changes, palette open, and exit do not.
- The helper strips ANSI. Assert visible labels and messages, not screen coordinates or colors.
