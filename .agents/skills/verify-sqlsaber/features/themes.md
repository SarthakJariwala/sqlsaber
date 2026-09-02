# Themes

Theme commands let a user choose a Pygments style for terminal syntax highlighting, browse choices interactively, and reset the saved theme.

## Sub-features

- `theme-set-named` validates and saves a named Pygments theme.
- `theme-set-interactive` opens a searchable theme selector when no name is supplied.
- `theme-reset` removes the saved theme after confirmation.

## How to get to it (user POV)

- Run `saber theme set THEME`.
- Run `saber theme set` in a terminal to browse choices.
- Run `saber theme reset`, with `--yes` for automation.

## Driving it with verify-sqlsaber

Preconditions:

- Doctor reports `HEALTHY`.
- No `theme.json` exists in the isolated config directory.

- **Named selection.** Capture `saber theme set dracula`. It exits `0` and prints `Theme set to: dracula`.
- **Persisted proof.** Copy the file from `path theme-config`. It records `dracula` as both the selected name and Pygments style.
- **Reset.** Capture `saber theme reset --yes`. It prints `Theme reset to default: nord`. Confirm `theme.json` no longer exists, then run a help command to show the CLI still starts.
- **Interactive route.** In a fresh run, start `saber theme set` through `drive`, select or cancel one visible choice, and retain the transcript. Do not substitute the named path when claiming selector behavior.

## Gotchas

- A non-TTY call without `THEME` is a usage error.
- `SQLSABER_THEME` wins over the saved file in normal use, but this skill does not claim that precedence from ANSI-stripped evidence.
- Available names come from installed Pygments styles and can change with that dependency.
- ANSI colors are not stable proof. Use the success text and persisted JSON.
