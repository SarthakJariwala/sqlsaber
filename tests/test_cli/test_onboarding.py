from io import StringIO

from sqlsaber.cli.onboarding import welcome_screen
from sqlsaber.render import reset_io


def test_welcome_screen_prints_compact_greeting() -> None:
    buf = StringIO()
    reset_io(stdout=buf, stderr=StringIO(), tty=False)
    try:
        welcome_screen()
        text = buf.getvalue()
    finally:
        reset_io()

    assert "Welcome to SQLsaber!" in text
    assert "Let's get you set up" in text
    assert "█" not in text
