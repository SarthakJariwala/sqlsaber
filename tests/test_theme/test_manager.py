import pytest

from sqlsaber.theme import manager


@pytest.fixture(autouse=True)
def clear_theme_cache():
    manager.get_theme_manager.cache_clear()
    yield
    manager.get_theme_manager.cache_clear()


def test_dynamic_palette_uses_pygments_colors(monkeypatch):
    monkeypatch.setenv("SQLSABER_THEME", "zenburn")

    tm = manager.get_theme_manager()

    assert tm.pygments_style_name == "zenburn"
    assert tm.style("primary") == "#efdcbc"
    assert tm.style("accent") == "#e89393"
    assert tm.style("success") == "#cc9393"
    assert tm.style("error") == "#e37170"
    assert tm.style("info") == "#efef8f"
    assert tm.style("muted") == "#7f9f7f"

    monkeypatch.delenv("SQLSABER_THEME", raising=False)


def test_unknown_theme_falls_back_to_defaults(monkeypatch):
    monkeypatch.setenv("SQLSABER_THEME", "does-not-exist")

    tm = manager.get_theme_manager()

    assert tm.pygments_style_name == "nord"  # nord is the default

    monkeypatch.delenv("SQLSABER_THEME", raising=False)
