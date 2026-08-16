# tests/test_gui_import.py
import pytest


def test_gui_module_imports_without_a_display():
    # Importing must not require GTK to be initialised, so `hunch --help`
    # works on a headless box.
    mod = pytest.importorskip("hunch.gui.app")
    assert hasattr(mod, "run_gui")


def test_run_gui_reports_missing_pygobject(monkeypatch):
    import hunch.gui.app as app

    def boom(name, *a, **k):
        if name.startswith("gi"):
            raise ImportError("No module named 'gi'")
        return __import__(name, *a, **k)

    monkeypatch.setattr("builtins.__import__", boom)
    rc = app.run_gui("test")
    assert rc == 1        # a clear failure, not a traceback
