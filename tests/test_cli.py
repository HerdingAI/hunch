import json

import pytest

import hunch.cli as cli


def test_search_json_output(tmp_path, capsys, monkeypatch):
    from hunch.search import Result
    monkeypatch.setattr(cli, "_open", lambda: (object(), object()))
    monkeypatch.setattr(
        "hunch.search.search",
        lambda *a, **k: [Result("/x/a.pdf", "a.pdf", 100, 0.9, "snip", "document")])
    rc = cli.main(["search", "lease", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["filename"] == "a.pdf"
    assert payload[0]["score"] == 0.9


def test_search_with_no_results_returns_nonzero(capsys, monkeypatch):
    monkeypatch.setattr(cli, "_open", lambda: (object(), object()))
    monkeypatch.setattr("hunch.search.search", lambda *a, **k: [])
    assert cli.main(["search", "nothing"]) == 1


def test_doctor_reports_capabilities(capsys, monkeypatch):
    monkeypatch.setattr(cli, "_open", lambda: (object(), object()))
    rc = cli.main(["doctor"])
    out = capsys.readouterr().out.lower()
    assert rc == 0
    assert "cpu" in out and ("gpu" in out or "documents" in out)


def test_unknown_command_exits_nonzero():
    with pytest.raises(SystemExit):
        cli.main(["nonsense"])


def test_auth_openrouter_requires_confirmation_before_storing_key(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    assert cli.main(["auth", "openrouter"]) == 1


def test_auth_openrouter_stores_key_and_switches_backend_on_confirmation(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt="": "y")
    monkeypatch.setenv("HUNCH_OPENROUTER_KEY", "sk-test")
    stored = {}
    monkeypatch.setattr("hunch.backends.openrouter.store_api_key",
                        lambda key: stored.setdefault("key", key))
    saved = {}
    monkeypatch.setattr("hunch.config.save_config", lambda cfg: saved.setdefault("cfg", cfg))
    assert cli.main(["auth", "openrouter"]) == 0
    assert stored["key"] == "sk-test"
    assert saved["cfg"].backend == "openrouter"


def test_scheduled_index_tracks_cumulative_daily_spend(tmp_path, monkeypatch):
    from hunch import config as config_mod, db as db_mod

    cfg = config_mod.Config()
    cfg.folders = [tmp_path]
    cfg.daily_budget_seconds = 100
    conn = db_mod.connect(tmp_path / "i.db", dim=cfg.embed_dim)
    monkeypatch.setattr(cli, "_open", lambda: (conn, cfg))
    monkeypatch.setattr(
        cli.catalog, "crawl",
        lambda c, cf: {"seen": 0, "added": 0, "updated": 0, "tombstoned": 0,
                       "seconds": 0.0, "skipped_roots": []})
    # Hermetic: this machine's real AC-power state must never decide
    # whether this test's budget math runs.
    monkeypatch.setattr(cli.probe, "on_ac_power", lambda: True)

    seen_budgets = []

    def fake_run(conn, cfg, budget_seconds, limit=0):
        seen_budgets.append(budget_seconds)
        return {"processed": 0, "counts": {}, "seconds": 40.0}

    monkeypatch.setattr(cli.worker, "run", fake_run)

    cli.main(["index", "--scheduled"])
    cli.main(["index", "--scheduled"])
    cli.main(["index", "--scheduled"])

    assert seen_budgets == [100, 60, 20]   # each run gets what's left of today's 100s


def test_scheduled_index_defers_enrichment_on_battery(tmp_path, monkeypatch, capsys):
    from hunch import config as config_mod, db as db_mod

    cfg = config_mod.Config()
    cfg.folders = [tmp_path]
    conn = db_mod.connect(tmp_path / "i.db", dim=cfg.embed_dim)
    monkeypatch.setattr(cli, "_open", lambda: (conn, cfg))
    monkeypatch.setattr(
        cli.catalog, "crawl",
        lambda c, cf: {"seen": 0, "added": 0, "updated": 0, "tombstoned": 0,
                       "seconds": 0.0, "skipped_roots": []})
    monkeypatch.setattr(cli.probe, "on_ac_power", lambda: False)

    called = []
    monkeypatch.setattr(cli.worker, "run", lambda *a, **k: called.append(True))

    rc = cli.main(["index", "--scheduled"])
    assert rc == 0
    assert called == []                # the expensive pass never ran
    assert "battery" in capsys.readouterr().out.lower()


def test_reindex_embeddings_actually_rebuilds_vectors(tmp_path, monkeypatch):
    # Regression test for a real bug: the original design cleared
    # vec_embedding and waited for the next `hunch index` to rebuild it,
    # but enrich_one's dedup fast path ("file_embedding already has this
    # content_hash") short-circuits before ever regenerating a vector --
    # silently leaving semantic search permanently empty, and stamping
    # embed_model to a value worker.run() could never self-heal from.
    from hunch import config as config_mod, db as db_mod, worker as worker_mod
    from tests.test_worker import StubBackend

    cfg = config_mod.Config()
    conn = db_mod.connect(tmp_path / "i.db", dim=4)
    backend = StubBackend()
    f = tmp_path / "doc.txt"
    f.write_text("a lease agreement")
    conn.execute("INSERT INTO file_catalog(path, filename, ext, size_bytes, status) "
                 "VALUES (?,?,?,?,'pending')", (str(f), "doc.txt", "txt", f.stat().st_size))
    conn.commit()
    row = conn.execute("SELECT id, path, ext, size_bytes FROM file_catalog").fetchone()
    worker_mod.enrich_one(conn, backend, cfg, row)
    db_mod.set_meta(conn, "embed_model", backend.model_id)
    db_mod.set_meta(conn, "embed_dim", str(backend.dim))

    monkeypatch.setattr(cli, "_open", lambda: (conn, cfg))
    monkeypatch.setattr(cli, "get_backend", lambda cfg: backend)

    rc = cli.main(["reindex", "--embeddings"])
    assert rc == 0
    assert conn.execute("SELECT count(*) FROM vec_embedding").fetchone()[0] == 1
    # The index must not be left permanently mismatched afterward -- this
    # is what previously locked worker.run() out forever.
    assert db_mod.get_meta(conn, "embed_model") == backend.model_id
