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
    # Hermetic: this machine's real dev-venv package set must never decide
    # what this test asserts.
    monkeypatch.setattr(cli.config, "load_config", lambda: cli.config.Config())
    monkeypatch.setattr(cli.probe, "local_backend_importable", lambda: True)
    monkeypatch.setattr(cli.probe, "media_importable", lambda: True)
    monkeypatch.setattr(cli.probe, "gui_importable", lambda: True)
    rc = cli.main(["doctor"])
    out = capsys.readouterr().out.lower()
    assert rc == 0
    assert "cpu" in out and ("gpu" in out or "documents" in out)


def test_doctor_flags_gtk_bindings_the_window_cannot_reach(monkeypatch, capsys):
    # PyGObject is a distro package, not a pip extra, so the usual failure
    # is "installed system-wide but invisible to this venv" (a pipx install
    # without --system-site-packages). Terminal search keeps working, so
    # nothing crashes -- doctor is the only place that names it.
    from hunch.setup.probe import Capabilities

    fake_caps = Capabilities(has_gpu=False, vram_mb=0, ram_mb=8000,
                             free_disk_mb=50000, cpu_count=4,
                             has_tesseract=True, has_poppler=True, has_ffmpeg=True)
    monkeypatch.setattr(cli, "_open", lambda: (object(), object()))
    monkeypatch.setattr(cli.probe, "probe", lambda: fake_caps)
    monkeypatch.setattr(cli.probe, "local_backend_importable", lambda: True)
    monkeypatch.setattr(cli.probe, "media_importable", lambda: True)
    monkeypatch.setattr(cli.probe, "gui_importable", lambda: False)
    monkeypatch.setattr(cli.config, "load_config", lambda: cli.config.Config())

    rc = cli.main(["doctor"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "MISSING -- sudo apt install python3-gi" in out


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


def test_a_second_concurrent_index_refuses_to_run_instead_of_crashing(
        tmp_path, monkeypatch, capsys):
    # Reproduced live: the hourly scheduled timer crashed three runs in a
    # row with "sqlite3.OperationalError: database is locked" while a long
    # manual `hunch index` was still active -- each failure in well under a
    # second, because worker.py holds a write transaction open across the
    # actual transcribe/caption/embed call, and a second writer's read
    # snapshot goes stale before it can retry its way in. A repeatedly
    # failing systemd service can eventually hit its start-limit and stop
    # being retried at all, silently ending scheduled indexing for good.
    import fcntl
    from hunch import config as config_mod, db as db_mod

    cfg = config_mod.Config()
    cfg.folders = [tmp_path]
    conn = db_mod.connect(tmp_path / "i.db", dim=cfg.embed_dim)
    monkeypatch.setattr(cli, "_open", lambda: (conn, cfg))

    crawled = []
    monkeypatch.setattr(cli.catalog, "crawl", lambda c, cf: crawled.append(1))

    # Simulate another `hunch index` already holding the lock this run
    # would need -- exactly what a concurrent scheduled/manual overlap
    # looks like.
    lock_path = tmp_path / "i.db.lock"
    held = open(lock_path, "w")
    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        rc = cli.main(["index", "--scheduled"])
    finally:
        fcntl.flock(held, fcntl.LOCK_UN)
        held.close()

    assert rc == 0
    assert "already running" in capsys.readouterr().out
    assert crawled == []          # refused before touching the catalog at all


def test_index_runs_normally_once_the_other_instance_is_done(tmp_path, monkeypatch):
    from hunch import config as config_mod, db as db_mod

    cfg = config_mod.Config()
    cfg.folders = [tmp_path]
    conn = db_mod.connect(tmp_path / "i.db", dim=cfg.embed_dim)
    monkeypatch.setattr(cli, "_open", lambda: (conn, cfg))

    crawled = []
    monkeypatch.setattr(
        cli.catalog, "crawl",
        lambda c, cf: crawled.append(1) or {
            "seen": 0, "added": 0, "updated": 0, "tombstoned": 0,
            "seconds": 0.0, "skipped_roots": []})
    monkeypatch.setattr(cli, "_prune", lambda c: None)

    rc = cli.main(["index", "--catalog-only"])

    assert rc == 0
    assert crawled == [1]         # no lock held elsewhere -- runs for real


def test_scheduled_index_tracks_cumulative_daily_spend(tmp_path, monkeypatch):
    from hunch import config as config_mod, db as db_mod

    cfg = config_mod.Config()
    cfg.folders = [tmp_path]
    cfg.daily_budget_seconds = 100
    conn = db_mod.connect(tmp_path / "i.db", dim=cfg.embed_dim)
    # Past the first-index pass already -- this test verifies steady-state
    # daily tracking, not first-pass budgeting (covered separately).
    db_mod.set_meta(conn, "first_pass_done", "1")
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


def test_scheduled_index_spends_first_run_budget_until_backlog_clears(tmp_path, monkeypatch):
    # Regression test for a real bug: hunch setup never indexes directly --
    # only the hourly timer does, and it always passes --scheduled -- so
    # without first-pass tracking, the 5-hour first_run_budget_seconds was
    # unreachable and every firing (including the very first) used the
    # 20-min daily budget instead, turning a "few hours" first index into
    # potentially weeks on a large corpus.
    from hunch import config as config_mod, db as db_mod

    cfg = config_mod.Config()
    cfg.folders = [tmp_path]
    cfg.first_run_budget_seconds = 100
    cfg.daily_budget_seconds = 20
    conn = db_mod.connect(tmp_path / "i.db", dim=cfg.embed_dim)
    monkeypatch.setattr(cli, "_open", lambda: (conn, cfg))
    monkeypatch.setattr(
        cli.catalog, "crawl",
        lambda c, cf: {"seen": 0, "added": 0, "updated": 0, "tombstoned": 0,
                       "seconds": 0.0, "skipped_roots": []})
    monkeypatch.setattr(cli.probe, "on_ac_power", lambda: True)

    seen_budgets = []

    def fake_run(conn, cfg, budget_seconds, limit=0):
        seen_budgets.append(budget_seconds)
        return {"processed": 0, "counts": {}, "seconds": 40.0}

    monkeypatch.setattr(cli.worker, "run", fake_run)
    # Backlog still pending for the first two firings, clears on the third.
    phase_calls = []

    def fake_next_phase(conn, cfg=None):
        phase_calls.append(True)
        return None if len(phase_calls) >= 3 else "document"

    monkeypatch.setattr(cli.budget_mod, "next_phase", fake_next_phase)

    cli.main(["index", "--scheduled"])
    cli.main(["index", "--scheduled"])
    cli.main(["index", "--scheduled"])   # backlog clears here
    cli.main(["index", "--scheduled"])   # steady-state daily budget now

    assert seen_budgets == [100, 60, 20, 20]
    assert db_mod.get_meta(conn, "first_pass_done") == "1"


def test_setup_reports_a_failed_timer_install_instead_of_claiming_success(
        tmp_path, monkeypatch, capsys):
    # Regression test for a real bug: install_user_units() failures (no
    # systemctl, or `enable --now` erroring) were never surfaced -- cmd_setup
    # claimed "Nothing else is required. Indexing starts automatically" even
    # though the *only* thing that ever triggers enrichment never got
    # installed, leaving a "successfully installed" system that silently
    # never indexes anything.
    from hunch.setup.probe import Capabilities

    fake_caps = Capabilities(has_gpu=False, vram_mb=0, ram_mb=8000,
                             free_disk_mb=50000, cpu_count=4,
                             has_tesseract=True, has_poppler=True, has_ffmpeg=True)
    monkeypatch.setattr(cli.probe, "probe", lambda: fake_caps)
    monkeypatch.setattr(cli.probe, "local_backend_importable", lambda: True)
    monkeypatch.setattr(cli.probe, "media_importable", lambda: True)
    monkeypatch.setattr(cli.config, "load_config", lambda: cli.config.Config())
    monkeypatch.setattr(cli.config, "save_config", lambda cfg: None)
    monkeypatch.setattr(
        cli.install, "install_user_units",
        lambda: (_ for _ in ()).throw(
            RuntimeError("systemctl --user enable --now hunch-index.timer "
                        "failed: exit code 1")))
    monkeypatch.setattr(cli.install, "install_launcher", lambda: tmp_path / "x.desktop")
    monkeypatch.setattr(cli.install, "install_nautilus_script", lambda: tmp_path / "script")
    monkeypatch.setattr(cli.install, "bind_shortcut", lambda: False)

    rc = cli.main(["setup"])
    out = capsys.readouterr().out
    assert rc == 0                                     # not a hard failure
    assert "could not install the background indexing timer" in out
    assert "Nothing else is required" not in out


def test_setup_warns_instead_of_claiming_success_when_embedding_deps_are_missing(
        tmp_path, monkeypatch, capsys):
    # Regression test for a real bug: `pipx install hunch-search` with no
    # extras passes every hardware check in cmd_setup and then silently
    # fails every single file at embed time -- the only trace was an
    # error_reason buried in a DB column. cmd_setup must say so up front,
    # in the terminal the user is already looking at, and must not claim
    # "Nothing else is required" when nothing will actually work.
    from hunch.setup.probe import Capabilities

    fake_caps = Capabilities(has_gpu=False, vram_mb=0, ram_mb=8000,
                             free_disk_mb=50000, cpu_count=4,
                             has_tesseract=True, has_poppler=True, has_ffmpeg=True)
    monkeypatch.setattr(cli.probe, "probe", lambda: fake_caps)
    monkeypatch.setattr(cli.probe, "local_backend_importable", lambda: False)
    monkeypatch.setattr(cli.probe, "media_importable", lambda: True)
    monkeypatch.setattr(cli.config, "load_config", lambda: cli.config.Config())
    monkeypatch.setattr(cli.config, "save_config", lambda cfg: None)
    monkeypatch.setattr(cli.install, "install_user_units", lambda: [])
    monkeypatch.setattr(cli.install, "install_launcher", lambda: tmp_path / "x.desktop")
    monkeypatch.setattr(cli.install, "install_nautilus_script", lambda: tmp_path / "script")
    monkeypatch.setattr(cli.install, "bind_shortcut", lambda: False)

    rc = cli.main(["setup"])
    out = capsys.readouterr().out
    assert rc == 0                                     # not a hard failure
    assert 'pipx install --force "hunch-search[local]"' in out
    assert "Nothing else is required" not in out
    # The timer itself installed fine -- only the embedding deps are the
    # problem, so this warning must not be mistaken for the timer one.
    assert "could not install the background indexing timer" not in out


def test_setup_warns_when_media_deps_are_missing_even_though_local_is_fine(
        tmp_path, monkeypatch, capsys):
    # Regression test for a real gap the re-review caught: cmd_setup only
    # checked local_backend_importable(), not media_importable() -- a
    # narrower variant of the same C1 failure mode (audio/video files, not
    # every file) went unwarned by `hunch setup` even though `hunch doctor`
    # already covered it.
    from hunch.setup.probe import Capabilities

    fake_caps = Capabilities(has_gpu=False, vram_mb=0, ram_mb=8000,
                             free_disk_mb=50000, cpu_count=4,
                             has_tesseract=True, has_poppler=True, has_ffmpeg=True)
    monkeypatch.setattr(cli.probe, "probe", lambda: fake_caps)
    monkeypatch.setattr(cli.probe, "local_backend_importable", lambda: True)
    monkeypatch.setattr(cli.probe, "media_importable", lambda: False)
    monkeypatch.setattr(cli.config, "load_config", lambda: cli.config.Config())
    monkeypatch.setattr(cli.config, "save_config", lambda cfg: None)
    monkeypatch.setattr(cli.install, "install_user_units", lambda: [])
    monkeypatch.setattr(cli.install, "install_launcher", lambda: tmp_path / "x.desktop")
    monkeypatch.setattr(cli.install, "install_nautilus_script", lambda: tmp_path / "script")
    monkeypatch.setattr(cli.install, "bind_shortcut", lambda: False)

    rc = cli.main(["setup"])
    out = capsys.readouterr().out
    assert rc == 0
    assert 'pipx install --force "hunch-search[media]"' in out
    assert "Nothing else is required" not in out
    assert 'hunch-search[local]' not in out    # local deps were fine


def test_doctor_flags_a_missing_local_embedding_dependency(monkeypatch, capsys):
    from hunch.setup.probe import Capabilities

    fake_caps = Capabilities(has_gpu=False, vram_mb=0, ram_mb=8000,
                             free_disk_mb=50000, cpu_count=4,
                             has_tesseract=True, has_poppler=True, has_ffmpeg=True)
    monkeypatch.setattr(cli, "_open", lambda: (object(), object()))
    monkeypatch.setattr(cli.probe, "probe", lambda: fake_caps)
    monkeypatch.setattr(cli.probe, "local_backend_importable", lambda: False)
    monkeypatch.setattr(cli.probe, "media_importable", lambda: True)
    monkeypatch.setattr(cli.config, "load_config", lambda: cli.config.Config())

    rc = cli.main(["doctor"])
    out = capsys.readouterr().out
    assert rc == 0
    assert 'MISSING -- pipx install "hunch-search[local]"' in out


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


def test_reindex_embeddings_survives_a_failed_row(tmp_path, monkeypatch):
    # Regression test for a real bug: reindex --embeddings had no per-row
    # error handling and a single end-of-loop commit, so one transient
    # embed() failure (e.g. a network hiccup on a remote backend) would
    # crash the whole command and discard every vector already rebuilt in
    # that same run -- the exact command a user reaches for to recover a
    # broken index must not itself be this fragile.
    from hunch import config as config_mod, db as db_mod, worker as worker_mod
    from tests.test_worker import StubBackend

    cfg = config_mod.Config()
    conn = db_mod.connect(tmp_path / "i.db", dim=4)
    seed_backend = StubBackend()
    for name, body in [("good.txt", "a lease agreement"), ("bad.txt", "a recipe")]:
        f = tmp_path / name
        f.write_text(body)
        conn.execute("INSERT INTO file_catalog(path, filename, ext, size_bytes, status) "
                     "VALUES (?,?,?,?,'pending')", (str(f), name, "txt", f.stat().st_size))
    conn.commit()
    for row in conn.execute("SELECT id, path, ext, size_bytes FROM file_catalog").fetchall():
        worker_mod.enrich_one(conn, seed_backend, cfg, row)
    db_mod.set_meta(conn, "embed_model", seed_backend.model_id)
    db_mod.set_meta(conn, "embed_dim", str(seed_backend.dim))

    class FlakyBackend(StubBackend):
        def embed(self, texts):
            if any("recipe" in t for t in texts):
                raise RuntimeError("simulated transient failure")
            return super().embed(texts)

    flaky = FlakyBackend()
    monkeypatch.setattr(cli, "_open", lambda: (conn, cfg))
    monkeypatch.setattr(cli, "get_backend", lambda cfg: flaky)

    rc = cli.main(["reindex", "--embeddings"])
    assert rc == 0                                    # no crash
    # Both rows still have a vector: the good one freshly rebuilt, the bad
    # one left untouched rather than deleted with nothing to replace it.
    assert conn.execute("SELECT count(*) FROM vec_embedding").fetchone()[0] == 2
    # An incomplete migration must not be reported as a completed one --
    # vectors from different models aren't comparable (embedding_model_matches's
    # own contract), so stamping embed_model here would silently mix them.
    assert db_mod.get_meta(conn, "embed_model") == seed_backend.model_id


def test_reindex_rebuilds_the_vector_table_when_the_dimension_changes(
        tmp_path, monkeypatch):
    # Regression test for a real bug: vec0 fixes dimensionality at CREATE
    # and db.connect uses IF NOT EXISTS, so changing embed_dim left the old
    # table in place and every insert failed with "Dimension mismatch" --
    # including this command, which is exactly what the mismatch message
    # tells people to run. There was no way out of a dim change at all.
    from hunch import config as config_mod, db as db_mod, worker as worker_mod
    from tests.test_worker import StubBackend

    cfg = config_mod.Config()
    conn = db_mod.connect(tmp_path / "i.db", dim=4)
    f = tmp_path / "doc.txt"
    f.write_text("a lease agreement")
    conn.execute("INSERT INTO file_catalog(path, filename, ext, size_bytes, status) "
                 "VALUES (?,?,?,?,'pending')", (str(f), "doc.txt", "txt", f.stat().st_size))
    conn.commit()
    row = conn.execute("SELECT id, path, ext, size_bytes FROM file_catalog").fetchone()
    worker_mod.enrich_one(conn, StubBackend(), cfg, row)
    assert db_mod.vec_dim(conn) == 4

    class WiderBackend(StubBackend):
        model_id, dim = "wider", 8

        def embed(self, texts):
            return [[0.5] * 8 for _ in texts]

    monkeypatch.setattr(cli, "_open", lambda: (conn, cfg))
    monkeypatch.setattr(cli, "get_backend", lambda cfg: WiderBackend())

    assert cli.main(["reindex", "--embeddings"]) == 0
    assert db_mod.vec_dim(conn) == 8
    assert conn.execute("SELECT count(*) FROM vec_embedding").fetchone()[0] == 1


def test_prune_leaves_a_fresh_tombstones_embedding_alone(tmp_path):
    # Regression test for a real bug: _prune's orphan-detection queries
    # filtered on deleted_at IS NULL, so a file tombstoned this second (not
    # yet past PRUNE_TOMBSTONE_DAYS) was treated as if it didn't reference
    # its content_hash at all -- its embedding was deleted immediately
    # instead of waiting out the retention window, defeating the point of
    # having one.
    from hunch import catalog, config as config_mod, db as db_mod
    from hunch import worker as worker_mod
    from tests.test_worker import StubBackend

    root = tmp_path / "root"
    root.mkdir()
    cfg = config_mod.Config()
    cfg.folders = [root]
    conn = db_mod.connect(tmp_path / "i.db", dim=4)
    f = root / "a.txt"
    f.write_text("a lease agreement for the flat")
    catalog.crawl(conn, cfg)
    row = conn.execute(
        "SELECT id, path, ext, size_bytes FROM file_catalog").fetchone()
    worker_mod.enrich_one(conn, StubBackend(), cfg, row)

    f.unlink()
    catalog.crawl(conn, cfg)      # tombstones the row
    cli._prune(conn)

    assert conn.execute("SELECT count(*) FROM file_embedding").fetchone()[0] == 1
    assert conn.execute("SELECT count(*) FROM vec_embedding").fetchone()[0] == 1


def test_prune_purges_old_tombstones_and_their_orphaned_embeddings(tmp_path):
    from hunch import catalog, config as config_mod, db as db_mod
    from hunch import worker as worker_mod
    from tests.test_worker import StubBackend

    root = tmp_path / "root"
    root.mkdir()
    cfg = config_mod.Config()
    cfg.folders = [root]
    conn = db_mod.connect(tmp_path / "i.db", dim=4)
    f = root / "a.txt"
    f.write_text("a lease agreement for the flat")
    catalog.crawl(conn, cfg)
    row = conn.execute(
        "SELECT id, path, ext, size_bytes FROM file_catalog").fetchone()
    worker_mod.enrich_one(conn, StubBackend(), cfg, row)

    f.unlink()
    catalog.crawl(conn, cfg)
    conn.execute(
        "UPDATE file_catalog SET deleted_at = unixepoch() - ?",
        (cli.PRUNE_TOMBSTONE_DAYS * 86400 + 1,))
    conn.commit()
    stats = cli._prune(conn)

    assert stats == {"orphaned_vectors": 1, "purged_tombstones": 1}
    assert conn.execute("SELECT count(*) FROM file_catalog").fetchone()[0] == 0
    assert conn.execute("SELECT count(*) FROM file_embedding").fetchone()[0] == 0
    assert conn.execute("SELECT count(*) FROM vec_embedding").fetchone()[0] == 0


def test_prune_before_retention_lets_a_restored_file_keep_its_embedding(tmp_path):
    # The scenario _prune's docstring calls out by name: delete, prune
    # (before the retention window), then restore the exact same content
    # (a Trash restore, a re-added git-tracked file). A restore always gets
    # a fresh mtime, so catalog.crawl treats it as "changed" and requeues it
    # (content_hash cleared, status='pending') rather than leaving it
    # untouched -- the payoff of keeping the embedding alive is enrich_one's
    # dedup fast path (worker.py, "SELECT 1 FROM file_embedding WHERE
    # content_hash = ?"): the restored content hashes identically, so it
    # completes without ever calling the backend again. If prune had
    # already deleted that embedding, this same restore would silently pay
    # for a full re-embed instead of a free dedup hit.
    from hunch import catalog, config as config_mod, db as db_mod
    from hunch import worker as worker_mod
    from tests.test_worker import StubBackend

    root = tmp_path / "root"
    root.mkdir()
    cfg = config_mod.Config()
    cfg.folders = [root]
    conn = db_mod.connect(tmp_path / "i.db", dim=4)
    f = root / "a.txt"
    body = "a lease agreement for the flat"
    f.write_text(body)
    catalog.crawl(conn, cfg)
    row = conn.execute(
        "SELECT id, path, ext, size_bytes FROM file_catalog").fetchone()
    worker_mod.enrich_one(conn, StubBackend(), cfg, row)

    f.unlink()
    catalog.crawl(conn, cfg)
    cli._prune(conn)

    f.write_text(body)          # byte-identical restore, fresh mtime
    catalog.crawl(conn, cfg)

    class BackendThatMustNotBeCalled(StubBackend):
        def embed(self, texts):
            raise AssertionError(
                "dedup fast path should have skipped re-embedding")

    row = conn.execute(
        "SELECT id, path, ext, size_bytes FROM file_catalog").fetchone()
    status = worker_mod.enrich_one(conn, BackendThatMustNotBeCalled(), cfg, row)

    assert status == "done"
    assert conn.execute("SELECT count(*) FROM file_embedding").fetchone()[0] == 1


def test_prune_does_not_orphan_a_hash_still_held_by_an_unpurged_tombstone_during_a_rename(
        tmp_path):
    # Models the other scenario the docstring calls out: a move/rename
    # tombstones the old path and inserts a new row for the new path in the
    # SAME crawl, but the new row's content_hash starts out NULL until the
    # worker gets to it. If prune runs in that gap, the *only* thing keeping
    # the shared embedding alive is the old row's still-unpurged tombstone --
    # it must not be treated as "doesn't count."
    from hunch import catalog, config as config_mod, db as db_mod
    from hunch import worker as worker_mod
    from tests.test_worker import StubBackend

    root = tmp_path / "root"
    root.mkdir()
    cfg = config_mod.Config()
    cfg.folders = [root]
    conn = db_mod.connect(tmp_path / "i.db", dim=4)
    old = root / "old.txt"
    old.write_text("a lease agreement for the flat")
    catalog.crawl(conn, cfg)
    row = conn.execute(
        "SELECT id, path, ext, size_bytes FROM file_catalog").fetchone()
    worker_mod.enrich_one(conn, StubBackend(), cfg, row)

    new = root / "new.txt"
    old.rename(new)
    catalog.crawl(conn, cfg)    # tombstones old.txt, inserts new.txt pending

    cli._prune(conn)            # runs before the worker reaches new.txt

    assert conn.execute("SELECT count(*) FROM file_embedding").fetchone()[0] == 1
    new_status = conn.execute(
        "SELECT status FROM file_catalog WHERE filename='new.txt'").fetchone()[0]
    assert new_status == "pending"


def test_setup_says_which_configured_folders_do_not_exist(
        tmp_path, monkeypatch, capsys):
    # Found on a real machine: the default folder list is hardcoded to
    # ~/Documents, ~/Downloads, ~/Desktop, ~/Pictures, but this machine's
    # Documents and Downloads live on another drive. setup listed all four
    # as "Folders to index" and then said "Nothing else is required",
    # while the user's actual document library was never touched.
    from hunch.setup.probe import Capabilities

    real = tmp_path / "Pictures"
    real.mkdir()
    gone = tmp_path / "Documents"          # deliberately not created

    cfg = cli.config.Config()
    cfg.folders = [gone, real]
    fake_caps = Capabilities(has_gpu=False, vram_mb=0, ram_mb=8000,
                             free_disk_mb=50000, cpu_count=4,
                             has_tesseract=True, has_poppler=True, has_ffmpeg=True)
    monkeypatch.setattr(cli.probe, "probe", lambda: fake_caps)
    monkeypatch.setattr(cli.probe, "local_backend_importable", lambda: True)
    monkeypatch.setattr(cli.probe, "media_importable", lambda: True)
    monkeypatch.setattr(cli.config, "load_config", lambda: cfg)
    monkeypatch.setattr(cli.config, "save_config", lambda c: None)
    monkeypatch.setattr(cli.install, "install_user_units", lambda: [])
    monkeypatch.setattr(cli.install, "install_launcher", lambda: tmp_path / "x.desktop")
    monkeypatch.setattr(cli.install, "install_nautilus_script", lambda: tmp_path / "s")
    monkeypatch.setattr(cli.install, "bind_shortcut", lambda: False)

    assert cli.main(["setup"]) == 0
    out = capsys.readouterr().out
    assert "not found -- skipped" in out
    assert "does not exist" in out          # and names the count
    assert str(cli.config.config_path()) in out    # and where to fix it


def test_status_shows_which_folder_dominates_the_index(tmp_path, monkeypatch, capsys):
    # From a real machine: one scraped-data tree on the Desktop was 96% of a
    # 149,058-file index. It consumed the first-index budget and buried the
    # user's own documents in every search, and nothing in the tool said so
    # -- the status counts look identical whether an index is mostly your
    # documents or mostly a directory you forgot was there.
    from hunch import config as config_mod, db as db_mod

    big, small = tmp_path / "Desktop", tmp_path / "Pictures"
    cfg = config_mod.Config()
    cfg.folders = [big, small]
    conn = db_mod.connect(tmp_path / "i.db", dim=4)
    for i in range(90):
        conn.execute("INSERT INTO file_catalog(path, filename, ext, size_bytes, status) "
                     "VALUES (?,?,?,?,'done')",
                     (f"{big}/data/f{i}.json", f"f{i}.json", "json", 100))
    for i in range(10):
        conn.execute("INSERT INTO file_catalog(path, filename, ext, size_bytes, status) "
                     "VALUES (?,?,?,?,'done')",
                     (f"{small}/p{i}.jpg", f"p{i}.jpg", "jpg", 100))
    conn.commit()
    monkeypatch.setattr(cli, "_open", lambda: (conn, cfg))

    assert cli.main(["status"]) == 0
    out = capsys.readouterr().out
    assert "Where they came from" in out
    assert "90.0%" in out
    assert "most of your index" in out            # and says what to do
    assert str(cli.config.config_path()) in out


def test_index_says_so_when_the_first_pass_ends_with_work_left(tmp_path, monkeypatch, capsys):
    # Measured on a real 149,058-file index: 134 files/min against a 5-hour
    # first-pass budget leaves ~77,000 files unindexed, which then take about
    # 29 days at 20 minutes a day. The handoff was silent, so the user would
    # reasonably believe indexing had finished when most of it had not.
    from hunch import config as config_mod, db as db_mod

    cfg = config_mod.Config()
    cfg.folders = [tmp_path]
    cfg.first_run_budget_seconds = 100
    conn = db_mod.connect(tmp_path / "i.db", dim=cfg.embed_dim)
    conn.execute("INSERT INTO file_catalog(path, filename, ext, size_bytes, status) "
                 "VALUES (?,?,?,?,'pending')", (str(tmp_path / "a.txt"), "a.txt", "txt", 99))
    conn.commit()
    monkeypatch.setattr(cli, "_open", lambda: (conn, cfg))
    monkeypatch.setattr(cli.catalog, "crawl",
                        lambda c, cf: {"seen": 0, "added": 0, "updated": 0,
                                       "tombstoned": 0, "seconds": 0.0,
                                       "skipped_roots": []})
    monkeypatch.setattr(cli.probe, "on_ac_power", lambda: True)
    # Spends the whole first-pass budget without clearing the backlog.
    monkeypatch.setattr(cli.worker, "run",
                        lambda *a, **k: {"processed": 5, "counts": {"done": 5},
                                         "seconds": 100.0})
    monkeypatch.setattr(cli.budget_mod, "next_phase", lambda conn, cfg=None: "document")

    assert cli.main(["index", "--scheduled"]) == 0
    out = capsys.readouterr().out
    assert "still unindexed" in out
    assert "minutes a day" in out
    assert str(cli.config.config_path()) in out
    # And the state machine really did hand off, so it says this once.
    assert db_mod.get_meta(conn, "first_pass_done") == "1"


def test_prune_bounds_the_timing_table(tmp_path):
    # enrich_timing gained ~2.4 rows per enriched file and nothing removed
    # any: 106,322 rows part-way through one real 149k-file index, growing
    # forever across every future re-enrichment.
    from hunch import db as db_mod

    conn = db_mod.connect(tmp_path / "i.db", dim=4)
    for i in range(cli.TIMING_RETAINED_ROWS + 500):
        conn.execute("INSERT INTO enrich_timing(source_kind, stage, seconds) "
                     "VALUES ('document','hash',0.1)")
    conn.commit()
    cli._prune(conn)
    kept = conn.execute("SELECT count(*) FROM enrich_timing").fetchone()[0]
    assert kept <= cli.TIMING_RETAINED_ROWS + 1
    # And still enough for the estimate that reads it.
    from hunch import budget as budget_mod
    assert kept > budget_mod.RATE_SAMPLE


def test_status_estimates_how_long_the_current_phase_will_take(tmp_path, monkeypatch, capsys):
    # The counts alone never answered "how long will this take", which on a
    # large corpus is the difference between waiting up and going to bed.
    from hunch import config as config_mod, db as db_mod

    cfg = config_mod.Config()
    cfg.folders = [tmp_path]
    conn = db_mod.connect(tmp_path / "i.db", dim=4)
    for i in range(50):
        conn.execute("INSERT INTO file_catalog(path, filename, ext, size_bytes, status) "
                     "VALUES (?,?,?,?,'pending')",
                     (f"{tmp_path}/f{i}.txt", f"f{i}.txt", "txt", 100))
    for _ in range(10):
        conn.execute("INSERT INTO enrich_timing(source_kind, stage, seconds) "
                     "VALUES ('document','hash',1.0)")
    conn.commit()
    monkeypatch.setattr(cli, "_open", lambda: (conn, cfg))

    assert cli.main(["status"]) == 0
    out = capsys.readouterr().out
    assert "to go" in out
    assert "measured" in out
