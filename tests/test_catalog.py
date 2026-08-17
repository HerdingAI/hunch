import time
from pathlib import Path

import hunch.catalog as catalog
import hunch.db as db
from hunch.config import Config


def _cfg(tmp_path):
    cfg = Config()
    cfg.folders = [tmp_path]
    return cfg


def _connect(tmp_path):
    # The index db must live outside the folder being crawled: db.connect()
    # runs in WAL mode, which drops -wal/-shm files next to it on first
    # write. Putting the db inside tmp_path (the same root _cfg() points
    # the crawl at) would make the crawl "discover" its own storage files
    # as regular content -- a self-indexing artifact of this test fixture,
    # not something that can happen in real use (the production db path is
    # always in XDG_DATA_HOME, never inside an indexed folder).
    return db.connect(tmp_path.parent / (tmp_path.name + "-db") / "i.db")


def test_walk_skips_excluded_dirs_and_junk_files(tmp_path):
    (tmp_path / "keep.txt").write_text("hello world, this is real content")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "dep.js").write_text("x" * 100)
    (tmp_path / "pkg.dist-info").mkdir()
    (tmp_path / "pkg.dist-info" / "METADATA").write_text("y" * 100)
    (tmp_path / "._resourcefork.pdf").write_text("z" * 100)
    (tmp_path / "~$lockfile.docx").write_text("z" * 100)
    (tmp_path / "tiny.txt").write_text("x")          # below min_size_bytes

    found = {Path(p).name for p, _, _ in catalog.iter_files(tmp_path, _cfg(tmp_path))}
    assert found == {"keep.txt"}


def test_crawl_inserts_and_sets_status_by_type(tmp_path):
    (tmp_path / "a.txt").write_text("some searchable words here")
    (tmp_path / "b.xyz").write_text("unknown type payload here")
    conn = _connect(tmp_path)
    stats = catalog.crawl(conn, _cfg(tmp_path))
    assert stats["seen"] == 2
    assert stats["added"] == 2
    rows = dict(conn.execute("SELECT filename, status FROM file_catalog"))
    assert rows["a.txt"] == "pending"
    assert rows["b.xyz"] == "unsupported"


def test_recrawl_requeues_changed_and_leaves_unchanged_alone(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("first version of the content")
    conn = _connect(tmp_path)
    catalog.crawl(conn, _cfg(tmp_path))
    conn.execute("UPDATE file_catalog SET status='done', content_hash='h1'")
    conn.commit()

    # Unchanged file must stay done.
    catalog.crawl(conn, _cfg(tmp_path))
    assert conn.execute("SELECT status FROM file_catalog").fetchone()[0] == "done"

    # Changed file must be re-queued with its stale hash cleared.
    time.sleep(0.01)
    f.write_text("second version, noticeably different content")
    catalog.crawl(conn, _cfg(tmp_path))
    status, chash = conn.execute(
        "SELECT status, content_hash FROM file_catalog").fetchone()
    assert status == "pending"
    assert chash is None


def test_recrawl_gives_a_failed_file_another_chance_under_the_retry_cap(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("content behind a permission error that gets fixed")
    conn = _connect(tmp_path)
    catalog.crawl(conn, _cfg(tmp_path))
    conn.execute("UPDATE file_catalog SET status='failed', retry_count=1")
    conn.commit()
    catalog.crawl(conn, _cfg(tmp_path))
    assert conn.execute("SELECT status FROM file_catalog").fetchone()[0] == "pending"


def test_recrawl_leaves_a_failed_file_alone_once_retry_cap_is_hit(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("content behind a permanently broken extractor")
    conn = _connect(tmp_path)
    cfg = _cfg(tmp_path)
    catalog.crawl(conn, cfg)
    conn.execute("UPDATE file_catalog SET status='failed', retry_count=?",
                 (cfg.max_enrich_retries,))
    conn.commit()
    catalog.crawl(conn, cfg)
    assert conn.execute("SELECT status FROM file_catalog").fetchone()[0] == "failed"


def test_recrawl_resets_retry_count_when_content_actually_changes(tmp_path):
    # Regression test: a file that previously exhausted its retry cap
    # (e.g. a corrupt PDF) must get a fresh retry budget once its content
    # genuinely changes (e.g. the user re-saves it fixed) -- otherwise a
    # stale retry_count inherited from the old content's failure history
    # would wrongly delay enriching content that was never even attempted
    # (worker.py's _phase_pending_rows cooldown gates on retry_count > 0).
    f = tmp_path / "a.txt"
    f.write_text("content behind a permanently broken extractor")
    conn = _connect(tmp_path)
    cfg = _cfg(tmp_path)
    catalog.crawl(conn, cfg)
    conn.execute("UPDATE file_catalog SET status='failed', retry_count=?",
                 (cfg.max_enrich_retries,))
    conn.commit()

    time.sleep(0.01)
    f.write_text("a completely different, working replacement")
    catalog.crawl(conn, cfg)
    status, retry_count = conn.execute(
        "SELECT status, retry_count FROM file_catalog").fetchone()
    assert status == "pending"
    assert retry_count == 0


def test_crawl_tombstones_deleted_files(tmp_path):
    f = tmp_path / "gone.txt"
    f.write_text("this file will be deleted shortly")
    conn = _connect(tmp_path)
    catalog.crawl(conn, _cfg(tmp_path))
    f.unlink()
    stats = catalog.crawl(conn, _cfg(tmp_path))
    assert stats["tombstoned"] == 1
    assert conn.execute(
        "SELECT deleted_at IS NOT NULL FROM file_catalog").fetchone()[0] == 1


def test_walk_survives_unreadable_directory(tmp_path):
    (tmp_path / "ok.txt").write_text("readable content lives here")
    bad = tmp_path / "locked"
    bad.mkdir()
    (bad / "hidden.txt").write_text("unreachable content")
    bad.chmod(0o000)
    try:
        names = {Path(p).name for p, _, _ in
                 catalog.iter_files(tmp_path, _cfg(tmp_path))}
        assert "ok.txt" in names          # walk continued despite the error
    finally:
        bad.chmod(0o755)


def test_recrawl_same_second_does_not_tombstone(tmp_path):
    # Regression test: comparing a sub-second time.time() run-start against
    # integer-second last_seen values would tombstone any row touched in the
    # same wall-clock second as the crawl began.
    (tmp_path / "a.txt").write_text("stable content across fast recrawls")
    conn = _connect(tmp_path)
    catalog.crawl(conn, _cfg(tmp_path))
    stats = catalog.crawl(conn, _cfg(tmp_path))
    assert stats["tombstoned"] == 0


def test_unreachable_folder_does_not_tombstone_its_files(tmp_path):
    live = tmp_path / "live"
    live.mkdir()
    missing = tmp_path / "missing"
    missing.mkdir()
    (live / "a.txt").write_text("content that stays reachable")
    (missing / "b.txt").write_text("content behind a folder that goes away")
    conn = _connect(tmp_path)
    catalog.crawl(conn, _cfg(tmp_path), folders=[live, missing])
    assert conn.execute(
        "SELECT status FROM file_catalog WHERE filename='b.txt'").fetchone()[0] == "pending"

    # Simulate an unmounted drive: the folder goes away, but the content
    # underneath it still exists somewhere -- it has not been deleted.
    moved_away = tmp_path / "missing_elsewhere"
    missing.rename(moved_away)
    stats = catalog.crawl(conn, _cfg(tmp_path), folders=[live, missing])
    assert stats["skipped_roots"] == [str(missing)]
    assert conn.execute(
        "SELECT deleted_at FROM file_catalog WHERE filename='b.txt'").fetchone()[0] is None

    # And it self-heals once the folder is reachable again.
    moved_away.rename(missing)
    catalog.crawl(conn, _cfg(tmp_path), folders=[live, missing])
    assert conn.execute(
        "SELECT deleted_at FROM file_catalog WHERE filename='b.txt'").fetchone()[0] is None


def test_tombstone_scoping_does_not_confuse_sibling_prefixes(tmp_path):
    # A root "nas" must not accidentally prefix-match a sibling root "nas2".
    nas = tmp_path / "nas"
    nas.mkdir()
    nas2 = tmp_path / "nas2"
    nas2.mkdir()
    (nas / "a.txt").write_text("lives under the short-named folder")
    (nas2 / "b.txt").write_text("lives under the folder with a shared prefix")
    conn = _connect(tmp_path)
    catalog.crawl(conn, _cfg(tmp_path), folders=[nas, nas2])
    (nas / "a.txt").unlink()
    stats = catalog.crawl(conn, _cfg(tmp_path), folders=[nas, nas2])
    assert stats["tombstoned"] == 1
    assert conn.execute(
        "SELECT deleted_at IS NOT NULL FROM file_catalog "
        "WHERE filename='b.txt'").fetchone()[0] == 0


def test_root_reachable_gives_up_on_a_hang(tmp_path, monkeypatch):
    stuck = tmp_path / "stuck"
    stuck.mkdir()
    orig_is_dir = Path.is_dir

    def hanging_is_dir(self):
        if self == stuck:
            time.sleep(5)
        return orig_is_dir(self)

    monkeypatch.setattr(Path, "is_dir", hanging_is_dir)
    started = time.time()
    assert catalog._root_reachable(stuck, timeout=0.1) is False
    assert time.time() - started < 1
