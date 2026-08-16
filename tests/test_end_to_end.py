"""The full lifecycle: add -> index -> search -> delete -> disappear."""
import hunch.catalog as catalog
import hunch.db as db
import hunch.search as search_mod
import hunch.worker as worker
from hunch.config import Config
from tests.test_search import VariedBackend


def test_add_index_search_delete_disappear(tmp_path):
    docs = tmp_path / "Documents"
    docs.mkdir()
    target = docs / "tenancy.txt"
    target.write_text("a lease agreement for the flat, twelve month term")
    (docs / "unrelated.txt").write_text("how to bake sourdough bread at home")

    cfg = Config()
    cfg.folders = [docs]
    conn = db.connect(tmp_path / "index.db", dim=4)
    backend = VariedBackend()

    # 1. catalog
    stats = catalog.crawl(conn, cfg)
    assert stats["added"] == 2

    # 2. enrich
    worker.run(conn, cfg, budget_seconds=60, backend=backend)
    done = conn.execute(
        "SELECT count(*) FROM file_catalog WHERE status='done'").fetchone()[0]
    assert done == 2

    # 3. search finds it
    results = search_mod.search(conn, cfg, "lease", backend=backend)
    assert results[0].filename == "tenancy.txt"

    # 4. delete and re-crawl
    target.unlink()
    stats = catalog.crawl(conn, cfg)
    assert stats["tombstoned"] == 1

    # 5. it is gone from search
    names = {r.filename for r in search_mod.search(conn, cfg, "lease", backend=backend)}
    assert "tenancy.txt" not in names


def test_changed_file_is_requeued_and_reembedded(tmp_path):
    docs = tmp_path / "Documents"
    docs.mkdir()
    f = docs / "note.txt"
    f.write_text("original content about gardening")
    cfg = Config()
    cfg.folders = [docs]
    conn = db.connect(tmp_path / "index.db", dim=4)
    backend = VariedBackend()

    catalog.crawl(conn, cfg)
    worker.run(conn, cfg, budget_seconds=60, backend=backend)

    import time
    time.sleep(0.01)
    f.write_text("a lease agreement replaced the gardening note")
    catalog.crawl(conn, cfg)
    assert conn.execute(
        "SELECT status FROM file_catalog").fetchone()[0] == "pending"

    worker.run(conn, cfg, budget_seconds=60, backend=backend)
    results = search_mod.search(conn, cfg, "lease", backend=backend)
    assert results and results[0].filename == "note.txt"
