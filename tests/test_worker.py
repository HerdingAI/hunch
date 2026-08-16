import hunch.db as db
import hunch.worker as worker
from hunch.backends.base import Backend
from hunch.config import Config


class StubBackend(Backend):
    model_id, dim = "stub", 4

    def embed(self, texts):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    def describe_image(self, path):
        return "a described image", ""

    def transcribe(self, path):
        return "a transcript", ""


def _row(conn, path, ext, size, status="pending"):
    conn.execute("INSERT INTO file_catalog(path, filename, ext, size_bytes, status) "
                 "VALUES (?,?,?,?,?)", (path, path.split("/")[-1], ext, size, status))
    conn.commit()
    return conn.execute("SELECT id, path, ext, size_bytes FROM file_catalog "
                        "WHERE path=?", (path,)).fetchone()


def test_document_is_embedded_and_marked_done(tmp_path):
    conn = db.connect(tmp_path / "i.db", dim=4)
    f = tmp_path / "a.txt"
    f.write_text("a lease agreement for a flat in the city")
    row = _row(conn, str(f), "txt", f.stat().st_size)
    status = worker.enrich_one(conn, StubBackend(), Config(), row)
    assert status == "done"
    assert conn.execute("SELECT count(*) FROM file_embedding").fetchone()[0] == 1


def test_identical_content_is_enriched_once(tmp_path):
    conn = db.connect(tmp_path / "i.db", dim=4)
    body = "identical content in two places"
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_text(body)
    b.write_text(body)
    cfg, backend = Config(), StubBackend()
    worker.enrich_one(conn, backend, cfg, _row(conn, str(a), "txt", a.stat().st_size))
    status = worker.enrich_one(conn, backend, cfg, _row(conn, str(b), "txt", b.stat().st_size))
    # Dedup is a fast path to the same terminal state as normal enrichment,
    # not a separate status -- the status vocabulary in Global Constraints
    # has no "dedup" value, so it must land on "done" like any other success.
    assert status == "done"
    assert conn.execute("SELECT count(*) FROM file_embedding").fetchone()[0] == 1


def test_tiny_image_is_skipped_not_failed(tmp_path):
    conn = db.connect(tmp_path / "i.db", dim=4)
    f = tmp_path / "icon.png"
    f.write_bytes(b"0" * 100)
    row = _row(conn, str(f), "png", 100)
    # A scope decision must not pollute the failure signal.
    assert worker.enrich_one(conn, StubBackend(), Config(), row) == "skipped"


def test_missing_file_is_requeued_not_immediately_failed(tmp_path):
    # A file missing at enrichment time isn't proof of deletion -- that
    # judgment belongs to catalog.crawl()'s authoritative directory listing,
    # not to the worker guessing from one failed stat() call.
    conn = db.connect(tmp_path / "i.db", dim=4)
    row = _row(conn, str(tmp_path / "ghost.txt"), "txt", 50)
    assert worker.enrich_one(conn, StubBackend(), Config(), row) == "pending"
    status, deleted_at, retries = conn.execute(
        "SELECT status, deleted_at, retry_count FROM file_catalog "
        "WHERE id=?", (row[0],)).fetchone()
    assert status == "pending"
    assert deleted_at is None
    assert retries == 1


def test_missing_file_fails_without_tombstoning_once_retries_exhausted(tmp_path):
    conn = db.connect(tmp_path / "i.db", dim=4)
    cfg = Config()
    row = _row(conn, str(tmp_path / "ghost.txt"), "txt", 50)
    status = "pending"
    for _ in range(cfg.max_enrich_retries):
        status = worker.enrich_one(conn, StubBackend(), cfg, row)
    assert status == "failed"
    deleted_at = conn.execute(
        "SELECT deleted_at FROM file_catalog WHERE id=?", (row[0],)).fetchone()[0]
    assert deleted_at is None      # the worker never decides deletion, only crawl does


def test_run_respects_the_time_budget(tmp_path):
    conn = db.connect(tmp_path / "i.db", dim=4)
    for i in range(20):
        f = tmp_path / f"f{i}.txt"
        f.write_text(f"document number {i} with some words")
        _row(conn, str(f), "txt", f.stat().st_size)
    stats = worker.run(conn, Config(), budget_seconds=0.0, backend=StubBackend())
    assert stats["processed"] == 0        # no budget, no work
    assert "seconds" in stats             # actual elapsed time, for daily-budget tracking


def test_phase_stalled_detects_a_backlog_that_never_shrinks(tmp_path):
    conn = db.connect(tmp_path / "i.db", dim=4)
    _row(conn, str(tmp_path / "a.txt"), "txt", 10)
    _row(conn, str(tmp_path / "b.txt"), "txt", 10)
    assert worker._phase_stalled(conn, "document") is False   # no baseline yet
    assert worker._phase_stalled(conn, "document") is True    # unchanged -> stalled

    conn.execute("UPDATE file_catalog SET status='done' WHERE filename='a.txt'")
    conn.commit()
    assert worker._phase_stalled(conn, "document") is False   # backlog shrank
