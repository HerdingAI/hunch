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


def test_content_failure_increments_retry_count(tmp_path):
    # Regression test for a real bug: _finish() never touched retry_count,
    # so a content failure (bad extraction, no text, embed error) stayed
    # at retry_count=0 forever -- catalog.crawl's self-heal requeue
    # ("WHEN status='failed' AND retry_count < max THEN 'pending'") never
    # capped, meaning the file was re-extracted from scratch on every
    # single crawl indefinitely, wasting real work with no bound.
    conn = db.connect(tmp_path / "i.db", dim=4)
    f = tmp_path / "empty.txt"
    f.write_text(" " * 30)     # extractable size, but no real content
    row = _row(conn, str(f), "txt", f.stat().st_size)
    status = worker.enrich_one(conn, StubBackend(), Config(), row)
    assert status == "failed"
    retry_count = conn.execute(
        "SELECT retry_count FROM file_catalog WHERE id=?", (row[0],)).fetchone()[0]
    assert retry_count == 1


def test_good_outcome_resets_a_stale_retry_count(tmp_path):
    conn = db.connect(tmp_path / "i.db", dim=4)
    f = tmp_path / "a.txt"
    f.write_text("a lease agreement for the flat")
    row = _row(conn, str(f), "txt", f.stat().st_size)
    conn.execute("UPDATE file_catalog SET retry_count=2 WHERE id=?", (row[0],))
    conn.commit()
    status = worker.enrich_one(conn, StubBackend(), Config(), row)
    assert status == "done"
    retry_count = conn.execute(
        "SELECT retry_count FROM file_catalog WHERE id=?", (row[0],)).fetchone()[0]
    assert retry_count == 0


def test_missing_file_does_not_burn_all_retries_within_one_run(tmp_path):
    # Regression test for a real bug: enrich_one's missing-file branch sets
    # status back to 'pending' so it can be retried later, but drain()'s
    # own loop immediately reselected the same row again within the SAME
    # run -- a persistently-missing file could exhaust all of
    # cfg.max_enrich_retries's attempts in one pass instead of one attempt
    # per run, permanently landing on status='failed' before the file even
    # got a real chance to become available again across separate runs.
    conn = db.connect(tmp_path / "i.db", dim=4)
    cfg = Config()
    _row(conn, str(tmp_path / "ghost.txt"), "txt", 50)
    worker.run(conn, cfg, budget_seconds=5.0, backend=StubBackend())
    status, retry_count = conn.execute(
        "SELECT status, retry_count FROM file_catalog").fetchone()
    assert status == "pending"
    assert retry_count == 1
    assert cfg.max_enrich_retries > 1     # otherwise this test proves nothing


def test_changed_content_is_not_delayed_by_the_retry_cooldown(tmp_path):
    # The cooldown that fixes the bug above must not also delay brand-new
    # or freshly-changed content (always retry_count=0 the moment it
    # becomes pending) -- only repeat attempts on already-failing content
    # should ever wait.
    conn = db.connect(tmp_path / "i.db", dim=4)
    f = tmp_path / "a.txt"
    f.write_text("a lease agreement for the flat")
    _row(conn, str(f), "txt", f.stat().st_size)
    stats = worker.run(conn, Config(), budget_seconds=5.0, backend=StubBackend())
    assert stats["counts"] == {"done": 1}


def test_stage_models_are_released_so_whisper_and_vision_never_stack(tmp_path):
    # Regression test for a real bug: local_inprocess.py's stated premise is
    # that "a 4 GB card cannot hold the embedder, the vision model and
    # Whisper at once", but nothing ever called release() during a run --
    # only the GUI's idle timer did. `audio` (Whisper) runs immediately
    # before `image_caption` (vision) with the embedder resident throughout,
    # so on the 4 GB card the project targets all three stacked.
    released = []

    class TrackingBackend(StubBackend):
        def release(self):
            released.append(True)

    conn = db.connect(tmp_path / "i.db", dim=4)
    f = tmp_path / "note.mp3"
    f.write_bytes(b"ID3" + b"\x00" * 200)
    _row(conn, str(f), "mp3", f.stat().st_size)

    worker.run(conn, Config(), budget_seconds=5.0, backend=TrackingBackend())
    assert released, "audio phase must free Whisper before the next phase loads"


def test_embedder_only_phases_do_not_pay_a_pointless_reload(tmp_path):
    # The flip side: `document` and `image_meta` use the embedder alone, so
    # releasing after them would just re-pay the load cost for no VRAM win.
    released = []

    class TrackingBackend(StubBackend):
        def release(self):
            released.append(True)

    conn = db.connect(tmp_path / "i.db", dim=4)
    f = tmp_path / "a.txt"
    f.write_text("a lease agreement for the flat")
    _row(conn, str(f), "txt", f.stat().st_size)

    worker.run(conn, Config(), budget_seconds=5.0, backend=TrackingBackend())
    assert released == []


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


def test_caption_one_demotes_source_kind_when_file_is_missing(tmp_path):
    conn = db.connect(tmp_path / "i.db", dim=4)
    conn.execute(
        "INSERT INTO file_embedding(content_hash, extracted_text, source_kind, model) "
        "VALUES (?,?,?,?)", ("abc123", "ocr text", "image_meta", "stub"))
    conn.commit()
    status = worker.caption_one(conn, StubBackend(), Config(), "abc123",
                                str(tmp_path / "gone.jpg"))
    assert status == "failed"
    source_kind = conn.execute(
        "SELECT source_kind FROM file_embedding WHERE content_hash=?", ("abc123",)).fetchone()[0]
    assert source_kind == "image"


def test_run_survives_an_unhandled_caption_backend_exception(tmp_path):
    class CrashingCaptionBackend(StubBackend):
        def describe_image(self, path):
            raise RuntimeError("simulated captioning backend crash")

    conn = db.connect(tmp_path / "i.db", dim=4)
    img = tmp_path / "real.jpg"
    img.write_bytes(b"fake image bytes")
    conn.execute(
        "INSERT INTO file_catalog(path, filename, ext, size_bytes, status, content_hash) "
        "VALUES (?,?,?,?,?,?)", (str(img), "real.jpg", "jpg", 1000, "done", "def456"))
    conn.execute(
        "INSERT INTO file_embedding(content_hash, extracted_text, source_kind, model) "
        "VALUES (?,?,?,?)", ("def456", "ocr text", "image_meta", "stub"))
    conn.commit()
    result = worker.run(conn, Config(), budget_seconds=5.0, backend=CrashingCaptionBackend())
    assert result["processed"] == 1
    assert result["counts"] == {"failed": 1}
    source_kind = conn.execute(
        "SELECT source_kind FROM file_embedding WHERE content_hash=?", ("def456",)).fetchone()[0]
    assert source_kind == "image"


def test_a_run_stops_instead_of_failing_an_entire_corpus_identically(tmp_path):
    # Regression test for the failure that wasted a real first index: the
    # configured embedder was larger than the GPU, so backend.embed() raised
    # CUDA OOM on every file. The run kept going, marking each file failed
    # in turn -- on a 147k-file corpus that is hours of thrashing to index
    # exactly nothing, and it drives every file toward its retry cap for a
    # reason that has nothing to do with the file.
    class AlwaysOOM(StubBackend):
        def embed(self, texts):
            raise RuntimeError("CUDA out of memory. Tried to allocate 40.00 MiB")

    conn = db.connect(tmp_path / "i.db", dim=4)
    for i in range(worker.SYSTEMIC_FAILURE_STREAK + 20):
        f = tmp_path / f"doc{i}.txt"
        f.write_text(f"document {i} with real words in it")
        _row(conn, str(f), "txt", f.stat().st_size)

    try:
        worker.run(conn, Config(), budget_seconds=30.0, backend=AlwaysOOM())
        assert False, "expected SystemicFailure"
    except worker.SystemicFailure as exc:
        assert "consecutive backend failures" in str(exc)

    # Stopped early rather than working through every file...
    attempted = conn.execute(
        "SELECT count(*) FROM file_catalog WHERE status='failed'").fetchone()[0]
    assert attempted <= worker.SYSTEMIC_FAILURE_STREAK
    # ...leaving the rest pending for a run that might succeed.
    assert conn.execute(
        "SELECT count(*) FROM file_catalog WHERE status='pending'").fetchone()[0] > 0


def test_genuinely_bad_files_do_not_stop_a_run_that_is_working(tmp_path):
    # The guard must key on "nothing is succeeding", not on a failure count:
    # a corpus with many broken files mixed among good ones has to keep going.
    conn = db.connect(tmp_path / "i.db", dim=4)
    for i in range(worker.SYSTEMIC_FAILURE_STREAK + 10):
        f = tmp_path / f"empty{i}.txt"
        f.write_text(" " * 30)              # extractable size, no real text
        _row(conn, str(f), "txt", f.stat().st_size)
    good = tmp_path / "good.txt"
    good.write_text("a lease agreement for the flat")
    _row(conn, str(good), "txt", good.stat().st_size)

    stats = worker.run(conn, Config(), budget_seconds=30.0, backend=StubBackend())
    assert stats["counts"].get("done") == 1        # the good one still landed


def test_silence_is_skipped_not_failed(tmp_path):
    # Found live: a synthetic tone and a silent video both transcribed fine
    # and were marked 'failed' for containing no speech. An image with no
    # OCR text a few lines away in the same function lands on 'skipped',
    # and the spec is explicit about why the distinction exists -- "skipped
    # is distinct from failed on purpose... conflating them would corrupt
    # the failure-rate signal". Instrumental music, ambient video and
    # silent screen recordings are content facts, not errors.
    #
    # It costs work too: catalog.crawl requeues failed files under the
    # retry cap, so every silent file was re-transcribed on each crawl
    # until it exhausted its retries.
    class SilentBackend(StubBackend):
        def transcribe(self, path):
            return "", ""                  # transcribed fine, nothing said

    conn = db.connect(tmp_path / "i.db", dim=4)
    f = tmp_path / "instrumental.mp3"
    f.write_bytes(b"ID3" + b"\x00" * 400)
    row = _row(conn, str(f), "mp3", f.stat().st_size)

    assert worker.enrich_one(conn, SilentBackend(), Config(), row) == "skipped"
    status, reason, retries = conn.execute(
        "SELECT status, error_reason, retry_count FROM file_catalog "
        "WHERE id=?", (row[0],)).fetchone()
    assert status == "skipped"
    assert "no speech" in reason
    assert retries == 0            # a scope decision must not burn retries


def test_a_real_transcription_failure_is_still_a_failure(tmp_path):
    # The distinction must stay sharp in the other direction: a backend
    # that could not transcribe at all is an error, not a content fact.
    class BrokenBackend(StubBackend):
        def transcribe(self, path):
            return "", "transcribe failed: model could not load"

    conn = db.connect(tmp_path / "i.db", dim=4)
    f = tmp_path / "speech.mp3"
    f.write_bytes(b"ID3" + b"\x00" * 400)
    row = _row(conn, str(f), "mp3", f.stat().st_size)
    assert worker.enrich_one(conn, BrokenBackend(), Config(), row) == "failed"
