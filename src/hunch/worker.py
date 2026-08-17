"""Pass 2 -- turn catalogued files into searchable vectors.

Resumability is structural: `status` is the only durable state and each file's
embedding write plus status flip happens in one transaction, so killing the
process mid-run leaves nothing to reconcile.
"""
from __future__ import annotations

import hashlib
import os
import time

from . import budget as budget_mod
from . import db as db_mod
from . import extract, imagemeta
from .backends import get_backend
from .config import Config, classify

QUICK_HASH_THRESHOLD = 8 * 1024 * 1024
BATCH = 50
# Phases that load a heavy model of their own on top of the embedder --
# Whisper for `audio`, the vision model for `image_caption`. Released when
# the phase ends so the two never stack (see drain()'s finally block).
STAGE_MODEL_PHASES = {"audio", "image_caption"}
# Consecutive *backend* failures that mean "stop the run" rather than "these
# files are bad". Comfortably above any plausible transient blip, well below
# the cost of failing a whole corpus one file at a time.
SYSTEMIC_FAILURE_STREAK = 25


class SystemicFailure(RuntimeError):
    """The backend keeps failing; the run stops instead of grinding."""


# Reasons written by the backend rather than by the extractor. A file with
# no text is a fact about that file; a failed embed is a fact about the run.
BACKEND_ERROR_PREFIXES = ("embed failed", "caption failed", "transcribe failed")


def _is_backend_failure(conn, phase: str, item) -> bool:
    if phase == "image_caption":
        return True          # this phase's only failure mode is the backend
    reason = conn.execute(
        "SELECT error_reason FROM file_catalog WHERE id = ?",
        (item[0],)).fetchone()
    return bool(reason and reason[0]
                and reason[0].startswith(BACKEND_ERROR_PREFIXES))


def content_hash(path: str, size: int) -> tuple[str | None, str]:
    """Identity for dedup.

    Files under 8 MB are hashed whole. Larger ones use size plus the first and
    last 1 MB: hashing every byte of a multi-GB video would cost more I/O than
    the enrichment it is meant to save.
    """
    h = hashlib.sha256()
    h.update(str(size).encode())
    try:
        with open(path, "rb") as fh:
            if size <= QUICK_HASH_THRESHOLD:
                for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                    h.update(chunk)
            else:
                h.update(fh.read(1024 * 1024))
                fh.seek(-1024 * 1024, os.SEEK_END)
                h.update(fh.read(1024 * 1024))
    except OSError as exc:
        return None, f"read failed: {exc}"
    return h.hexdigest(), ""


def _finish(conn, fid, status, chash=None, reason=None):
    # No tombstone flag here by design: deletion is catalog.crawl()'s call
    # alone, made from an authoritative directory listing, never a judgment
    # the worker makes from a failed operation.
    #
    # retry_count moves here too: catalog.crawl()'s self-heal requeue
    # ("WHEN status='failed' AND retry_count < max THEN 'pending'") only
    # works if something actually increments retry_count on every failure.
    # Before this, only enrich_one's missing-file branch did -- a content
    # failure (bad PDF, no extractable text, embed error) left retry_count
    # at 0 forever, so crawl requeued it every single run with no cap,
    # burning real extraction work on an unfixable file indefinitely. A
    # good terminal outcome resets the counter so a stale failure from
    # long ago doesn't cost a fresh future failure any of its retry budget.
    if status == "failed":
        retry_expr = "retry_count + 1"
    elif status in ("done", "skipped", "unsupported"):
        retry_expr = "0"
    else:
        retry_expr = "retry_count"
    conn.execute(
        f"UPDATE file_catalog SET last_attempt = unixepoch(), status = ?, "
        f"content_hash = COALESCE(?, content_hash), error_reason = ?, "
        f"retry_count = {retry_expr} WHERE id = ?",
        (status, chash, extract.clean_text(reason) or None, fid))
    conn.commit()
    return status


def _store(conn, chash, vector, text, kind, model):
    conn.execute(
        "INSERT INTO file_embedding(content_hash, extracted_text, source_kind, "
        "model, char_count) VALUES (?,?,?,?,?) "
        "ON CONFLICT(content_hash) DO UPDATE SET extracted_text=excluded.extracted_text, "
        "source_kind=excluded.source_kind, char_count=excluded.char_count",
        (chash, text[:2_000_000], kind, model, len(text)))
    rowid = conn.execute(
        "SELECT rowid FROM file_embedding WHERE content_hash = ?", (chash,)).fetchone()[0]
    conn.execute("DELETE FROM vec_embedding WHERE rowid = ?", (rowid,))
    conn.execute("INSERT INTO vec_embedding(rowid, embedding) VALUES (?, ?)",
                 (rowid, db_mod.serialize(vector)))


def _timing(conn, kind, stage, seconds, nbytes=None):
    conn.execute("INSERT INTO enrich_timing(source_kind, stage, seconds, bytes) "
                 "VALUES (?,?,?,?)", (kind, stage, seconds, nbytes))


def enrich_one(conn, backend, cfg: Config, row) -> str:
    fid, path, ext, size = row
    size = size or 0
    kind = classify(ext)

    if kind == "unsupported":
        return _finish(conn, fid, "unsupported")
    if not os.path.exists(path):
        # A missing file at enrichment time is not proof of deletion --
        # permission errors and stale network handles look identical to a
        # real delete from here. Deletion is catalog.crawl()'s call alone,
        # made from an authoritative directory listing; this only requeues,
        # bounded by cfg.max_enrich_retries so a persistently unreadable
        # file eventually stops being retried on every run instead of
        # blocking budget forever.
        retries = conn.execute(
            "SELECT retry_count FROM file_catalog WHERE id=?", (fid,)
        ).fetchone()[0] + 1
        if retries >= cfg.max_enrich_retries:
            conn.execute(
                "UPDATE file_catalog SET status='failed', retry_count=?, "
                "last_attempt=unixepoch(), error_reason=? WHERE id=?",
                (retries, "file unreadable after repeated attempts", fid))
            conn.commit()
            return "failed"
        conn.execute(
            "UPDATE file_catalog SET status='pending', retry_count=?, "
            "last_attempt=unixepoch(), error_reason=? WHERE id=?",
            (retries, "file unreadable at enrichment time", fid))
        conn.commit()
        return "pending"

    t0 = time.time()
    chash, err = content_hash(path, size)
    if err:
        return _finish(conn, fid, "failed", reason=err)
    _timing(conn, kind, "hash", time.time() - t0, size)

    if conn.execute("SELECT 1 FROM file_embedding WHERE content_hash = ?",
                    (chash,)).fetchone():
        return _finish(conn, fid, "done", chash=chash)

    text, err, source_kind = "", "", kind
    t0 = time.time()

    if kind == "document":
        text, err = extract.extract_document(path, ext, size, cfg)
        _timing(conn, kind, "extract", time.time() - t0, size)

    elif kind == "image":
        if size < cfg.image_skip_below:
            # Icons and sprites: still findable by name, not worth GPU time.
            return _finish(conn, fid, "skipped", chash=chash,
                           reason="below image size threshold")
        text = imagemeta.describe(path, cfg)
        source_kind = "image_meta"
        _timing(conn, kind, "image_meta", time.time() - t0, size)
        if not text.strip():
            return _finish(conn, fid, "skipped", chash=chash,
                           reason="no metadata or OCR text")

    elif kind in ("audio", "video"):
        text, err = backend.transcribe(path)
        _timing(conn, kind, "transcribe", time.time() - t0, size)
        if not err and not text.strip():
            # Transcription worked and there was simply nothing said --
            # instrumental music, ambient video, a silent screen capture.
            # That is a fact about the content, exactly like an image with
            # no OCR text a few lines above, which is why it lands on
            # 'skipped' rather than 'failed'. The spec is explicit that
            # conflating the two "would corrupt the failure-rate signal",
            # and it also costs real work: catalog.crawl requeues failed
            # files under the retry cap, so every silent file would be
            # re-transcribed on each crawl until it exhausted its retries.
            return _finish(conn, fid, "skipped", chash=chash,
                           reason="no speech detected")

    text = extract.clean_text(text)
    if err or not text.strip():
        return _finish(conn, fid, "failed", chash=chash,
                       reason=err or "extractor produced no text")

    t0 = time.time()
    try:
        vector = backend.embed([text])[0]
    except Exception as exc:                       # noqa: BLE001
        return _finish(conn, fid, "failed", chash=chash, reason=f"embed failed: {exc}")
    _timing(conn, kind, "embed", time.time() - t0, len(text))

    _store(conn, chash, vector, text, source_kind, backend.model_id)
    return _finish(conn, fid, "done", chash=chash)


def caption_one(conn, backend, cfg: Config, chash: str, path: str) -> str:
    """Phase 4: upgrade an existing metadata record with a visual description."""
    if not os.path.exists(path):
        # Demoted the same way a genuine captioning failure is, below: this
        # phase deliberately has no retry bookkeeping (unlike enrich_one),
        # so without this the row would be re-selected and re-attempted on
        # every remaining iteration of this same run, potentially burning
        # the whole budget on one file that's already vanished by the time
        # this phase runs. catalog.crawl() still tombstones the row for
        # real on its own schedule; this only stops the loop.
        conn.execute("UPDATE file_embedding SET source_kind='image' "
                     "WHERE content_hash=?", (chash,))
        conn.commit()
        return "failed"
    t0 = time.time()
    caption, err = backend.describe_image(path)
    _timing(conn, "image", "caption", time.time() - t0, None)
    if err or not caption.strip():
        conn.execute("UPDATE file_embedding SET source_kind='image' "
                     "WHERE content_hash=?", (chash,))
        conn.commit()
        return "failed"
    old = conn.execute("SELECT extracted_text FROM file_embedding "
                       "WHERE content_hash=?", (chash,)).fetchone()[0] or ""
    text = extract.clean_text(f"{caption}\n\n{old}")
    vector = backend.embed([text])[0]
    _store(conn, chash, vector, text, "image", backend.model_id)
    conn.commit()
    return "done"


def _phase_pending_rows(conn, phase: str, batch: int):
    if phase == "image_caption":
        return conn.execute(
            "SELECT e.content_hash, c.path FROM file_embedding e "
            "JOIN file_catalog c ON c.content_hash = e.content_hash "
            "WHERE e.source_kind = 'image_meta' AND c.deleted_at IS NULL "
            "ORDER BY c.size_bytes DESC LIMIT ?", (batch,)).fetchall()
    exts = budget_mod.phase_exts(phase)
    marks = ",".join("?" * len(exts))
    # last_attempt cooldown, gated on retry_count > 0: a persistently-
    # missing file's enrich_one call sets status back to 'pending' (see
    # its missing-file branch) so it can be retried later -- without this
    # floor, drain()'s own loop reselects it again within the same run,
    # moments later, potentially burning all of cfg.max_enrich_retries's
    # attempts in one pass. Gating on retry_count > 0 (rather than
    # last_attempt alone) is what keeps this from also delaying brand-new
    # or just-changed content, which is always retry_count=0 the moment
    # it becomes pending and must be eligible immediately, not after a
    # cooldown meant for repeat failures of the *same* content. 60s is
    # far shorter than the hourly scheduled timer, so a genuine cross-run
    # retry of already-failing content is unaffected.
    return conn.execute(
        f"SELECT id, path, ext, size_bytes FROM file_catalog "
        f"WHERE status='pending' AND deleted_at IS NULL AND ext IN ({marks}) "
        f"AND (retry_count = 0 OR last_attempt IS NULL "
        f"     OR last_attempt < unixepoch() - 60) "
        f"ORDER BY size_bytes ASC LIMIT ?", (*exts, batch)).fetchall()


def _phase_stalled(conn, phase: str) -> bool:
    """True if `phase`'s own backlog has not shrunk since the last check.

    A one-time large backlog that simply takes many runs to clear is *not*
    stalled -- its count goes down each time even if slowly. This only fires
    when new work is arriving at least as fast as it's being cleared, which
    is the case strict phase-completion ordering has no answer for.
    """
    key = f"phase_pending_{phase}"
    prev = db_mod.get_meta(conn, key)
    current = budget_mod.pending_count(conn, phase)
    db_mod.set_meta(conn, key, str(current))
    return prev is not None and current >= int(prev)


def run(conn, cfg: Config, budget_seconds: float, limit: int = 0, backend=None) -> dict:
    backend = backend or get_backend(cfg)
    if not db_mod.embedding_model_matches(conn, backend.model_id, backend.dim):
        raise RuntimeError(
            "index was built with a different embedding model. "
            "Run `hunch reindex --embeddings` to rebuild vectors from stored text.")
    db_mod.set_meta(conn, "embed_model", backend.model_id)
    db_mod.set_meta(conn, "embed_dim", str(backend.dim))

    run_started = time.time()
    overall = budget_mod.Budget(budget_seconds)
    counts: dict[str, int] = {}
    processed = 0
    consecutive_backend_failures = 0

    def stats() -> dict:
        # Actual elapsed wall-clock, not the requested budget -- the caller
        # (cli.cmd_index) needs this to track cumulative same-day spend
        # across the hourly scheduled timer, since a run can finish well
        # under budget when there's simply nothing left to do.
        return {"processed": processed, "counts": counts,
                "seconds": time.time() - run_started}

    def process(phase, item) -> bool:
        """Process one row/item for `phase`. Returns True once `limit` is hit."""
        nonlocal processed
        if phase == "image_caption":
            chash, path = item
            try:
                status = caption_one(conn, backend, cfg, chash, path)
            except Exception as exc:               # noqa: BLE001
                # An unforeseen captioning bug (OOM, decode failure, backend
                # timeout) must cost one file, not the run -- captioning is
                # the least predictable phase by design. Demote the same way
                # caption_one's own known-failure path does, so a
                # permanently-broken image doesn't re-crash every future pass.
                conn.rollback()
                conn.execute("UPDATE file_embedding SET source_kind='image' "
                             "WHERE content_hash=?", (chash,))
                conn.commit()
                status = "failed"
        else:
            try:
                status = enrich_one(conn, backend, cfg, item)
            except Exception as exc:               # noqa: BLE001
                # An unforeseen extractor bug must cost one file, not the run.
                conn.rollback()
                status = _finish(conn, item[0], "failed",
                                 reason=f"unhandled: {exc}")
        # Backend failures in a row mean the model/service itself is
        # unusable -- not that these files are bad. Observed for real: an
        # embedder larger than the GPU OOMed on every file while the run
        # kept going, hours of thrashing to index nothing out of 147k
        # files, driving every one of them toward its retry cap for a
        # reason that had nothing to do with the file. Counting only
        # backend errors is what separates that from a corpus whose first
        # files happen to be junk: "no extractable text" is a fact about a
        # file, "embed failed" is a fact about the run. Stopping leaves the
        # rest pending for a run that might work.
        nonlocal consecutive_backend_failures
        if status == "failed" and _is_backend_failure(conn, phase, item):
            consecutive_backend_failures += 1
        elif status != "failed":
            consecutive_backend_failures = 0
        counts[status] = counts.get(status, 0) + 1
        processed += 1
        if consecutive_backend_failures >= SYSTEMIC_FAILURE_STREAK:
            raise SystemicFailure(
                f"stopped after {consecutive_backend_failures} consecutive "
                f"backend failures -- the model or service looks unusable, "
                f"rather than these files being bad. Run `hunch doctor`; the "
                f"underlying error is recorded against the affected files.")
        return bool(limit and processed >= limit)

    def drain(phase, phase_budget) -> bool:
        """Process `phase` until phase_budget or the overall budget runs out,
        or its queue runs dry. Returns True if `limit` was hit."""
        try:
            while not phase_budget.exhausted() and not overall.exhausted():
                rows = _phase_pending_rows(conn, phase, BATCH)
                if not rows:
                    return False
                for item in rows:
                    if phase_budget.exhausted() or overall.exhausted():
                        return False
                    if process(phase, item):
                        return True
            return False
        finally:
            # Free the stage model this phase loaded before the next phase
            # loads its own. local_inprocess.py's own premise is that "a 4 GB
            # card cannot hold the embedder, the vision model and Whisper at
            # once" -- but nothing released them, and `audio` (Whisper) runs
            # immediately before `image_caption` (vision) with the embedder
            # resident throughout, which is precisely that three-way pileup.
            # Only these two phases hold a heavy stage model; `document` and
            # `image_meta` use the embedder alone, so releasing after them
            # would just re-pay the reload for nothing. In a `finally` so an
            # early return on an exhausted budget frees VRAM too.
            if phase in STAGE_MODEL_PHASES:
                backend.release()

    leading = budget_mod.next_phase(conn)
    if leading is None:
        return stats()

    # Detect starvation before doing any work: if the leading phase's own
    # backlog didn't shrink since the last run, strict phase-completion
    # ordering would starve every phase behind it forever. Guarantee those
    # phases a floor of this run's budget instead. In the normal case
    # (backlog shrinking, or this is the first run) this is a no-op and
    # phases are processed strictly in order to completion, as designed.
    later_active = [p for p in budget_mod.PHASES[budget_mod.PHASES.index(leading) + 1:]
                    if budget_mod.phase_has_pending(conn, p)]
    if later_active and _phase_stalled(conn, leading):
        floor = overall.total * 0.25
        if drain(leading, budget_mod.Budget(max(0.0, overall.total - floor))):
            return stats()
        share = overall.remaining() / len(later_active) if not overall.exhausted() else 0.0
        for phase in later_active:
            if overall.exhausted():
                break
            if drain(phase, budget_mod.Budget(min(share, overall.remaining()))):
                return stats()
        # Falls through to normal ordering for whatever's left, below.

    while not overall.exhausted():
        phase = budget_mod.next_phase(conn)
        if phase is None:
            break
        if drain(phase, overall):
            return stats()

    return stats()
