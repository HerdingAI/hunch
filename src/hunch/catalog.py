"""Pass 1 — walk folders and record what exists.

Deliberately cheap: it stats files but never opens them, so a full pass costs
directory metadata rather than reading every byte on disk. Content hashing is
the enrichment worker's job, because that stage has to open the file anyway.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Iterator

from .config import Config, classify

BATCH = 2000
ROOT_TIMEOUT = 10.0


def _clean(value: str) -> str:
    """Make a filesystem string safe to store.

    Paths can carry bytes that are not valid UTF-8; os.scandir surfaces those
    as lone surrogates, which cannot be encoded. Replace them so the row is
    still storable, and let the caller notice the path changed.
    """
    return value.encode("utf-8", "replace").decode("utf-8", "replace").replace("\x00", "")


def iter_files(root: Path, cfg: Config) -> Iterator[tuple[str, int, float]]:
    stack = [str(root)]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                for entry in it:
                    try:
                        if entry.is_symlink():
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            if entry.name in cfg.exclude_dirs:
                                continue
                            if entry.name.endswith(cfg.exclude_dir_suffixes):
                                continue
                            stack.append(entry.path)
                        elif entry.is_file(follow_symlinks=False):
                            if entry.name.startswith(cfg.exclude_file_prefixes):
                                continue
                            if entry.name in cfg.exclude_filenames:
                                continue
                            st = entry.stat(follow_symlinks=False)
                            if st.st_size < cfg.min_size_bytes:
                                continue
                            yield entry.path, st.st_size, st.st_mtime
                    except (OSError, ValueError):
                        continue      # one bad entry must not end the walk
        except (OSError, PermissionError):
            continue


def _root_reachable(root: Path, timeout: float = ROOT_TIMEOUT) -> bool:
    """A bounded reachability check for one configured folder.

    Path.is_dir() on a stale or dropped network mount can block for the
    kernel's own mount timeout (often well past a minute) rather than
    failing fast. Running the check in a daemon thread with a deadline lets
    the crawl give up on that one root and move on instead of stalling the
    whole budgeted run.
    """
    result: dict[str, bool] = {}

    def check():
        try:
            result["ok"] = root.is_dir()
        except OSError:
            result["ok"] = False

    t = threading.Thread(target=check, daemon=True)
    t.start()
    t.join(timeout)
    return result.get("ok", False)


def crawl(conn, cfg: Config, folders: list[Path] | None = None) -> dict:
    folders = folders or cfg.folders
    started = time.time()
    seen = added = updated = 0
    batch: list[tuple] = []
    walked_roots: list[str] = []
    skipped_roots: list[str] = []

    # Row ids this run's walk actually confirmed present. The tombstone
    # sweep below deletes anything under a walked root that is *not* in
    # this set, rather than comparing "when was this last touched" against
    # "when did this run start": SQLite's unixepoch() is second-precision,
    # so two crawls that both land in the same wall-clock second (e.g. a
    # test with no sleep(), or a user re-running the CLI immediately) are
    # indistinguishable by timestamp alone -- a row this run just confirmed
    # and a row the *previous* run last confirmed can carry the exact same
    # last_seen value. An id-based "was it seen this run" check has no such
    # ambiguity, so it correctly tombstones a same-second deletion without
    # ever risking a false tombstone of a row this run just touched.
    conn.execute("CREATE TEMP TABLE IF NOT EXISTS _crawl_touched (id INTEGER PRIMARY KEY)")
    conn.execute("DELETE FROM _crawl_touched")

    def flush(rows):
        nonlocal added, updated
        if not rows:
            return
        for path, filename, ext, size, mtime, status in rows:
            cur = conn.execute(
                "SELECT id, size_bytes, mtime FROM file_catalog WHERE path = ?",
                (path,))
            row = cur.fetchone()
            if row is None:
                cur = conn.execute(
                    "INSERT INTO file_catalog(path, filename, ext, size_bytes, "
                    "mtime, status, last_seen) VALUES (?,?,?,?,?,?,unixepoch())",
                    (path, filename, ext, size, mtime, status))
                fid = cur.lastrowid
                added += 1
            else:
                fid, old_size, old_mtime = row
                changed = (old_size != size) or (old_mtime != mtime)
                if changed:
                    # Clear the hash too: a changed file is different content,
                    # so its old embedding must not be reused. retry_count
                    # resets too -- new content deserves a fresh retry
                    # budget, not one still carrying failures against
                    # whatever the file used to contain (and worker.py's
                    # _phase_pending_rows cooldown gates purely on
                    # retry_count > 0, so a stale nonzero count here would
                    # wrongly delay enriching content that was never even
                    # attempted).
                    conn.execute(
                        "UPDATE file_catalog SET size_bytes=?, mtime=?, "
                        "status=?, content_hash=NULL, error_reason=NULL, "
                        "deleted_at=NULL, retry_count=0, last_seen=unixepoch() "
                        "WHERE id=?",
                        (size, mtime, status, fid))
                else:
                    # A file stuck at 'failed' from a bounded run of
                    # unreadable-at-enrichment-time attempts (see worker.py's
                    # enrich_one) gets one more shot each time it's still
                    # seen here, as long as it's under the retry cap -- so a
                    # permission fix or a remounted share self-heals without
                    # waiting on a separate maintenance job.
                    conn.execute(
                        "UPDATE file_catalog SET last_seen=unixepoch(), "
                        "deleted_at=NULL, status = CASE "
                        "WHEN status='failed' AND retry_count < ? "
                        "THEN 'pending' ELSE status END WHERE id=?",
                        (cfg.max_enrich_retries, fid))
                updated += 1
            conn.execute("INSERT OR IGNORE INTO _crawl_touched(id) VALUES (?)", (fid,))
        conn.commit()

    for root in folders:
        root = Path(root)
        if not _root_reachable(root):
            skipped_roots.append(str(root))
            continue
        walked_roots.append(str(root))
        for path, size, mtime in iter_files(root, cfg):
            safe = _clean(path)
            filename = os.path.basename(safe)
            ext = os.path.splitext(filename)[1][1:].lower()
            if safe != path:
                # Undecodable bytes in the name: the stored path can no longer
                # round-trip to the real file, so enrichment could never open
                # it. Mark it rather than let it look like a real failure.
                status = "skipped"
            elif classify(ext) == "unsupported":
                status = "unsupported"
            else:
                status = "pending"
            batch.append((safe, filename, ext, size, mtime, status))
            seen += 1
            if len(batch) >= BATCH:
                flush(batch)
                batch = []
    flush(batch)

    if walked_roots:
        # Scope the sweep to folders actually walked this run. A folder that
        # was unreachable (unplugged drive, dropped network share) must
        # leave its previously-indexed rows untouched, not get treated as a
        # mass deletion just because it couldn't be visited this time. Each
        # prefix carries a trailing separator so "nas" can't prefix-match a
        # sibling root like "nas2".
        conds = " OR ".join(["substr(path, 1, ?) = ?"] * len(walked_roots))
        params: list = []
        for root in walked_roots:
            prefix = root if root.endswith(os.sep) else root + os.sep
            params += [len(prefix), prefix]
        cur = conn.execute(
            "UPDATE file_catalog SET deleted_at = unixepoch() "
            "WHERE deleted_at IS NULL "
            "AND id NOT IN (SELECT id FROM _crawl_touched) "
            f"AND ({conds})",
            params)
        tombstoned = cur.rowcount
    else:
        # Nothing was reachable this run (e.g. every mount is down). Treat
        # that as "we don't know," not "everything was deleted."
        tombstoned = 0
    conn.commit()

    return {"seen": seen, "added": added, "updated": updated,
            "tombstoned": tombstoned, "seconds": time.time() - started,
            "skipped_roots": skipped_roots}
