"""Query the index. Shared by the CLI and the GUI so both rank identically."""
from __future__ import annotations

from dataclasses import dataclass

from . import db as db_mod
from .backends import get_backend
from .config import Config

# A file matching both lexically and semantically is a stronger hit than
# either alone.
HYBRID_BOOST = 0.15

# vec0's KNN picks its k nearest neighbors *before* this query's outer
# WHERE/JOIN filters run, so a tombstoned row within the k-nearest set
# consumes a slot invisibly and never reaches the deleted_at filter to be
# replaced. Over-fetch so live results have headroom to survive that
# filtering; the final result list is still capped at `limit` by search()'s
# own sort+slice. cli.py's scheduled _prune keeps tombstone volume bounded
# over time -- this margin only has to absorb churn between prune runs, not
# unbounded growth.
_FETCH_MULTIPLIER = 8
_FETCH_CAP = 400


@dataclass
class Result:
    path: str
    filename: str
    size: int
    score: float
    snippet: str = ""
    kind: str = ""


def _literal(conn, query: str, limit: int) -> list[Result]:
    like = f"%{query}%"
    rows = conn.execute(
        "SELECT path, filename, size_bytes FROM file_catalog "
        "WHERE deleted_at IS NULL AND (filename LIKE ? OR path LIKE ?) "
        "ORDER BY length(filename) ASC LIMIT ?", (like, like, limit)).fetchall()
    out = []
    for path, filename, size in rows:
        # Cheap, explainable relevance: an exact filename hit beats a path hit.
        score = 0.9 if query.lower() in (filename or "").lower() else 0.6
        out.append(Result(path, filename, size or 0, score))
    return out


def _semantic(conn, vector, limit: int) -> list[Result]:
    fetch_k = min(limit * _FETCH_MULTIPLIER, _FETCH_CAP)
    # One row per *content*, not per file. Enrichment dedups by content_hash
    # -- identical files share a single embedding -- but joining catalog rows
    # back onto it re-expands them: on a real index one embedding was shared
    # by 11,283 files, so a single vector entering the top-k emitted 11,283
    # rows at an identical score and buried every other match. Picking the
    # shortest path per hash keeps a stable, canonical-looking
    # representative, and it makes the over-fetch above mean what it says:
    # k vectors now yield up to k distinct results rather than k copies of
    # one file.
    rows = conn.execute(
        "SELECT c.path, c.filename, c.size_bytes, v.distance, "
        "       substr(e.extracted_text, 1, 240), e.source_kind "
        "FROM vec_embedding v "
        "JOIN file_embedding e ON e.rowid = v.rowid "
        "JOIN file_catalog c ON c.id = ("
        "    SELECT c2.id FROM file_catalog c2 "
        "    WHERE c2.content_hash = e.content_hash AND c2.deleted_at IS NULL "
        "    ORDER BY length(c2.path), c2.path LIMIT 1) "
        "WHERE v.embedding MATCH ? AND k = ?",
        (db_mod.serialize(vector), fetch_k)).fetchall()
    out = []
    for path, filename, size, distance, snippet, kind in rows:
        # vec_embedding's distance is Euclidean (vec0's default), not
        # cosine -- `1 - distance` was silently treating it as cosine
        # distance, which for unit vectors ranges roughly 0-2, while true
        # Euclidean distance between unit vectors ranges roughly 0-1.4.
        # That mismatch clamped nearly every real match to 0.0. For unit
        # vectors (db.serialize L2-normalizes every embedding), cosine
        # similarity = 1 - euclidean_distance**2 / 2 exactly.
        cos_sim = 1.0 - (float(distance) ** 2) / 2.0
        out.append(Result(path, filename, size or 0, max(0.0, cos_sim),
                          (snippet or "").strip(), kind or ""))
    return out


def search(conn, cfg: Config, query: str, mode: str = "hybrid",
           limit: int = 20, backend=None) -> list[Result]:
    if mode not in ("literal", "semantic", "hybrid"):
        raise ValueError(f"unknown search mode: {mode!r}")

    query = (query or "").strip()
    if not query:
        return []

    results: list[Result] = []
    if mode in ("literal", "hybrid"):
        results.extend(_literal(conn, query, limit))

    if mode in ("semantic", "hybrid"):
        backend = backend or get_backend(cfg)
        try:
            vector = backend.embed([query])[0]
        except Exception as exc:                       # noqa: BLE001
            # A remote backend (openrouter/ollama) can fail at any time --
            # network errors, auth failures, a down model server. Degrade to
            # whatever the literal pass already found rather than crash the
            # caller with a raw traceback, matching enrich_one's established
            # graceful-degrade pattern for this exact backend.embed() call.
            results.sort(key=lambda r: r.score, reverse=True)
            return results[:limit]
        by_path = {r.path: r for r in results}
        for hit in _semantic(conn, vector, limit):
            existing = by_path.get(hit.path)
            if existing is None:
                results.append(hit)
            else:
                # Clamped to 1.0: an already-high literal or semantic score
                # plus the boost would otherwise display as e.g. "115%" in
                # the GUI, which reads as a bug rather than a strong match.
                existing.score = min(max(existing.score, hit.score) + HYBRID_BOOST, 1.0)
                existing.snippet = existing.snippet or hit.snippet
                existing.kind = existing.kind or hit.kind

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:limit]
