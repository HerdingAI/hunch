"""Query the index. Shared by the CLI and the GUI so both rank identically."""
from __future__ import annotations

from dataclasses import dataclass

from . import db as db_mod
from .backends import get_backend
from .config import Config

# A file matching both lexically and semantically is a stronger hit than
# either alone.
HYBRID_BOOST = 0.15


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
    rows = conn.execute(
        "SELECT c.path, c.filename, c.size_bytes, v.distance, "
        "       substr(e.extracted_text, 1, 240), e.source_kind "
        "FROM vec_embedding v "
        "JOIN file_embedding e ON e.rowid = v.rowid "
        "JOIN file_catalog c ON c.content_hash = e.content_hash "
        "WHERE v.embedding MATCH ? AND k = ? AND c.deleted_at IS NULL",
        (db_mod.serialize(vector), limit)).fetchall()
    out = []
    for path, filename, size, distance, snippet, kind in rows:
        out.append(Result(path, filename, size or 0,
                          max(0.0, 1.0 - float(distance)),
                          (snippet or "").strip(), kind or ""))
    return out


def search(conn, cfg: Config, query: str, mode: str = "hybrid",
           limit: int = 20, backend=None) -> list[Result]:
    query = (query or "").strip()
    if not query:
        return []

    results: list[Result] = []
    if mode in ("literal", "hybrid"):
        results.extend(_literal(conn, query, limit))

    if mode in ("semantic", "hybrid"):
        backend = backend or get_backend(cfg)
        vector = backend.embed([query])[0]
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
