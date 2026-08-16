import hunch.db as db
import hunch.search as search
import hunch.worker as worker
from hunch.config import Config
from tests.test_worker import StubBackend


class VariedBackend(StubBackend):
    """Embeds the literal query 'lease' exactly on-axis, on-topic content
    near but not exactly on it (so there's headroom under the 1.0 clamp
    for the hybrid boost to show), and off-topic content far away."""

    def embed(self, texts):
        out = []
        for t in texts:
            tl = t.lower()
            if tl == "lease":
                out.append([1.0, 0.0, 0.0, 0.0])
            elif "lease" in tl:
                out.append([0.95, 0.05, 0.0, 0.0])
            else:
                out.append([0.0, 1.0, 0.0, 0.0])
        return out


def _seed(tmp_path):
    conn = db.connect(tmp_path / "i.db", dim=4)
    cfg, backend = Config(), VariedBackend()
    for name, body in [("lease.txt", "a lease agreement for the flat"),
                       ("recipe.txt", "how to bake sourdough bread")]:
        f = tmp_path / name
        f.write_text(body)
        conn.execute("INSERT INTO file_catalog(path, filename, ext, size_bytes, status) "
                     "VALUES (?,?,?,?,'pending')",
                     (str(f), name, "txt", f.stat().st_size))
    conn.commit()
    for row in conn.execute("SELECT id, path, ext, size_bytes FROM file_catalog").fetchall():
        worker.enrich_one(conn, backend, cfg, row)
    return conn, cfg, backend


def test_literal_mode_matches_filename(tmp_path):
    conn, cfg, backend = _seed(tmp_path)
    results = search.search(conn, cfg, "recipe", mode="literal", backend=backend)
    assert results and results[0].filename == "recipe.txt"


def test_semantic_mode_ranks_by_vector_distance(tmp_path):
    conn, cfg, backend = _seed(tmp_path)
    results = search.search(conn, cfg, "lease", mode="semantic", backend=backend)
    assert results[0].filename == "lease.txt"


def test_hybrid_boosts_a_file_matching_both_ways(tmp_path):
    conn, cfg, backend = _seed(tmp_path)
    hybrid = search.search(conn, cfg, "lease", mode="hybrid", backend=backend)
    semantic = search.search(conn, cfg, "lease", mode="semantic", backend=backend)
    top_h = next(r for r in hybrid if r.filename == "lease.txt")
    top_s = next(r for r in semantic if r.filename == "lease.txt")
    assert top_h.score > top_s.score
    # A file matching both ways combines a near-1.0 semantic score with the
    # boost; the display-facing percentage must never exceed 100%.
    assert top_h.score <= 1.0


def test_tombstoned_files_are_never_returned(tmp_path):
    conn, cfg, backend = _seed(tmp_path)
    conn.execute("UPDATE file_catalog SET deleted_at = unixepoch() "
                 "WHERE filename='lease.txt'")
    conn.commit()
    names = {r.filename for r in
             search.search(conn, cfg, "lease", mode="hybrid", backend=backend)}
    assert "lease.txt" not in names


def test_results_carry_a_snippet(tmp_path):
    conn, cfg, backend = _seed(tmp_path)
    results = search.search(conn, cfg, "lease", mode="semantic", backend=backend)
    # The snippet is what proves the match was semantic rather than lexical.
    assert "lease agreement" in results[0].snippet


class PositionedBackend(StubBackend):
    """Places each text at an exact, caller-chosen vector -- unlike
    VariedBackend's two-bucket scheme, this lets a test put tombstoned
    files precisely at the query's nearest neighbors."""

    def __init__(self, vectors):
        self.vectors = vectors

    def embed(self, texts):
        return [self.vectors.get(t, [0.0, 0.0, 0.0, 1.0]) for t in texts]


class CrashingBackend(StubBackend):
    def embed(self, texts):
        raise RuntimeError("simulated network failure")


def test_tombstoned_neighbors_do_not_shrink_the_result_count(tmp_path):
    # vec0's KNN picks its k nearest neighbors *before* the deleted_at
    # filter runs, so without over-fetching, tombstoned files placed at
    # the query's nearest neighbors consume k-slots and are then dropped
    # by the JOIN -- silently returning fewer than `limit` results even
    # though enough live, relevant files exist.
    conn = db.connect(tmp_path / "i.db", dim=4)
    cfg = Config()
    vectors = {
        "content of dead1.txt": [1.0, 0.0, 0.0, 0.0],
        "content of dead2.txt": [1.0, 0.0, 0.0, 0.0],
        "content of live1.txt": [0.9, 0.1, 0.0, 0.0],
        "content of live2.txt": [0.9, 0.1, 0.0, 0.0],
        "content of live3.txt": [0.9, 0.1, 0.0, 0.0],
        "content": [1.0, 0.0, 0.0, 0.0],
    }
    backend = PositionedBackend(vectors)
    for name in ["dead1.txt", "dead2.txt", "live1.txt", "live2.txt", "live3.txt"]:
        f = tmp_path / name
        f.write_text(f"content of {name}")
        conn.execute("INSERT INTO file_catalog(path, filename, ext, size_bytes, status) "
                     "VALUES (?,?,?,?,'pending')", (str(f), name, "txt", f.stat().st_size))
    conn.commit()
    for row in conn.execute("SELECT id, path, ext, size_bytes FROM file_catalog").fetchall():
        worker.enrich_one(conn, backend, cfg, row)
    conn.execute("UPDATE file_catalog SET deleted_at = unixepoch() "
                 "WHERE filename IN ('dead1.txt', 'dead2.txt')")
    conn.commit()

    results = search.search(conn, cfg, "content", mode="semantic", limit=2, backend=backend)
    assert len(results) == 2
    # Which 2 of the 3 tied live files come back is vec0's internal
    # tie-break to make, not something this test should pin down -- the
    # bug under test is the *count*, not the choice among equidistant ties.
    assert {r.filename for r in results}.issubset({"live1.txt", "live2.txt", "live3.txt"})


def test_hybrid_search_degrades_to_literal_on_backend_failure(tmp_path):
    conn = db.connect(tmp_path / "i.db", dim=4)
    cfg = Config()
    f = tmp_path / "doc.txt"
    f.write_text("a lease document")
    conn.execute("INSERT INTO file_catalog(path, filename, ext, size_bytes, status) "
                 "VALUES (?,?,?,?,'pending')", (str(f), "doc.txt", "txt", f.stat().st_size))
    conn.commit()
    results = search.search(conn, cfg, "doc", mode="hybrid", backend=CrashingBackend())
    assert [r.filename for r in results] == ["doc.txt"]


def test_semantic_search_returns_empty_on_backend_failure(tmp_path):
    conn = db.connect(tmp_path / "i.db", dim=4)
    cfg = Config()
    results = search.search(conn, cfg, "doc", mode="semantic", backend=CrashingBackend())
    assert results == []


def test_unknown_mode_raises_instead_of_silently_returning_nothing(tmp_path):
    conn = db.connect(tmp_path / "i.db", dim=4)
    cfg = Config()
    try:
        search.search(conn, cfg, "doc", mode="bogus")
        assert False, "expected ValueError"
    except ValueError:
        pass
