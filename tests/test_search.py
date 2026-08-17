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


class SixtyDegreeBackend(StubBackend):
    """query=[1,0] and doc=[0.5, sqrt(3)/2] are both unit vectors at
    exactly 60 degrees -- true cosine similarity is exactly 0.5. Found via
    live end-to-end testing: vec_embedding's `distance` column is
    Euclidean (vec0's untouched default), not cosine, so for these two
    unit vectors the Euclidean distance is exactly 1.0
    (sqrt(2-2*cos(60))). The old formula `max(0.0, 1.0 - distance)`
    computed max(0.0, 1.0 - 1.0) = 0.0 -- a moderately-similar match
    reported as a *complete* mismatch. Every real search on a real
    machine showed this: true matches scored ~0.0-0.07 regardless of how
    relevant they actually were."""
    dim = 2

    def embed(self, texts):
        return [[1.0, 0.0] if t == "query" else [0.5, 3 ** 0.5 / 2] for t in texts]


def test_semantic_score_reflects_true_cosine_similarity_not_euclidean_distance(tmp_path):
    conn = db.connect(tmp_path / "i.db", dim=2)
    cfg, backend = Config(), SixtyDegreeBackend()
    f = tmp_path / "doc.txt"
    f.write_text("a moderately related document")
    conn.execute("INSERT INTO file_catalog(path, filename, ext, size_bytes, status) "
                 "VALUES (?,?,?,?,'pending')", (str(f), "doc.txt", "txt", f.stat().st_size))
    conn.commit()
    row = conn.execute("SELECT id, path, ext, size_bytes FROM file_catalog").fetchone()
    worker.enrich_one(conn, backend, cfg, row)

    results = search.search(conn, cfg, "query", mode="semantic", backend=backend)
    assert len(results) == 1
    assert abs(results[0].score - 0.5) < 1e-4


def test_serialize_normalizes_vectors_from_backends_that_do_not(tmp_path):
    # openrouter.py and ollama.py return whatever their API gives back with
    # no normalization guarantee -- unlike local_inprocess.py's
    # normalize_embeddings=True. serialize() must normalize regardless of
    # source, since search.py's score formula only holds for unit vectors.
    raw = db.serialize([3.0, 4.0])           # magnitude 5, not unit
    import struct
    unpacked = struct.unpack("2f", raw)
    norm = sum(x * x for x in unpacked) ** 0.5
    assert abs(norm - 1.0) < 1e-6


def test_duplicate_files_do_not_fill_the_results_with_one_content(tmp_path):
    # Regression test from a real 149k-file index: enrichment dedups by
    # content_hash, so identical files share one embedding -- but joining
    # catalog rows back onto that embedding re-expanded them. One embedding
    # there was shared by 11,283 files, so a single vector entering the
    # top-k emitted 11,283 rows at an identical score and buried every
    # other match. Searching returned the same content over and over.
    conn = db.connect(tmp_path / "i.db", dim=4)
    cfg, backend = Config(), VariedBackend()
    body = "a lease agreement for the flat"
    for i in range(12):                      # same bytes, twelve paths
        f = tmp_path / f"copy{i}.txt"
        f.write_text(body)
        conn.execute("INSERT INTO file_catalog(path, filename, ext, size_bytes, status) "
                     "VALUES (?,?,?,?,'pending')",
                     (str(f), f.name, "txt", f.stat().st_size))
    other = tmp_path / "recipe.txt"
    other.write_text("how to bake sourdough bread")
    conn.execute("INSERT INTO file_catalog(path, filename, ext, size_bytes, status) "
                 "VALUES (?,?,?,?,'pending')",
                 (str(other), other.name, "txt", other.stat().st_size))
    conn.commit()
    for row in conn.execute("SELECT id, path, ext, size_bytes FROM file_catalog").fetchall():
        worker.enrich_one(conn, backend, cfg, row)

    # One embedding for the twelve copies, one for the outlier.
    assert conn.execute("SELECT count(*) FROM file_embedding").fetchone()[0] == 2

    results = search.search(conn, cfg, "lease", mode="semantic", backend=backend)
    paths = [r.path for r in results]
    assert len(paths) == len(set(paths))          # no path repeated
    assert len(results) == 2                      # one per distinct content
    # The outlier is still reachable rather than crowded out.
    assert any(r.filename == "recipe.txt" for r in results)
