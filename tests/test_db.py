import hunch.db as db


def test_connect_creates_tables(tmp_path):
    conn = db.connect(tmp_path / "index.db")
    names = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"file_catalog", "file_embedding", "enrich_timing", "meta"} <= names


def test_vec_table_accepts_and_searches_vectors(tmp_path):
    conn = db.connect(tmp_path / "index.db", dim=4)
    conn.execute("INSERT INTO file_embedding(content_hash) VALUES ('a')")
    conn.execute("INSERT INTO vec_embedding(rowid, embedding) VALUES (1, ?)",
                 (db.serialize([1.0, 0.0, 0.0, 0.0]),))
    conn.commit()
    rows = list(conn.execute(
        "SELECT rowid, distance FROM vec_embedding "
        "WHERE embedding MATCH ? AND k = 1", (db.serialize([1.0, 0.0, 0.0, 0.0]),)))
    assert rows[0][0] == 1
    assert rows[0][1] < 1e-6


def test_status_check_constraint_rejects_unknown_status(tmp_path):
    import sqlite3
    conn = db.connect(tmp_path / "index.db")
    try:
        conn.execute("INSERT INTO file_catalog(path, filename, status) "
                     "VALUES ('/x', 'x', 'bogus')")
        raised = False
    except sqlite3.IntegrityError:
        raised = True
    assert raised


def test_embedding_model_mismatch_is_detected(tmp_path):
    conn = db.connect(tmp_path / "index.db")
    db.set_meta(conn, "embed_model", "modelA")
    db.set_meta(conn, "embed_dim", "1024")
    # A vector must exist for there to be a vector space worth protecting --
    # see test_an_index_with_no_embeddings_is_not_claimed_by_a_model below.
    conn.execute("INSERT INTO file_embedding(content_hash, extracted_text, "
                 "source_kind, model) VALUES ('h','t','document','modelA')")
    conn.commit()
    assert db.embedding_model_matches(conn, "modelA", 1024)
    assert not db.embedding_model_matches(conn, "modelB", 1024)
    assert not db.embedding_model_matches(conn, "modelA", 768)


def test_index_file_and_dir_are_owner_only(tmp_path):
    import stat
    db_path = tmp_path / "sub" / "index.db"
    db.connect(db_path)
    assert stat.S_IMODE(db_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(db_path.parent.stat().st_mode) == 0o700


def test_a_reader_never_takes_the_write_lock(tmp_path):
    # Regression test for the worst bug live testing found: db.connect()
    # ran executescript(), CREATE VIRTUAL TABLE and set_meta() on *every*
    # open, so `hunch status`, `hunch search` and the GUI each tried to take
    # the write lock the indexer already held, and died with a raw
    # sqlite3.OperationalError "database is locked" traceback. Reproduced
    # against a real 149k-file index mid-run: every search crashed for the
    # entire multi-hour first pass -- precisely the failure WAL exists to
    # prevent, defeated by schema setup sitting in the common path.
    import sqlite3

    path = tmp_path / "i.db"
    db.connect(path, dim=4)

    # Hold the write lock the way a long enrichment transaction does.
    writer = sqlite3.connect(path)
    writer.execute("BEGIN EXCLUSIVE")
    writer.execute("INSERT INTO meta(key, value) VALUES ('holding', '1')")
    try:
        reader = db.connect(path, dim=4)
        assert reader.execute(
            "SELECT count(*) FROM file_catalog").fetchone()[0] == 0
    finally:
        writer.rollback()
        writer.close()


def test_schema_is_still_created_on_a_fresh_database(tmp_path):
    # The other half of the fix: skipping the writes must not skip setup on
    # a database that genuinely has none.
    conn = db.connect(tmp_path / "fresh.db", dim=4)
    names = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table','view')")}
    assert {"file_catalog", "file_embedding", "meta"} <= names
    assert db.get_meta(conn, "schema_version") == str(db.SCHEMA_VERSION)
    assert db.vec_dim(conn) == 4


def test_an_index_with_no_embeddings_is_not_claimed_by_a_model(tmp_path):
    # Regression test for a real wedge: worker.run stamps embed_model at the
    # *start* of a run, before embedding anything, so a run that dies early
    # claims a vector space it never wrote into. Hit for real -- the first
    # index chose an embedder too big for the GPU, failed every file, and
    # stamped the index anyway; fixing the model then made every future run
    # abort with "run `hunch reindex --embeddings`", a recovery command with
    # zero vectors and zero stored text to rebuild from.
    conn = db.connect(tmp_path / "i.db", dim=4)
    db.set_meta(conn, "embed_model", "some/old-model")
    db.set_meta(conn, "embed_dim", "4")
    assert conn.execute("SELECT count(*) FROM file_embedding").fetchone()[0] == 0
    assert db.embedding_model_matches(conn, "a/different-model", 4) is True


def test_a_populated_index_still_refuses_a_different_model(tmp_path):
    # The guard must keep doing its actual job: once vectors exist, mixing
    # models silently produces meaningless search results.
    conn = db.connect(tmp_path / "i.db", dim=4)
    db.set_meta(conn, "embed_model", "some/old-model")
    db.set_meta(conn, "embed_dim", "4")
    conn.execute("INSERT INTO file_embedding(content_hash, extracted_text, "
                 "source_kind, model) VALUES ('h','t','document','some/old-model')")
    conn.commit()
    assert db.embedding_model_matches(conn, "a/different-model", 4) is False
    assert db.embedding_model_matches(conn, "some/old-model", 4) is True
