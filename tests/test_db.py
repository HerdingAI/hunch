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
    assert db.embedding_model_matches(conn, "modelA", 1024)
    assert not db.embedding_model_matches(conn, "modelB", 1024)
    assert not db.embedding_model_matches(conn, "modelA", 768)


def test_index_file_and_dir_are_owner_only(tmp_path):
    import stat
    db_path = tmp_path / "sub" / "index.db"
    db.connect(db_path)
    assert stat.S_IMODE(db_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(db_path.parent.stat().st_mode) == 0o700
