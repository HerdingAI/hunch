"""SQLite storage. One file, no server, no CREATE EXTENSION, no superuser."""
from __future__ import annotations

import os
import re
import sqlite3
import struct
from pathlib import Path

import sqlite_vec

from . import config

SCHEMA_VERSION = 1


def serialize(vector) -> bytes:
    """Pack a float sequence into sqlite-vec's expected binary layout.

    Also L2-normalizes: vec_embedding's `distance` column is Euclidean (its
    default; nothing configures cosine), and search.py's score formula
    converts that to cosine similarity via `1 - distance**2/2`, an identity
    that only holds for unit vectors. local_inprocess.py already normalizes
    (SentenceTransformer's normalize_embeddings=True), but openrouter.py and
    ollama.py return whatever their API gives back with no guarantee -- this
    is the one place every embedding (stored or query) passes through, so
    normalizing here makes the identity hold everywhere instead of only for
    one of three backends.
    """
    norm = sum(x * x for x in vector) ** 0.5
    if norm > 0:
        vector = [x / norm for x in vector]
    return struct.pack(f"{len(vector)}f", *vector)


def connect(path: Path | None = None, dim: int | None = None,
            check_same_thread: bool = True) -> sqlite3.Connection:
    path = Path(path) if path else config.db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    # The index aggregates extracted text from every indexed file into one
    # place, so it must not inherit a permissive umask (0o644 would let any
    # other local account on the machine read it).
    is_new = not path.exists()
    conn = sqlite3.connect(path, timeout=30.0, check_same_thread=check_same_thread)
    if is_new:
        os.chmod(path, 0o600)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    # Connection-local settings: these configure this handle, they do not
    # write to the database file, so they are safe on every open.
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 30000")

    # Everything below writes. Doing it unconditionally made every reader a
    # writer: `hunch status`, `hunch search` and the GUI all opened the db,
    # immediately tried to take the write lock, and lost it to the indexer
    # -- crashing with a raw "database is locked" traceback for the whole
    # multi-hour first index. That is the exact failure WAL is meant to
    # prevent (see the journal_mode line below), defeated by the schema
    # setup sitting in the common path. An already-initialised database
    # needs none of it, so readers now take the lock-free WAL read path.
    if _needs_init(conn):
        # Persistent in the file itself, so it only has to be set once --
        # and setting it is a write, which is why it lives in here.
        conn.execute("PRAGMA journal_mode = WAL")
        schema = (Path(__file__).parent / "schema.sql").read_text()
        conn.executescript(schema)

        if dim is None:
            stored = get_meta(conn, "embed_dim")
            dim = int(stored) if stored else config.Config().embed_dim
        # Virtual table dimension is fixed at creation, so it lives outside
        # schema.sql where the value can be interpolated.
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_embedding "
            f"USING vec0(embedding float[{dim}])"
        )
        set_meta(conn, "schema_version", str(SCHEMA_VERSION))
        conn.commit()
    return conn


def _needs_init(conn: sqlite3.Connection) -> bool:
    """True when this database still needs its schema written.

    Read-only by design: it must not be the thing that takes a write lock.
    A missing `meta` table raises OperationalError, which is simply what a
    brand-new (or pre-schema) database looks like from here.
    """
    try:
        row = conn.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    except sqlite3.OperationalError:
        return True
    return row is None or row[0] != str(SCHEMA_VERSION)


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )
    conn.commit()


def vec_dim(conn: sqlite3.Connection) -> int | None:
    """The dimension vec_embedding was actually created with, or None."""
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'vec_embedding'").fetchone()
    if not row or not row[0]:
        return None
    found = re.search(r"float\s*\[\s*(\d+)\s*\]", row[0])
    return int(found.group(1)) if found else None


def recreate_vec_table(conn: sqlite3.Connection, dim: int) -> None:
    """Rebuild vec_embedding at a new dimension.

    vec0 fixes dimensionality at CREATE time, and connect() uses IF NOT
    EXISTS -- so changing embed_dim otherwise leaves the old table in place
    and every insert fails with "Dimension mismatch", including the
    `hunch reindex --embeddings` run meant to recover. Dropping loses
    nothing: vectors of the wrong dimension are unusable by definition, and
    reindex regenerates all of them from the text it kept.
    """
    conn.execute("DROP TABLE IF EXISTS vec_embedding")
    conn.execute(
        f"CREATE VIRTUAL TABLE vec_embedding USING vec0(embedding float[{dim}])")
    conn.commit()


def embedding_model_matches(conn: sqlite3.Connection, model: str, dim: int) -> bool:
    """Guard against mixing vector spaces.

    Vectors from different models are not comparable even at equal dimension,
    so a mismatch must surface as a prompt to reindex rather than as silently
    meaningless search results.
    """
    stored_model = get_meta(conn, "embed_model")
    stored_dim = get_meta(conn, "embed_dim")
    if stored_model is None or stored_dim is None:
        return True     # fresh index; caller will stamp it
    if stored_model == model and int(stored_dim) == dim:
        return True
    # The stamp is written at the start of a run, before any file is
    # embedded, so a run that dies early claims a vector space it never
    # wrote into. Seen for real: the first index picked an embedder too
    # large for the GPU and failed every file, then fixing the model left
    # the index permanently wedged -- refusing to run and demanding
    # `hunch reindex --embeddings` to rebuild zero vectors from zero stored
    # text. With nothing embedded there is no vector space to protect and
    # nothing to mix, so any model is free to claim it.
    return conn.execute("SELECT NOT EXISTS(SELECT 1 FROM file_embedding)").fetchone()[0] == 1
