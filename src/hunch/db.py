"""SQLite storage. One file, no server, no CREATE EXTENSION, no superuser."""
from __future__ import annotations

import os
import sqlite3
import struct
from pathlib import Path

import sqlite_vec

from . import config

SCHEMA_VERSION = 1


def serialize(vector) -> bytes:
    """Pack a float sequence into sqlite-vec's expected binary layout."""
    return struct.pack(f"{len(vector)}f", *vector)


def connect(path: Path | None = None, dim: int | None = None) -> sqlite3.Connection:
    path = Path(path) if path else config.db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    # The index aggregates extracted text from every indexed file into one
    # place, so it must not inherit a permissive umask (0o644 would let any
    # other local account on the machine read it).
    is_new = not path.exists()
    conn = sqlite3.connect(path, timeout=30.0)
    if is_new:
        os.chmod(path, 0o600)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    # WAL lets the GUI read while the worker writes; without it a long
    # enrichment transaction would block every search.
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA foreign_keys = ON")

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
    return stored_model == model and int(stored_dim) == dim
