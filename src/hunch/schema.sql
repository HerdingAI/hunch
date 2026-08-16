PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS file_catalog (
    id            INTEGER PRIMARY KEY,
    path          TEXT NOT NULL UNIQUE,
    filename      TEXT NOT NULL,
    ext           TEXT,
    size_bytes    INTEGER,
    mtime         REAL,
    content_hash  TEXT,
    status        TEXT NOT NULL DEFAULT 'pending',
    error_reason  TEXT,
    retry_count   INTEGER NOT NULL DEFAULT 0,
    last_attempt  REAL,
    first_seen    REAL NOT NULL DEFAULT (unixepoch()),
    last_seen     REAL NOT NULL DEFAULT (unixepoch()),
    deleted_at    REAL,
    CHECK (status IN ('pending','done','failed','unsupported','skipped'))
);

CREATE INDEX IF NOT EXISTS idx_catalog_status   ON file_catalog(status) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_catalog_hash     ON file_catalog(content_hash);
CREATE INDEX IF NOT EXISTS idx_catalog_ext      ON file_catalog(ext);
CREATE INDEX IF NOT EXISTS idx_catalog_filename ON file_catalog(filename);

-- Keyed by content_hash, not file id: identical bytes are enriched once and
-- every catalog row sharing that hash reuses the result.
CREATE TABLE IF NOT EXISTS file_embedding (
    content_hash   TEXT PRIMARY KEY,
    extracted_text TEXT,
    source_kind    TEXT,
    model          TEXT,
    char_count     INTEGER,
    created_at     REAL NOT NULL DEFAULT (unixepoch())
);

CREATE TABLE IF NOT EXISTS enrich_timing (
    id           INTEGER PRIMARY KEY,
    source_kind  TEXT,
    stage        TEXT,
    seconds      REAL,
    bytes        INTEGER,
    at           REAL NOT NULL DEFAULT (unixepoch())
);
CREATE INDEX IF NOT EXISTS idx_timing_kind_stage ON enrich_timing(source_kind, stage);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
