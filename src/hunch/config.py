"""Configuration, XDG paths, and file-type routing."""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, asdict
from pathlib import Path

import tomli_w

APP = "hunch"


def config_path() -> Path:
    base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / APP / "config.toml"


def data_dir() -> Path:
    base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / APP


def cache_dir() -> Path:
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / APP


def db_path() -> Path:
    return data_dir() / "index.db"


# --- file-type routing ----------------------------------------------------
# Documents a person wrote or reads. Machine formats are deliberately absent:
# json, xml, yaml/yml and log are, in practice, config and data dumps rather
# than anything anyone searches for by meaning, and they arrive in numbers no
# document format does. Measured on one real machine: 150,432 .json against
# 3,017 .md and 1,668 .txt -- indexing them spent the entire first-run budget
# embedding scraped API payloads while the user's actual documents waited
# behind them, and search returned `{"jobs": []}` for every query. Add them
# back per-machine via `doc_ext` in config.toml if yours are meaningful.
DOC_EXT = {
    "pdf", "docx", "doc", "odt", "rtf", "txt", "md", "markdown", "epub",
    "pptx", "ppt", "xlsx", "xls", "csv", "tsv", "html", "htm",
    "tex", "org", "eml", "msg", "mbox",
}
# Enriched first: these carry the content people actually search for.
RICH_DOC_EXT = {
    "pdf", "docx", "doc", "odt", "rtf", "epub", "pptx", "ppt",
    "xlsx", "xls", "eml", "msg", "mbox", "md", "markdown", "tex",
}
IMAGE_EXT = {"jpg", "jpeg", "png", "gif", "bmp", "tif", "tiff", "webp", "heic", "heif"}
AUDIO_EXT = {"mp3", "wav", "flac", "m4a", "aac", "ogg", "opus", "wma"}
VIDEO_EXT = {"mp4", "mkv", "avi", "mov", "wmv", "webm", "flv", "m4v", "mpg", "mpeg"}


def classify(ext: str, cfg: "Config | None" = None) -> str:
    """Route an extension to a phase, honouring per-machine overrides.

    Which extensions are worth indexing is genuinely machine-specific -- a
    writer's folder and a developer's differ by orders of magnitude in what
    they contain -- so the defaults above are a starting point, not a
    verdict. cfg is optional so existing callers and tests keep working
    against the defaults.
    """
    ext = (ext or "").lower().lstrip(".")
    docs = cfg.doc_ext if cfg else DOC_EXT
    images = cfg.image_ext if cfg else IMAGE_EXT
    audio = cfg.audio_ext if cfg else AUDIO_EXT
    video = cfg.video_ext if cfg else VIDEO_EXT
    if ext in docs:
        return "document"
    if ext in images:
        return "image"
    if ext in audio:
        return "audio"
    if ext in video:
        return "video"
    return "unsupported"


# Pruned during the walk. Not housekeeping: on a real corpus, excluding these
# made the crawl 25x faster and stopped dependency files crowding real
# documents out of results.
DEFAULT_EXCLUDE_DIRS = {
    # os / filesystem
    "System Volume Information", "$RECYCLE.BIN", "$Recycle.Bin", "lost+found",
    ".Trash", ".Trash-1000", ".zfs", ".fseventsd", ".Spotlight-V100",
    "Windows", "Program Files", "Program Files (x86)", "ProgramData", "AppData",
    # vcs / build caches
    ".git", ".svn", ".hg", "__pycache__", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", ".tox", ".ipynb_checkpoints", ".terraform",
    # language / package-manager caches
    "node_modules", ".npm", ".bun", ".yarn", ".pnpm-store", ".deno", ".nvm",
    ".cargo", ".rustup", ".gradle", ".m2", ".nuget", ".gem", ".pub-cache",
    ".pyenv", ".rbenv", "site-packages", "dist-packages", "venv", ".venv",
    "virtualenvs", "anaconda3", "miniconda3", ".conda", ".cache", "Cache", "Caches",
    # editor / agent server trees
    ".vscode-server", ".vscode", ".cursor-server", ".cursor", ".idea",
    ".jupyter", ".ipython",
    # app runtimes
    "snap", ".steam", ".wine", ".docker", ".ollama", ".local-cache",
}
DEFAULT_EXCLUDE_DIR_SUFFIXES = (".dist-info", ".egg-info")
# "._" are macOS AppleDouble forks; "~$" are Office lock stubs. Both shadow a
# real document with the same extension and always fail extraction.
DEFAULT_EXCLUDE_FILE_PREFIXES = ("._", "~$", ".~lock.")
DEFAULT_EXCLUDE_FILENAMES = {".DS_Store", "Thumbs.db", "desktop.ini", ".localized"}


def _default_folders() -> list[Path]:
    home = Path.home()
    wanted = ["Documents", "Downloads", "Desktop", "Pictures"]
    return [home / n for n in wanted]


@dataclass
class Config:
    folders: list[Path] = field(default_factory=_default_folders)
    backend: str = "local_inprocess"
    embed_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embed_dim: int = 1024
    vision_model: str = "microsoft/Florence-2-large"
    whisper_model: str = "base"
    openrouter_embed_model: str = "qwen/qwen3-embedding-4b"
    openrouter_vision_model: str = "google/gemma-3-4b-it"
    openrouter_transcribe_model: str = "openai/whisper-large-v3"
    first_run_budget_seconds: int = 5 * 3600
    daily_budget_seconds: int = 20 * 60
    image_skip_below: int = 20 * 1024
    # A file that's missing at enrichment time might be genuinely deleted, or
    # might just be a transient permission error or a stale network handle --
    # they look identical from here. Deletion is catalog.crawl()'s call
    # alone, made from an authoritative directory listing; this only bounds
    # how many worker attempts a persistently-unreadable file gets before it
    # stops being retried on every run.
    max_enrich_retries: int = 3
    ocr_languages: str = "eng+spa"
    # Editable per machine: see classify() and DOC_EXT's note on why the
    # defaults exclude json/xml/yaml/log.
    doc_ext: set[str] = field(default_factory=lambda: set(DOC_EXT))
    image_ext: set[str] = field(default_factory=lambda: set(IMAGE_EXT))
    audio_ext: set[str] = field(default_factory=lambda: set(AUDIO_EXT))
    video_ext: set[str] = field(default_factory=lambda: set(VIDEO_EXT))
    min_size_bytes: int = 16
    exclude_dirs: set[str] = field(default_factory=lambda: set(DEFAULT_EXCLUDE_DIRS))
    exclude_dir_suffixes: tuple[str, ...] = DEFAULT_EXCLUDE_DIR_SUFFIXES
    exclude_file_prefixes: tuple[str, ...] = DEFAULT_EXCLUDE_FILE_PREFIXES
    exclude_filenames: set[str] = field(default_factory=lambda: set(DEFAULT_EXCLUDE_FILENAMES))


def load_config(path: Path | None = None) -> Config:
    path = path or config_path()
    cfg = Config()
    if not path.exists():
        return cfg
    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    for key, value in raw.items():
        if not hasattr(cfg, key):
            continue           # forward-compatible: ignore unknown keys
        if key == "folders":
            setattr(cfg, key, [Path(p).expanduser() for p in value])
        elif key in ("exclude_dirs", "exclude_filenames", "doc_ext",
                     "image_ext", "audio_ext", "video_ext"):
            setattr(cfg, key, set(value))
        elif key in ("exclude_dir_suffixes", "exclude_file_prefixes"):
            setattr(cfg, key, tuple(value))
        else:
            setattr(cfg, key, value)
    return cfg


def save_config(cfg: Config, path: Path | None = None) -> None:
    path = path or config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = asdict(cfg)
    raw["folders"] = [str(p) for p in cfg.folders]
    raw["exclude_dirs"] = sorted(cfg.exclude_dirs)
    raw["exclude_filenames"] = sorted(cfg.exclude_filenames)
    for key in ("doc_ext", "image_ext", "audio_ext", "video_ext"):
        raw[key] = sorted(getattr(cfg, key))
    raw["exclude_dir_suffixes"] = list(cfg.exclude_dir_suffixes)
    raw["exclude_file_prefixes"] = list(cfg.exclude_file_prefixes)
    with path.open("wb") as fh:
        tomli_w.dump(raw, fh)
