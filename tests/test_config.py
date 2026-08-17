from pathlib import Path
import hunch.config as c


def test_defaults_use_xdg_user_dirs():
    cfg = c.load_config(Path("/nonexistent/config.toml"))
    names = {p.name for p in cfg.folders}
    assert names == {"Documents", "Downloads", "Desktop", "Pictures"}
    assert cfg.backend == "local_inprocess"
    assert cfg.first_run_budget_seconds == 5 * 3600
    assert cfg.daily_budget_seconds == 20 * 60


def test_home_itself_is_never_a_default_folder():
    # Indexing $HOME would drag in .cache, .local and every dependency tree.
    cfg = c.load_config(Path("/nonexistent/config.toml"))
    assert Path.home() not in cfg.folders


def test_roundtrip_preserves_values(tmp_path):
    cfg = c.load_config(Path("/nonexistent/config.toml"))
    cfg.backend = "openrouter"
    cfg.folders = [tmp_path / "a", tmp_path / "b"]
    dest = tmp_path / "config.toml"
    c.save_config(cfg, dest)
    again = c.load_config(dest)
    assert again.backend == "openrouter"
    assert again.folders == [tmp_path / "a", tmp_path / "b"]


def test_unknown_keys_in_an_existing_config_are_ignored(tmp_path):
    # save_config writes every field, so a config written by an older
    # version can name keys this one no longer has -- image_caption_above
    # was exactly that: a knob nothing ever read, removed rather than
    # left lying to users. Upgrading past it must not break the install.
    path = tmp_path / "config.toml"
    path.write_text('backend = "local_inprocess"\n'
                    'image_caption_above = 512000\n'
                    'some_future_key = "whatever"\n')
    cfg = c.load_config(path)
    assert cfg.backend == "local_inprocess"
    assert not hasattr(cfg, "image_caption_above")


def test_classify_routes_extensions():
    assert c.classify("pdf") == "document"
    assert c.classify("eml") == "document"
    assert c.classify("jpg") == "image"
    assert c.classify("mp3") == "audio"
    assert c.classify("mkv") == "video"
    assert c.classify("xyz") == "unsupported"


def test_exclusions_cover_dependency_and_junk_patterns():
    cfg = c.load_config(Path("/nonexistent/config.toml"))
    for d in ("node_modules", ".venv", "site-packages", ".cargo", "__pycache__",
              ".git", ".vscode-server"):
        assert d in cfg.exclude_dirs
    assert cfg.exclude_dir_suffixes == (".dist-info", ".egg-info")
    assert cfg.exclude_file_prefixes == ("._", "~$", ".~lock.")
