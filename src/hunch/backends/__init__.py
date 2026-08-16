from __future__ import annotations

from ..config import Config
from .base import Backend


def get_backend(cfg: Config) -> Backend:
    if cfg.backend == "local_inprocess":
        from .local_inprocess import LocalBackend
        return LocalBackend(cfg)
    if cfg.backend == "openrouter":
        from .openrouter import OpenRouterBackend
        return OpenRouterBackend(cfg)
    if cfg.backend == "ollama":
        from .ollama import OllamaBackend
        return OllamaBackend(cfg)
    raise ValueError(f"unknown backend: {cfg.backend!r}")


__all__ = ["Backend", "get_backend"]
