"""Optional backend for people already running Ollama.

Never a prerequisite -- the local tier runs in-process. Ollama does not serve
speech-to-text, so transcription falls back to faster-whisper.
"""
from __future__ import annotations

import base64
import json
import urllib.request

from ..config import Config
from .base import Backend, check_dim


def _post(url: str, payload: dict, timeout: int = 300) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


class OllamaBackend(Backend):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.url = "http://127.0.0.1:11434"
        self.model_id = "qwen3-embedding"
        self.dim = cfg.embed_dim
        self._local = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        data = _post(f"{self.url}/api/embed",
                     {"model": self.model_id, "input": [t[:8000] for t in texts]})
        vectors = data["embeddings"]
        check_dim(vectors, self.dim, self.model_id)
        return vectors

    def describe_image(self, path: str) -> tuple[str, str]:
        try:
            with open(path, "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode()
            data = _post(f"{self.url}/api/generate",
                         {"model": "gemma3:4b",
                          "prompt": "Describe this image, including any visible text.",
                          "images": [b64], "stream": False}, timeout=600)
            return (data.get("response") or "").strip(), ""
        except Exception as exc:                   # noqa: BLE001
            return "", f"caption failed: {exc}"

    def transcribe(self, path: str) -> tuple[str, str]:
        if self._local is None:
            from .local_inprocess import LocalBackend
            self._local = LocalBackend(self.cfg)
        return self._local.transcribe(path)
