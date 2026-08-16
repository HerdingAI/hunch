"""OpenRouter: all three modalities from one key and one base URL."""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import urllib.request

from ..config import Config
from .base import Backend

BASE = "https://openrouter.ai/api/v1"


def _keyring_get() -> str | None:
    try:
        import keyring
        return keyring.get_password("hunch", "openrouter")
    except Exception:                              # noqa: BLE001
        return None


def store_api_key(key: str) -> None:
    """The write side of api_key()'s keyring read -- without this, the env
    var documented as a fallback was the only path that actually worked."""
    import keyring
    keyring.set_password("hunch", "openrouter", key)


def _post_json(url: str, payload: dict, key: str, timeout: int = 120) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}",
                 "X-Title": "Hunch"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


class OpenRouterBackend(Backend):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model_id = cfg.openrouter_embed_model
        self.dim = cfg.embed_dim

    def api_key(self) -> str:
        key = os.environ.get("HUNCH_OPENROUTER_KEY") or _keyring_get()
        if not key:
            raise RuntimeError(
                "no OpenRouter API key. Run `hunch auth openrouter` or set "
                "HUNCH_OPENROUTER_KEY.")
        return key

    def embed(self, texts: list[str]) -> list[list[float]]:
        data = _post_json(f"{BASE}/embeddings",
                          {"model": self.cfg.openrouter_embed_model,
                           "input": [t[:8000] for t in texts]},
                          self.api_key())
        return [row["embedding"] for row in data["data"]]

    def describe_image(self, path: str) -> tuple[str, str]:
        try:
            mime = mimetypes.guess_type(path)[0] or "image/jpeg"
            with open(path, "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode()
            data = _post_json(
                f"{BASE}/chat/completions",
                {"model": self.cfg.openrouter_vision_model,
                 "messages": [{"role": "user", "content": [
                     {"type": "text", "text":
                      "Describe this image, including any visible text, "
                      "objects, people, setting and document type."},
                     {"type": "image_url",
                      "image_url": {"url": f"data:{mime};base64,{b64}"}}]}]},
                self.api_key())
            return data["choices"][0]["message"]["content"].strip(), ""
        except Exception as exc:                   # noqa: BLE001
            return "", f"caption failed: {exc}"

    def transcribe(self, path: str) -> tuple[str, str]:
        try:
            with open(path, "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode()
            data = _post_json(f"{BASE}/audio/transcriptions",
                              {"model": self.cfg.openrouter_transcribe_model,
                               "file": b64},
                              self.api_key(), timeout=900)
            return (data.get("text") or "").strip(), ""
        except Exception as exc:                   # noqa: BLE001
            return "", f"transcribe failed: {exc}"
