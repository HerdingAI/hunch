"""OpenRouter: all three modalities from one key and one base URL."""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import shutil
import subprocess
import tempfile
import urllib.request

from ..config import Config
from ..extract import FFMPEG_TIMEOUT
from .base import Backend

BASE = "https://openrouter.ai/api/v1"
# Containers OpenRouter's /audio/transcriptions accepts directly. Video
# containers are absent by design -- they must be demuxed to audio first.
AUDIO_FORMATS = {"wav", "mp3", "flac", "m4a", "ogg", "webm", "aac"}


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
        """Transcribe via OpenRouter's /audio/transcriptions endpoint.

        Two things this has to get right that the obvious implementation
        does not. The wire format is OpenRouter's own, not OpenAI's: a JSON
        body carrying `input_audio: {data, format}`, where OpenAI would
        take a multipart file upload. A body of `{"file": <base64>}` --
        which is neither -- is silently rejected, so every audio and video
        file fails with the backend set to openrouter.

        And the bytes have to be audio. worker.py routes both audio and
        video here, but the endpoint accepts audio containers only (wav,
        mp3, flac, m4a, ogg, webm, aac), so an mp4 or mkv would be refused
        even with the field names right. faster-whisper hides this for the
        local backend by decoding video itself; over HTTP the decoding has
        to happen on this side. Transcoding everything to mono 16 kHz mp3
        handles video and audio through one path, guarantees `format`
        matches the payload, and shrinks the base64 body by an order of
        magnitude -- which matters, because the transfer counts against
        the upstream provider's 60-second processing timeout.
        """
        audio_b64, fmt, err = self._audio_payload(path)
        if err:
            return "", err
        try:
            data = _post_json(f"{BASE}/audio/transcriptions",
                              {"model": self.cfg.openrouter_transcribe_model,
                               "input_audio": {"data": audio_b64, "format": fmt}},
                              self.api_key(), timeout=900)
            return (data.get("text") or "").strip(), ""
        except Exception as exc:                   # noqa: BLE001
            return "", f"transcribe failed: {exc}"

    def _audio_payload(self, path: str) -> tuple[str, str, str]:
        """(base64, format, error). Prefers ffmpeg; falls back to raw bytes."""
        if shutil.which("ffmpeg"):
            with tempfile.TemporaryDirectory() as tmp:
                out = os.path.join(tmp, "audio.mp3")
                proc = subprocess.run(
                    ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", path,
                     "-vn", "-ac", "1", "-ar", "16000", "-b:a", "32k", out],
                    capture_output=True, text=True, timeout=FFMPEG_TIMEOUT,
                    check=False)
                if proc.returncode == 0 and os.path.exists(out):
                    with open(out, "rb") as fh:
                        return base64.b64encode(fh.read()).decode(), "mp3", ""
                return "", "", f"transcribe failed: ffmpeg: {proc.stderr.strip()[:200]}"
        # No ffmpeg: only already-supported audio containers can be sent as-is.
        # Video without ffmpeg is genuinely unsupported here -- saying so beats
        # uploading megabytes the endpoint will reject.
        ext = os.path.splitext(path)[1].lstrip(".").lower()
        if ext not in AUDIO_FORMATS:
            return "", "", (f"transcribe failed: {ext or 'this file'} needs ffmpeg "
                            f"to extract audio (sudo apt install ffmpeg)")
        with open(path, "rb") as fh:
            return base64.b64encode(fh.read()).decode(), ext, ""
