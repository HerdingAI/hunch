"""Measure the machine, then say plainly what the user gets.

Graded per capability, never pass/fail: text embeddings run on a CPU, so every
machine gets working semantic document search. The GPU check gates images and
audio alone.
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

MIN_VRAM_MB = 4096
# Module-level so tests can point it at a fake sysfs tree via monkeypatch
# instead of depending on this machine's actual power state.
_POWER_SUPPLY_DIR = Path("/sys/class/power_supply")


@dataclass
class Capabilities:
    has_gpu: bool
    vram_mb: int
    ram_mb: int
    free_disk_mb: int
    cpu_count: int
    has_tesseract: bool
    has_poppler: bool
    has_ffmpeg: bool

    def verdict(self) -> dict:
        """Graded per capability -- never a single pass/fail.

        image_text (OCR + EXIF + offline geocoding) needs only tesseract and
        costs ~0.2s/image, so it must never be bundled with the GPU check.
        Only photo_descriptions (vision captioning, ~5s/image) is genuinely
        impractical without a GPU or OpenRouter. Conflating the two would
        tell a CPU-only user their photos are unsupported, when phase 3
        already makes every photo findable by date, place and any text on it.
        """
        documents = self.has_poppler                       # CPU-only, always on
        image_text = self.has_tesseract                     # CPU-only, always on
        gpu_ok = self.has_gpu and self.vram_mb >= MIN_VRAM_MB
        photo_descriptions = gpu_ok                          # expensive without a GPU
        transcription = self.has_ffmpeg                      # works on CPU, slower

        if documents and image_text and photo_descriptions and transcription:
            summary = ("You are running fully local. Documents, images and audio "
                       "are all understood on this machine. Nothing is uploaded "
                       "and there is nothing to pay for.")
        elif documents:
            summary = ("Documents and the text in your images will work well on "
                       "this machine — they run on the CPU. Describing photos in "
                       "detail needs a GPU with 4 GB or more, or an OpenRouter key; "
                       "audio and video transcription will work but may be slow "
                       "without a GPU.")
        else:
            summary = ("Install poppler-utils and tesseract-ocr to read documents: "
                       "sudo apt install poppler-utils tesseract-ocr")
        return {"documents": documents, "image_text": image_text,
                "photo_descriptions": photo_descriptions,
                "transcription": transcription, "summary": summary}


def _vram_mb() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=False)
        if out.returncode == 0 and out.stdout.strip():
            return int(out.stdout.strip().splitlines()[0])
    except Exception:                              # noqa: BLE001
        pass
    return 0


def _ram_mb() -> int:
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return 0


def probe() -> Capabilities:
    vram = _vram_mb()
    usage = shutil.disk_usage(Path.home())
    return Capabilities(
        has_gpu=vram > 0,
        vram_mb=vram,
        ram_mb=_ram_mb(),
        free_disk_mb=usage.free // (1024 * 1024),
        cpu_count=os.cpu_count() or 1,
        has_tesseract=shutil.which("tesseract") is not None,
        has_poppler=shutil.which("pdftotext") is not None,
        has_ffmpeg=shutil.which("ffmpeg") is not None,
    )


def on_ac_power() -> bool:
    """True if plugged in, or if this machine has no battery at all.

    A desktop with no battery is always "on AC" for budget purposes.
    cmd_index (cli.py) uses this to defer the expensive enrichment pass
    on a scheduled run while still letting the cheap catalog pass go
    ahead -- gating via a systemd ConditionACPower= on hunch-index.service
    would have blocked both together, since one service invocation runs
    both passes.
    """
    if not _POWER_SUPPLY_DIR.is_dir():
        return True
    found_mains = False
    for supply in _POWER_SUPPLY_DIR.iterdir():
        try:
            kind = (supply / "type").read_text().strip()
        except OSError:
            continue
        if kind != "Mains":
            continue
        found_mains = True
        try:
            if (supply / "online").read_text().strip() == "1":
                return True
        except OSError:
            continue
    return not found_mains


def local_backend_importable() -> bool:
    """Whether the local_inprocess backend's embedding dependency (the
    `local` extra) is actually installed -- distinct from has_gpu/
    has_tesseract/etc above, which probe system binaries, not Python
    packages. `pipx install hunch-search` with no extras passes every
    hardware check and then silently fails every single file at embed
    time (worker.py's broad except around backend.embed()), leaving only
    an error_reason buried in a DB column no one looks at.

    find_spec, not a real import: sentence_transformers pulls in torch,
    which costs real seconds to load -- a presence check must not pay
    that just to answer "is this installed".
    """
    return importlib.util.find_spec("sentence_transformers") is not None


def media_importable() -> bool:
    """Whether faster-whisper (the `media` extra) is installed. Needed by
    both the local_inprocess and ollama backends (ollama.py: "Ollama does
    not serve speech-to-text, so transcription falls back to
    faster-whisper") -- only the openrouter backend transcribes without
    it. Same silent-failure shape as local_backend_importable(), scoped to
    audio/video files instead of every file.
    """
    return importlib.util.find_spec("faster_whisper") is not None
