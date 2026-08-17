"""Everything in one Python process: no daemon, no server, no Ollama.

Models load lazily and per stage. A 4 GB card cannot hold the embedder, the
vision model and Whisper at once, so each is loaded when first needed and can
be released when idle.
"""
from __future__ import annotations

import concurrent.futures
import os

from ..config import Config
from .base import Backend

EMBED_CHARS = 8000          # keep inside the embedding model's context
# Matches extract.FFMPEG_TIMEOUT: long recordings on CPU fallback are
# legitimately slow, but a malformed container can also hang faster-whisper
# indefinitely, so transcription needs the same bounded-wait treatment every
# other risky format gets.
TRANSCRIBE_TIMEOUT = 900


def _device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:                              # noqa: BLE001
        return "cpu"


class LocalBackend(Backend):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model_id = cfg.embed_model
        self.dim = cfg.embed_dim
        self.device = _device()
        # Tracked separately from self.device: ctranslate2 (faster-whisper's
        # backend) links its own CUDA stack independent of torch's, so a
        # library mismatch there (see transcribe()) must not force the
        # embedder/vision model -- which use torch's CUDA and are unaffected
        # -- onto CPU too.
        self._whisper_device = self.device
        # Tracked separately for the same reason, but a different cause: the
        # background indexer and an interactive search are two processes
        # sharing one card. The indexer legitimately holds most of it, so a
        # search launched mid-index can find no room -- see embed().
        self._embed_device = self.device
        self._embedder = None
        self._vision = None
        self._vision_proc = None
        self._whisper = None

    # --- embeddings -------------------------------------------------------
    def _load_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(
                self.cfg.embed_model, device=self._embed_device,
                truncate_dim=self.cfg.embed_dim)
        return self._embedder

    def _encode(self, model, clipped):
        vecs = model.encode(clipped, batch_size=16, normalize_embeddings=True,
                            show_progress_bar=False)
        return [list(map(float, v)) for v in vecs]

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch, falling back to CPU if the GPU has no room.

        Two processes share one card by design: the background indexer runs
        for hours while the user searches, which the README promises works.
        On a real 8 GB card the indexer settled at 5.7 GB, leaving 21 MB --
        so a search launched mid-index could not load the model at all. It
        did not crash: search.py catches embed() failures and degrades to
        literal matching, so the user simply got "no matches" for files that
        were sitting right there, with nothing to suggest the semantic half
        had never run.

        Falling back is cheap where it matters. Embedding one query on CPU
        measured 0.07 s (against a 4.9 s one-time model load) -- fine for a
        search, and the indexer's throughput is unaffected because it is the
        process that still holds the GPU. Mirrors transcribe()'s existing
        CPU-retry, including staying on CPU afterward rather than re-paying
        a load we now know fails.
        """
        clipped = [(t or "")[:EMBED_CHARS] for t in texts]
        try:
            return self._encode(self._load_embedder(), clipped)
        except Exception:                          # noqa: BLE001
            if self._embed_device == "cpu":
                raise
            self._embed_device = "cpu"
            self._embedder = None
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:                      # noqa: BLE001
                pass
            return self._encode(self._load_embedder(), clipped)

    # --- vision -----------------------------------------------------------
    def _load_vision(self):
        if self._vision is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self._vision = AutoModelForCausalLM.from_pretrained(
                self.cfg.vision_model, torch_dtype=dtype,
                trust_remote_code=True).to(self.device).eval()
            self._vision_proc = AutoProcessor.from_pretrained(
                self.cfg.vision_model, trust_remote_code=True)
        return self._vision, self._vision_proc

    def describe_image(self, path: str) -> tuple[str, str]:
        try:
            from PIL import Image
            model, proc = self._load_vision()
            image = Image.open(path).convert("RGB")
            # Florence-2 is a task model: ask for a caption, get a caption --
            # no conversational prose to strip.
            task = "<DETAILED_CAPTION>"
            inputs = proc(text=task, images=image, return_tensors="pt").to(self.device)
            if self.device == "cuda":
                inputs = {k: (v.half() if v.dtype.is_floating_point else v)
                          for k, v in inputs.items()}
            out = model.generate(**inputs, max_new_tokens=128, num_beams=1,
                                 do_sample=False)
            text = proc.batch_decode(out, skip_special_tokens=True)[0]
            parsed = proc.post_process_generation(
                text, task=task, image_size=image.size)
            return str(parsed.get(task, text)).strip(), ""
        except Exception as exc:                   # noqa: BLE001
            return "", f"caption failed: {exc}"

    # --- audio ------------------------------------------------------------
    def _load_whisper(self):
        if self._whisper is None:
            from faster_whisper import WhisperModel
            try:
                self._whisper = WhisperModel(self.cfg.whisper_model,
                                             device=self._whisper_device, compute_type="int8")
            except Exception:                      # noqa: BLE001
                # CUDA runtime libs missing is common; degraded beats broken.
                # (Construction alone rarely raises this -- see transcribe()
                # for the more common case where it surfaces later.)
                self._whisper_device = "cpu"
                self._whisper = WhisperModel(self.cfg.whisper_model,
                                             device="cpu", compute_type="int8")
        return self._whisper

    def _transcribe_once(self, model, path: str) -> tuple[str, str]:
        def _run():
            # Transcribe an opening slice rather than the whole recording.
            # The embedder reads the first EMBED_CHARS characters and speech
            # runs about 1,000 characters per minute, so roughly the first
            # eight minutes is all that can reach a vector -- past that the
            # audio is decoded at full cost and then truncated away. Measured
            # on a real library: 2,301 files averaging 32 minutes, so about
            # three quarters of every transcription was doing nothing.
            #
            # The tradeoff is real and deliberate: a topic first raised late
            # in a long recording is no longer findable by that topic, only
            # by its opening. Set transcribe_max_seconds to 0 for the whole
            # file if that matters more than the time.
            cap = getattr(self.cfg, "transcribe_max_seconds", 0) or 0
            kwargs = {"beam_size": 1}
            if cap > 0:
                kwargs["clip_timestamps"] = [0.0, float(cap)]
            segments, _info = model.transcribe(path, **kwargs)
            return " ".join(s.text for s in segments).strip()

        # A thread with a deadline can't kill a stuck native call outright,
        # but it unblocks the worker loop so one malformed file doesn't stall
        # the whole budgeted run -- the model stays loaded for the next file
        # either way. shutdown(wait=False) is deliberate: waiting here would
        # block on the very hang this exists to escape.
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = pool.submit(_run)
        try:
            text = future.result(timeout=TRANSCRIBE_TIMEOUT)
            pool.shutdown(wait=False)
            return text, ""
        except concurrent.futures.TimeoutError:
            pool.shutdown(wait=False)
            return "", f"transcribe timeout after {TRANSCRIBE_TIMEOUT}s"
        except Exception as exc:                   # noqa: BLE001
            pool.shutdown(wait=False)
            return "", f"transcribe failed: {exc}"

    def transcribe(self, path: str) -> tuple[str, str]:
        try:
            model = self._load_whisper()
        except Exception as exc:                   # noqa: BLE001
            return "", f"transcribe failed: {exc}"

        text, err = self._transcribe_once(model, path)
        if err.startswith("transcribe failed") and self._whisper_device == "cuda":
            # ctranslate2 links a CUDA stack independent of torch's and
            # defers loading it past model construction to first inference
            # -- unlike torch, which resolves its own bundled CUDA libs
            # eagerly. A missing/mismatched libcublas (a common local-GPU
            # misconfiguration, e.g. a CUDA-13 torch install alongside
            # ctranslate2's CUDA-12 requirement) therefore surfaces here,
            # not in _load_whisper()'s try/except, which never actually
            # triggers for this -- the most common real case. Retry once on
            # CPU rather than failing every file for the rest of the run;
            # sticking to CPU afterward (no reset in release()) avoids
            # repeating a load we now know fails.
            self._whisper_device = "cpu"
            self._whisper = None
            try:
                model = self._load_whisper()
            except Exception as exc:               # noqa: BLE001
                return "", f"transcribe failed: {exc}"
            text, err = self._transcribe_once(model, path)
        return text, err

    def release(self) -> None:
        """Actually give the VRAM back.

        Dropping the references is not enough. Torch models hold reference
        cycles, so the tensors stay alive until a collection runs, and
        empty_cache() then has nothing to reclaim. Measured: 2,296 MiB
        reserved after load, still 2,296 MiB after dropping the refs and
        calling empty_cache(), and 20 MiB once gc.collect() ran first.

        Everything that relies on this was therefore a no-op: the GUI's
        idle release ("so the app does not sit on hundreds of megabytes
        indefinitely") and worker.drain()'s release between stage phases,
        which exists so Whisper and the vision model never occupy a 4 GB
        card at the same time.
        """
        self._embedder = self._vision = self._vision_proc = self._whisper = None
        try:
            import gc

            import torch
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()
        except Exception:                          # noqa: BLE001
            pass
