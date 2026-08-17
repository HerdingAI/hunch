"""The seam that makes local-vs-cloud a configuration choice.

Only `embed` produces vectors. describe_image and transcribe produce text,
which is then embedded -- which is why the embedding model must match across
backends but the vision and transcription models need not.
"""
from __future__ import annotations

import abc


def check_dim(vectors: list[list[float]], expected: int, model_id: str) -> None:
    """Fail loudly when a remote model's real width isn't the configured one.

    local_inprocess passes truncate_dim to the model, so its vectors are the
    configured width by construction. A remote API just returns whatever its
    model produces: config says 1024, the model returns 2560, and the
    mismatch only surfaces several layers down as sqlite-vec's "Dimension
    mismatch for inserted vector" -- once per file, from a stack that names
    neither the backend, the model, nor the fix. Raising here turns that
    into one actionable sentence.
    """
    if not vectors:
        return
    actual = len(vectors[0])
    if actual != expected:
        raise RuntimeError(
            f"{model_id} returns {actual}-dimensional vectors but this index "
            f"expects {expected}. Set embed_dim = {actual} in the config file "
            f"(hunch doctor prints its path) and run "
            f"`hunch reindex --embeddings`.")


class Backend(abc.ABC):
    #: Identifies the vector space. Stored in the index and compared on open.
    model_id: str = ""
    dim: int = 0

    @abc.abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch. Batching is the main throughput lever -- callers
        should pass many texts, not one."""

    @abc.abstractmethod
    def describe_image(self, path: str) -> tuple[str, str]:
        """Return (description, error)."""

    @abc.abstractmethod
    def transcribe(self, path: str) -> tuple[str, str]:
        """Return (transcript, error)."""

    def release(self) -> None:
        """Free model memory. Called when idle; safe to override or ignore."""
        return None
