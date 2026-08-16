"""The seam that makes local-vs-cloud a configuration choice.

Only `embed` produces vectors. describe_image and transcribe produce text,
which is then embedded -- which is why the embedding model must match across
backends but the vision and transcription models need not.
"""
from __future__ import annotations

import abc


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
