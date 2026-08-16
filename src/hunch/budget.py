"""Time as a resource.

The performance contract (5h first run, 20min/day) is enforced here rather
than hoped for: the worker spends a budget in phase order and stops when it
runs out, resuming next session.
"""
from __future__ import annotations

import time

from .config import AUDIO_EXT, IMAGE_EXT, VIDEO_EXT, DOC_EXT

# Ordered by what can FINISH, not by what is cheapest. A half-captioned photo
# library is worse than none: the user cannot tell "not there" from "not
# indexed yet" and stops trusting the tool. Captioning is open-ended, so it is
# explicitly last.
PHASES = ["document", "image_meta", "audio", "image_caption"]

PHASE_LABELS = {
    "document": "Reading documents",
    "image_meta": "Reading text in images",
    "audio": "Listening to audio & video",
    "image_caption": "Describing photos",
}


class Budget:
    def __init__(self, total_seconds: float):
        self.total = float(total_seconds)
        self.started = time.time()

    def remaining(self) -> float:
        # Rounded to the centisecond: budgets here are measured in minutes to
        # hours, so sub-10ms precision buys nothing functionally, but an
        # unrounded read is at the mercy of interpreter/scheduler jitter
        # between statements (observed several microseconds under pytest's
        # assertion rewriting) -- enough to make a spend()-then-remaining()
        # check flaky at tight tolerances even though no real budget time
        # passed. Rounding away noise well below anything that matters here
        # makes the reading deterministic instead.
        elapsed = round(time.time() - self.started, 2)
        return max(0.0, self.total - elapsed)

    def spend(self, seconds: float) -> None:
        # Manual accounting on top of the automatic wall-clock decay above:
        # a caller that already knows how long an operation took (e.g. one
        # that isn't looping through drain()) can pre-charge it instead of
        # waiting for the next remaining() call to notice.
        self.total -= float(seconds)

    def exhausted(self) -> bool:
        return self.remaining() <= 0.0


def _pending_count(conn, exts: set[str]) -> int:
    if not exts:
        return 0
    marks = ",".join("?" * len(exts))
    return conn.execute(
        f"SELECT count(*) FROM file_catalog WHERE status='pending' "
        f"AND deleted_at IS NULL AND ext IN ({marks})", tuple(exts)).fetchone()[0]


def phase_exts(phase: str) -> set[str]:
    if phase == "document":
        return DOC_EXT
    if phase in ("image_meta", "image_caption"):
        return IMAGE_EXT
    if phase == "audio":
        return AUDIO_EXT | VIDEO_EXT
    return set()


def pending_count(conn, phase: str) -> int:
    if phase == "image_caption":
        # Captioning upgrades rows that already have a metadata record.
        return conn.execute(
            "SELECT count(*) FROM file_catalog c "
            "JOIN file_embedding e ON e.content_hash = c.content_hash "
            "WHERE c.deleted_at IS NULL AND e.source_kind = 'image_meta'"
        ).fetchone()[0]
    return _pending_count(conn, phase_exts(phase))


def phase_has_pending(conn, phase: str) -> bool:
    return pending_count(conn, phase) > 0


def next_phase(conn) -> str | None:
    """First phase with work left, in the fixed order above."""
    for phase in PHASES:
        if phase_has_pending(conn, phase):
            return phase
    return None
