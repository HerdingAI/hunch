import sys
import pytest

from hunch.backends import base, get_backend
from hunch.config import Config


class FakeBackend(base.Backend):
    model_id = "fake"
    dim = 4

    def embed(self, texts):
        return [[float(len(t)), 0.0, 0.0, 0.0] for t in texts]

    def describe_image(self, path):
        return "a fake description", ""

    def transcribe(self, path):
        return "fake transcript", ""


def test_backend_contract_is_enforced():
    with pytest.raises(TypeError):
        base.Backend()          # abstract; cannot instantiate


def test_fake_backend_embeds_in_batch():
    b = FakeBackend()
    vecs = b.embed(["ab", "abcd"])
    assert len(vecs) == 2
    assert vecs[0][0] == 2.0 and vecs[1][0] == 4.0


def test_release_is_optional_and_safe():
    FakeBackend().release()     # base provides a no-op


def test_get_backend_rejects_unknown_name():
    cfg = Config()
    cfg.backend = "nonsense"
    with pytest.raises(ValueError, match="unknown backend"):
        get_backend(cfg)


def test_get_backend_returns_local_by_default(monkeypatch):
    cfg = Config()
    created = {}

    class Stub(FakeBackend):
        def __init__(self, cfg):
            created["yes"] = True

    monkeypatch.setattr("hunch.backends.local_inprocess.LocalBackend", Stub)
    get_backend(cfg)
    assert created["yes"]


def test_local_backend_transcribe_times_out_without_hanging_forever(monkeypatch):
    import time as _time

    import hunch.backends.local_inprocess as mod

    class HangingModel:
        def transcribe(self, path, beam_size=1, **kw):
            _time.sleep(5)
            return [], None

    b = mod.LocalBackend(Config())
    b._whisper = HangingModel()
    monkeypatch.setattr(mod, "TRANSCRIBE_TIMEOUT", 0.1)
    text, err = b.transcribe("irrelevant.wav")
    assert text == ""
    assert "timeout" in err


def test_local_backend_transcribe_falls_back_to_cpu_on_cuda_failure(monkeypatch):
    # ctranslate2 (faster-whisper's backend) links its own CUDA stack
    # independently of torch's, and defers loading it past model
    # construction to first inference -- so a CUDA/cuBLAS mismatch (a real,
    # fairly common misconfiguration) never trips _load_whisper()'s own
    # try/except. It only surfaces here, on the first real transcribe call.
    import hunch.backends.local_inprocess as mod

    class StubModel:
        def __init__(self, device):
            self.device = device

        def transcribe(self, path, beam_size=1, **kw):
            if self.device == "cuda":
                raise RuntimeError(
                    "Library libcublas.so.12 is not found or cannot be loaded")
            return [type("Seg", (), {"text": "fallback ok"})()], None

    monkeypatch.setattr(
        "faster_whisper.WhisperModel",
        lambda model_size, device, compute_type: StubModel(device))

    b = mod.LocalBackend(Config())
    b.device = "cuda"
    b._whisper_device = "cuda"
    text, err = b.transcribe("irrelevant.wav")
    assert text == "fallback ok"
    assert err == ""
    assert b._whisper_device == "cpu"      # stuck on cpu, not retried every call


def test_embed_falls_back_to_cpu_when_the_gpu_is_full(monkeypatch):
    # Regression test from the live system: the background indexer settled
    # at 5.7 GB of an 8 GB card, leaving 21 MB, so a search launched during
    # indexing could not load the embedder at all. It did not crash --
    # search.py degrades to literal matching -- so the user was told "no
    # matches" for files that were indexed and would have matched. The
    # README promises searching works while indexing runs, and two
    # processes sharing one card is the normal case, not an edge case.
    from hunch.backends.local_inprocess import LocalBackend
    from hunch.config import Config

    b = LocalBackend.__new__(LocalBackend)        # skip __init__'s probing
    b.cfg = Config()
    b.device = b._embed_device = "cuda"
    b._embedder = None
    loaded = []

    class FakeModel:
        def __init__(self, device):
            self.device = device

        def encode(self, texts, **kw):
            if self.device == "cuda":
                raise RuntimeError("CUDA out of memory. Tried to allocate 20.00 MiB")
            return [[0.5, 0.5] for _ in texts]

    def fake_load():
        loaded.append(b._embed_device)
        return FakeModel(b._embed_device)

    monkeypatch.setattr(b, "_load_embedder", fake_load)
    out = b.embed(["a query"])

    assert out == [[0.5, 0.5]]                    # the search still works
    assert loaded == ["cuda", "cpu"]              # tried the GPU, then fell back
    assert b._embed_device == "cpu"               # and stays there this session


def test_embed_does_not_loop_when_cpu_also_fails(monkeypatch):
    from hunch.backends.local_inprocess import LocalBackend
    from hunch.config import Config

    b = LocalBackend.__new__(LocalBackend)
    b.cfg = Config()
    b.device = b._embed_device = "cuda"
    b._embedder = None

    class Broken:
        def encode(self, texts, **kw):
            raise RuntimeError("still broken")

    monkeypatch.setattr(b, "_load_embedder", lambda: Broken())
    try:
        b.embed(["q"])
        assert False, "expected the error to surface"
    except RuntimeError as exc:
        assert "still broken" in str(exc)


def test_release_collects_cycles_so_the_memory_is_actually_freed(monkeypatch):
    # Regression test for a bug that made every existing release() call a
    # no-op. Dropping the model references is not enough: torch models hold
    # reference cycles, so the tensors stay alive until a collection runs
    # and empty_cache() finds nothing to reclaim. Measured on a real GPU --
    # 2,296 MiB reserved after load, still 2,296 MiB after release(), and
    # 20 MiB once gc.collect() ran first.
    #
    # This silently disabled the GUI's idle release and worker.drain()'s
    # release between stage phases, which is what keeps Whisper and the
    # vision model off a 4 GB card at the same time. The earlier test for
    # that fix asserted release() was *called*, which it always was -- the
    # call just did nothing.
    from hunch.backends.local_inprocess import LocalBackend
    from hunch.config import Config

    order = []
    b = LocalBackend.__new__(LocalBackend)
    b.cfg = Config()
    b.device = "cuda"
    b._embedder = object()
    b._vision = b._vision_proc = b._whisper = None

    import gc as real_gc
    monkeypatch.setattr(real_gc, "collect", lambda *a: order.append("gc") or 0)

    fake_torch = type(sys)("torch")
    fake_torch.cuda = type("cuda", (), {
        "empty_cache": staticmethod(lambda: order.append("empty_cache"))})
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    b.release()

    assert b._embedder is None
    # Order matters: collecting after empty_cache() would free the cycles
    # too late for that call to reclaim anything.
    assert order == ["gc", "empty_cache"]


def test_transcription_is_capped_to_what_can_reach_a_vector():
    # Measured on a real library: 2,301 audio files averaging 32 minutes.
    # The embedder reads the first 8,000 characters and speech runs about
    # 1,000 chars/minute, so roughly three quarters of every transcription
    # was decoded at full cost and then truncated away -- the same shape as
    # reading 200 MB of a file to embed 8 KB, except transcription costs
    # orders of magnitude more per byte.
    import hunch.backends.local_inprocess as mod

    seen = {}

    class RecordingModel:
        def transcribe(self, path, beam_size=1, **kw):
            seen.update(kw)
            return [], None

    cfg = Config()
    cfg.transcribe_max_seconds = 300
    b = mod.LocalBackend(cfg)
    b._whisper = RecordingModel()
    b.transcribe("/tmp/whatever.mp3")
    assert seen["clip_timestamps"] == [0.0, 300.0]


def test_transcription_cap_of_zero_means_the_whole_recording():
    # The tradeoff is real -- a topic first raised late in a long recording
    # stops being findable by it -- so opting out has to work.
    import hunch.backends.local_inprocess as mod

    seen = {}

    class RecordingModel:
        def transcribe(self, path, beam_size=1, **kw):
            seen.update(kw)
            return [], None

    cfg = Config()
    cfg.transcribe_max_seconds = 0
    b = mod.LocalBackend(cfg)
    b._whisper = RecordingModel()
    b.transcribe("/tmp/whatever.mp3")
    assert "clip_timestamps" not in seen
