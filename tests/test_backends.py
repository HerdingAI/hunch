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
        def transcribe(self, path, beam_size=1):
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

        def transcribe(self, path, beam_size=1):
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
