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
