import json

import pytest

from hunch.backends.openrouter import OpenRouterBackend
from hunch.config import Config


@pytest.fixture
def cfg(monkeypatch):
    monkeypatch.setenv("HUNCH_OPENROUTER_KEY", "test-key")
    c = Config()
    c.backend = "openrouter"
    return c


def test_embed_posts_to_embeddings_endpoint(cfg, monkeypatch):
    seen = {}

    def fake_post(url, payload, key, timeout=120):
        seen["url"] = url
        seen["payload"] = payload
        return {"data": [{"embedding": [0.1, 0.2, 0.3]},
                         {"embedding": [0.4, 0.5, 0.6]}]}

    monkeypatch.setattr("hunch.backends.openrouter._post_json", fake_post)
    out = OpenRouterBackend(cfg).embed(["a", "b"])
    assert seen["url"].endswith("/embeddings")
    assert seen["payload"]["input"] == ["a", "b"]      # batched in one call
    assert out == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]


def test_describe_image_uses_chat_completions_with_image_url(cfg, monkeypatch, tmp_path):
    img = tmp_path / "x.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)
    seen = {}

    def fake_post(url, payload, key, timeout=120):
        seen["url"] = url
        seen["payload"] = payload
        return {"choices": [{"message": {"content": "a grey square"}}]}

    monkeypatch.setattr("hunch.backends.openrouter._post_json", fake_post)
    text, err = OpenRouterBackend(cfg).describe_image(str(img))
    assert err == ""
    assert text == "a grey square"
    assert seen["url"].endswith("/chat/completions")
    content = seen["payload"]["messages"][0]["content"]
    assert any(part["type"] == "image_url" for part in content)


def test_transcribe_sends_openrouters_input_audio_shape_not_openais_file_field(
        cfg, monkeypatch, tmp_path):
    # Regression test for a real bug: the body was {"model": ..., "file":
    # <base64>}, which is neither OpenRouter's schema nor OpenAI's. Verified
    # against OpenRouter's live docs and API: the endpoint exists (nonsense
    # paths 404, this one 401s without a key) and takes a JSON body with
    # `input_audio: {data, format}` -- OpenAI's multipart `file` upload is a
    # different API. With the wrong shape every audio and video file failed
    # for anyone using the openrouter backend.
    audio = tmp_path / "note.mp3"
    audio.write_bytes(b"ID3" + b"\x00" * 64)
    seen = {}

    def fake_post(url, payload, key, timeout=120):
        seen["url"] = url
        seen["payload"] = payload
        return {"text": "  the recorded words  "}

    monkeypatch.setattr("hunch.backends.openrouter._post_json", fake_post)
    # No ffmpeg: exercises the raw-passthrough path for an already-supported
    # container, which must still use the documented field names.
    monkeypatch.setattr("hunch.backends.openrouter.shutil.which", lambda n: None)

    text, err = OpenRouterBackend(cfg).transcribe(str(audio))
    assert err == ""
    assert text == "the recorded words"
    assert seen["url"].endswith("/audio/transcriptions")
    assert "file" not in seen["payload"]              # OpenAI's shape, not this API's
    assert seen["payload"]["input_audio"]["format"] == "mp3"
    assert seen["payload"]["input_audio"]["data"]     # base64, non-empty


def test_transcribe_demuxes_video_because_the_endpoint_takes_audio_only(
        cfg, monkeypatch, tmp_path):
    # worker.py routes video here too, but OpenRouter accepts audio
    # containers only -- an .mp4 sent as-is is refused no matter how correct
    # the field names are. ffmpeg must extract the audio first.
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00\x00\x00 ftypmp42" + b"\x00" * 64)
    seen = {}

    monkeypatch.setattr("hunch.backends.openrouter._post_json",
                        lambda url, payload, key, timeout=120: (
                            seen.update(payload=payload) or {"text": "spoken"}))
    monkeypatch.setattr("hunch.backends.openrouter.shutil.which",
                        lambda n: "/usr/bin/ffmpeg")

    class FakeProc:
        returncode, stderr = 0, ""

    def fake_run(cmd, **kw):
        # ffmpeg must be told to drop video and produce the mp3 we claim.
        assert "-vn" in cmd
        open(cmd[-1], "wb").write(b"fake mp3 bytes")
        return FakeProc()

    monkeypatch.setattr("hunch.backends.openrouter.subprocess.run", fake_run)

    text, err = OpenRouterBackend(cfg).transcribe(str(video))
    assert err == ""
    assert text == "spoken"
    assert seen["payload"]["input_audio"]["format"] == "mp3"   # not "mp4"


def test_transcribe_video_without_ffmpeg_says_so_instead_of_uploading_junk(
        cfg, monkeypatch, tmp_path):
    video = tmp_path / "clip.mkv"
    video.write_bytes(b"\x1a\x45\xdf\xa3" + b"\x00" * 64)
    posted = []
    monkeypatch.setattr("hunch.backends.openrouter._post_json",
                        lambda *a, **k: posted.append(1) or {"text": ""})
    monkeypatch.setattr("hunch.backends.openrouter.shutil.which", lambda n: None)

    text, err = OpenRouterBackend(cfg).transcribe(str(video))
    assert text == ""
    assert "ffmpeg" in err
    assert posted == []            # never upload bytes the endpoint will reject


def test_missing_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("HUNCH_OPENROUTER_KEY", raising=False)
    monkeypatch.setattr("hunch.backends.openrouter._keyring_get", lambda: None)
    c = Config()
    c.backend = "openrouter"
    with pytest.raises(RuntimeError, match="no OpenRouter API key"):
        OpenRouterBackend(c).embed(["a"])


def test_http_failure_surfaces_as_error_not_exception(cfg, monkeypatch, tmp_path):
    img = tmp_path / "x.png"
    img.write_bytes(b"0" * 32)

    def boom(*a, **k):
        raise OSError("network unreachable")

    monkeypatch.setattr("hunch.backends.openrouter._post_json", boom)
    text, err = OpenRouterBackend(cfg).describe_image(str(img))
    assert text == ""
    assert "network unreachable" in err


def test_store_api_key_writes_the_same_keyring_entry_api_key_reads(monkeypatch):
    from hunch.backends import openrouter

    written = {}

    class FakeKeyring:
        def set_password(self, service, username, value):
            written["args"] = (service, username, value)

    import sys
    monkeypatch.setitem(sys.modules, "keyring", FakeKeyring())
    openrouter.store_api_key("sk-test-123")
    assert written["args"] == ("hunch", "openrouter", "sk-test-123")
