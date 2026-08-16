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
