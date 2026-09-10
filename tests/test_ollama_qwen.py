"""
Tests for Ollama / Qwen3-4B client integration.

Verifies:
- Ollama available + qwen3:4b available
- Ollama unavailable
- qwen3:4b unavailable
- HTTP 404
- HTTP 500
- Empty response
- Malformed response
- Mock mode explicitly enabled (via mock_mode=True or LLM_PROVIDER=mock)
- Strict mode with silent fallback disabled (no automatic fallback to mock or MAC RTL)
"""

from io import BytesIO
import json
import os
import urllib.error
from unittest.mock import MagicMock, patch
import pytest

from llm.qwen import OllamaQwenClient


class MockHttpResponse:
    def __init__(self, data: bytes, status: int = 200):
        self._data = data
        self.status = status

    def read(self) -> bytes:
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


@pytest.fixture
def ollama_client():
    return OllamaQwenClient(
        model="qwen3:4b",
        base_url="http://localhost:11434",
        mock_mode=False,
    )


def test_ollama_available_and_qwen3_available(ollama_client):
    """Test model verification when Ollama is reachable and qwen3:4b is present."""
    tags_payload = json.dumps({
        "models": [
            {"name": "qwen3:4b", "model": "qwen3:4b"},
            {"name": "llama3:latest", "model": "llama3"},
        ]
    }).encode("utf-8")

    with patch("urllib.request.urlopen", return_value=MockHttpResponse(tags_payload)):
        # Should complete without error
        ollama_client.verify_model_installed()


@pytest.mark.asyncio
async def test_ollama_generation_success(ollama_client):
    """Test successful generation with metadata provenance and observability."""
    gen_payload = json.dumps({
        "response": "module mac;\nendmodule",
        "done": True,
        "prompt_eval_count": 10,
        "eval_count": 25,
    }).encode("utf-8")

    with patch("urllib.request.urlopen", return_value=MockHttpResponse(gen_payload)):
        resp = await ollama_client.generate("Design a MAC unit")
        assert "module mac" in resp.text
        assert resp.metadata["provider"] == "ollama"
        assert resp.metadata["model"] == "qwen3:4b"
        assert resp.metadata["fallback_used"] is False
        assert resp.prompt_tokens == 10
        assert resp.completion_tokens == 25


def test_ollama_unavailable(ollama_client):
    """Test behavior when Ollama server is unreachable."""
    url_err = urllib.error.URLError("Connection refused")

    with patch("urllib.request.urlopen", side_effect=url_err):
        with pytest.raises(RuntimeError, match="Cannot connect to Ollama at http://localhost:11434"):
            ollama_client.verify_model_installed()


@pytest.mark.asyncio
async def test_ollama_generation_unavailable(ollama_client):
    """Test generation raises RuntimeError when Ollama is unreachable (no silent fallback)."""
    url_err = urllib.error.URLError("Connection refused")

    with patch("urllib.request.urlopen", side_effect=url_err):
        with pytest.raises(RuntimeError, match="Cannot connect to Ollama at http://localhost:11434"):
            await ollama_client.generate("Design a MAC")


def test_qwen3_4b_unavailable(ollama_client):
    """Test verification failure when qwen3:4b is not in /api/tags."""
    tags_payload = json.dumps({
        "models": [
            {"name": "deepseek-coder:6.7b", "model": "deepseek-coder"},
            {"name": "mistral:7b", "model": "mistral"},
        ]
    }).encode("utf-8")

    with patch("urllib.request.urlopen", return_value=MockHttpResponse(tags_payload)):
        with pytest.raises(RuntimeError, match="Qwen3-4B is not installed in Ollama"):
            ollama_client.verify_model_installed()


@pytest.mark.asyncio
async def test_http_404_error(ollama_client):
    """Test HTTP 404 response handling."""
    http_err = urllib.error.HTTPError(
        "http://localhost:11434/api/generate",
        404,
        "Not Found",
        {},
        BytesIO(b"model 'qwen3:4b' not found"),
    )

    with patch("urllib.request.urlopen", side_effect=http_err):
        with pytest.raises(RuntimeError, match="Ollama request failed: HTTP 404"):
            await ollama_client.generate("Hello")


@pytest.mark.asyncio
async def test_http_500_error(ollama_client):
    """Test HTTP 500 response handling."""
    http_err = urllib.error.HTTPError(
        "http://localhost:11434/api/generate",
        500,
        "Internal Server Error",
        {},
        BytesIO(b"server crash"),
    )

    with patch("urllib.request.urlopen", side_effect=http_err):
        with pytest.raises(RuntimeError, match="Ollama request failed: HTTP 500"):
            await ollama_client.generate("Hello")


@pytest.mark.asyncio
async def test_empty_response(ollama_client):
    """Test empty response body raises RuntimeError."""
    with patch("urllib.request.urlopen", return_value=MockHttpResponse(b"")):
        with pytest.raises(RuntimeError, match="Ollama returned empty response"):
            await ollama_client.generate("Hello")


@pytest.mark.asyncio
async def test_malformed_json_response(ollama_client):
    """Test non-JSON response raises RuntimeError."""
    with patch("urllib.request.urlopen", return_value=MockHttpResponse(b"invalid json <>")):
        with pytest.raises(RuntimeError, match="malformed JSON response"):
            await ollama_client.generate("Hello")


@pytest.mark.asyncio
async def test_malformed_response_missing_field(ollama_client):
    """Test JSON response missing 'response' field raises RuntimeError."""
    payload = json.dumps({"done": True, "eval_count": 0}).encode("utf-8")
    with patch("urllib.request.urlopen", return_value=MockHttpResponse(payload)):
        with pytest.raises(RuntimeError, match="missing 'response' field"):
            await ollama_client.generate("Hello")


@pytest.mark.asyncio
async def test_mock_mode_explicitly_enabled():
    """Test explicit mock mode via mock_mode=True."""
    client = OllamaQwenClient(mock_mode=True)
    client.set_mock_response("special query", {"action": "CREATE_RTL"})

    # verify_model_installed should do nothing in mock mode
    client.verify_model_installed()

    # generate should return mock response without calling network
    resp = await client.generate("This is a special query for testing")
    assert resp.metadata["provider"] == "mock"
    assert resp.metadata["fallback_used"] is True
    parsed = json.loads(resp.text)
    assert parsed["action"] == "CREATE_RTL"


@pytest.mark.asyncio
async def test_mock_mode_via_env_var(monkeypatch):
    """Test explicit mock mode via LLM_PROVIDER=mock."""
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    client = OllamaQwenClient(mock_mode=False)

    # Should not call network
    client.verify_model_installed()
    resp = await client.generate("Any prompt")
    assert resp.metadata["fallback_used"] is True
    assert resp.metadata["provider"] == "mock"


@pytest.mark.asyncio
async def test_strict_mode_fallback_disabled(ollama_client):
    """Test that when Ollama fails, NO fallback MAC RTL is ever returned."""
    url_err = urllib.error.URLError("Ollama is down")

    with patch("urllib.request.urlopen", side_effect=url_err):
        with pytest.raises(RuntimeError):
            await ollama_client.generate("Design an 8-bit signed MAC")

    # Verify no _offline_fallback exists to produce deterministic MAC
    assert not hasattr(ollama_client, "_offline_fallback")
