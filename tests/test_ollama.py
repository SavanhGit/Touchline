import io
import json
from urllib.error import URLError

import pytest

from soccer_tactics.agent.ollama import OllamaChatClient, model_from_env


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


def test_chat_posts_messages_tools_and_json_format_to_local_api():
    calls = []
    result = {"message": {"content": "{}", "tool_calls": []}}

    def transport(request, timeout):
        calls.append((request, timeout))
        return FakeResponse(result)

    tools = [{"type": "function", "function": {"name": "x", "parameters": {"type": "object"}}}]
    client = OllamaChatClient(transport=transport)
    response = client.chat([{"role": "user", "content": "hello"}], tools, format="json")

    request, timeout = calls[0]
    assert request.full_url == "http://localhost:11434/api/chat"
    assert request.get_method() == "POST"
    assert request.get_header("Content-type") == "application/json"
    assert json.loads(request.data) == {
        "model": "qwen3:8b",
        "messages": [{"role": "user", "content": "hello"}],
        "tools": tools,
        "format": "json",
        "stream": False,
    }
    assert timeout > 0
    assert response == result


def test_model_can_be_selected_with_touchline_environment(monkeypatch):
    monkeypatch.setenv("TOUCHLINE_OLLAMA_MODEL", "llama3.1:8b")
    assert model_from_env() == "llama3.1:8b"


def test_unavailable_server_error_is_actionable():
    def transport(request, timeout):
        raise URLError(ConnectionRefusedError())

    with pytest.raises(RuntimeError, match="Ollama.*localhost:11434.*ollama serve"):
        OllamaChatClient(transport=transport).chat([])


def test_missing_model_error_is_actionable():
    from urllib.error import HTTPError

    def transport(request, timeout):
        raise HTTPError(request.full_url, 404, "Not Found", {}, io.BytesIO(b'{"error":"model not found"}'))

    with pytest.raises(RuntimeError, match="qwen3:8b.*ollama pull"):
        OllamaChatClient(transport=transport).chat([])
