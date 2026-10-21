"""Small stdlib HTTP client for a local Ollama chat server."""

from __future__ import annotations

import json
import os
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_MODEL = "qwen3:8b"
CHAT_URL = "http://localhost:11434/api/chat"


def model_from_env() -> str:
    """Return the configured local model, falling back to the default."""
    return os.environ.get("TOUCHLINE_OLLAMA_MODEL", "").strip() or DEFAULT_MODEL


class OllamaChatClient:
    """Post chat requests to Ollama; inject ``transport`` for deterministic tests."""

    def __init__(
        self,
        model: str | None = None,
        *,
        transport: Callable[..., Any] | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.model = model or model_from_env()
        self.transport = transport or urlopen
        self.timeout = timeout

    def chat(self, messages, tools=None, *, format=None) -> dict:
        """Send one non-streaming chat request and return Ollama's response."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            # Qwen3 defaults to thinking mode, which is unnecessary for this
            # evidence lookup flow and can push ordinary questions past timeout.
            "think": False,
        }
        if tools is not None:
            payload["tools"] = tools
        if format is not None:
            payload["format"] = format

        request = Request(
            CHAT_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self.transport(request, timeout=self.timeout) as response:
                body = response.read()
        except HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            if exc.code == 404 and "model" in error_body.lower():
                raise RuntimeError(
                    f"Ollama model '{self.model}' was not found. Install it locally with `ollama pull {self.model}`."
                ) from exc
            raise RuntimeError(f"Ollama chat request failed (HTTP {exc.code}): {error_body}") from exc
        except TimeoutError as exc:
            raise RuntimeError(
                f"Ollama timed out after {self.timeout:g} seconds while generating with "
                f"'{self.model}'. The server is reachable, but the model did not finish; "
                "try a smaller local model such as qwen3:4b."
            ) from exc
        except (URLError, ConnectionError, OSError) as exc:
            raise RuntimeError(
                "Ollama is unavailable at localhost:11434. Start the local Ollama server (run `ollama serve`) and try again."
            ) from exc

        try:
            result = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RuntimeError("Ollama returned an invalid JSON response.") from exc
        if not isinstance(result, dict):
            raise RuntimeError("Ollama returned an unexpected chat response.")
        return result
