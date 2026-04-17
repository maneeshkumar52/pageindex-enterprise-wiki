"""
tests/test_ollama_client.py — Unit tests for ollama_client.py
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from ollama_client import OllamaClient


@pytest.fixture
def client() -> OllamaClient:
    cfg = {
        "ollama": {
            "base_url": "http://localhost:11434",
            "model": "llama3",
            "embedding_model": "nomic-embed-text",
            "temperature": 0.3,
            "top_p": 0.9,
            "timeout": 30,
        }
    }
    return OllamaClient(cfg)


class TestOllamaAvailability:
    @patch("ollama_client.requests.get")
    def test_is_available_true(self, mock_get: MagicMock, client: OllamaClient) -> None:
        mock_get.return_value = MagicMock(status_code=200)
        assert client.is_available() is True

    @patch("ollama_client.requests.get")
    def test_is_available_false_on_error(self, mock_get: MagicMock, client: OllamaClient) -> None:
        mock_get.side_effect = requests.ConnectionError("refused")
        assert client.is_available() is False

    @patch("ollama_client.requests.get")
    def test_list_models(self, mock_get: MagicMock, client: OllamaClient) -> None:
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"models": [{"name": "llama3"}, {"name": "mistral"}]},
        )
        mock_get.return_value.raise_for_status = MagicMock()
        models = client.list_models()
        assert "llama3" in models
        assert "mistral" in models


class TestOllamaChat:
    @patch("ollama_client.requests.post")
    def test_chat_returns_content(self, mock_post: MagicMock, client: OllamaClient) -> None:
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"message": {"content": "Hello from LLM"}},
        )
        mock_post.return_value.raise_for_status = MagicMock()
        result = client.chat([{"role": "user", "content": "hi"}])
        assert result == "Hello from LLM"

    @patch("ollama_client.requests.post")
    def test_chat_with_system_prompt(self, mock_post: MagicMock, client: OllamaClient) -> None:
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"message": {"content": "response"}},
        )
        mock_post.return_value.raise_for_status = MagicMock()
        client.chat(
            [{"role": "user", "content": "test"}],
            system_prompt="You are helpful.",
        )
        call_data = mock_post.call_args[1]["json"]
        assert call_data["messages"][0]["role"] == "system"


class TestOllamaEmbed:
    @patch("ollama_client.requests.post")
    def test_embed_returns_vector(self, mock_post: MagicMock, client: OllamaClient) -> None:
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"embeddings": [[0.1, 0.2, 0.3]]},
        )
        mock_post.return_value.raise_for_status = MagicMock()
        vec = client.embed("test text")
        assert vec == [0.1, 0.2, 0.3]

    @patch("ollama_client.requests.post")
    def test_embed_batch(self, mock_post: MagicMock, client: OllamaClient) -> None:
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"embeddings": [[0.1, 0.2], [0.3, 0.4]]},
        )
        mock_post.return_value.raise_for_status = MagicMock()
        vecs = client.embed_batch(["a", "b"])
        assert len(vecs) == 2


class TestOllamaRAG:
    @patch("ollama_client.requests.post")
    def test_ask_with_context(self, mock_post: MagicMock, client: OllamaClient) -> None:
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"message": {"content": "The answer is 42."}},
        )
        mock_post.return_value.raise_for_status = MagicMock()
        answer = client.ask_with_context("What is the answer?", "Context: 42")
        assert "42" in answer
