"""
ollama_client.py — Thin wrapper around the Ollama HTTP API.

Provides helpers for:
  • Chat completion (streaming & non-streaming)
  • Embedding generation
  • Model health checks
"""

from __future__ import annotations

import json
import logging
from typing import Any, Generator

import requests

from utils import load_config, get_nested

logger = logging.getLogger("pageindex_wiki.ollama")


class OllamaClient:
    """Lightweight client for the Ollama REST API."""

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        if cfg is None:
            cfg = load_config()
        ollama_cfg = cfg.get("ollama", {})
        self.base_url: str = ollama_cfg.get("base_url", "http://localhost:11434").rstrip("/")
        self.model: str = ollama_cfg.get("model", "llama3")
        self.embedding_model: str = ollama_cfg.get("embedding_model", "nomic-embed-text")
        self.temperature: float = ollama_cfg.get("temperature", 0.3)
        self.top_p: float = ollama_cfg.get("top_p", 0.9)
        self.timeout: int = ollama_cfg.get("timeout", 120)

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return True if the Ollama server is reachable."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def list_models(self) -> list[str]:
        """Return names of locally available models."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=10)
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except Exception as exc:
            logger.warning("Failed to list models: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Chat completion
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """
        Send a chat completion request and return the full response text.

        Parameters
        ----------
        messages : list of {"role": str, "content": str}
        model : override the default model
        system_prompt : optional system message prepended to the conversation
        """
        model = model or self.model
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
            },
        }
        logger.info("Ollama chat → model=%s, messages=%d", model, len(messages))
        resp = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "")

    def chat_stream(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        system_prompt: str | None = None,
    ) -> Generator[str, None, None]:
        """
        Stream chat completion tokens as a generator.
        Yields one token-string at a time.
        """
        model = model or self.model
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages

        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
            },
        }
        logger.info("Ollama chat_stream → model=%s", model)
        with requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout,
            stream=True,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                chunk = json.loads(line)
                token = chunk.get("message", {}).get("content", "")
                if token:
                    yield token

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def embed(self, text: str, model: str | None = None) -> list[float]:
        """Generate an embedding vector for a single text string."""
        model = model or self.embedding_model
        payload = {"model": model, "input": text}
        resp = requests.post(
            f"{self.base_url}/api/embed",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        embeddings = data.get("embeddings", [])
        if embeddings:
            return embeddings[0]
        return []

    def embed_batch(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        model = model or self.embedding_model
        payload = {"model": model, "input": texts}
        resp = requests.post(
            f"{self.base_url}/api/embed",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json().get("embeddings", [])

    # ------------------------------------------------------------------
    # RAG helper
    # ------------------------------------------------------------------

    def ask_with_context(
        self,
        question: str,
        context: str,
        model: str | None = None,
    ) -> str:
        """
        Answer a question given retrieved context.

        Uses a structured prompt template to ground the LLM response.
        """
        system = (
            "You are an expert enterprise knowledge assistant. "
            "Answer the user's question using ONLY the provided context. "
            "If the context does not contain enough information, say so clearly. "
            "Cite the relevant section or page when possible. "
            "Be concise and professional."
        )
        user_msg = (
            f"### Context\n\n{context}\n\n"
            f"### Question\n\n{question}"
        )
        return self.chat(
            messages=[{"role": "user", "content": user_msg}],
            model=model,
            system_prompt=system,
        )
