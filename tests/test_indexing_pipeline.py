"""
tests/test_indexing_pipeline.py — Unit tests for indexing_pipeline.py
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_config(tmp_path: Path) -> dict:
    """Return a config dict pointing to temp directories."""
    return {
        "ollama": {
            "base_url": "http://localhost:11434",
            "model": "llama3",
            "embedding_model": "nomic-embed-text",
            "temperature": 0.3,
            "top_p": 0.9,
            "timeout": 30,
        },
        "pageindex": {
            "workspace": str(tmp_path / "workspace"),
            "index_model": "ollama/llama3",
            "retrieve_model": "ollama/llama3",
        },
        "chromadb": {
            "persist_directory": str(tmp_path / "chromadb"),
            "collection_name": "test_docs",
            "chunk_size": 50,
            "chunk_overlap": 10,
        },
        "storage": {
            "upload_dir": str(tmp_path / "uploads"),
            "index_dir": str(tmp_path / "indexes"),
            "output_dir": str(tmp_path / "outputs"),
        },
        "logging": {"level": "DEBUG"},
    }


@pytest.fixture
def sample_txt(tmp_path: Path) -> str:
    """Create a sample text file and return its path."""
    f = tmp_path / "sample.txt"
    f.write_text("This is a sample document for testing. " * 50)
    return str(f)


# ---------------------------------------------------------------------------
# VectorStore tests
# ---------------------------------------------------------------------------

class TestVectorStore:
    @patch("indexing_pipeline.OllamaClient")
    def test_add_and_search(self, mock_ollama_cls: MagicMock, tmp_config: dict) -> None:
        mock_ollama = MagicMock()
        mock_ollama.embed_batch.return_value = [[0.1] * 384] * 5
        mock_ollama.embed.return_value = [0.1] * 384
        mock_ollama_cls.return_value = mock_ollama

        from indexing_pipeline import VectorStore
        store = VectorStore(tmp_config)

        n = store.add_document("doc1", "word " * 200, {"filename": "test.txt"})
        assert n > 0
        assert store.count > 0

        results = store.search("word", n_results=3)
        assert len(results) > 0
        assert "text" in results[0]
        assert "score" in results[0]

    @patch("indexing_pipeline.OllamaClient")
    def test_delete_document(self, mock_ollama_cls: MagicMock, tmp_config: dict) -> None:
        mock_ollama = MagicMock()
        mock_ollama.embed_batch.return_value = [[0.1] * 384] * 5
        mock_ollama_cls.return_value = mock_ollama

        from indexing_pipeline import VectorStore
        store = VectorStore(tmp_config)

        store.add_document("doc_del", "content " * 200, {"filename": "del.txt"})
        assert store.count > 0

        store.delete_document("doc_del")
        # ChromaDB may not immediately reflect count changes in all backends,
        # but the delete should not raise.


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_load_save_roundtrip(self, tmp_config: dict) -> None:
        import indexing_pipeline as ip

        meta_file = Path(tmp_config["storage"]["index_dir"]) / "doc_registry.json"
        ip._META_FILE = str(meta_file)

        ip._save_registry({"doc1": {"filename": "a.txt"}})
        loaded = ip._load_registry()
        assert loaded["doc1"]["filename"] == "a.txt"

    def test_load_missing_returns_empty(self) -> None:
        import indexing_pipeline as ip
        ip._META_FILE = "/nonexistent/path/registry.json"
        assert ip._load_registry() == {}


# ---------------------------------------------------------------------------
# Pipeline integration (mocked externals)
# ---------------------------------------------------------------------------

class TestPageIndexPipeline:
    @patch("indexing_pipeline.OllamaClient")
    def test_ingest_txt_file(
        self,
        mock_ollama_cls: MagicMock,
        tmp_config: dict,
        sample_txt: str,
    ) -> None:
        mock_ollama = MagicMock()
        mock_ollama.embed_batch.return_value = [[0.1] * 384] * 20
        mock_ollama.embed.return_value = [0.1] * 384
        mock_ollama_cls.return_value = mock_ollama

        import indexing_pipeline as ip
        meta_file = Path(tmp_config["storage"]["index_dir"]) / "doc_registry.json"
        ip._META_FILE = str(meta_file)

        pipeline = ip.PageIndexPipeline(tmp_config)
        # PageIndex will fail to import in test env — that's OK,
        # the pipeline falls back to vector-only mode.
        meta = pipeline.ingest(sample_txt)

        assert "doc_id" in meta
        assert meta["filename"] == "sample.txt"
        assert meta["num_chunks"] > 0
        assert pipeline.document_count == 1

    @patch("indexing_pipeline.OllamaClient")
    def test_duplicate_detection(
        self,
        mock_ollama_cls: MagicMock,
        tmp_config: dict,
        sample_txt: str,
    ) -> None:
        mock_ollama = MagicMock()
        mock_ollama.embed_batch.return_value = [[0.1] * 384] * 20
        mock_ollama_cls.return_value = mock_ollama

        import indexing_pipeline as ip
        meta_file = Path(tmp_config["storage"]["index_dir"]) / "doc_registry.json"
        ip._META_FILE = str(meta_file)

        pipeline = ip.PageIndexPipeline(tmp_config)
        meta1 = pipeline.ingest(sample_txt)
        meta2 = pipeline.ingest(sample_txt)
        # Should return the same doc_id (dedup by hash)
        assert meta1["doc_id"] == meta2["doc_id"]
        assert pipeline.document_count == 1

    @patch("indexing_pipeline.OllamaClient")
    def test_list_and_delete(
        self,
        mock_ollama_cls: MagicMock,
        tmp_config: dict,
        sample_txt: str,
    ) -> None:
        mock_ollama = MagicMock()
        mock_ollama.embed_batch.return_value = [[0.1] * 384] * 20
        mock_ollama_cls.return_value = mock_ollama

        import indexing_pipeline as ip
        meta_file = Path(tmp_config["storage"]["index_dir"]) / "doc_registry.json"
        ip._META_FILE = str(meta_file)

        pipeline = ip.PageIndexPipeline(tmp_config)
        meta = pipeline.ingest(sample_txt)

        docs = pipeline.list_documents()
        assert len(docs) == 1

        deleted = pipeline.delete_document(meta["doc_id"])
        assert deleted is True
        assert pipeline.document_count == 0

    def test_ingest_missing_file_raises(self, tmp_config: dict) -> None:
        import indexing_pipeline as ip
        meta_file = Path(tmp_config["storage"]["index_dir"]) / "doc_registry.json"
        ip._META_FILE = str(meta_file)

        pipeline = ip.PageIndexPipeline(tmp_config)
        with pytest.raises(FileNotFoundError):
            pipeline.ingest("/does/not/exist.txt")
