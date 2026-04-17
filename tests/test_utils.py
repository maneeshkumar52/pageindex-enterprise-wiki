"""
tests/test_utils.py — Unit tests for utils.py
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from utils import (
    build_doc_metadata,
    chunk_text,
    export_to_markdown,
    extract_text,
    extract_text_from_txt,
    file_hash,
    get_nested,
    load_config,
)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_loads_yaml(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "test_config.yaml"
        cfg_file.write_text("ollama:\n  model: testmodel\n")
        # Reset cache
        import utils
        utils._CONFIG_CACHE = None
        cfg = load_config(str(cfg_file))
        assert cfg["ollama"]["model"] == "testmodel"
        utils._CONFIG_CACHE = None  # cleanup

    def test_missing_file_raises(self) -> None:
        import utils
        utils._CONFIG_CACHE = None
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path.yaml")
        utils._CONFIG_CACHE = None


class TestGetNested:
    def test_simple_path(self) -> None:
        d = {"a": {"b": {"c": 42}}}
        assert get_nested(d, "a", "b", "c") == 42

    def test_missing_returns_default(self) -> None:
        d = {"a": 1}
        assert get_nested(d, "x", "y", default="nope") == "nope"

    def test_empty_dict(self) -> None:
        assert get_nested({}, "a", default=0) == 0


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

class TestExtractText:
    def test_extract_txt(self, tmp_path: Path) -> None:
        f = tmp_path / "hello.txt"
        f.write_text("Hello, World!")
        assert extract_text(str(f)) == "Hello, World!"

    def test_extract_md(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.md"
        f.write_text("# Title\n\nBody text.")
        assert "Title" in extract_text(str(f))

    def test_unsupported_ext_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "data.xyz"
        f.write_text("data")
        with pytest.raises(ValueError, match="Unsupported"):
            extract_text(str(f))

    def test_extract_from_txt_encoding(self, tmp_path: Path) -> None:
        f = tmp_path / "utf8.txt"
        f.write_text("café résumé", encoding="utf-8")
        assert "café" in extract_text_from_txt(str(f))


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

class TestChunking:
    def test_single_chunk(self) -> None:
        text = "word " * 100
        chunks = chunk_text(text, chunk_size=200, chunk_overlap=20)
        assert len(chunks) == 1
        assert chunks[0]["index"] == 0

    def test_multiple_chunks(self) -> None:
        text = "word " * 1000
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
        assert len(chunks) > 1
        # Each chunk should have non-empty text
        for c in chunks:
            assert c["text"].strip()

    def test_overlap(self) -> None:
        words = [f"w{i}" for i in range(200)]
        text = " ".join(words)
        chunks = chunk_text(text, chunk_size=50, chunk_overlap=10)
        # Second chunk should start at word 40 (50 - 10)
        second_chunk_words = chunks[1]["text"].split()
        assert second_chunk_words[0] == "w40"

    def test_empty_text(self) -> None:
        assert chunk_text("") == []


# ---------------------------------------------------------------------------
# File hashing
# ---------------------------------------------------------------------------

class TestFileHash:
    def test_deterministic(self, tmp_path: Path) -> None:
        f = tmp_path / "data.txt"
        f.write_text("deterministic content")
        h1 = file_hash(str(f))
        h2 = file_hash(str(f))
        assert h1 == h2
        assert len(h1) == 64  # SHA-256

    def test_different_content_different_hash(self, tmp_path: Path) -> None:
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("content A")
        f2.write_text("content B")
        assert file_hash(str(f1)) != file_hash(str(f2))


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

class TestBuildDocMetadata:
    def test_fields_present(self, tmp_path: Path) -> None:
        f = tmp_path / "test.pdf"
        f.write_bytes(b"%PDF-1.4 fake content")
        meta = build_doc_metadata(str(f))
        assert meta["filename"] == "test.pdf"
        assert meta["extension"] == ".pdf"
        assert meta["size_bytes"] > 0
        assert len(meta["sha256"]) == 64
        assert "indexed_at" in meta


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

class TestExportToMarkdown:
    def test_creates_file(self, tmp_path: Path) -> None:
        out = str(tmp_path / "exports")
        path = export_to_markdown(
            query="What is our leave policy?",
            answer="New employees get 15 days per year.",
            sources=[{"filename": "hr_policy.txt", "page": 3, "snippet": "15 days"}],
            output_dir=out,
        )
        assert Path(path).exists()
        content = Path(path).read_text()
        assert "leave policy" in content.lower()
        assert "15 days" in content

    def test_no_sources(self, tmp_path: Path) -> None:
        out = str(tmp_path / "exports")
        path = export_to_markdown("q", "a", [], output_dir=out)
        assert Path(path).exists()
        content = Path(path).read_text()
        assert "Sources" not in content
