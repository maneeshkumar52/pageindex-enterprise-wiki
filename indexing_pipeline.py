"""
indexing_pipeline.py — Document ingestion, PageIndex tree indexing,
and ChromaDB vector search pipeline.

Two complementary retrieval layers:
  1. **PageIndex** — Reasoning-based tree search (primary, vectorless).
  2. **ChromaDB** — Embedding-based semantic search (secondary, dense retrieval).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

import chromadb

from ollama_client import OllamaClient
from utils import (
    build_doc_metadata,
    chunk_text,
    docx_to_markdown,
    ensure_dirs,
    extract_text,
    file_hash,
    get_nested,
    load_config,
    txt_to_markdown,
)

logger = logging.getLogger("pageindex_wiki.indexing")

# ---------------------------------------------------------------------------
# Metadata store  (lightweight JSON-based)
# ---------------------------------------------------------------------------

_META_FILE = "data/indexes/doc_registry.json"


def _load_registry() -> dict[str, Any]:
    p = Path(_META_FILE)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def _save_registry(registry: dict[str, Any]) -> None:
    Path(_META_FILE).parent.mkdir(parents=True, exist_ok=True)
    Path(_META_FILE).write_text(
        json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# ChromaDB helper
# ---------------------------------------------------------------------------

class VectorStore:
    """Manages a ChromaDB collection for semantic search."""

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        if cfg is None:
            cfg = load_config()
        chroma_cfg = cfg.get("chromadb", {})
        persist_dir = chroma_cfg.get("persist_directory", "data/chromadb")
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection_name = chroma_cfg.get("collection_name", "wiki_documents")
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.chunk_size: int = chroma_cfg.get("chunk_size", 512)
        self.chunk_overlap: int = chroma_cfg.get("chunk_overlap", 64)
        self._ollama = OllamaClient(cfg)

    def add_document(self, doc_id: str, text: str, metadata: dict[str, Any]) -> int:
        """Chunk, embed, and store a document. Returns the number of chunks."""
        chunks = chunk_text(text, self.chunk_size, self.chunk_overlap)
        if not chunks:
            return 0

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []
        embeddings: list[list[float]] = []

        batch_texts = [c["text"] for c in chunks]
        batch_embeds = self._ollama.embed_batch(batch_texts)

        for chunk, emb in zip(chunks, batch_embeds):
            chunk_id = f"{doc_id}_chunk_{chunk['index']}"
            ids.append(chunk_id)
            documents.append(chunk["text"])
            metadatas.append({
                "doc_id": doc_id,
                "chunk_index": chunk["index"],
                "filename": metadata.get("filename", ""),
            })
            embeddings.append(emb)

        self._collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        logger.info("VectorStore: added %d chunks for doc_id=%s", len(chunks), doc_id)
        return len(chunks)

    def search(self, query: str, n_results: int = 5) -> list[dict[str, Any]]:
        """Semantic search against the vector store."""
        query_emb = self._ollama.embed(query)
        if not query_emb:
            return []
        results = self._collection.query(
            query_embeddings=[query_emb],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        hits: list[dict[str, Any]] = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            hits.append({
                "text": doc,
                "metadata": meta,
                "score": 1 - dist,  # cosine similarity
            })
        return hits

    def delete_document(self, doc_id: str) -> None:
        """Remove all chunks belonging to a document."""
        self._collection.delete(where={"doc_id": doc_id})
        logger.info("VectorStore: deleted chunks for doc_id=%s", doc_id)

    @property
    def count(self) -> int:
        return self._collection.count()


# ---------------------------------------------------------------------------
# PageIndex tree builder
# ---------------------------------------------------------------------------

class PageIndexPipeline:
    """
    Orchestrates document ingestion using both PageIndex (tree index)
    and ChromaDB (vector embeddings).
    """

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        if cfg is None:
            cfg = load_config()
        self.cfg = cfg
        ensure_dirs(cfg)

        pi_cfg = cfg.get("pageindex", {})
        self.workspace = Path(pi_cfg.get("workspace", "data/workspace"))
        self.workspace.mkdir(parents=True, exist_ok=True)

        self.index_model: str = pi_cfg.get("index_model", "ollama/llama3")
        self.retrieve_model: str = pi_cfg.get("retrieve_model", "ollama/llama3")
        self.upload_dir = Path(get_nested(cfg, "storage", "upload_dir", default="data/uploads"))
        self.index_dir = Path(get_nested(cfg, "storage", "index_dir", default="data/indexes"))

        self.vector_store = VectorStore(cfg)
        self.ollama = OllamaClient(cfg)

        self._pi_client: Any | None = None
        self._registry = _load_registry()

    # ------------------------------------------------------------------
    # PageIndex client (lazy init)
    # ------------------------------------------------------------------

    def _get_pi_client(self) -> Any:
        """Lazily initialise the self-hosted PageIndexClient."""
        if self._pi_client is None:
            try:
                from pageindex import PageIndexClient
                os.environ.setdefault("OLLAMA_API_BASE", self.cfg.get("ollama", {}).get("base_url", "http://localhost:11434"))
                self._pi_client = PageIndexClient(
                    workspace=str(self.workspace),
                    index_model=self.index_model,
                    retrieve_model=self.retrieve_model,
                )
                logger.info("PageIndexClient initialised (workspace=%s)", self.workspace)
            except ImportError:
                logger.warning(
                    "pageindex package not installed — tree indexing disabled. "
                    "Install with: pip install pageindex"
                )
                raise
        return self._pi_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(self, file_path: str) -> dict[str, Any]:
        """
        Full ingestion pipeline for a single document.

        Steps
        -----
        1. Copy file into upload_dir.
        2. Build PageIndex tree (PDF / Markdown).
        3. Store embeddings in ChromaDB.
        4. Register metadata.

        Returns a metadata dict including ``doc_id``.
        """
        src = Path(file_path)
        if not src.exists():
            raise FileNotFoundError(file_path)

        # Deduplicate by content hash
        sha = file_hash(file_path)
        for doc_id, entry in self._registry.items():
            if entry.get("sha256") == sha:
                logger.info("Duplicate detected — returning existing doc_id=%s", doc_id)
                return entry

        # Copy to upload dir
        dest = self.upload_dir / src.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, dest)

        metadata = build_doc_metadata(str(dest))
        ext = src.suffix.lower()

        # --- PageIndex tree indexing ---
        pi_doc_id: str | None = None
        tree_structure: Any = None
        indexable_path = str(dest)

        try:
            client = self._get_pi_client()

            # PageIndex supports PDF and Markdown natively.
            # Convert DOCX / TXT → Markdown first.
            if ext == ".docx":
                indexable_path = docx_to_markdown(str(dest), str(self.upload_dir))
            elif ext == ".txt":
                indexable_path = txt_to_markdown(str(dest), str(self.upload_dir))

            pi_doc_id = client.index(indexable_path)
            tree_structure = client.get_document_structure(pi_doc_id)
            logger.info("PageIndex tree built: doc_id=%s", pi_doc_id)
        except Exception as exc:
            logger.warning("PageIndex indexing failed (%s): %s", src.name, exc)

        # --- ChromaDB vector embeddings ---
        doc_id = pi_doc_id or sha[:16]
        raw_text = extract_text(str(dest))
        n_chunks = self.vector_store.add_document(doc_id, raw_text, metadata)

        # --- Persist metadata ---
        metadata.update({
            "doc_id": doc_id,
            "pi_doc_id": pi_doc_id,
            "tree_structure": tree_structure,
            "num_chunks": n_chunks,
            "source_path": str(dest),
        })
        self._registry[doc_id] = metadata
        _save_registry(self._registry)
        logger.info("Ingested %s → doc_id=%s (%d chunks)", src.name, doc_id, n_chunks)
        return metadata

    def search(self, query: str, n_results: int = 5) -> list[dict[str, Any]]:
        """
        Multi-layer retrieval.

        1. ChromaDB semantic search returns top chunks.
        2. (If PageIndex is available) tree-search enriches results.
        """
        hits = self.vector_store.search(query, n_results=n_results)

        # Enrich hits with PageIndex page content when available
        try:
            client = self._get_pi_client()
            enriched: list[dict[str, Any]] = []
            seen_docs: set[str] = set()
            for hit in hits:
                doc_id = hit["metadata"].get("doc_id", "")
                if doc_id and doc_id not in seen_docs:
                    seen_docs.add(doc_id)
                    pi_meta = self._registry.get(doc_id, {})
                    pi_doc_id = pi_meta.get("pi_doc_id")
                    if pi_doc_id:
                        try:
                            structure = client.get_document_structure(pi_doc_id)
                            hit["tree_structure"] = structure
                        except Exception:
                            pass
                enriched.append(hit)
            return enriched
        except Exception:
            return hits

    def ask(self, question: str, n_results: int = 5) -> dict[str, Any]:
        """
        End-to-end RAG: retrieve context → generate answer via Ollama.

        Returns {"answer": str, "sources": list[dict]}
        """
        hits = self.search(question, n_results=n_results)

        # Build context from retrieved chunks
        context_parts: list[str] = []
        sources: list[dict[str, Any]] = []
        for i, hit in enumerate(hits):
            filename = hit["metadata"].get("filename", "unknown")
            chunk_idx = hit["metadata"].get("chunk_index", "?")
            score = hit.get("score", 0)
            text = hit["text"]
            context_parts.append(
                f"[Source {i+1}: {filename}, chunk {chunk_idx}, relevance {score:.2f}]\n{text}"
            )
            sources.append({
                "filename": filename,
                "chunk_index": chunk_idx,
                "score": score,
                "snippet": text[:300],
            })

        context = "\n\n---\n\n".join(context_parts) if context_parts else ""
        if not context:
            return {
                "answer": "No relevant documents found. Please upload and index documents first.",
                "sources": [],
            }

        answer = self.ollama.ask_with_context(question, context)
        return {"answer": answer, "sources": sources}

    # ------------------------------------------------------------------
    # Registry queries
    # ------------------------------------------------------------------

    def list_documents(self) -> list[dict[str, Any]]:
        """Return metadata for all indexed documents."""
        return list(self._registry.values())

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        return self._registry.get(doc_id)

    def delete_document(self, doc_id: str) -> bool:
        """Remove a document from both stores."""
        if doc_id not in self._registry:
            return False
        self.vector_store.delete_document(doc_id)
        del self._registry[doc_id]
        _save_registry(self._registry)
        logger.info("Deleted doc_id=%s", doc_id)
        return True

    @property
    def document_count(self) -> int:
        return len(self._registry)

    @property
    def chunk_count(self) -> int:
        return self.vector_store.count

    @property
    def storage_size(self) -> int:
        """Total size of all indexed documents in bytes."""
        return sum(doc.get("size_bytes", 0) for doc in self._registry.values())

    def get_document_chunks(self, doc_id: str) -> list[dict[str, Any]]:
        """Retrieve all chunks for a document from ChromaDB."""
        results = self.vector_store._collection.get(
            where={"doc_id": doc_id},
            include=["documents", "metadatas"],
        )
        chunks: list[dict[str, Any]] = []
        docs = results.get("documents", [])
        metas = results.get("metadatas", [])
        for doc_text, meta in zip(docs, metas):
            chunks.append({
                "text": doc_text,
                "chunk_index": meta.get("chunk_index", 0),
            })
        chunks.sort(key=lambda c: c["chunk_index"])
        return chunks
