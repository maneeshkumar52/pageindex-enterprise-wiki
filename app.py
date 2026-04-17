"""
app.py — Enterprise Knowledge Portal

Professional Streamlit UI for document intelligence powered by
PageIndex (reasoning-based RAG) + Ollama (local LLM) + ChromaDB (vector search).

Features:
  • Dashboard with real-time system metrics and document statistics
  • Multi-strategy document search with relevance scoring
  • Conversational RAG chat grounded in indexed documents
  • Analytics dashboard with document insights and query tracking
  • Document explorer with chunk-level inspection
  • PDF and Markdown export of Q&A results
  • Real-time system health monitoring
"""

from __future__ import annotations

import os
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from indexing_pipeline import PageIndexPipeline
from ollama_client import OllamaClient
from utils import export_to_markdown, export_to_pdf, load_config, setup_logging, ensure_dirs

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

cfg = load_config()
logger = setup_logging(cfg)
ensure_dirs(cfg)

OUTPUT_DIR = cfg.get("storage", {}).get("output_dir", "outputs")
MAX_UPLOAD_MB = cfg.get("streamlit", {}).get("max_upload_mb", 200)

st.set_page_config(
    page_title=cfg.get("streamlit", {}).get("page_title", "PageIndex Enterprise Wiki"),
    page_icon=cfg.get("streamlit", {}).get("page_icon", "📚"),
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — Modern production-grade theme
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    .block-container { padding-top: 1.5rem; }

    /* ── Header ── */
    .portal-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #fff; padding: 2.5rem 3rem; border-radius: 20px;
        margin-bottom: 1.5rem; position: relative; overflow: hidden;
        box-shadow: 0 10px 40px rgba(102,126,234,0.3);
    }
    .portal-header::before {
        content: ''; position: absolute; top: -50%; right: -30%;
        width: 80%; height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 60%);
    }
    .portal-header h1 {
        margin: 0 0 0.4rem 0; font-size: 2.2rem;
        font-weight: 800; letter-spacing: -0.5px;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .portal-header p {
        margin: 0; opacity: 0.9; font-size: 1.02rem;
        line-height: 1.6; max-width: 700px;
    }
    .tech-badges { margin-top: 1rem; display: flex; gap: 0.6rem; flex-wrap: wrap; }
    .tech-badge {
        background: rgba(255,255,255,0.18); border: 1px solid rgba(255,255,255,0.3);
        color: #fff; padding: 5px 14px; border-radius: 25px;
        font-size: 0.78rem; font-weight: 600; letter-spacing: 0.3px;
    }

    /* ── Metric cards ── */
    div[data-testid="stMetric"] {
        background: white; border: 1px solid #e8ecf4; border-radius: 14px;
        padding: 16px 20px; box-shadow: 0 2px 12px rgba(0,0,0,0.04);
    }
    div[data-testid="stMetric"] label {
        color: #8899aa !important; font-weight: 700 !important;
        text-transform: uppercase; font-size: 0.68rem !important; letter-spacing: 0.8px;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #1a1a2e !important; font-weight: 800 !important;
    }

    /* ── Source cards ── */
    .source-card {
        background: white; border: 1px solid #e8ecf4;
        border-left: 5px solid #667eea;
        padding: 1rem 1.4rem; margin: 0.6rem 0;
        border-radius: 0 12px 12px 0; line-height: 1.6;
        transition: all 0.2s ease; box-shadow: 0 1px 6px rgba(0,0,0,0.03);
    }
    .source-card:hover {
        transform: translateX(4px); box-shadow: 0 4px 16px rgba(0,0,0,0.06);
        border-left-color: #764ba2;
    }
    .source-card .source-title { font-weight: 700; color: #1a1a2e; font-size: 0.92rem; }
    .source-card .score-badge {
        display: inline-block; background: linear-gradient(135deg, #667eea, #764ba2);
        color: white; padding: 2px 10px; border-radius: 12px;
        font-size: 0.72rem; font-weight: 700; margin-left: 6px;
    }
    .source-card .snippet { color: #5a6577; font-size: 0.85rem; margin-top: 6px; }

    /* ── Relevance bar ── */
    .rel-bar-bg {
        background: #f0f2f6; border-radius: 6px; height: 6px;
        margin: 6px 0 4px 0; overflow: hidden;
    }
    .rel-bar {
        height: 100%; border-radius: 6px;
        background: linear-gradient(90deg, #667eea, #764ba2);
    }

    /* ── Status badges ── */
    .status-badge {
        display: inline-flex; align-items: center; gap: 5px;
        padding: 4px 14px; border-radius: 25px;
        font-size: 0.78rem; font-weight: 700;
    }
    .badge-ok  { background: #d4edda; color: #155724; }
    .badge-err { background: #f8d7da; color: #721c24; }

    /* ── File type badges ── */
    .file-badge {
        display: inline-block; padding: 2px 10px; border-radius: 8px;
        font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #fafbff 0%, #f0f2f8 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown h1 {
        font-size: 1.4rem; color: #1a1a2e; font-weight: 800;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px; background: #f8f9fc; border-radius: 12px; padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px; padding: 10px 20px; font-weight: 700;
    }
    .stTabs [aria-selected="true"] {
        background: white !important; box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
    }

    /* ── Chat messages ── */
    .stChatMessage {
        border-radius: 14px; box-shadow: 0 1px 6px rgba(0,0,0,0.04);
        border: 1px solid #f0f2f6;
    }

    /* ── Answer container ── */
    .answer-container {
        background: linear-gradient(135deg, #f8f9ff, #f0f4ff);
        border: 1px solid #d4dff7; border-radius: 16px;
        padding: 1.5rem 2rem; margin: 0.5rem 0 1rem 0;
        line-height: 1.8; box-shadow: 0 2px 12px rgba(102,126,234,0.06);
    }

    /* ── Footer ── */
    .portal-footer {
        text-align: center; color: #8899aa; font-size: 0.78rem;
        padding: 2.5rem 0 1.5rem 0; border-top: 2px solid #f0f2f6; margin-top: 3rem;
    }
    .portal-footer a { color: #667eea; text-decoration: none; font-weight: 600; }
    .portal-footer a:hover { text-decoration: underline; }

    /* ── Empty state ── */
    .empty-state { text-align: center; padding: 4rem 2rem; color: #8899aa; }
    .empty-state .icon { font-size: 4rem; margin-bottom: 1rem; }
    .empty-state h3 {
        color: #5a6577; font-weight: 700; margin-bottom: 0.5rem; font-size: 1.3rem;
    }

    /* ── Doc cards ── */
    .doc-card {
        background: white; border: 1px solid #e8ecf4; border-radius: 12px;
        padding: 0.8rem 1.2rem; margin: 0.5rem 0;
        transition: all 0.2s ease; box-shadow: 0 1px 4px rgba(0,0,0,0.03);
    }
    .doc-card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.06); border-color: #667eea; }

    /* ── Pipeline flow diagram ── */
    .flow-diagram {
        display: flex; align-items: center; justify-content: center; gap: 6px;
        padding: 1.2rem; background: white; border-radius: 14px;
        border: 1px solid #e8ecf4; margin: 0.8rem 0; flex-wrap: wrap;
    }
    .flow-step {
        text-align: center; padding: 10px 14px;
        background: linear-gradient(135deg, #f8f9ff, #f0f4ff);
        border-radius: 10px; font-size: 0.8rem; font-weight: 600;
        color: #1a1a2e; min-width: 70px; border: 1px solid #e8ecf4;
    }
    .flow-arrow { color: #667eea; font-size: 1.1rem; font-weight: 700; }

    /* ── Tech stack cards ── */
    .tech-card {
        background: white; border: 1px solid #e8ecf4; border-radius: 14px;
        padding: 1.2rem; text-align: center; transition: all 0.2s;
    }
    .tech-card:hover { border-color: #667eea; box-shadow: 0 4px 20px rgba(102,126,234,0.1); }
    .tech-card strong { display: block; color: #1a1a2e; font-size: 0.95rem; margin: 0.3rem 0; }
    .tech-card p { color: #8899aa; font-size: 0.78rem; margin: 0; }

    /* ── Chunk viewer ── */
    .chunk-card {
        background: #f8f9fc; border: 1px solid #e8ecf4; border-radius: 10px;
        padding: 0.8rem 1rem; margin: 0.4rem 0; font-size: 0.85rem; line-height: 1.5;
    }
    .chunk-header { font-weight: 700; color: #667eea; font-size: 0.78rem; margin-bottom: 4px; }

    /* ── Query history ── */
    .query-item {
        background: white; border: 1px solid #e8ecf4; border-radius: 10px;
        padding: 0.5rem 0.8rem; margin: 0.3rem 0; font-size: 0.82rem;
    }
    .query-item .q-time { color: #aab4c0; font-size: 0.7rem; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _init_state() -> None:
    defaults: dict[str, Any] = {
        "pipeline": None,
        "ollama": None,
        "chat_history": [],
        "last_answer": None,
        "query_history": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()

# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------

def get_pipeline() -> PageIndexPipeline:
    if st.session_state["pipeline"] is None:
        st.session_state["pipeline"] = PageIndexPipeline(cfg)
    return st.session_state["pipeline"]


def get_ollama() -> OllamaClient:
    if st.session_state["ollama"] is None:
        st.session_state["ollama"] = OllamaClient(cfg)
    return st.session_state["ollama"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FILE_ICONS = {".pdf": "📕", ".docx": "📘", ".txt": "📄", ".md": "📝"}
_FILE_COLORS = {".pdf": "#e74c3c", ".docx": "#2980b9", ".txt": "#27ae60", ".md": "#8e44ad"}


def _file_icon(ext: str) -> str:
    return _FILE_ICONS.get(ext, "📄")


def _file_color(ext: str) -> str:
    return _FILE_COLORS.get(ext, "#95a5a6")


def _fmt_size(b: int) -> str:
    if b < 1024:
        return f"{b} B"
    if b < 1048576:
        return f"{b / 1024:.1f} KB"
    return f"{b / 1048576:.1f} MB"


def _time_ago(iso_ts: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_ts)
        delta = datetime.now(timezone.utc) - dt
        if delta.days > 0:
            return f"{delta.days}d ago"
        h = delta.seconds // 3600
        if h > 0:
            return f"{h}h ago"
        m = delta.seconds // 60
        return f"{m}m ago" if m > 0 else "just now"
    except (ValueError, TypeError):
        return "—"


def _track_query(query: str, elapsed: float, n_sources: int) -> None:
    st.session_state["query_history"].append({
        "query": query,
        "time": elapsed,
        "sources": n_sources,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


def _get_suggestions(docs: list[dict[str, Any]]) -> list[str]:
    if not docs:
        return []
    suggestions: list[str] = []
    for doc in docs[:3]:
        stem = Path(doc.get("filename", "doc")).stem.replace("_", " ").replace("-", " ")
        suggestions.append(f"Key points in {stem}?")
    if len(docs) > 1:
        suggestions.append("Compare all documents")
    return suggestions[:4]

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("# 📚 Enterprise Wiki")
        st.caption("Reasoning-Based RAG · 100% On-Premise")
        st.divider()

        # ── Upload ──
        st.markdown("### 📤 Upload Documents")
        uploaded = st.file_uploader(
            "Upload",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=True,
            help=f"Max {MAX_UPLOAD_MB} MB · PDF, DOCX, TXT, Markdown",
            label_visibility="collapsed",
        )

        if uploaded:
            st.caption(f"**{len(uploaded)}** file(s) selected")
            if st.button("🔄 Index All Files", use_container_width=True, type="primary"):
                pipeline = get_pipeline()
                bar = st.progress(0, text="Preparing…")
                ok = 0
                for i, uf in enumerate(uploaded):
                    bar.progress(
                        i / len(uploaded),
                        text=f"Indexing **{uf.name}** ({i + 1}/{len(uploaded)})",
                    )
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=Path(uf.name).suffix
                    ) as tmp:
                        tmp.write(uf.read())
                        tmp_path = tmp.name
                    try:
                        meta = pipeline.ingest(tmp_path)
                        st.toast(f"✅ {uf.name} — {meta.get('num_chunks', 0)} chunks", icon="✅")
                        ok += 1
                    except Exception as exc:
                        st.error(f"**{uf.name}**: {exc}", icon="❌")
                        logger.exception("Indexing failed: %s", uf.name)
                    finally:
                        os.unlink(tmp_path)
                bar.progress(1.0, text="Complete!")
                if ok:
                    st.success(f"Indexed **{ok}/{len(uploaded)}** files.")
                    time.sleep(0.8)
                    st.rerun()

        st.divider()

        # ── Indexed Documents ──
        st.markdown("### 📂 Indexed Documents")
        pipeline = get_pipeline()
        docs = pipeline.list_documents()
        if docs:
            for doc in docs:
                fn = doc.get("filename", "N/A")
                ext = doc.get("extension", "")
                chunks = doc.get("num_chunks", 0)
                size = doc.get("size_bytes", 0)
                doc_id = doc.get("doc_id", "")
                icon = _file_icon(ext)
                color = _file_color(ext)
                when = _time_ago(doc.get("indexed_at", ""))

                st.markdown(
                    f'<div class="doc-card">'
                    f'{icon} <strong>{fn}</strong> '
                    f'<span class="file-badge" style="background:{color}20;color:{color};">'
                    f'{ext.lstrip(".")}</span><br/>'
                    f'<span style="color:#8899aa;font-size:0.78rem;">'
                    f'{chunks} chunks · {_fmt_size(size)} · {when}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if st.button("🗑️ Remove", key=f"del_{doc_id}", use_container_width=True):
                    pipeline.delete_document(doc_id)
                    st.toast(f"Removed {fn}", icon="🗑️")
                    st.rerun()
        else:
            st.info("No documents indexed yet. Upload files above.", icon="📄")

        # ── Quick Actions ──
        if docs:
            st.divider()
            if st.button("🗑️ Clear All Documents", use_container_width=True):
                for doc in docs:
                    pipeline.delete_document(doc.get("doc_id", ""))
                st.toast("All documents removed", icon="🗑️")
                st.rerun()

        st.divider()

        # ── System Status ──
        with st.expander("⚙️ System Status", expanded=False):
            ollama = get_ollama()
            ok = ollama.is_available()
            if ok:
                st.markdown(
                    '**Ollama** <span class="status-badge badge-ok">● Online</span>',
                    unsafe_allow_html=True,
                )
                models = ollama.list_models()
                if models:
                    st.caption(f"Models: {', '.join(models[:6])}")
            else:
                st.markdown(
                    '**Ollama** <span class="status-badge badge-err">● Offline</span>',
                    unsafe_allow_html=True,
                )
                st.error("Run `ollama serve` to start.", icon="⚠️")
            c1, c2 = st.columns(2)
            c1.metric("Docs", pipeline.document_count)
            c2.metric("Chunks", pipeline.chunk_count)
            st.metric("Storage", _fmt_size(pipeline.storage_size))

        # ── Recent Queries ──
        qh = st.session_state.get("query_history", [])
        if qh:
            with st.expander(f"🕒 Recent Queries ({len(qh)})", expanded=False):
                for q in reversed(qh[-10:]):
                    when = _time_ago(q["timestamp"])
                    st.markdown(
                        f'<div class="query-item">'
                        f'🔍 {q["query"][:60]}{"…" if len(q["query"]) > 60 else ""}<br/>'
                        f'<span class="q-time">{q["time"]:.1f}s · '
                        f'{q["sources"]} sources · {when}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        st.divider()
        st.caption(
            "Built with [PageIndex](https://github.com/VectifyAI/PageIndex) · "
            "[Ollama](https://ollama.com) · [Streamlit](https://streamlit.io)"
        )

# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

def _render_main() -> None:
    # ── Header ──
    st.markdown("""
    <div class="portal-header">
        <h1>🔍 Enterprise Knowledge Portal</h1>
        <p>Ask natural-language questions about your organisation's documents.
        PageIndex builds reasoning trees — Ollama generates grounded, cited answers
        entirely on-premise.</p>
        <div class="tech-badges">
            <span class="tech-badge">🌲 PageIndex RAG</span>
            <span class="tech-badge">🧠 Ollama LLM</span>
            <span class="tech-badge">📊 ChromaDB Vectors</span>
            <span class="tech-badge">🔒 100% Local</span>
            <span class="tech-badge">📄 Multi-Format</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Metrics ribbon ──
    pipeline = get_pipeline()
    ollama = get_ollama()
    ollama_ok = ollama.is_available()
    qh = st.session_state.get("query_history", [])

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("📄 Documents", pipeline.document_count)
    m2.metric("🧩 Chunks", pipeline.chunk_count)
    m3.metric("🤖 LLM", "Online" if ollama_ok else "Offline")
    m4.metric("🔧 Model", cfg.get("ollama", {}).get("model", "—"))
    m5.metric("💾 Storage", _fmt_size(pipeline.storage_size))
    m6.metric("🔍 Queries", len(qh))

    st.divider()

    # ── Tabs ──
    tab_search, tab_chat, tab_analytics, tab_docs, tab_about = st.tabs(
        ["🔎 Search", "💬 Chat", "📊 Analytics", "📄 Documents", "ℹ️ About"]
    )

    # ───────────────────────── SEARCH TAB ─────────────────────────
    with tab_search:
        if pipeline.document_count == 0:
            st.markdown(
                '<div class="empty-state"><div class="icon">📤</div>'
                '<h3>No Documents Indexed</h3>'
                '<p>Upload documents using the sidebar to start searching.</p></div>',
                unsafe_allow_html=True,
            )
        else:
            all_docs = pipeline.list_documents()
            suggestions = _get_suggestions(all_docs)

            # Suggested questions
            trigger: str | None = None
            if suggestions:
                st.caption("💡 **Quick Questions** — click to search instantly")
                scols = st.columns(min(len(suggestions), 4))
                for i, (col, s) in enumerate(zip(scols, suggestions)):
                    with col:
                        if st.button(s, key=f"suggest_{i}", use_container_width=True):
                            trigger = s

            # Search input + controls
            query = st.text_input(
                "Search",
                placeholder="e.g. What is our leave policy for new employees?",
                key="search_input",
                label_visibility="collapsed",
            )
            effective_query = trigger if trigger else query

            col_btn, col_n = st.columns([4, 1])
            search_clicked = col_btn.button(
                "🔍 Search Documents", type="primary", use_container_width=True
            )
            n_results = col_n.number_input(
                "Sources", 1, 10, 5, key="n_results"
            )

            if (search_clicked or trigger) and effective_query:
                if not ollama_ok:
                    st.error("Ollama is offline. Run `ollama serve`.", icon="⚠️")
                else:
                    with st.spinner("🔍 Searching documents and generating answer…"):
                        t0 = time.time()
                        result = pipeline.ask(effective_query, n_results=n_results)
                        elapsed = time.time() - t0

                    answer = result["answer"]
                    sources = result["sources"]
                    _track_query(effective_query, elapsed, len(sources))
                    st.session_state["last_answer"] = {
                        "query": effective_query,
                        "answer": answer,
                        "sources": sources,
                        "time": elapsed,
                    }

            # Display persisted result
            res = st.session_state.get("last_answer")
            if res:
                st.markdown(
                    f"⏱️ **{res['time']:.1f}s** &nbsp;·&nbsp; "
                    f"📌 **{len(res['sources'])}** sources &nbsp;·&nbsp; "
                    f"📊 **{pipeline.chunk_count}** chunks searched"
                )

                st.markdown("#### 📝 Answer")
                st.markdown(
                    f'<div class="answer-container">{res["answer"]}</div>',
                    unsafe_allow_html=True,
                )

                # Export buttons
                ex1, ex2 = st.columns(2)
                with ex1:
                    if st.button("📥 Export Markdown", key="exp_md", use_container_width=True):
                        p = export_to_markdown(
                            res["query"], res["answer"], res["sources"], OUTPUT_DIR
                        )
                        st.success(f"Saved to `{p}`", icon="📥")
                with ex2:
                    if st.button("📄 Export PDF", key="exp_pdf", use_container_width=True):
                        p = export_to_pdf(
                            res["query"], res["answer"], res["sources"], OUTPUT_DIR
                        )
                        st.success(f"Saved to `{p}`", icon="📄")

                # Sources with relevance bars
                if res["sources"]:
                    st.markdown("#### 📌 Sources")
                    for i, src in enumerate(res["sources"], 1):
                        pct = src["score"] * 100
                        fn = src["filename"]
                        ext = Path(fn).suffix.lower()
                        icon = _file_icon(ext)
                        st.markdown(
                            f'<div class="source-card">'
                            f'<span class="source-title">{icon} Source {i}: {fn}</span>'
                            f' · chunk {src["chunk_index"]}'
                            f' <span class="score-badge">{pct:.0f}%</span>'
                            f'<div class="rel-bar-bg">'
                            f'<div class="rel-bar" style="width:{pct}%;"></div></div>'
                            f'<div class="snippet">{src["snippet"][:300]}…</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

    # ───────────────────────── CHAT TAB ─────────────────────────
    with tab_chat:
        if pipeline.document_count == 0:
            st.markdown(
                '<div class="empty-state"><div class="icon">💬</div>'
                '<h3>Start a Conversation</h3>'
                '<p>Upload and index documents first, then chat here.</p></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "Multi-turn conversation grounded in your documents. "
                "Every response is backed by retrieved evidence."
            )

            # Display history
            for msg in st.session_state["chat_history"]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if msg.get("sources"):
                        with st.expander(f"📌 {len(msg['sources'])} source(s)"):
                            for src in msg["sources"]:
                                pct = src.get("score", 0) * 100
                                st.caption(
                                    f"**{src['filename']}** · chunk {src['chunk_index']}"
                                    f" · {pct:.0f}%"
                                )

            # Chat input
            if prompt := st.chat_input("Ask about your documents…"):
                if not ollama_ok:
                    st.error("Ollama is offline.", icon="⚠️")
                else:
                    st.session_state["chat_history"].append(
                        {"role": "user", "content": prompt}
                    )
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    with st.chat_message("assistant"):
                        with st.spinner("Thinking…"):
                            t0 = time.time()
                            result = pipeline.ask(prompt)
                            elapsed = time.time() - t0
                        answer = result["answer"]
                        sources = result["sources"]
                        _track_query(prompt, elapsed, len(sources))
                        st.markdown(answer)
                        if sources:
                            with st.expander(f"📌 {len(sources)} source(s)"):
                                for src in sources:
                                    pct = src.get("score", 0) * 100
                                    st.caption(
                                        f"**{src['filename']}** · "
                                        f"chunk {src['chunk_index']} · {pct:.0f}%"
                                    )

                    st.session_state["chat_history"].append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )

            if st.session_state["chat_history"]:
                if st.button("🧹 Clear Conversation", use_container_width=True):
                    st.session_state["chat_history"] = []
                    st.rerun()

    # ───────────────────────── ANALYTICS TAB ─────────────────────────
    with tab_analytics:
        all_docs = pipeline.list_documents()
        if not all_docs:
            st.markdown(
                '<div class="empty-state"><div class="icon">📊</div>'
                '<h3>No Data Yet</h3>'
                '<p>Index documents to see analytics.</p></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown("#### 📊 Document Insights")

            # Summary metrics
            total_chunks = sum(d.get("num_chunks", 0) for d in all_docs)
            total_size = sum(d.get("size_bytes", 0) for d in all_docs)
            avg_chunks = total_chunks / len(all_docs)
            types = Counter(
                d.get("extension", "").lstrip(".").upper() or "?"
                for d in all_docs
            )

            a1, a2, a3, a4 = st.columns(4)
            a1.metric("Total Documents", len(all_docs))
            a2.metric("Total Chunks", total_chunks)
            a3.metric("Avg Chunks/Doc", f"{avg_chunks:.1f}")
            a4.metric("Total Storage", _fmt_size(total_size))

            st.divider()

            # Charts row 1
            ch1, ch2 = st.columns(2)
            with ch1:
                st.markdown("##### File Type Distribution")
                type_df = pd.DataFrame(
                    [{"Type": k, "Count": v} for k, v in types.items()]
                )
                donut = (
                    alt.Chart(type_df)
                    .mark_arc(innerRadius=50, outerRadius=100, cornerRadius=4)
                    .encode(
                        theta=alt.Theta("Count:Q"),
                        color=alt.Color(
                            "Type:N",
                            scale=alt.Scale(scheme="tableau10"),
                        ),
                        tooltip=["Type", "Count"],
                    )
                    .properties(height=280)
                )
                st.altair_chart(donut, use_container_width=True)

            with ch2:
                st.markdown("##### Chunks per Document")
                chunk_df = pd.DataFrame(
                    [
                        {
                            "Document": d.get("filename", "?")[:25],
                            "Chunks": d.get("num_chunks", 0),
                        }
                        for d in all_docs
                    ]
                )
                bar = (
                    alt.Chart(chunk_df)
                    .mark_bar(
                        cornerRadiusTopLeft=6,
                        cornerRadiusTopRight=6,
                        color="#667eea",
                    )
                    .encode(
                        x=alt.X("Document:N", sort="-y", axis=alt.Axis(labelAngle=-30)),
                        y="Chunks:Q",
                        tooltip=["Document", "Chunks"],
                    )
                    .properties(height=280)
                )
                st.altair_chart(bar, use_container_width=True)

            st.divider()

            # Document sizes
            st.markdown("##### Document Sizes")
            size_df = pd.DataFrame(
                [
                    {
                        "Document": d.get("filename", "?")[:25],
                        "Size (KB)": round(d.get("size_bytes", 0) / 1024, 1),
                    }
                    for d in all_docs
                ]
            )
            size_bar = (
                alt.Chart(size_df)
                .mark_bar(cornerRadiusEnd=6, color="#764ba2")
                .encode(
                    y=alt.Y("Document:N", sort="-x"),
                    x="Size (KB):Q",
                    tooltip=["Document", "Size (KB)"],
                )
                .properties(height=max(150, len(all_docs) * 45))
            )
            st.altair_chart(size_bar, use_container_width=True)

            # Query history table
            qh = st.session_state.get("query_history", [])
            if qh:
                st.divider()
                st.markdown("##### 🕒 Session Query Log")
                qh_df = pd.DataFrame(
                    [
                        {
                            "#": i + 1,
                            "Query": q["query"][:50],
                            "Time (s)": f'{q["time"]:.1f}',
                            "Sources": q["sources"],
                        }
                        for i, q in enumerate(qh)
                    ]
                )
                st.dataframe(qh_df, use_container_width=True, hide_index=True)

    # ───────────────────────── DOCUMENTS TAB ─────────────────────────
    with tab_docs:
        all_docs = pipeline.list_documents()
        if not all_docs:
            st.markdown(
                '<div class="empty-state"><div class="icon">📄</div>'
                '<h3>No Documents</h3>'
                '<p>Upload documents using the sidebar to explore them here.</p></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(f"#### 📄 Document Explorer — {len(all_docs)} document(s)")

            # Filter
            filter_q = st.text_input(
                "Filter",
                placeholder="🔎 Type to filter documents…",
                key="doc_filter",
                label_visibility="collapsed",
            )
            filtered = (
                [d for d in all_docs if filter_q.lower() in d.get("filename", "").lower()]
                if filter_q
                else all_docs
            )

            for doc in filtered:
                fn = doc.get("filename", "N/A")
                ext = doc.get("extension", "")
                icon = _file_icon(ext)
                color = _file_color(ext)
                chunks_n = doc.get("num_chunks", 0)
                size = doc.get("size_bytes", 0)
                doc_id = doc.get("doc_id", "")
                indexed = doc.get("indexed_at", "")
                sha = doc.get("sha256", "")[:12]

                with st.expander(f"{icon} {fn}", expanded=False):
                    # Metadata
                    st.markdown("**Metadata**")
                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric("Size", _fmt_size(size))
                    mc2.metric("Chunks", chunks_n)
                    mc3.metric("Indexed", _time_ago(indexed))
                    st.caption(
                        f"**Type:** {ext.lstrip('.')} &nbsp;·&nbsp; "
                        f"**Hash:** `{sha}…` &nbsp;·&nbsp; "
                        f"**ID:** `{doc_id[:12]}…`"
                    )

                    # Chunks preview
                    try:
                        chunks = pipeline.get_document_chunks(doc_id)
                        if chunks:
                            st.markdown(f"**Content Preview** ({len(chunks)} chunks)")
                            for chunk in chunks[:8]:
                                text = chunk["text"]
                                st.markdown(
                                    f'<div class="chunk-card">'
                                    f'<div class="chunk-header">Chunk {chunk["chunk_index"]}</div>'
                                    f'{text[:250]}{"…" if len(text) > 250 else ""}'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )
                            if len(chunks) > 8:
                                st.caption(f"Showing 8 of {len(chunks)} chunks")
                    except Exception:
                        st.caption("Chunk preview unavailable")

                    if st.button(
                        f"🗑️ Delete {fn}",
                        key=f"deldoc_{doc_id}",
                        use_container_width=True,
                    ):
                        pipeline.delete_document(doc_id)
                        st.toast(f"Removed {fn}", icon="🗑️")
                        st.rerun()

    # ───────────────────────── ABOUT TAB ─────────────────────────
    with tab_about:
        st.markdown("### How It Works")

        # Indexing pipeline flow
        st.markdown("**Indexing Pipeline**")
        st.markdown(
            '<div class="flow-diagram">'
            '<div class="flow-step">📄<br/>Upload</div><div class="flow-arrow">→</div>'
            '<div class="flow-step">📝<br/>Parse</div><div class="flow-arrow">→</div>'
            '<div class="flow-step">🧩<br/>Chunk</div><div class="flow-arrow">→</div>'
            '<div class="flow-step">🧮<br/>Embed</div><div class="flow-arrow">→</div>'
            '<div class="flow-step">🌲<br/>Index</div><div class="flow-arrow">→</div>'
            '<div class="flow-step">💾<br/>Store</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        # Query pipeline flow
        st.markdown("**Query Pipeline**")
        st.markdown(
            '<div class="flow-diagram">'
            '<div class="flow-step">❓<br/>Query</div><div class="flow-arrow">→</div>'
            '<div class="flow-step">🔍<br/>Search</div><div class="flow-arrow">→</div>'
            '<div class="flow-step">📊<br/>Rank</div><div class="flow-arrow">→</div>'
            '<div class="flow-step">🧠<br/>Generate</div><div class="flow-arrow">→</div>'
            '<div class="flow-step">📝<br/>Answer</div><div class="flow-arrow">→</div>'
            '<div class="flow-step">📌<br/>Cite</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.divider()

        st.markdown("### Dual Retrieval Architecture")
        st.markdown("""
| Layer | Technology | Approach |
|-------|-----------|----------|
| **Primary** | PageIndex | Hierarchical reasoning tree — LLM navigates the tree to locate relevant sections |
| **Secondary** | ChromaDB | Dense-vector cosine search over overlapping text chunks |
        """)

        st.code(
            "┌──────────────┐    ┌───────────────┐    ┌─────────────────┐    ┌────────────┐\n"
            "│  Browser     │───▶│  Streamlit    │───▶│  PageIndex      │───▶│  Ollama    │\n"
            "│  (Employee)  │    │  Portal       │    │  + ChromaDB     │    │  LLM       │\n"
            "│              │◀───│              │◀───│                 │◀───│  (Local)   │\n"
            "└──────────────┘    └───────────────┘    └─────────────────┘    └────────────┘",
            language=None,
        )

        st.divider()

        st.markdown("### 🛠️ Tech Stack")
        t1, t2, t3, t4 = st.columns(4)
        with t1:
            st.markdown(
                '<div class="tech-card"><div style="font-size:2rem;">🌲</div>'
                '<strong>PageIndex</strong><p>Reasoning tree RAG</p></div>',
                unsafe_allow_html=True,
            )
        with t2:
            st.markdown(
                '<div class="tech-card"><div style="font-size:2rem;">🧠</div>'
                '<strong>Ollama</strong><p>Local LLM inference</p></div>',
                unsafe_allow_html=True,
            )
        with t3:
            st.markdown(
                '<div class="tech-card"><div style="font-size:2rem;">📊</div>'
                '<strong>ChromaDB</strong><p>Vector embeddings</p></div>',
                unsafe_allow_html=True,
            )
        with t4:
            st.markdown(
                '<div class="tech-card"><div style="font-size:2rem;">🎯</div>'
                '<strong>Streamlit</strong><p>Interactive UI</p></div>',
                unsafe_allow_html=True,
            )

        st.divider()

        st.markdown("### Configuration")
        st.json({
            "LLM Model": cfg.get("ollama", {}).get("model", "—"),
            "Embedding Model": cfg.get("ollama", {}).get("embedding_model", "—"),
            "Chunk Size": cfg.get("chromadb", {}).get("chunk_size", 512),
            "Chunk Overlap": cfg.get("chromadb", {}).get("chunk_overlap", 64),
            "Temperature": cfg.get("ollama", {}).get("temperature", 0.3),
            "Max Upload (MB)": MAX_UPLOAD_MB,
        })

    # ── Footer ──
    st.markdown(
        '<div class="portal-footer">'
        'Enterprise Knowledge Portal · Powered by '
        '<a href="https://github.com/VectifyAI/PageIndex">PageIndex</a> · '
        '<a href="https://ollama.com">Ollama</a> · '
        '<a href="https://streamlit.io">Streamlit</a><br/>'
        '© 2024 Maneesh Kumar · MIT License · '
        '<a href="https://github.com/maneeshkumar52/pageindex-enterprise-wiki">GitHub</a>'
        '</div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    _render_sidebar()
    _render_main()


main()
