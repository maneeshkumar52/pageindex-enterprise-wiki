"""
app.py — Streamlit UI for the PageIndex Enterprise Knowledge Portal.

Provides:
  • Sidebar: document upload, indexed-file browser, system status.
  • Main panel: natural-language search bar, conversational chat,
    retrieved context display with source metadata, and Markdown export.
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Any

import streamlit as st

from indexing_pipeline import PageIndexPipeline
from ollama_client import OllamaClient
from utils import export_to_markdown, load_config, setup_logging, ensure_dirs

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

cfg = load_config()
logger = setup_logging(cfg)
ensure_dirs(cfg)

OUTPUT_DIR = cfg.get("storage", {}).get("output_dir", "outputs")
MAX_UPLOAD_MB = cfg.get("streamlit", {}).get("max_upload_mb", 200)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title=cfg.get("streamlit", {}).get("page_title", "PageIndex Enterprise Wiki"),
    page_icon=cfg.get("streamlit", {}).get("page_icon", "📚"),
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — Professional dark-accent theme
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    /* ── Global ── */
    .block-container { padding-top: 2rem; }
    
    /* ── Header ── */
    .portal-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        color: #ffffff;
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    .portal-header h1 {
        margin: 0 0 0.3rem 0;
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .portal-header p {
        margin: 0;
        opacity: 0.85;
        font-size: 1rem;
        line-height: 1.5;
    }
    .portal-header .tech-badges {
        margin-top: 0.8rem;
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
    }
    .portal-header .tech-badge {
        background: rgba(255,255,255,0.15);
        border: 1px solid rgba(255,255,255,0.25);
        color: #fff;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    
    /* ── Metric cards ── */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #f8f9ff, #eef1f8);
        border: 1px solid #e0e5f0;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    div[data-testid="stMetric"] label {
        color: #5a6577 !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        font-size: 0.7rem !important;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #1a1a2e !important;
        font-weight: 700 !important;
    }
    
    /* ── Source cards ── */
    .source-card {
        background: #f8f9fa;
        border-left: 4px solid #4A90D9;
        padding: 0.8rem 1.2rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.88rem;
        line-height: 1.5;
        transition: transform 0.15s ease;
    }
    .source-card:hover {
        transform: translateX(4px);
        background: #f0f4ff;
    }
    .source-card strong { color: #2a6cb6; }
    .source-card .score {
        display: inline-block;
        background: #e3f2fd;
        color: #1565c0;
        padding: 1px 8px;
        border-radius: 10px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 4px;
    }
    
    /* ── Status indicators ── */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .badge-ok  { background: #d4edda; color: #155724; }
    .badge-err { background: #f8d7da; color: #721c24; }
    
    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #fafbff 0%, #f0f2f8 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown h1 {
        font-size: 1.3rem;
        color: #1a1a2e;
    }
    
    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
        font-weight: 600;
    }
    
    /* ── Chat messages ── */
    .stChatMessage {
        border-radius: 12px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    
    /* ── Answer box ── */
    .answer-container {
        background: #f6fbff;
        border: 1px solid #d4e6f7;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 0.5rem 0 1rem 0;
        line-height: 1.7;
    }
    
    /* ── Footer ── */
    .portal-footer {
        text-align: center;
        color: #8899aa;
        font-size: 0.78rem;
        padding: 2rem 0 1rem 0;
        border-top: 1px solid #e8ecf1;
        margin-top: 3rem;
    }
    .portal-footer a { color: #4A90D9; text-decoration: none; }
    .portal-footer a:hover { text-decoration: underline; }
    
    /* ── Empty state ── */
    .empty-state {
        text-align: center;
        padding: 3rem 2rem;
        color: #8899aa;
    }
    .empty-state .icon { font-size: 3rem; margin-bottom: 0.5rem; }
    .empty-state h3 { color: #5a6577; margin-bottom: 0.5rem; }
    
    /* ── Doc card ── */
    .doc-card {
        background: white;
        border: 1px solid #e8ecf1;
        border-radius: 10px;
        padding: 0.7rem 1rem;
        margin: 0.4rem 0;
        transition: box-shadow 0.15s ease;
    }
    .doc-card:hover {
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------

def _init_state() -> None:
    defaults: dict[str, Any] = {
        "pipeline": None,
        "ollama": None,
        "chat_history": [],
        "search_results": None,
        "last_answer": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_state()


# ---------------------------------------------------------------------------
# Lazy singletons
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
# Sidebar
# ---------------------------------------------------------------------------

def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("# 📚 Enterprise Wiki")
        st.caption("Reasoning-Based RAG · 100% On-Premise")
        st.divider()

        # ---------- System status ----------
        with st.expander("⚙️ System Status", expanded=True):
            ollama = get_ollama()
            ollama_ok = ollama.is_available()
            
            if ollama_ok:
                st.markdown(
                    '**Ollama** &nbsp; <span class="status-badge badge-ok">● Online</span>',
                    unsafe_allow_html=True,
                )
                models = ollama.list_models()
                if models:
                    st.caption(f"**{len(models)} model(s):** {', '.join(models[:6])}")
            else:
                st.markdown(
                    '**Ollama** &nbsp; <span class="status-badge badge-err">● Offline</span>',
                    unsafe_allow_html=True,
                )
                st.error("Run `ollama serve` to start the LLM server.", icon="⚠️")

            pipeline = get_pipeline()
            c1, c2 = st.columns(2)
            c1.metric("Documents", pipeline.document_count)
            c2.metric("Chunks", pipeline.chunk_count)

        st.divider()

        # ---------- Upload ----------
        st.markdown("### 📤 Upload Documents")
        uploaded_files = st.file_uploader(
            "Drag & drop files here",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=True,
            help=f"Max {MAX_UPLOAD_MB} MB per file · PDF, DOCX, TXT, Markdown",
            label_visibility="collapsed",
        )

        if uploaded_files:
            st.caption(f"**{len(uploaded_files)}** file(s) selected")
            if st.button("🔄 Index All Files", use_container_width=True, type="primary"):
                pipeline = get_pipeline()
                progress = st.progress(0, text="Preparing…")
                success_count = 0
                for i, uf in enumerate(uploaded_files):
                    progress.progress(
                        (i) / len(uploaded_files),
                        text=f"Indexing **{uf.name}** ({i+1}/{len(uploaded_files)})",
                    )
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=Path(uf.name).suffix
                    ) as tmp:
                        tmp.write(uf.read())
                        tmp_path = tmp.name
                    try:
                        meta = pipeline.ingest(tmp_path)
                        st.toast(f"✅ {uf.name} — {meta.get('num_chunks', 0)} chunks", icon="✅")
                        success_count += 1
                        logger.info("Indexed via UI: %s", uf.name)
                    except Exception as exc:
                        st.error(f"**{uf.name}**: {exc}", icon="❌")
                        logger.exception("Indexing failed: %s", uf.name)
                    finally:
                        os.unlink(tmp_path)
                progress.progress(1.0, text="Complete!")
                if success_count > 0:
                    st.success(f"Indexed **{success_count}** of **{len(uploaded_files)}** files.")
                    time.sleep(1)
                    st.rerun()

        st.divider()

        # ---------- Indexed documents ----------
        st.markdown("### 📂 Indexed Documents")
        pipeline = get_pipeline()
        docs = pipeline.list_documents()
        if docs:
            for doc in docs:
                fn = doc.get("filename", "N/A")
                chunks = doc.get("num_chunks", 0)
                size_kb = doc.get("size_bytes", 0) / 1024
                doc_id = doc.get("doc_id", "")
                
                st.markdown(
                    f'<div class="doc-card">'
                    f'📄 <strong>{fn}</strong><br/>'
                    f'<span style="color:#8899aa;font-size:0.8rem;">'
                    f'{chunks} chunks · {size_kb:.1f} KB</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if st.button("🗑️ Remove", key=f"del_{doc_id}", use_container_width=True):
                    pipeline.delete_document(doc_id)
                    st.toast(f"Removed {fn}", icon="🗑️")
                    st.rerun()
        else:
            st.info("No documents indexed yet. Upload files above to get started.", icon="📄")

        st.divider()
        st.caption(
            "Built with [PageIndex](https://github.com/VectifyAI/PageIndex) · "
            "[Ollama](https://ollama.com) · "
            "[Streamlit](https://streamlit.io)"
        )


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

def _render_main() -> None:
    # ── Header ──
    st.markdown("""
    <div class="portal-header">
        <h1>🔍 Enterprise Knowledge Portal</h1>
        <p>Ask natural-language questions about your organisation's documents.<br/>
        PageIndex builds reasoning trees from your content. Ollama generates grounded, cited answers — entirely on-premise.</p>
        <div class="tech-badges">
            <span class="tech-badge">🌲 PageIndex RAG</span>
            <span class="tech-badge">🧠 Ollama LLM</span>
            <span class="tech-badge">📊 ChromaDB Vectors</span>
            <span class="tech-badge">🔒 100% Local</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Metrics ──
    pipeline = get_pipeline()
    ollama = get_ollama()
    ollama_ok = ollama.is_available()
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("📄 Documents", pipeline.document_count)
    m2.metric("🧩 Chunks", pipeline.chunk_count)
    m3.metric("🤖 LLM", "Online" if ollama_ok else "Offline")
    m4.metric("🔧 Model", cfg.get("ollama", {}).get("model", "N/A"))

    st.divider()

    # ── Tabs ──
    tab_search, tab_chat, tab_about = st.tabs(["🔎 Search", "💬 Chat", "ℹ️ About"])

    # ────────────────────── Search Tab ──────────────────────
    with tab_search:
        if pipeline.document_count == 0:
            st.markdown(
                '<div class="empty-state">'
                '<div class="icon">📤</div>'
                '<h3>No Documents Indexed</h3>'
                '<p>Upload documents using the sidebar to start searching.</p>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            query = st.text_input(
                "🔍 Ask a question",
                placeholder="e.g. What is our leave policy for new employees?",
                key="search_query",
                label_visibility="collapsed",
            )
            
            c_search, c_export = st.columns([3, 1])
            search_clicked = c_search.button(
                "🔍 Search Documents", type="primary", use_container_width=True
            )

            if search_clicked and query:
                if not ollama_ok:
                    st.error("Ollama is offline. Run `ollama serve` to start.", icon="⚠️")
                    return

                with st.spinner("🔍 Searching documents and generating answer…"):
                    start = time.time()
                    result = pipeline.ask(query)
                    elapsed = time.time() - start

                answer = result["answer"]
                sources = result["sources"]
                st.session_state["last_answer"] = {
                    "query": query, "answer": answer,
                    "sources": sources, "time": elapsed,
                }

            # ── Display persisted result ──
            if st.session_state.get("last_answer"):
                res = st.session_state["last_answer"]
                
                st.markdown("### 📝 Answer")
                st.markdown(
                    f'<div class="answer-container">{res["answer"]}</div>',
                    unsafe_allow_html=True,
                )
                st.caption(f"⏱️ Generated in {res['time']:.1f}s")

                if res["sources"]:
                    st.markdown("### 📌 Sources Used")
                    for i, src in enumerate(res["sources"], 1):
                        score_pct = src['score'] * 100
                        st.markdown(
                            f'<div class="source-card">'
                            f'<strong>Source {i}:</strong> {src["filename"]} '
                            f'· chunk {src["chunk_index"]} '
                            f'<span class="score">{score_pct:.0f}% match</span>'
                            f'<br/><span style="color:#555;">{src["snippet"][:250]}…</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                # Export button
                if c_export.button("📥 Export", use_container_width=True):
                    path = export_to_markdown(
                        res["query"], res["answer"], res["sources"], OUTPUT_DIR
                    )
                    st.success(f"Saved to `{path}`", icon="📥")

    # ────────────────────── Chat Tab ──────────────────────
    with tab_chat:
        if pipeline.document_count == 0:
            st.markdown(
                '<div class="empty-state">'
                '<div class="icon">💬</div>'
                '<h3>Start a Conversation</h3>'
                '<p>Upload and index documents first, then ask questions here.</p>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "Have a multi-turn conversation grounded in your documents. "
                "Each response is backed by retrieved evidence."
            )

            # Display history
            for msg in st.session_state["chat_history"]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if msg.get("sources"):
                        with st.expander(f"📌 {len(msg['sources'])} source(s) used"):
                            for src in msg["sources"]:
                                score_pct = src.get('score', 0) * 100
                                st.caption(
                                    f"**{src['filename']}** · chunk {src['chunk_index']} · "
                                    f"{score_pct:.0f}% match"
                                )

            # Chat input
            if prompt := st.chat_input("Ask a question about your documents…"):
                if not ollama_ok:
                    st.error("Ollama is offline. Run `ollama serve` to start.", icon="⚠️")
                    return

                # User message
                st.session_state["chat_history"].append(
                    {"role": "user", "content": prompt}
                )
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Assistant response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking…"):
                        result = pipeline.ask(prompt)
                    answer = result["answer"]
                    sources = result["sources"]
                    st.markdown(answer)
                    if sources:
                        with st.expander(f"📌 {len(sources)} source(s) used"):
                            for src in sources:
                                score_pct = src.get('score', 0) * 100
                                st.caption(
                                    f"**{src['filename']}** · chunk {src['chunk_index']} · "
                                    f"{score_pct:.0f}% match"
                                )

                st.session_state["chat_history"].append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )

            # Clear chat
            if st.session_state["chat_history"]:
                if st.button("🧹 Clear Conversation", use_container_width=True):
                    st.session_state["chat_history"] = []
                    st.rerun()

    # ────────────────────── About Tab ──────────────────────
    with tab_about:
        st.markdown("### How It Works")
        st.markdown("""
        This portal uses **two complementary retrieval strategies** for maximum accuracy:
        
        | Layer | Technology | Approach |
        |-------|-----------|----------|
        | **Primary** | PageIndex | Builds a hierarchical reasoning tree from each document. At query time, the LLM navigates the tree to find the most relevant sections. |
        | **Secondary** | ChromaDB | Classic dense-vector search over overlapping text chunks for fast semantic matching. |
        
        The **Ollama LLM** runs entirely on your local machine — no data ever leaves your network.
        """)
        
        st.markdown("### Architecture")
        st.code("""
┌──────────────┐    ┌───────────────┐    ┌─────────────────┐    ┌────────────┐
│  Employee    │───▶│  Streamlit    │───▶│  PageIndex      │───▶│  Ollama    │
│  (Browser)   │    │  UI           │    │  Tree Index +   │    │  LLM       │
│              │◀───│              │◀───│  ChromaDB       │◀───│  (Local)   │
└──────────────┘    └───────────────┘    └─────────────────┘    └────────────┘
        """, language=None)
        
        st.markdown("### Configuration")
        st.json({
            "LLM Model": cfg.get("ollama", {}).get("model", "N/A"),
            "Embedding Model": cfg.get("ollama", {}).get("embedding_model", "N/A"),
            "Chunk Size": cfg.get("chromadb", {}).get("chunk_size", 512),
            "Chunk Overlap": cfg.get("chromadb", {}).get("chunk_overlap", 64),
            "Temperature": cfg.get("ollama", {}).get("temperature", 0.3),
        })

    # ── Footer ──
    st.markdown(
        '<div class="portal-footer">'
        'Enterprise Knowledge Portal · Powered by '
        '<a href="https://github.com/VectifyAI/PageIndex">PageIndex</a> · '
        '<a href="https://ollama.com">Ollama</a> · '
        '<a href="https://streamlit.io">Streamlit</a><br/>'
        'Designed for on-premise enterprise knowledge management'
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
