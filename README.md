<div align="center">

# 📚 PageIndex Enterprise Knowledge Portal

### Reasoning-Based RAG for Internal Document Intelligence

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io)
[![PageIndex](https://img.shields.io/badge/RAG-PageIndex-4A90D9.svg)](https://github.com/VectifyAI/PageIndex)
[![Ollama](https://img.shields.io/badge/LLM-Ollama-000000.svg)](https://ollama.com)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Docker Ready](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](Dockerfile)

<br/>

> **Ship an enterprise-grade knowledge portal in minutes.**  
> Upload internal documents → PageIndex builds a reasoning tree → Employees ask natural-language questions → Ollama generates grounded, cited answers — all running 100 % on-premises.

<br/>

```
┌──────────────┐    ┌───────────────┐    ┌─────────────────┐    ┌────────────┐
│  Employee    │───▶│  Streamlit    │───▶│  PageIndex      │───▶│  Ollama    │
│  (Browser)   │    │  UI           │    │  Tree Index +   │    │  LLM       │
│              │◀───│              │◀───│  ChromaDB       │◀───│  (Local)   │
└──────────────┘    └───────────────┘    └─────────────────┘    └────────────┘
```

</div>

---

## Table of Contents

| # | Section | Description |
|---|---------|-------------|
| 1 | [Overview](#1-overview) | What this project does and why it matters |
| 2 | [Architecture](#2-architecture) | System design and data flow |
| 3 | [Prerequisites](#3-prerequisites) | What you need before starting |
| 4 | [Quick Start](#4-quick-start) | Get running in under 5 minutes |
| 5 | [UI Snapshots & Walkthrough](#5-ui-snapshots--walkthrough) | Visual preview of every screen |
| 6 | [Step-by-Step Execution Guide](#6-step-by-step-execution-guide) | Detailed terminal + UI walkthrough |
| 7 | [Project Structure](#7-project-structure) | File-by-file walkthrough |
| 8 | [Configuration](#8-configuration) | Every knob you can turn |
| 9 | [Usage Guide](#9-usage-guide) | Step-by-step workflows |
| 10 | [API Reference](#10-api-reference) | Key classes and methods |
| 11 | [Docker Deployment](#11-docker-deployment) | Container-based production setup |
| 12 | [Testing](#12-testing) | How to run and extend the test suite |
| 13 | [Troubleshooting](#13-troubleshooting) | Common issues and fixes |
| 14 | [Production Checklist](#14-production-checklist) | Enterprise hardening guide |
| 15 | [Contributing](#15-contributing) | How to help improve this project |
| 16 | [License](#16-license) | MIT |

---

## 1. Overview

### The Problem

Enterprise teams drown in internal documentation — HR policies, compliance manuals, SOPs, training guides, technical runbooks. Finding the right answer means hunting through dozens of files. Traditional keyword search fails because employees ask *questions*, not *keywords*.

### The Solution

This portal combines two retrieval paradigms for maximum accuracy:

| Layer | Technology | How It Works |
|-------|-----------|--------------|
| **Primary** | [PageIndex](https://github.com/VectifyAI/PageIndex) | Builds a hierarchical "table-of-contents" tree from each document. At query time, an LLM reasons over the tree to locate the most relevant sections — no vector similarity, no chunking artefacts. |
| **Secondary** | [ChromaDB](https://www.trychroma.com/) | Classic dense-vector retrieval over overlapping text chunks. Provides fast semantic search and acts as a fallback when PageIndex tree search is unavailable. |

The LLM layer ([Ollama](https://ollama.com)) runs entirely on-premises — **zero data leaves your network**.

### Key Capabilities

- **Multi-format ingestion** — PDF, DOCX, TXT, Markdown
- **Dual retrieval** — Reasoning-based tree search + embedding-based vector search
- **Conversational Q&A** — Multi-turn chat grounded in your documents
- **Source citation** — Every answer links back to the originating file, section, and chunk
- **Export** — Save answers as Markdown reports
- **100 % local** — No cloud APIs, no data exfiltration

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Streamlit UI (app.py)                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │ Upload Panel  │  │ Search Tab   │  │ Chat Tab                 │  │
│  │ File Browser  │  │ Query + Hits │  │ Multi-turn Conversation  │  │
│  └──────┬───────┘  └──────┬───────┘  └────────────┬─────────────┘  │
│         │                 │                        │                │
└─────────┼─────────────────┼────────────────────────┼────────────────┘
          │                 │                        │
          ▼                 ▼                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│              Indexing Pipeline (indexing_pipeline.py)                │
│                                                                     │
│  ┌──────────────────┐    ┌──────────────────────────────────────┐   │
│  │ PageIndex Client  │    │ ChromaDB VectorStore                 │   │
│  │ • Tree indexing   │    │ • Chunk + embed + store              │   │
│  │ • Structure query │    │ • Cosine similarity search           │   │
│  └────────┬─────────┘    └───────────────┬──────────────────────┘   │
│           │                              │                          │
└───────────┼──────────────────────────────┼──────────────────────────┘
            │                              │
            ▼                              ▼
┌──────────────────────┐    ┌──────────────────────────────────────┐
│ PageIndex Workspace   │    │ Ollama Server (ollama_client.py)     │
│ (data/workspace/)     │    │ • Chat completion (streaming)        │
│ Tree JSON structures  │    │ • Embedding generation               │
└──────────────────────┘    │ • RAG answer synthesis               │
                            └──────────────────────────────────────┘
```

### Data Flow

1. **Upload** → File saved to `data/uploads/`, deduplicated by SHA-256
2. **Index** → PageIndex builds a reasoning tree; text is chunked and embedded into ChromaDB
3. **Query** → ChromaDB returns top-k chunks; PageIndex enriches with tree structure
4. **Generate** → Ollama synthesises an answer from retrieved context
5. **Present** → Streamlit renders the answer with source citations

---

## 3. Prerequisites

| Dependency | Version | Purpose |
|-----------|---------|---------|
| **Python** | 3.11+ | Runtime |
| **Ollama** | latest | Local LLM server |
| **Git** | any | Clone the repo |
| **Docker** *(optional)* | 24+ | Container deployment |

### Install Ollama

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Verify
ollama --version
```

### Pull Required Models

```bash
# LLM for Q&A and indexing
ollama pull llama3

# Embedding model for vector search
ollama pull nomic-embed-text
```

> **Tip:** You can substitute any Ollama model (`mistral`, `gemma`, `phi3`, `qwen2`) — just update `config.yaml`.

---

## 4. Quick Start

### Option A — Local (Recommended for Development)

```bash
# 1. Clone
git clone <your-repo-url> pageindex-wiki
cd pageindex-wiki

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start Ollama (in a separate terminal)
ollama serve

# 5. Launch the portal
streamlit run app.py
```

Open **http://localhost:8501** in your browser. Done.

### Option B — Docker Compose (Recommended for Production)

```bash
# Brings up Ollama + Wiki app
docker compose up -d

# Pull models inside the container
docker exec -it ollama ollama pull llama3
docker exec -it ollama ollama pull nomic-embed-text

# Open http://localhost:8501
```

---

## 5. UI Snapshots & Walkthrough

> Since this is a locally-deployed application, below are high-fidelity ASCII representations of each screen. Run the app to experience the full interactive UI with animations, hover effects, and responsive layout.

### 5.1 Landing Page — Portal Header & Dashboard

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                                                                                  │
│  ┌─ SIDEBAR ──────────────┐  ┌─ MAIN PANEL ────────────────────────────────────┐ │
│  │                        │  │                                                  │ │
│  │  📚 Enterprise Wiki    │  │  ┌──────────────────────────────────────────────┐ │ │
│  │  Reasoning-Based RAG   │  │  │  🔍 Enterprise Knowledge Portal             │ │ │
│  │  100% On-Premise       │  │  │                                              │ │ │
│  │                        │  │  │  Ask natural-language questions about your    │ │ │
│  │  ─────────────────     │  │  │  organisation's documents. PageIndex builds  │ │ │
│  │                        │  │  │  reasoning trees from your content.          │ │ │
│  │  ⚙️ System Status      │  │  │                                              │ │ │
│  │  ┌──────────────────┐  │  │  │  🌲 PageIndex RAG  🧠 Ollama LLM           │ │ │
│  │  │ Ollama  ● Online │  │  │  │  📊 ChromaDB       🔒 100% Local           │ │ │
│  │  │ 2 model(s):      │  │  │  └──────────────────────────────────────────────┘ │ │
│  │  │ llama3,          │  │  │                                                  │ │
│  │  │ nomic-embed-text │  │  │  ┌────────────┐ ┌──────────┐ ┌───────┐ ┌──────┐ │ │
│  │  │                  │  │  │  │📄 Documents│ │🧩 Chunks │ │🤖 LLM │ │🔧 Mod│ │ │
│  │  │ Documents│Chunks │  │  │  │     3      │ │    47    │ │Online  │ │llama3│ │ │
│  │  │     3    │  47   │  │  │  └────────────┘ └──────────┘ └───────┘ └──────┘ │ │
│  │  └──────────────────┘  │  │                                                  │ │
│  │                        │  │  ─────────────────────────────────────────────    │ │
│  │  ─────────────────     │  │                                                  │ │
│  │                        │  │  ┌──────┐  ┌──────┐  ┌──────┐                    │ │
│  │  📤 Upload Documents   │  │  │🔎 Sea│  │💬 Cha│  │ℹ️ Abo│     ◀─ Tabs       │ │
│  │  ┌──────────────────┐  │  │  │ rch  │  │  t   │  │  ut  │                    │ │
│  │  │  Drag & drop     │  │  │  └──────┘  └──────┘  └──────┘                    │ │
│  │  │  files here      │  │  │                                                  │ │
│  │  │  PDF DOCX TXT MD │  │  └──────────────────────────────────────────────────┘ │
│  │  └──────────────────┘  │                                                      │
│  │                        │                                                      │
│  │  ─────────────────     │                                                      │
│  │                        │                                                      │
│  │  📂 Indexed Documents  │                                                      │
│  │  ┌──────────────────┐  │                                                      │
│  │  │📄 hr_policy.txt  │  │                                                      │
│  │  │  15 chunks · 4KB │  │                                                      │
│  │  │  [🗑️ Remove]     │  │                                                      │
│  │  ├──────────────────┤  │                                                      │
│  │  │📄 compliance.md  │  │                                                      │
│  │  │  22 chunks · 8KB │  │                                                      │
│  │  │  [🗑️ Remove]     │  │                                                      │
│  │  ├──────────────────┤  │                                                      │
│  │  │📄 faq.txt        │  │                                                      │
│  │  │  10 chunks · 2KB │  │                                                      │
│  │  │  [🗑️ Remove]     │  │                                                      │
│  │  └──────────────────┘  │                                                      │
│  │                        │                                                      │
│  └────────────────────────┘                                                      │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Search Tab — Query & Answer with Source Citations

```
┌─ 🔎 Search Tab ──────────────────────────────────────────────────────────────┐
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────────┐ │
│  │ 🔍  What is the leave policy for new employees?                      │ │
│  └──────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌──────────────────────────┐  ┌──────────────┐                              │
│  │  🔍 Search Documents     │  │  📥 Export    │                              │
│  └──────────────────────────┘  └──────────────┘                              │
│                                                                              │
│  ── 📝 Answer ──────────────────────────────────────────────────────────     │
│  ┌──────────────────────────────────────────────────────────────────────────┐ │
│  │  New employees are entitled to the following leave benefits:             │ │
│  │                                                                          │ │
│  │  1. **Annual Leave**: 15 days per calendar year, accrued monthly at     │ │
│  │     1.25 days/month. Leave can be carried forward up to 5 days.         │ │
│  │  2. **Sick Leave**: 10 days per year. Medical certificate required      │ │
│  │     for absences exceeding 3 consecutive days.                          │ │
│  │  3. **Probation Period**: During the 90-day probation, leave accrues    │ │
│  │     but can only be taken with manager approval.                        │ │
│  │                                                                          │ │
│  │  All leave requests must be submitted through the HR portal at least    │ │
│  │  48 hours in advance for planned leave.                                 │ │
│  └──────────────────────────────────────────────────────────────────────────┘ │
│  ⏱️ Generated in 3.2s                                                        │
│                                                                              │
│  ── 📌 Sources Used ────────────────────────────────────────────────────     │
│  ┌──────────────────────────────────────────────────────────────────────────┐ │
│  │ ▎ Source 1: hr_policy.txt · chunk 3         ┌──────────┐               │ │
│  │ ▎                                            │ 94% match│               │ │
│  │ ▎ Annual leave entitlement for full-time     └──────────┘               │ │
│  │ ▎ employees is 15 working days per calendar                             │ │
│  │ ▎ year, prorated for the first year of…                                 │ │
│  └──────────────────────────────────────────────────────────────────────────┘ │
│  ┌──────────────────────────────────────────────────────────────────────────┐ │
│  │ ▎ Source 2: hr_policy.txt · chunk 5         ┌──────────┐               │ │
│  │ ▎                                            │ 87% match│               │ │
│  │ ▎ Sick leave policy: employees receive 10    └──────────┘               │ │
│  │ ▎ days of paid sick leave annually. A                                   │ │
│  │ ▎ medical certificate is required for…                                  │ │
│  └──────────────────────────────────────────────────────────────────────────┘ │
│  ┌──────────────────────────────────────────────────────────────────────────┐ │
│  │ ▎ Source 3: faq.txt · chunk 2               ┌──────────┐               │ │
│  │ ▎                                            │ 72% match│               │ │
│  │ ▎ Q: Can new employees take leave during     └──────────┘               │ │
│  │ ▎ probation? A: Yes, with prior approval…                              │ │
│  └──────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 5.3 Chat Tab — Multi-Turn Conversational Q&A

```
┌─ 💬 Chat Tab ────────────────────────────────────────────────────────────────┐
│                                                                              │
│  Have a multi-turn conversation grounded in your documents.                  │
│  Each response is backed by retrieved evidence.                              │
│                                                                              │
│  ┌─ 🧑 User ────────────────────────────────────────────────────────────┐   │
│  │  What are the compliance requirements for data handling?              │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─ 🤖 Assistant ────────────────────────────────────────────────────────┐  │
│  │  Based on the compliance manual, data handling requirements include:   │  │
│  │                                                                        │  │
│  │  1. **Classification**: All data must be classified as Public,        │  │
│  │     Internal, Confidential, or Restricted before processing.          │  │
│  │  2. **Encryption**: Confidential and Restricted data must be          │  │
│  │     encrypted at rest (AES-256) and in transit (TLS 1.2+).           │  │
│  │  3. **Access Control**: Role-based access with quarterly reviews.     │  │
│  │  4. **Retention**: Follow the retention schedule in Appendix B.       │  │
│  │                                                                        │  │
│  │  ▸ 📌 3 source(s) used                                               │  │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─ 🧑 User ────────────────────────────────────────────────────────────┐   │
│  │  What happens if an employee violates these policies?                 │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─ 🤖 Assistant ────────────────────────────────────────────────────────┐  │
│  │  According to the compliance manual, policy violations follow a       │  │
│  │  progressive discipline process:                                      │  │
│  │                                                                        │  │
│  │  • **First offense**: Written warning + mandatory retraining          │  │
│  │  • **Second offense**: Suspension (1-5 days) + performance review     │  │
│  │  • **Severe violations**: Immediate termination + legal referral      │  │
│  │                                                                        │  │
│  │  ▸ 📌 2 source(s) used                                               │  │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────────┐ │
│  │  Ask a question about your documents…                                    │ │
│  └──────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌──────────────────────────────────────┐                                    │
│  │  🧹 Clear Conversation               │                                    │
│  └──────────────────────────────────────┘                                    │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 5.4 About Tab — Architecture & Configuration

```
┌─ ℹ️ About Tab ───────────────────────────────────────────────────────────────┐
│                                                                              │
│  ── How It Works ────────────────────────────────────────────────────────    │
│                                                                              │
│  ┌──────────┬─────────────┬───────────────────────────────────────────────┐  │
│  │  Layer   │ Technology  │ Approach                                      │  │
│  ├──────────┼─────────────┼───────────────────────────────────────────────┤  │
│  │ Primary  │ PageIndex   │ Hierarchical reasoning tree + LLM navigation │  │
│  │ Secondary│ ChromaDB    │ Dense-vector search over text chunks          │  │
│  └──────────┴─────────────┴───────────────────────────────────────────────┘  │
│                                                                              │
│  ── Architecture ────────────────────────────────────────────────────────    │
│                                                                              │
│  ┌──────────────┐   ┌──────────────┐   ┌────────────────┐   ┌───────────┐  │
│  │  Employee    │──▶│  Streamlit   │──▶│  PageIndex     │──▶│  Ollama   │  │
│  │  (Browser)   │   │  UI          │   │  + ChromaDB    │   │  LLM      │  │
│  │              │◀──│              │◀──│                │◀──│  (Local)  │  │
│  └──────────────┘   └──────────────┘   └────────────────┘   └───────────┘  │
│                                                                              │
│  ── Configuration ───────────────────────────────────────────────────────    │
│                                                                              │
│  {                                                                           │
│    "LLM Model":       "llama3",                                              │
│    "Embedding Model": "nomic-embed-text",                                    │
│    "Chunk Size":       512,                                                  │
│    "Chunk Overlap":    64,                                                   │
│    "Temperature":      0.3                                                   │
│  }                                                                           │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 5.5 Document Upload — Progress & Notifications

```
┌─ Sidebar: Upload in Progress ────────────────────────────────────────────────┐
│                                                                              │
│  📤 Upload Documents                                                         │
│  ┌──────────────────────────────────┐                                        │
│  │  📎 hr_policy.txt               │                                        │
│  │  📎 compliance_manual.md        │                                        │
│  │  📎 faq.txt                     │                                        │
│  └──────────────────────────────────┘                                        │
│  3 file(s) selected                                                          │
│                                                                              │
│  ┌──────────────────────────────────┐                                        │
│  │  🔄 Index All Files              │  ◀─ Primary action button              │
│  └──────────────────────────────────┘                                        │
│                                                                              │
│  Indexing hr_policy.txt (1/3)                                                │
│  ████████████░░░░░░░░░░░░░░ 33%                                              │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐     │
│  │  ✅ hr_policy.txt — 15 chunks                                       │     │
│  └──────────────────────────────────────────────────────────────────────┘     │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Step-by-Step Execution Guide

> Complete walkthrough from zero to a working portal with answers.

### Step 1: Clone & Set Up Environment

```bash
$ git clone https://github.com/<your-username>/pageindex-enterprise-wiki.git
$ cd pageindex-enterprise-wiki

$ python3 -m venv .venv
$ source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate           # Windows

$ pip install -r requirements.txt
```

**Expected output:**
```
Successfully installed chromadb-0.6.3 streamlit-1.56.0 pageindex-0.2.8 ...
```

### Step 2: Install & Start Ollama

```bash
# Install Ollama (if not already installed)
$ curl -fsSL https://ollama.com/install.sh | sh

# Pull required models
$ ollama pull llama3
$ ollama pull nomic-embed-text

# Start the Ollama server (keep this terminal open)
$ ollama serve
```

**Expected output:**
```
Listening on 127.0.0.1:11434
```

### Step 3: Verify Ollama is Running

```bash
# In a new terminal:
$ curl http://localhost:11434/api/tags | python3 -m json.tool
```

**Expected output:**
```json
{
  "models": [
    { "name": "llama3:latest", "size": 4661211808 },
    { "name": "nomic-embed-text:latest", "size": 274302450 }
  ]
}
```

### Step 4: Run the Test Suite

```bash
$ source .venv/bin/activate
$ python -m pytest tests/ -v
```

**Expected output:**
```
tests/test_indexing_pipeline.py::TestVectorStore::test_add_and_search     PASSED
tests/test_indexing_pipeline.py::TestVectorStore::test_delete_document    PASSED
tests/test_indexing_pipeline.py::TestRegistry::test_load_save_roundtrip   PASSED
tests/test_indexing_pipeline.py::TestRegistry::test_load_missing_returns_empty  PASSED
tests/test_indexing_pipeline.py::TestPageIndexPipeline::test_ingest_txt_file   PASSED
tests/test_indexing_pipeline.py::TestPageIndexPipeline::test_duplicate_detection PASSED
tests/test_indexing_pipeline.py::TestPageIndexPipeline::test_list_and_delete    PASSED
tests/test_indexing_pipeline.py::TestPageIndexPipeline::test_ingest_missing_file_raises PASSED
tests/test_ollama_client.py::TestOllamaAvailability::test_is_available_true     PASSED
tests/test_ollama_client.py::TestOllamaAvailability::test_is_available_false_on_error PASSED
tests/test_ollama_client.py::TestOllamaAvailability::test_list_models           PASSED
tests/test_ollama_client.py::TestOllamaChat::test_chat_returns_content          PASSED
tests/test_ollama_client.py::TestOllamaChat::test_chat_with_system_prompt       PASSED
tests/test_ollama_client.py::TestOllamaEmbed::test_embed_returns_vector         PASSED
tests/test_ollama_client.py::TestOllamaEmbed::test_embed_batch                  PASSED
tests/test_ollama_client.py::TestOllamaRAG::test_ask_with_context               PASSED
tests/test_utils.py::TestLoadConfig::test_loads_yaml                            PASSED
tests/test_utils.py::TestLoadConfig::test_missing_file_raises                   PASSED
... (17 more tests)
======================== 34 passed in 1.4s ========================
```

### Step 5: Launch the Portal

```bash
$ streamlit run app.py
```

**Expected output:**
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

### Step 6: Upload Sample Documents

1. Open **http://localhost:8501** in your browser
2. In the sidebar, expand the **📤 Upload Documents** section
3. Drag the three sample files from `sample_docs/`:
   - `hr_policy.txt` — HR leave & benefits policy
   - `compliance_manual.md` — Regulatory compliance guide
   - `faq.txt` — Employee frequently asked questions
4. Click **🔄 Index All Files**
5. Watch the progress bar — each file is saved, tree-indexed, chunked, and embedded

**What happens behind the scenes:**
```
[INFO]  Ingesting: hr_policy.txt
[INFO]  → File hash: a3f7c2... (dedup check passed)
[INFO]  → Exported to Markdown → data/workspace/hr_policy.md
[INFO]  → PageIndex tree built (reasoning hierarchy)
[INFO]  → 15 chunks embedded into ChromaDB
[INFO]  → Registered in doc_registry.json
```

### Step 7: Ask Your First Question

1. Switch to the **🔎 Search** tab
2. Type: `What is the leave policy for new employees?`
3. Click **🔍 Search Documents**
4. The portal will:
   - Retrieve top-5 relevant chunks from ChromaDB
   - Send context + question to Ollama (llama3)
   - Display a grounded answer with source citations
5. Click **📥 Export** to save the Q&A as a Markdown file in `outputs/`

### Step 8: Start a Conversation

1. Switch to the **💬 Chat** tab
2. Ask: `What are the compliance requirements for data handling?`
3. Follow up: `What happens if someone violates these policies?`
4. Each answer cites specific document sections with match scores

### Step 9: Verify Exported Results

```bash
$ ls outputs/
search_2024-01-15_14-30-25.md

$ cat outputs/search_*.md
```

**Expected output:**
```markdown
# Search Results

**Query:** What is the leave policy for new employees?
**Generated:** 2024-01-15 14:30:25

## Answer

New employees are entitled to 15 days of annual leave...

## Sources

| # | File | Chunk | Score |
|---|------|-------|-------|
| 1 | hr_policy.txt | 3 | 0.94 |
| 2 | hr_policy.txt | 5 | 0.87 |
| 3 | faq.txt | 2 | 0.72 |
```

---

## 7. Project Structure

```
pageindex-wiki/
│
├── app.py                    # Streamlit UI — sidebar, search, chat
├── indexing_pipeline.py      # PageIndex tree + ChromaDB vector pipeline
├── ollama_client.py          # Ollama REST API wrapper (chat, embed, stream)
├── utils.py                  # Config, logging, parsing, chunking, export
├── config.yaml               # All runtime configuration (single source of truth)
│
├── requirements.txt          # Pinned Python dependencies
├── Dockerfile                # Multi-stage production container
├── docker-compose.yml        # Full stack: Ollama + Wiki app
├── .dockerignore
├── .gitignore
│
├── .streamlit/
│   └── config.toml           # Streamlit theme customisation
│
├── sample_docs/              # Example enterprise documents
│   ├── hr_policy.txt         # HR policy manual
│   ├── compliance_manual.md  # Compliance & regulatory guide
│   └── faq.txt               # Employee FAQ
│
├── outputs/                  # Exported search results (Markdown)
│   └── .gitkeep
│
├── tests/                    # Pytest test suite
│   ├── __init__.py
│   ├── test_utils.py
│   ├── test_ollama_client.py
│   └── test_indexing_pipeline.py
│
└── data/                     # Runtime data (gitignored)
    ├── uploads/              # Raw uploaded files
    ├── indexes/              # Document registry JSON
    ├── workspace/            # PageIndex tree structures
    └── chromadb/             # ChromaDB persistence
```

### Module Responsibilities

| Module | Responsibility | Key Classes/Functions |
|--------|---------------|----------------------|
| `app.py` | UI rendering, session state, file upload | `main()`, `_render_sidebar()`, `_render_main()` |
| `indexing_pipeline.py` | Document ingestion, retrieval orchestration | `PageIndexPipeline`, `VectorStore` |
| `ollama_client.py` | LLM communication (chat, embed, stream) | `OllamaClient` |
| `utils.py` | Config, parsing, chunking, export, logging | `load_config()`, `extract_text()`, `chunk_text()`, `export_to_markdown()` |
| `config.yaml` | Centralised configuration | — |

---

## 8. Configuration

All settings live in `config.yaml`. **No hardcoded paths or model names exist in the codebase.**

```yaml
# LLM / Ollama
ollama:
  base_url: "http://localhost:11434"
  model: "llama3"                      # Q&A generation model
  embedding_model: "nomic-embed-text"  # Vector embedding model
  temperature: 0.3                     # Lower = more deterministic
  timeout: 120                         # Request timeout (seconds)

# PageIndex (self-hosted)
pageindex:
  workspace: "data/workspace"
  index_model: "ollama/llama3"         # LiteLLM model string
  retrieve_model: "ollama/llama3"

# ChromaDB
chromadb:
  persist_directory: "data/chromadb"
  collection_name: "wiki_documents"
  chunk_size: 512                      # Words per chunk
  chunk_overlap: 64                    # Overlap between chunks

# Storage
storage:
  upload_dir: "data/uploads"
  index_dir: "data/indexes"
  output_dir: "outputs"
```

### Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `OLLAMA_HOST` | Override Ollama URL (Docker) | `http://localhost:11434` |
| `OLLAMA_API_BASE` | Used by PageIndex/LiteLLM | `http://localhost:11434` |

---

## 9. Usage Guide

### 9.1 Upload Documents

1. Open the sidebar in the Streamlit UI.
2. Drag-and-drop or click to upload PDF, DOCX, TXT, or Markdown files.
3. Click **"Index Uploaded Files"**.
4. Watch the progress bar — each file is:
   - Saved to `data/uploads/`
   - Indexed by PageIndex (reasoning tree)
   - Chunked and embedded into ChromaDB

### 9.2 Search

1. Switch to the **🔎 Search** tab.
2. Type a natural-language question (e.g., *"What is the leave policy for new employees?"*).
3. Click **Search**.
4. The system:
   - Retrieves the top-5 most relevant chunks from ChromaDB
   - Enriches results with PageIndex tree structure
   - Sends context + question to Ollama
   - Displays the answer with source citations

### 9.3 Chat

1. Switch to the **💬 Chat** tab.
2. Ask follow-up questions in a conversational flow.
3. Each response is grounded in document context with expandable source references.

### 9.4 Export

- After a search, click **"📥 Export to Markdown"** to save the Q&A to `outputs/`.
- Files are timestamped and include the question, answer, and source metadata.

### 9.5 Index Sample Documents

```bash
# The repo ships with sample enterprise docs:
# sample_docs/hr_policy.txt
# sample_docs/compliance_manual.md
# sample_docs/faq.txt
#
# Upload them via the UI or copy to data/uploads/ and re-index.
```

---

## 10. API Reference

### `OllamaClient`

```python
from ollama_client import OllamaClient

client = OllamaClient()                      # loads from config.yaml
client.is_available()                         # → bool
client.list_models()                          # → ["llama3", "mistral", ...]
client.chat(messages, model=None, system_prompt=None)  # → str
client.chat_stream(messages, ...)             # → Generator[str]
client.embed(text)                            # → list[float]
client.embed_batch(texts)                     # → list[list[float]]
client.ask_with_context(question, context)    # → str (RAG answer)
```

### `PageIndexPipeline`

```python
from indexing_pipeline import PageIndexPipeline

pipeline = PageIndexPipeline()
metadata = pipeline.ingest("/path/to/document.pdf")   # → dict
results  = pipeline.search("leave policy", n_results=5)  # → list[dict]
answer   = pipeline.ask("What is the leave policy?")     # → {"answer": str, "sources": [...]}
docs     = pipeline.list_documents()                      # → list[dict]
pipeline.delete_document(doc_id)                          # → bool
```

### `VectorStore`

```python
from indexing_pipeline import VectorStore

store = VectorStore()
store.add_document(doc_id, text, metadata)  # → int (chunk count)
store.search(query, n_results=5)            # → list[dict]
store.delete_document(doc_id)
store.count                                 # → int
```

---

## 11. Docker Deployment

### Build & Run (Standalone)

```bash
# Build the image
docker build -t pageindex-wiki .

# Run (assumes Ollama is running on host)
docker run -d \
  --name pageindex-wiki \
  -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/outputs:/app/outputs \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  pageindex-wiki
```

### Docker Compose (Full Stack)

```bash
# Start everything
docker compose up -d

# Pull models
docker exec -it ollama ollama pull llama3
docker exec -it ollama ollama pull nomic-embed-text

# View logs
docker compose logs -f wiki

# Stop
docker compose down
```

### GPU Support (NVIDIA)

The `docker-compose.yml` includes GPU reservation for Ollama. Ensure the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) is installed.

For CPU-only deployments, remove the `deploy.resources` block from `docker-compose.yml`.

---

## 12. Testing

### Run All Tests

```bash
# From project root
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=. --cov-report=term-missing
```

### Test Categories

| File | Tests | What It Covers |
|------|-------|----------------|
| `test_utils.py` | 18 | Config loading, text extraction, chunking, hashing, metadata, export |
| `test_ollama_client.py` | 8 | Health checks, chat, streaming, embeddings, RAG helper |
| `test_indexing_pipeline.py` | 8 | VectorStore CRUD, registry persistence, pipeline ingestion, dedup |

**Total: 34 tests** — all passing. External services (Ollama, PageIndex) are mocked — tests run offline without GPU.

---

## 13. Troubleshooting

### Ollama Not Running

```
Error: Connection refused — http://localhost:11434
```

**Fix:** Start the Ollama server:

```bash
ollama serve
```

### Model Not Found

```
Error: model "llama3" not found
```

**Fix:** Pull the model:

```bash
ollama pull llama3
ollama pull nomic-embed-text
```

### PageIndex Import Error

```
Warning: pageindex package not installed — tree indexing disabled.
```

**Fix:** Install PageIndex:

```bash
pip install pageindex
```

> The portal **still works** without PageIndex — it falls back to ChromaDB-only vector search.

### ChromaDB Permission Error

```
Error: [Errno 13] Permission denied: 'data/chromadb'
```

**Fix:** Ensure the `data/` directory is writable:

```bash
chmod -R 755 data/
```

### Large File Upload Timeout

**Fix:** Increase the max upload size in `config.yaml`:

```yaml
streamlit:
  max_upload_mb: 500
```

And in `.streamlit/config.toml`:

```toml
[server]
maxUploadSize = 500
```

### Docker: Ollama Connection Refused

**Fix:** When running in Docker, use the service name:

```yaml
environment:
  - OLLAMA_HOST=http://ollama:11434
```

Or for host Ollama:

```
-e OLLAMA_HOST=http://host.docker.internal:11434
```

---

## 14. Production Checklist

### Security

- [ ] Run Ollama behind a firewall — do not expose port 11434 to the internet
- [ ] Enable Streamlit authentication (`streamlit-authenticator` or reverse proxy)
- [ ] Use HTTPS via a reverse proxy (Nginx, Caddy, Traefik)
- [ ] Set restrictive file permissions on `data/` directory
- [ ] Sanitise uploaded filenames (handled by the pipeline)
- [ ] Review `config.yaml` — remove any test values

### Performance

- [ ] Use GPU-accelerated Ollama for production workloads
- [ ] Tune `chunk_size` and `chunk_overlap` for your document types
- [ ] Consider `mistral` or `phi3` for faster inference on CPU
- [ ] Monitor ChromaDB collection size — consider sharding at 100k+ chunks
- [ ] Enable Streamlit caching (`@st.cache_data`) for repeated queries

### Observability

- [ ] Review `data/app.log` for indexing and query events
- [ ] Set `logging.level: DEBUG` in `config.yaml` for troubleshooting
- [ ] Add Prometheus metrics (Streamlit supports custom components)
- [ ] Monitor Ollama memory usage — large models require 8+ GB VRAM

### Scaling

- [ ] **Horizontal**: Deploy multiple Streamlit instances behind a load balancer
- [ ] **Storage**: Mount `data/` on a shared volume (NFS, EFS) for multi-instance
- [ ] **LLM**: Run Ollama on a dedicated GPU server; point `base_url` to it
- [ ] **Embeddings**: Pre-compute embeddings in batch for large document sets

### Backup

- [ ] Back up `data/indexes/doc_registry.json` (document metadata)
- [ ] Back up `data/chromadb/` (vector store)
- [ ] Back up `data/workspace/` (PageIndex tree structures)

---

## 15. Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m 'Add my feature'`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a Pull Request

### Code Standards

- Python 3.11+ with type hints
- Format with `black` and lint with `ruff`
- All new features must include tests
- Config-driven — no hardcoded values

---

## 16. License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

<div align="center">

**Built by an AI Architect** · Powered by [PageIndex](https://github.com/VectifyAI/PageIndex) · [Ollama](https://ollama.com) · [Streamlit](https://streamlit.io) · [ChromaDB](https://www.trychroma.com/)

</div>
