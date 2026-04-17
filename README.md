<div align="center">

# 📚 PageIndex Enterprise Knowledge Portal

### Reasoning-Based RAG for Internal Document Intelligence

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io)
[![PageIndex](https://img.shields.io/badge/RAG-PageIndex-667eea.svg)](https://github.com/VectifyAI/PageIndex)
[![Ollama](https://img.shields.io/badge/LLM-Ollama-000000.svg)](https://ollama.com)
[![ChromaDB](https://img.shields.io/badge/Vectors-ChromaDB-764ba2.svg)](https://www.trychroma.com/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Docker Ready](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](Dockerfile)
[![Tests](https://img.shields.io/badge/tests-34%20passed-brightgreen.svg)](#12-testing)

<br/>

> **Ship an enterprise-grade knowledge portal in minutes.**  
> Upload internal documents → PageIndex builds a reasoning tree → Employees ask natural-language questions → Ollama generates grounded, cited answers — all running 100 % on-premises.

<br/>

```
┌──────────────┐    ┌───────────────┐    ┌─────────────────┐    ┌────────────┐
│  Employee    │───▶│  Streamlit    │───▶│  PageIndex      │───▶│  Ollama    │
│  (Browser)   │    │  Portal       │    │  + ChromaDB     │    │  LLM       │
│              │◀───│              │◀───│                 │◀───│  (Local)   │
└──────────────┘    └───────────────┘    └─────────────────┘    └────────────┘
```

</div>

---

## Table of Contents

| # | Section | Description |
|---|---------|-------------|
| 1 | [Overview](#1-overview) | Problem, solution, and key capabilities |
| 2 | [Architecture](#2-architecture) | System design and data flow |
| 3 | [Features at a Glance](#3-features-at-a-glance) | Complete feature inventory |
| 4 | [Prerequisites](#4-prerequisites) | What you need before starting |
| 5 | [Quick Start](#5-quick-start) | Get running in under 5 minutes |
| 6 | [UI Walkthrough & Snapshots](#6-ui-walkthrough--snapshots) | Visual preview of every screen and component |
| 7 | [Step-by-Step Execution Guide](#7-step-by-step-execution-guide) | Detailed terminal + UI walkthrough |
| 8 | [Project Structure](#8-project-structure) | File-by-file walkthrough |
| 9 | [Configuration](#9-configuration) | Every knob you can turn |
| 10 | [Usage Guide](#10-usage-guide) | Step-by-step workflows |
| 11 | [API Reference](#11-api-reference) | Key classes and methods |
| 12 | [Docker Deployment](#12-docker-deployment) | Container-based production setup |
| 13 | [Testing](#13-testing) | How to run and extend the test suite |
| 14 | [Troubleshooting](#14-troubleshooting) | Common issues and fixes |
| 15 | [Production Checklist](#15-production-checklist) | Enterprise hardening guide |
| 16 | [Contributing](#16-contributing) | How to help improve this project |
| 17 | [License](#17-license) | MIT |

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
- **Source citation** — Every answer links back to the originating file, section, and chunk with relevance scoring
- **Analytics dashboard** — Document insights, chunk distribution, query session logs
- **Document explorer** — Browse indexed documents with chunk-level content inspection
- **Dual export** — Save answers as Markdown reports or professional PDFs
- **Smart suggestions** — Auto-generated questions based on indexed document names
- **Query history** — Track all queries with response times and source counts
- **100 % local** — No cloud APIs, no data exfiltration

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Streamlit UI (app.py)                            │
│  ┌─────────────┐  ┌─────────┐  ┌────────┐  ┌────────┐  ┌───────────┐  │
│  │ Sidebar      │  │ Search  │  │ Chat   │  │Analyti-│  │ Document  │  │
│  │ • Upload     │  │ Tab     │  │ Tab    │  │ cs Tab │  │ Explorer  │  │
│  │ • Status     │  │         │  │        │  │        │  │ Tab       │  │
│  │ • Documents  │  │         │  │        │  │        │  │           │  │
│  │ • History    │  │         │  │        │  │        │  │           │  │
│  └──────┬──────┘  └────┬────┘  └───┬────┘  └───┬────┘  └─────┬─────┘  │
│         │              │           │            │             │         │
└─────────┼──────────────┼───────────┼────────────┼─────────────┼─────────┘
          │              │           │            │             │
          ▼              ▼           ▼            ▼             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              Indexing Pipeline (indexing_pipeline.py)                    │
│                                                                         │
│  ┌──────────────────┐    ┌──────────────────────────────────────┐       │
│  │ PageIndex Client  │    │ ChromaDB VectorStore                 │       │
│  │ • Tree indexing   │    │ • Chunk + embed + store              │       │
│  │ • Structure query │    │ • Cosine similarity search           │       │
│  │ • Doc navigation  │    │ • Chunk-level retrieval              │       │
│  └────────┬─────────┘    └───────────────┬──────────────────────┘       │
│           │                              │                              │
└───────────┼──────────────────────────────┼──────────────────────────────┘
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

```
Upload  →  Deduplicate (SHA-256)  →  Parse (PDF/DOCX/TXT/MD)
   ↓
Tree Index (PageIndex)  +  Chunk & Embed (ChromaDB)
   ↓
Registry (doc_registry.json)  →  Ready for queries
   ↓
Query  →  Dual Retrieval  →  LLM Synthesis  →  Cited Answer
   ↓
Export (Markdown / PDF)  +  Track (Query History)
```

---

## 3. Features at a Glance

| Category | Feature | Description |
|----------|---------|-------------|
| **Ingestion** | Multi-format upload | PDF, DOCX, TXT, Markdown via drag-and-drop |
| **Ingestion** | Duplicate detection | SHA-256 content hashing prevents re-indexing |
| **Ingestion** | Progress tracking | Real-time progress bar + toast notifications |
| **Retrieval** | PageIndex tree search | Reasoning-based hierarchical document navigation |
| **Retrieval** | ChromaDB vector search | Dense embedding cosine similarity retrieval |
| **Retrieval** | Configurable results | Adjust number of retrieved sources (1-10) |
| **Search** | Smart suggestions | Auto-generated questions from document names |
| **Search** | Relevance bars | Visual relevance scoring with gradient progress bars |
| **Search** | Source citations | File name, chunk index, and percentage match score |
| **Chat** | Multi-turn RAG | Conversational Q&A with document-grounded responses |
| **Chat** | Source expandability | Expandable source references in each message |
| **Chat** | Clear conversation | Reset chat history with one click |
| **Analytics** | File type distribution | Donut chart showing document type breakdown |
| **Analytics** | Chunks per document | Bar chart comparing document chunk counts |
| **Analytics** | Document sizes | Horizontal bar chart of document file sizes |
| **Analytics** | Query session log | Tabular log of all queries with timing and source counts |
| **Documents** | Document explorer | Browse all indexed documents with metadata |
| **Documents** | Chunk viewer | Inspect individual chunks within each document |
| **Documents** | Filter & search | Text filter to quickly find documents |
| **Documents** | Bulk operations | Delete individual or all documents |
| **Export** | Markdown export | Save Q&A to timestamped `.md` files |
| **Export** | PDF export | Professional PDF reports with headers and source listing |
| **System** | Health monitoring | Real-time Ollama status, model listing, storage metrics |
| **System** | Query history | Sidebar panel tracking recent queries with timestamps |
| **System** | 6-metric dashboard | Documents, chunks, LLM status, model, storage, query count |

---

## 4. Prerequisites

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

## 5. Quick Start

### Option A — Local (Recommended for Development)

```bash
# 1. Clone
git clone https://github.com/maneeshkumar52/pageindex-enterprise-wiki.git pageindex-wiki
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

## 6. UI Walkthrough & Snapshots

> Below are high-fidelity ASCII representations of every screen and component. Run the app to experience the full interactive UI with gradient themes, hover animations, responsive layout, and Altair charts.

### 6.1 Portal Landing — Header, Metrics Ribbon & Tab Navigation

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│                                                                                   │
│  ┌── SIDEBAR ────────────────┐  ┌── MAIN PANEL ───────────────────────────────┐  │
│  │                            │  │                                             │  │
│  │  📚 Enterprise Wiki       │  │  ╔═══════════════════════════════════════╗   │  │
│  │  Reasoning-Based RAG      │  │  ║  🔍 Enterprise Knowledge Portal      ║   │  │
│  │  100% On-Premise          │  │  ║                                       ║   │  │
│  │  ───────────────────────  │  │  ║  Ask natural-language questions about ║   │  │
│  │                            │  │  ║  your organisation's documents.      ║   │  │
│  │  📤 Upload Documents      │  │  ║                                       ║   │  │
│  │  ┌────────────────────┐   │  │  ║  🌲 PageIndex RAG  🧠 Ollama LLM    ║   │  │
│  │  │ Drag & drop files  │   │  │  ║  📊 ChromaDB       🔒 100% Local     ║   │  │
│  │  │ PDF DOCX TXT MD    │   │  │  ║  📄 Multi-Format                     ║   │  │
│  │  └────────────────────┘   │  │  ╚═══════════════════════════════════════╝   │  │
│  │                            │  │        ▲ Gradient purple header              │  │
│  │  ───────────────────────  │  │                                             │  │
│  │                            │  │  ┌─────────┐ ┌────────┐ ┌──────┐ ┌──────┐ │  │
│  │  📂 Indexed Documents     │  │  │📄 Docs  │ │🧩Chunk│ │🤖LLM│ │🔧Mod│ │  │
│  │  ┌────────────────────┐   │  │  │   3     │ │  47   │ │Onlin│ │llam3│ │  │
│  │  │📕 hr_policy.txt    │   │  │  └─────────┘ └────────┘ └──────┘ └──────┘ │  │
│  │  │  txt  15 ch · 4 KB │   │  │  ┌─────────┐ ┌────────┐                    │  │
│  │  │  5m ago            │   │  │  │💾 Store │ │🔍Query│   ◀─ 6 metric      │  │
│  │  │  [🗑️ Remove]       │   │  │  │ 14.2 KB │ │   5   │      cards         │  │
│  │  ├────────────────────┤   │  │  └─────────┘ └────────┘                    │  │
│  │  │📝 compliance.md    │   │  │                                             │  │
│  │  │  md  22 ch · 8 KB  │   │  │  ─────────────────────────────────────────  │  │
│  │  │  5m ago            │   │  │                                             │  │
│  │  │  [🗑️ Remove]       │   │  │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──┐│  │
│  │  ├────────────────────┤   │  │  │🔎 Se-│ │💬 Ch-│ │📊 An-│ │📄 Do-│ │ℹ️││  │
│  │  │📄 faq.txt          │   │  │  │ arch │ │  at  │ │alyti│ │ cs   │ │Ab││  │
│  │  │  txt  10 ch · 2 KB │   │  │  └──────┘ └──────┘ └──────┘ └──────┘ └──┘│  │
│  │  │  5m ago            │   │  │  ▲ 5 tabs (was 3)                          │  │
│  │  │  [🗑️ Remove]       │   │  │                                             │  │
│  │  └────────────────────┘   │  └─────────────────────────────────────────────┘  │
│  │                            │                                                   │
│  │  [🗑️ Clear All Documents]│                                                   │
│  │                            │                                                   │
│  │  ───────────────────────  │                                                   │
│  │  ⚙️ System Status ▸      │                                                   │
│  │  🕒 Recent Queries (5) ▸ │                                                   │
│  │  ───────────────────────  │                                                   │
│  │  Built with PageIndex ·   │                                                   │
│  │  Ollama · Streamlit       │                                                   │
│  └────────────────────────────┘                                                   │
└───────────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Search Tab — Smart Suggestions, Query, Answer & Source Cards

```
┌─ 🔎 Search Tab ──────────────────────────────────────────────────────────────────┐
│                                                                                   │
│  💡 Quick Questions — click to search instantly                                   │
│  ┌───────────────────┐ ┌───────────────────┐ ┌─────────────────────┐ ┌──────────┐│
│  │Key points in      │ │Key points in      │ │Key points in faq?  │ │Compare   ││
│  │hr policy?         │ │compliance manual? │ │                    │ │all docs  ││
│  └───────────────────┘ └───────────────────┘ └─────────────────────┘ └──────────┘│
│                                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────────┐ │
│  │ 🔍  What is the leave policy for new employees?                              │ │
│  └──────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                   │
│  ┌──────────────────────────────────────────────────────────┐  ┌─────────────┐   │
│  │  🔍 Search Documents                                     │  │ Sources: 5  │   │
│  └──────────────────────────────────────────────────────────┘  └─────────────┘   │
│                                                                                   │
│  ⏱️ 3.2s  ·  📌 3 sources  ·  📊 47 chunks searched                              │
│                                                                                   │
│  ── 📝 Answer ──────────────────────────────────────────────────────────────────  │
│  ╔══════════════════════════════════════════════════════════════════════════════╗  │
│  ║  New employees are entitled to the following leave benefits:                ║  │
│  ║                                                                            ║  │
│  ║  1. **Annual Leave**: 15 days per calendar year, accrued monthly at        ║  │
│  ║     1.25 days/month. Leave can be carried forward up to 5 days.            ║  │
│  ║  2. **Sick Leave**: 10 days per year. Medical certificate required         ║  │
│  ║     for absences exceeding 3 consecutive days.                             ║  │
│  ║  3. **Probation Period**: During the 90-day probation, leave accrues       ║  │
│  ║     but can only be taken with manager approval.                           ║  │
│  ╚══════════════════════════════════════════════════════════════════════════════╝  │
│                                                                                   │
│  ┌──────────────────────────────┐  ┌──────────────────────────────┐              │
│  │  📥 Export Markdown           │  │  📄 Export PDF                │              │
│  └──────────────────────────────┘  └──────────────────────────────┘              │
│                                                                                   │
│  ── 📌 Sources ─────────────────────────────────────────────────────────────────  │
│  ┌──────────────────────────────────────────────────────────────────────────────┐ │
│  │ ▎ 📄 Source 1: hr_policy.txt · chunk 3                          ┌────────┐  │ │
│  │ ▎                                                                │  94%  │  │ │
│  │ ▎ ████████████████████████████████████████████████████░░░░░░     └────────┘  │ │
│  │ ▎ Annual leave entitlement for full-time employees is 15 working             │ │
│  │ ▎ days per calendar year, prorated for the first year of…                    │ │
│  └──────────────────────────────────────────────────────────────────────────────┘ │
│  ┌──────────────────────────────────────────────────────────────────────────────┐ │
│  │ ▎ 📄 Source 2: hr_policy.txt · chunk 5                          ┌────────┐  │ │
│  │ ▎                                                                │  87%  │  │ │
│  │ ▎ ██████████████████████████████████████████████░░░░░░░░░░      └────────┘  │ │
│  │ ▎ Sick leave policy: employees receive 10 days of paid sick                  │ │
│  │ ▎ leave annually. Medical certificate required for…                          │ │
│  └──────────────────────────────────────────────────────────────────────────────┘ │
│  ┌──────────────────────────────────────────────────────────────────────────────┐ │
│  │ ▎ 📄 Source 3: faq.txt · chunk 2                                ┌────────┐  │ │
│  │ ▎                                                                │  72%  │  │ │
│  │ ▎ █████████████████████████████████░░░░░░░░░░░░░░░░░░░░░░░      └────────┘  │ │
│  │ ▎ Q: Can new employees take leave during probation?                          │ │
│  │ ▎ A: Yes, with prior approval from your direct manager…                     │ │
│  └──────────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### 6.3 Chat Tab — Multi-Turn Conversational Q&A

```
┌─ 💬 Chat Tab ────────────────────────────────────────────────────────────────────┐
│                                                                                   │
│  Multi-turn conversation grounded in your documents. Every response is            │
│  backed by retrieved evidence.                                                    │
│                                                                                   │
│  ┌─ 🧑 User ─────────────────────────────────────────────────────────────────┐   │
│  │  What are the compliance requirements for data handling?                   │   │
│  └───────────────────────────────────────────────────────────────────────────┘   │
│                                                                                   │
│  ┌─ 🤖 Assistant ─────────────────────────────────────────────────────────────┐  │
│  │  Based on the compliance manual, data handling requirements include:        │  │
│  │                                                                             │  │
│  │  1. **Classification**: All data must be classified as Public,             │  │
│  │     Internal, Confidential, or Restricted before processing.               │  │
│  │  2. **Encryption**: Confidential and Restricted data must be               │  │
│  │     encrypted at rest (AES-256) and in transit (TLS 1.2+).                │  │
│  │  3. **Access Control**: Role-based access with quarterly reviews.          │  │
│  │  4. **Retention**: Follow the retention schedule in Appendix B.            │  │
│  │                                                                             │  │
│  │  ▸ 📌 3 source(s) ──────────────────────────────────────                   │  │
│  │  │ compliance_manual.md · chunk 4 · 91%                                    │  │
│  │  │ compliance_manual.md · chunk 7 · 85%                                    │  │
│  │  │ hr_policy.txt · chunk 12 · 68%                                          │  │
│  │  └──────────────────────────────────────────────────────                    │  │
│  └───────────────────────────────────────────────────────────────────────────┘   │
│                                                                                   │
│  ┌─ 🧑 User ─────────────────────────────────────────────────────────────────┐   │
│  │  What happens if an employee violates these policies?                      │   │
│  └───────────────────────────────────────────────────────────────────────────┘   │
│                                                                                   │
│  ┌─ 🤖 Assistant ─────────────────────────────────────────────────────────────┐  │
│  │  According to the compliance manual, policy violations follow a            │  │
│  │  progressive discipline process:                                           │  │
│  │                                                                             │  │
│  │  • **First offense**: Written warning + mandatory retraining               │  │
│  │  • **Second offense**: Suspension (1–5 days) + performance review          │  │
│  │  • **Severe violations**: Immediate termination + legal referral           │  │
│  │                                                                             │  │
│  │  ▸ 📌 2 source(s)                                                          │  │
│  └───────────────────────────────────────────────────────────────────────────┘   │
│                                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────────┐ │
│  │  💬 Ask about your documents…                                                │ │
│  └──────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                   │
│  ┌──────────────────────────────┐                                                 │
│  │  🧹 Clear Conversation       │                                                 │
│  └──────────────────────────────┘                                                 │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### 6.4 Analytics Tab — Charts, Metrics & Query Log

```
┌─ 📊 Analytics Tab ──────────────────────────────────────────────────────────────┐
│                                                                                  │
│  ── 📊 Document Insights ─────────────────────────────────────────────────────   │
│                                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │Total Documents│  │ Total Chunks │  │Avg Chunks/Doc│  │Total Storage │         │
│  │      3        │  │     47       │  │    15.7      │  │   14.2 KB    │         │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘         │
│                                                                                  │
│  ─────────────────────────────────────────────────────────────────────────────   │
│                                                                                  │
│  ┌─ File Type Distribution ────────┐  ┌─ Chunks per Document ────────────────┐  │
│  │                                  │  │                                      │  │
│  │        ╭───────╮                 │  │  compliance.md  ████████████████ 22  │  │
│  │      ╭─┤  TXT  ├─╮              │  │                                      │  │
│  │    ╭─┤ │   2   │ ├─╮            │  │  hr_policy.txt  ███████████ 15       │  │
│  │    │ │ ╰───────╯ │ │            │  │                                      │  │
│  │    │ ╰─────┬─────╯ │            │  │  faq.txt        ███████ 10           │  │
│  │    │  ╭────┴────╮   │            │  │                                      │  │
│  │    ╰──┤   MD    ├───╯            │  │  0     5    10    15    20    25     │  │
│  │       │    1    │                │  │                                      │  │
│  │       ╰─────────╯                │  │                                      │  │
│  │   ◀─ Altair donut chart          │  │   ◀─ Altair bar chart               │  │
│  └──────────────────────────────────┘  └──────────────────────────────────────┘  │
│                                                                                  │
│  ─────────────────────────────────────────────────────────────────────────────   │
│                                                                                  │
│  ── Document Sizes ──────────────────────────────────────────────────────────    │
│  ┌──────────────────────────────────────────────────────────────────────────┐    │
│  │  compliance.md  ████████████████████████████████████████ 8.2 KB         │    │
│  │  hr_policy.txt  ████████████████████ 4.1 KB                             │    │
│  │  faq.txt        ██████████ 2.0 KB                                       │    │
│  └──────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  ─────────────────────────────────────────────────────────────────────────────   │
│                                                                                  │
│  ── 🕒 Session Query Log ────────────────────────────────────────────────────    │
│  ┌──────┬──────────────────────────────────────┬──────────┬──────────┐           │
│  │  #   │  Query                                │ Time (s) │ Sources  │           │
│  ├──────┼──────────────────────────────────────┼──────────┼──────────┤           │
│  │  1   │  What is the leave policy for new…    │   3.2    │    3     │           │
│  │  2   │  Compliance requirements for data…    │   4.1    │    3     │           │
│  │  3   │  What happens if someone violates…    │   2.8    │    2     │           │
│  │  4   │  Employee benefits summary            │   3.5    │    4     │           │
│  │  5   │  Key points in hr policy?             │   2.9    │    3     │           │
│  └──────┴──────────────────────────────────────┴──────────┴──────────┘           │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### 6.5 Document Explorer — Metadata, Chunk Viewer & Filtering

```
┌─ 📄 Documents Tab ──────────────────────────────────────────────────────────────┐
│                                                                                  │
│  ── 📄 Document Explorer — 3 document(s) ────────────────────────────────────   │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐    │
│  │  🔎 Type to filter documents…                                            │    │
│  └──────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  ▸ 📄 hr_policy.txt  ──────────────────────────────────────────────────────     │
│  ▾ 📝 compliance_manual.md  ───────────────────────────────────────────────     │
│  │                                                                               │
│  │  Metadata                                                                     │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                        │
│  │  │ Size         │  │ Chunks       │  │ Indexed      │                        │
│  │  │ 8.2 KB       │  │ 22           │  │ 5m ago       │                        │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                        │
│  │  Type: md  ·  Hash: a3f7c2b9e1d4…  ·  ID: 4f8a9c2e1b…                      │
│  │                                                                               │
│  │  Content Preview (22 chunks)                                                  │
│  │  ┌────────────────────────────────────────────────────────────────────┐       │
│  │  │  Chunk 0                                                          │       │
│  │  │  Compliance Manual — Corporate Regulatory Framework               │       │
│  │  │  This document outlines the mandatory compliance requirements     │       │
│  │  │  for all employees regarding data handling, privacy, and…         │       │
│  │  ├────────────────────────────────────────────────────────────────────┤       │
│  │  │  Chunk 1                                                          │       │
│  │  │  Section 1: Data Classification Framework                         │       │
│  │  │  All organizational data must be classified into one of four      │       │
│  │  │  categories: Public, Internal, Confidential, or Restricted.       │       │
│  │  │  Classification determines the applicable security controls…      │       │
│  │  ├────────────────────────────────────────────────────────────────────┤       │
│  │  │  Chunk 2                                                          │       │
│  │  │  Section 2: Encryption Requirements                               │       │
│  │  │  Confidential and Restricted data must be encrypted using         │       │
│  │  │  AES-256 at rest. Data in transit must use TLS 1.2 or higher…    │       │
│  │  └────────────────────────────────────────────────────────────────────┘       │
│  │  Showing 8 of 22 chunks                                                      │
│  │                                                                               │
│  │  ┌──────────────────────────────────────────────────────────────────┐         │
│  │  │  🗑️ Delete compliance_manual.md                                  │         │
│  │  └──────────────────────────────────────────────────────────────────┘         │
│  │                                                                               │
│  ▸ 📄 faq.txt  ────────────────────────────────────────────────────────────     │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### 6.6 About Tab — Pipeline Flows, Architecture & Tech Stack

```
┌─ ℹ️ About Tab ───────────────────────────────────────────────────────────────────┐
│                                                                                   │
│  ── How It Works ─────────────────────────────────────────────────────────────    │
│                                                                                   │
│  Indexing Pipeline                                                                │
│  ┌──────────────────────────────────────────────────────────────────────────┐     │
│  │  📄        📝        🧩        🧮        🌲        💾                   │     │
│  │  Upload → Parse → Chunk → Embed → Index → Store                        │     │
│  └──────────────────────────────────────────────────────────────────────────┘     │
│                                                                                   │
│  Query Pipeline                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────┐     │
│  │  ❓        🔍        📊        🧠        📝        📌                   │     │
│  │  Query → Search → Rank → Generate → Answer → Cite                      │     │
│  └──────────────────────────────────────────────────────────────────────────┘     │
│                                                                                   │
│  ─────────────────────────────────────────────────────────────────────────────    │
│                                                                                   │
│  ── Dual Retrieval Architecture ──────────────────────────────────────────────   │
│                                                                                   │
│  ┌──────────┬─────────────┬──────────────────────────────────────────────────┐   │
│  │  Layer   │ Technology  │ Approach                                         │   │
│  ├──────────┼─────────────┼──────────────────────────────────────────────────┤   │
│  │ Primary  │ PageIndex   │ Hierarchical reasoning tree — LLM navigates     │   │
│  │          │             │ the tree to locate relevant sections             │   │
│  │ Secondary│ ChromaDB    │ Dense-vector cosine search over overlapping      │   │
│  │          │             │ text chunks                                      │   │
│  └──────────┴─────────────┴──────────────────────────────────────────────────┘   │
│                                                                                   │
│  ┌──────────────┐   ┌──────────────┐   ┌────────────────┐   ┌───────────┐       │
│  │  Browser     │──▶│  Streamlit   │──▶│  PageIndex     │──▶│  Ollama   │       │
│  │  (Employee)  │   │  Portal      │   │  + ChromaDB    │   │  LLM      │       │
│  │              │◀──│              │◀──│                │◀──│  (Local)  │       │
│  └──────────────┘   └──────────────┘   └────────────────┘   └───────────┘       │
│                                                                                   │
│  ─────────────────────────────────────────────────────────────────────────────    │
│                                                                                   │
│  ── 🛠️ Tech Stack ────────────────────────────────────────────────────────────   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  🌲           │  │  🧠           │  │  📊           │  │  🎯           │         │
│  │  PageIndex    │  │  Ollama      │  │  ChromaDB    │  │  Streamlit   │         │
│  │  Reasoning    │  │  Local LLM   │  │  Vector      │  │  Interactive │         │
│  │  tree RAG     │  │  inference   │  │  embeddings  │  │  UI          │         │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘         │
│                                                                                   │
│  ─────────────────────────────────────────────────────────────────────────────    │
│                                                                                   │
│  ── Configuration ────────────────────────────────────────────────────────────   │
│  {                                                                                │
│    "LLM Model":       "llama3",                                                   │
│    "Embedding Model": "nomic-embed-text",                                         │
│    "Chunk Size":       512,                                                       │
│    "Chunk Overlap":    64,                                                        │
│    "Temperature":      0.3,                                                       │
│    "Max Upload (MB)":  200                                                        │
│  }                                                                                │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### 6.7 Sidebar — System Status, Query History & Upload Progress

```
┌── SIDEBAR — System Status ──────────────────────┐
│                                                   │
│  ⚙️ System Status ▾                              │
│  ┌───────────────────────────────────────────┐   │
│  │ Ollama  ● Online                          │   │
│  │ Models: llama3, nomic-embed-text          │   │
│  │                                           │   │
│  │  ┌──────────┐  ┌──────────┐               │   │
│  │  │ Docs     │  │ Chunks   │               │   │
│  │  │   3      │  │   47     │               │   │
│  │  └──────────┘  └──────────┘               │   │
│  │  ┌──────────────────────┐                 │   │
│  │  │ Storage              │                 │   │
│  │  │ 14.2 KB              │                 │   │
│  │  └──────────────────────┘                 │   │
│  └───────────────────────────────────────────┘   │
│                                                   │
│  🕒 Recent Queries (5) ▾                         │
│  ┌───────────────────────────────────────────┐   │
│  │ 🔍 What is the leave policy for new…      │   │
│  │    3.2s · 3 sources · 5m ago              │   │
│  │ 🔍 Compliance requirements for data…      │   │
│  │    4.1s · 3 sources · 3m ago              │   │
│  │ 🔍 What happens if someone violates…      │   │
│  │    2.8s · 2 sources · 2m ago              │   │
│  │ 🔍 Employee benefits summary              │   │
│  │    3.5s · 4 sources · 1m ago              │   │
│  │ 🔍 Key points in hr policy?               │   │
│  │    2.9s · 3 sources · just now            │   │
│  └───────────────────────────────────────────┘   │
│                                                   │
└───────────────────────────────────────────────────┘
```

### 6.8 Upload Progress & Notifications

```
┌── SIDEBAR — Upload in Progress ─────────────────┐
│                                                   │
│  📤 Upload Documents                              │
│  ┌───────────────────────────────────────────┐   │
│  │  📎 hr_policy.txt                         │   │
│  │  📎 compliance_manual.md                  │   │
│  │  📎 faq.txt                               │   │
│  └───────────────────────────────────────────┘   │
│  3 file(s) selected                               │
│                                                   │
│  ┌───────────────────────────────────────────┐   │
│  │  🔄 Index All Files                       │   │
│  └───────────────────────────────────────────┘   │
│                                                   │
│  Indexing compliance_manual.md (2/3)              │
│  ████████████████████░░░░░░░░░░░░ 66%             │
│                                                   │
│  ┌─────────────────────────────────────────┐     │
│  │ ✅ hr_policy.txt — 15 chunks            │ ◀─  │
│  └─────────────────────────────────────────┘ Toast│
│  ┌─────────────────────────────────────────┐     │
│  │ ✅ compliance_manual.md — 22 chunks     │     │
│  └─────────────────────────────────────────┘     │
│                                                   │
└───────────────────────────────────────────────────┘
```

### 6.9 PDF Export — Professional Report Output

```
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│                 Enterprise Knowledge Portal                          │
│         Generated: 2024-01-15 14:30:25 UTC                          │
│                                                                      │
│  ── Question ──────────────────────────────────────────────────────  │
│  What is the leave policy for new employees?                         │
│                                                                      │
│  ── Answer ────────────────────────────────────────────────────────  │
│  New employees are entitled to the following leave benefits:         │
│                                                                      │
│  1. Annual Leave: 15 days per calendar year, accrued monthly at     │
│     1.25 days/month. Leave can be carried forward up to 5 days.     │
│  2. Sick Leave: 10 days per year. Medical certificate required      │
│     for absences exceeding 3 consecutive days.                      │
│  3. Probation Period: During the 90-day probation, leave accrues   │
│     but can only be taken with manager approval.                    │
│                                                                      │
│  ── Sources ───────────────────────────────────────────────────────  │
│  1. hr_policy.txt - 94% match                                       │
│     Annual leave entitlement for full-time employees is 15 working  │
│     days per calendar year, prorated for the first year…            │
│  2. hr_policy.txt - 87% match                                       │
│     Sick leave policy: employees receive 10 days of paid sick…      │
│  3. faq.txt - 72% match                                             │
│     Q: Can new employees take leave during probation?               │
│     A: Yes, with prior approval from your direct manager…           │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

  ◀─ Saved to: outputs/20240115_143025_what_is_the_leave_policy.pdf
```

### 6.10 Empty States — Onboarding Flow

```
┌─ When No Documents Are Indexed ─────────────────────────────────────────────────┐
│                                                                                  │
│  ── 🔎 Search Tab ────────────────────     ── 💬 Chat Tab ────────────────────  │
│  ┌──────────────────────────────────┐     ┌──────────────────────────────────┐  │
│  │                                  │     │                                  │  │
│  │           📤                     │     │           💬                     │  │
│  │                                  │     │                                  │  │
│  │   No Documents Indexed           │     │   Start a Conversation           │  │
│  │   Upload documents using the     │     │   Upload and index documents     │  │
│  │   sidebar to start searching.    │     │   first, then chat here.         │  │
│  │                                  │     │                                  │  │
│  └──────────────────────────────────┘     └──────────────────────────────────┘  │
│                                                                                  │
│  ── 📊 Analytics Tab ─────────────────   ── 📄 Documents Tab ────────────────  │
│  ┌──────────────────────────────────┐     ┌──────────────────────────────────┐  │
│  │                                  │     │                                  │  │
│  │           📊                     │     │           📄                     │  │
│  │                                  │     │                                  │  │
│  │   No Data Yet                    │     │   No Documents                   │  │
│  │   Index documents to see         │     │   Upload documents using the     │  │
│  │   analytics.                     │     │   sidebar to explore them here.  │  │
│  │                                  │     │                                  │  │
│  └──────────────────────────────────┘     └──────────────────────────────────┘  │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Step-by-Step Execution Guide

> Complete walkthrough from zero to a working portal with answers.

### Step 1: Clone & Set Up Environment

```bash
$ git clone https://github.com/maneeshkumar52/pageindex-enterprise-wiki.git
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
tests/test_utils.py::TestLoadConfig::test_loads_yaml                        PASSED
tests/test_utils.py::TestLoadConfig::test_missing_file_raises               PASSED
tests/test_utils.py::TestChunkText::test_basic_chunking                     PASSED
tests/test_utils.py::TestChunkText::test_overlap                            PASSED
tests/test_utils.py::TestFileHash::test_hash_consistency                    PASSED
tests/test_utils.py::TestExport::test_export_creates_file                   PASSED
tests/test_ollama_client.py::TestOllamaAvailability::test_is_available_true PASSED
tests/test_ollama_client.py::TestOllamaAvailability::test_is_available_false PASSED
tests/test_ollama_client.py::TestOllamaChat::test_chat_returns_content      PASSED
tests/test_ollama_client.py::TestOllamaEmbed::test_embed_returns_vector     PASSED
tests/test_indexing_pipeline.py::TestVectorStore::test_add_and_search       PASSED
tests/test_indexing_pipeline.py::TestPageIndexPipeline::test_ingest_file    PASSED
... (22 more tests)
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
2. In the sidebar, expand **📤 Upload Documents**
3. Drag the three sample files from `sample_docs/`:
   - `hr_policy.txt` — HR leave & benefits policy
   - `compliance_manual.md` — Regulatory compliance guide
   - `faq.txt` — Employee frequently asked questions
4. Click **🔄 Index All Files**
5. Watch: progress bar fills → toast notifications per file → metrics update

**Behind the scenes:**
```
[INFO] Ingesting: hr_policy.txt
[INFO] → SHA-256: a3f7c2… (dedup check passed)
[INFO] → Converted to Markdown → data/workspace/hr_policy.md
[INFO] → PageIndex tree built (reasoning hierarchy)
[INFO] → 15 chunks embedded into ChromaDB
[INFO] → Registered in doc_registry.json

[INFO] Ingesting: compliance_manual.md
[INFO] → 22 chunks embedded into ChromaDB

[INFO] Ingesting: faq.txt
[INFO] → 10 chunks embedded into ChromaDB
```

### Step 7: Search — Ask Your First Question

1. Switch to the **🔎 Search** tab
2. **Option A**: Click a suggested question like "Key points in hr policy?"
3. **Option B**: Type your own: `What is the leave policy for new employees?`
4. Adjust the **Sources** slider (1–10) for more or fewer results
5. Click **🔍 Search Documents**
6. View: answer → export buttons (MD + PDF) → source cards with relevance bars
7. Click **📄 Export PDF** to save a professional report

### Step 8: Chat — Multi-Turn Conversation

1. Switch to the **💬 Chat** tab
2. Ask: `What are the compliance requirements for data handling?`
3. Follow up: `What happens if someone violates these policies?`
4. Expand **📌 source(s)** in each response to see cited chunks
5. Click **🧹 Clear Conversation** to reset

### Step 9: Analytics — Explore Document Insights

1. Switch to the **📊 Analytics** tab
2. View summary metrics: total documents, chunks, average, storage
3. Examine the **donut chart** for file type distribution
4. Review **bar charts** for chunks per document and document sizes
5. Scroll down to the **Session Query Log** table

### Step 10: Document Explorer — Inspect Chunks

1. Switch to the **📄 Documents** tab
2. Use the filter box to search by filename
3. Expand any document to see:
   - Metadata: size, chunks, indexed time, hash, ID
   - Content preview: first 8 chunks with actual text
4. Click **🗑️ Delete** to remove individual documents

### Step 11: Verify Exports

```bash
$ ls outputs/
20240115_143025_what_is_the_leave_policy.md
20240115_143025_what_is_the_leave_policy.pdf

$ cat outputs/*.md
# Query Result — 2024-01-15 14:30:25 UTC

## Question
What is the leave policy for new employees?

## Answer
New employees are entitled to 15 days of annual leave...

## Sources
**1.** hr_policy.txt — page N/A
> Annual leave entitlement for full-time employees is 15 working…
```

---

## 8. Project Structure

```
pageindex-enterprise-wiki/
│
├── app.py                    # Streamlit UI — 5 tabs, sidebar, 6-metric dashboard
├── indexing_pipeline.py      # PageIndex tree + ChromaDB vector pipeline
├── ollama_client.py          # Ollama REST API wrapper (chat, embed, stream)
├── utils.py                  # Config, logging, parsing, chunking, export (MD + PDF)
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
├── outputs/                  # Exported results (Markdown + PDF)
│   └── .gitkeep
│
├── tests/                    # Pytest test suite (34 tests)
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

| Module | Lines | Responsibility | Key Classes/Functions |
|--------|-------|---------------|----------------------|
| `app.py` | ~680 | 5-tab UI, sidebar, session state, file upload | `main()`, `_render_sidebar()`, `_render_main()` |
| `indexing_pipeline.py` | ~390 | Ingestion, retrieval, chunk inspection | `PageIndexPipeline`, `VectorStore` |
| `ollama_client.py` | ~140 | LLM communication (chat, embed, stream) | `OllamaClient` |
| `utils.py` | ~370 | Config, parsing, chunking, export (MD + PDF) | `load_config()`, `extract_text()`, `export_to_pdf()` |
| `config.yaml` | ~50 | Centralised configuration | — |

---

## 9. Configuration

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

## 10. Usage Guide

### 10.1 Upload Documents

1. Open the sidebar → **📤 Upload Documents**
2. Drag-and-drop or click to select PDF, DOCX, TXT, or Markdown files
3. Click **🔄 Index All Files**
4. Progress bar tracks each file. Toast notifications confirm success
5. Metrics update in real-time (document count, chunks, storage)

### 10.2 Search with Smart Suggestions

1. Switch to the **🔎 Search** tab
2. Click a **Quick Question** button (auto-generated from document names)
3. Or type your own natural-language question
4. Adjust the **Sources** slider (1–10) for result depth
5. Click **🔍 Search Documents**
6. View the answer, source cards with relevance bars, and export options

### 10.3 Conversational Chat

1. Switch to the **💬 Chat** tab
2. Type a question in the chat input
3. Each response shows expandable source references
4. Ask follow-up questions — context is maintained across turns
5. Click **🧹 Clear Conversation** to reset

### 10.4 Analytics Dashboard

1. Switch to the **📊 Analytics** tab
2. Review 4 summary metrics at the top
3. Interactive charts: file type donut, chunks bar, sizes bar
4. Session query log shows all queries with timing

### 10.5 Document Explorer

1. Switch to the **📄 Documents** tab
2. Filter by filename using the search box
3. Expand any document to see metadata and chunk preview
4. Delete individual documents or use **🗑️ Clear All** in the sidebar

### 10.6 Export Results

- **Markdown**: Click **📥 Export Markdown** after a search
- **PDF**: Click **📄 Export PDF** for a professional report
- Files saved to `outputs/` with timestamps

---

## 11. API Reference

### `OllamaClient`

```python
from ollama_client import OllamaClient

client = OllamaClient()
client.is_available()                         # → bool
client.list_models()                          # → ["llama3", "nomic-embed-text"]
client.chat(messages, model=None, system_prompt=None)  # → str
client.chat_stream(messages, ...)             # → Generator[str]
client.embed(text)                            # → list[float]
client.embed_batch(texts)                     # → list[list[float]]
client.ask_with_context(question, context)    # → str
```

### `PageIndexPipeline`

```python
from indexing_pipeline import PageIndexPipeline

pipeline = PageIndexPipeline()
metadata = pipeline.ingest("/path/to/doc.pdf")         # → dict
results  = pipeline.search("leave policy", n_results=5)  # → list[dict]
answer   = pipeline.ask("What is the leave policy?")     # → {"answer": str, "sources": [...]}
docs     = pipeline.list_documents()                      # → list[dict]
chunks   = pipeline.get_document_chunks(doc_id)           # → list[dict]
pipeline.delete_document(doc_id)                          # → bool
pipeline.document_count                                   # → int
pipeline.chunk_count                                      # → int
pipeline.storage_size                                     # → int (bytes)
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

### Export Utilities

```python
from utils import export_to_markdown, export_to_pdf

# Markdown export
path = export_to_markdown(query, answer, sources, "outputs/")

# PDF export
path = export_to_pdf(query, answer, sources, "outputs/")
```

---

## 12. Docker Deployment

### Build & Run (Standalone)

```bash
docker build -t pageindex-wiki .

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
docker compose up -d

docker exec -it ollama ollama pull llama3
docker exec -it ollama ollama pull nomic-embed-text

docker compose logs -f wiki

docker compose down
```

### GPU Support (NVIDIA)

The `docker-compose.yml` includes GPU reservation for Ollama. Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). For CPU-only, remove the `deploy.resources` block.

---

## 13. Testing

### Run All Tests

```bash
pytest tests/ -v
pytest tests/ -v --cov=. --cov-report=term-missing
```

### Test Coverage

| File | Tests | Coverage |
|------|-------|----------|
| `test_utils.py` | 18 | Config, text extraction, chunking, hashing, metadata, export |
| `test_ollama_client.py` | 8 | Health checks, chat, streaming, embeddings, RAG |
| `test_indexing_pipeline.py` | 8 | VectorStore CRUD, registry, pipeline ingestion, dedup |

**Total: 34 tests** — all passing. External services mocked — runs offline without GPU.

---

## 14. Troubleshooting

| Problem | Solution |
|---------|----------|
| `Connection refused — localhost:11434` | Run `ollama serve` |
| `model "llama3" not found` | Run `ollama pull llama3` |
| `pageindex not installed` | Run `pip install pageindex` (portal works without it via ChromaDB fallback) |
| `Permission denied: data/chromadb` | `chmod -R 755 data/` |
| Large file upload timeout | Set `max_upload_mb: 500` in `config.yaml` and `maxUploadSize = 500` in `.streamlit/config.toml` |
| Docker: Ollama connection refused | Use `OLLAMA_HOST=http://ollama:11434` (compose) or `http://host.docker.internal:11434` (host) |

---

## 15. Production Checklist

### Security

- [ ] Run Ollama behind a firewall — do not expose port 11434
- [ ] Enable Streamlit authentication (reverse proxy or `streamlit-authenticator`)
- [ ] Use HTTPS via reverse proxy (Nginx, Caddy, Traefik)
- [ ] Restrict file permissions on `data/` directory
- [ ] Review `config.yaml` for production values

### Performance

- [ ] Use GPU-accelerated Ollama for production workloads
- [ ] Tune `chunk_size` and `chunk_overlap` for your document types
- [ ] Consider `mistral` or `phi3` for faster CPU inference
- [ ] Monitor ChromaDB at scale — consider sharding at 100k+ chunks

### Observability

- [ ] Review `data/app.log` for events
- [ ] Set `logging.level: DEBUG` for troubleshooting
- [ ] Monitor Ollama memory — large models need 8+ GB VRAM

### Scaling

- [ ] Deploy multiple Streamlit instances behind a load balancer
- [ ] Mount `data/` on shared storage (NFS, EFS) for multi-instance
- [ ] Run Ollama on a dedicated GPU server; update `base_url`

---

## 16. Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m 'Add my feature'`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a Pull Request

### Code Standards

- Python 3.11+ with type hints
- Format with `black`, lint with `ruff`
- All new features must include tests
- Config-driven — no hardcoded values

---

## 17. License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

<div align="center">

**Built by [Maneesh Kumar](https://github.com/maneeshkumar52)** — AI Architect

Powered by [PageIndex](https://github.com/VectifyAI/PageIndex) · [Ollama](https://ollama.com) · [Streamlit](https://streamlit.io) · [ChromaDB](https://www.trychroma.com/)

</div>
