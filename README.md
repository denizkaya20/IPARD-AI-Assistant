---
title: IPARD AI Assistant
emoji: 🌾
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# 🌾 IPARD III AI Assistant

An AI-powered **Retrieval-Augmented Generation (RAG)** assistant for exploring Turkey's IPARD III rural development program documents. It uses hybrid search with verified sources to deliver accurate, citation-backed answers.

> **Disclaimer:** This is an independent project and has no official affiliation with TKDK (Agriculture and Rural Development Support Institution). All data is sourced from publicly available documents.

## ✨ Features

- **Hybrid Search** — Combines BM25 (keyword) and semantic vector search with Reciprocal Rank Fusion (RRF)
- **Cross-Encoder Reranking** — Refines results using `BAAI/bge-reranker-base` for higher relevance
- **Turkish-Optimized Embeddings** — Uses `ytu-ce-cosmos/turkish-e5-large` for accurate Turkish language understanding
- **Source Citations** — Every answer includes references to the original document chunks
- **Streaming Responses** — Real-time answer generation via FastAPI streaming endpoint
- **Web Scraper** — Automated pipeline to download and process IPARD III PDFs and FAQ pages from official sources
- **Streamlit UI** — Interactive chat interface with conversation history
- **Docker Support** — Fully containerized with Docker Compose for easy deployment

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────┐
│   Streamlit UI      │  (app.py)
│   port 7860         │
└────────┬────────────┘
         │ HTTP
         ▼
┌─────────────────────┐
│   FastAPI Backend    │  (api.py)
│   port 8000         │
└────────┬────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│           RAG Pipeline                  │
│  ┌─────────┐  ┌──────────┐  ┌────────┐ │
│  │  BM25   │  │ Semantic │  │  RRF   │ │
│  │ Search  │  │  Search  │  │ Merge  │ │
│  └────┬────┘  └────┬─────┘  └───┬────┘ │
│       └────────────┴────────────┘      │
│                    │                    │
│            Cross-Encoder               │
│             Reranking                  │
│                    │                    │
│              LLM Generation            │
│            (Groq API)                  │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────┐
│     ChromaDB        │
│  (Vector Store)     │
└─────────────────────┘
```

## 📁 Project Structure

```
IPARD_QA_RAG/
├── ipard_rag/
│   └── src/
│       ├── rag_pipeline.py   # Core RAG engine (hybrid search + reranking + LLM)
│       ├── api.py            # FastAPI REST backend with streaming support
│       ├── app.py            # Streamlit chat interface
│       ├── scraper.py        # Web scraper for IPARD III documents & FAQs
│       ├── parser.py         # PDF parser with smart chunking
│       ├── embed.py          # Embedding generation with turkish-e5-large
│       ├── build_db.py       # ChromaDB vector database builder
│       ├── dedup.py          # Document deduplication utility
│       └── test_parse.py     # PDF parsing test script
├── .github/workflows/
│   └── huggingface.yml       # CI/CD pipeline for Hugging Face Spaces
├── Dockerfile                # Main container (Streamlit + FastAPI)
├── Dockerfile.api            # Standalone API container
├── Dockerfile.ui             # Standalone UI container
├── docker-compose.yml        # Multi-service orchestration
├── requirements.txt          # Backend dependencies
├── requirements-ui.txt       # Frontend dependencies
├── .env_example              # Environment variable template
└── .gitignore
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- A [Groq API key](https://console.groq.com/) for LLM inference

### 1. Clone the Repository

```bash
git clone https://github.com/denizkaya20/IPARD-AI-Assistant.git
cd IPARD-AI-Assistant
```

### 2. Set Up Environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
pip install -r requirements-ui.txt
```

### 3. Configure Environment Variables

```bash
cp .env_example .env
```

Edit `.env` and add your API key:

```
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Build the Knowledge Base

Run the data pipeline in order:

```bash
cd ipard_rag/src

# Step 1: Scrape IPARD III documents
python scraper.py

# Step 2: Parse PDFs into chunks
python parser.py

# Step 3: Generate embeddings
python embed.py

# Step 4: Build ChromaDB index
python build_db.py
```

### 5. Run the Application

**Start the API server:**

```bash
cd ipard_rag/src
uvicorn api:app --reload --port 8000
```

**Start the Streamlit UI** (in a separate terminal):

```bash
cd ipard_rag/src
streamlit run app.py --server.port 7860
```

### Docker Deployment

```bash
docker compose up --build
```

The UI will be available at `http://localhost:7860`.

## ⚙️ Technical Details

| Component | Technology |
|---|---|
| Embedding Model | `ytu-ce-cosmos/turkish-e5-large` |
| Reranker | `BAAI/bge-reranker-base` |
| Vector Store | ChromaDB |
| Keyword Search | BM25 (rank-bm25) |
| LLM Inference | Groq API |
| Backend | FastAPI |
| Frontend | Streamlit |
| Containerization | Docker / Docker Compose |

### RAG Pipeline Parameters

- **BM25 Top-N:** 20 candidates
- **Semantic Top-N:** 20 candidates
- **RRF Final Top-K:** 7 results
- **Reranker Top-K:** 5 final passages
- **Chunk Size:** ~600 chars (max 900) with 120-char overlap

## 📄 License

This project is licensed under the [Apache 2.0 License](LICENSE).
