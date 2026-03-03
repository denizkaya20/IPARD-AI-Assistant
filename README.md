---
title: IPARD III RAG
emoji: 🌾
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
app_port: 7860
---

# IPARD III RAG Assistant 🤖🌾

A powerful **Retrieval-Augmented Generation (RAG)** system designed specifically for **IPARD III** program documents. This assistant provides accurate, context-aware answers to questions about IPARD III grants, measures, and application processes by combining hybrid search with state-of-the-art language models.

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/denizkaya2022/IPARD-AI-Assistant)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black)](https://github.com/denizkaya20/IPARD-AI-Assistant)

## ✨ Key Features

- **Hybrid Search**: Combines BM25 keyword search with semantic embeddings (Turkish-E5-Large) for optimal retrieval
- **Cross-Encoder Reranking**: BGE-reranker-base model ensures the most relevant documents are selected
- **Real-time Streaming**: Server-Sent Events (SSE) for instant token-by-token responses
- **Filterable Queries**: Search by measure (Tedbir) and document type
- **Turkish Optimized**: Specifically tuned for Turkish IPARD III documentation

## 🏗️ System Architecture

```mermaid
graph TD
    A[User Query] --> B{Hybrid Search}
    
    subgraph B [Retrieval Stage]
        C[BM25<br/>Keyword Search]
        D[Semantic Search<br/>Turkish-E5-Large]
    end
    
    B --> E[Reciprocal Rank Fusion<br/>RRF Merge]
    E --> F[Cross-Encoder Reranking<br/>BGE-reranker-base]
    F --> G[Top-K Context Selection]
    G --> H[LLM Generation<br/>Groq API / OpenRouter]
    H --> I[Streaming Response]
    
    style A fill:#d4f1f9
    style H fill:#c9e7c9
    style I fill:#ffd966
🚀 Technology Stack
Component	Technology	Purpose
Frontend	Streamlit	Interactive web interface
Backend API	FastAPI	REST endpoints & streaming
Vector Database	ChromaDB	Embedding storage & retrieval
Keyword Search	BM25 (rank-bm25)	Lexical search fallback
Embeddings	Turkish-E5-Large	Semantic text representation
Reranking	BGE-reranker-base	Precision improvement
LLM	Groq API / OpenRouter	Answer generation
Deployment	Docker + Hugging Face Spaces	Containerized hosting
📁 Project Structure
text
.
├── 📁 src/
│   ├── 📄 api.py              # FastAPI backend endpoints
│   ├── 📄 app.py              # Streamlit frontend interface
│   ├── 📄 rag_pipeline.py      # Core RAG logic
│   ├── 📄 build_db.py          # ChromaDB builder
│   ├── 📄 embed.py             # Embedding generator
│   ├── 📄 parser.py            # PDF document parser
│   └── 📄 scraper_sss.py       # FAQ scraper
├── 📁 data/
│   ├── 📄 all_chunks.json      # Document chunks (LFS)
│   ├── 📄 embeddings.npy       # Pre-computed embeddings (LFS)
│   └── 📁 chromadb/            # Vector database
├── 📄 Dockerfile                # Container definition
├── 📄 requirements.txt          # Python dependencies
├── 📄 start.sh                  # Multi-service launcher
└── 📄 .gitattributes            # Git LFS configuration
🛠️ Local Development
Prerequisites
Python 3.11+

Git LFS (for large files)

Groq API key

Setup
bash
# Clone repository
git clone https://github.com/denizkaya20/IPARD-AI-Assistant.git
cd IPARD-AI-Assistant

# Install dependencies
pip install -r requirements.txt

# Set environment variables
echo "GROQ_API_KEY=your_key_here" > .env

# Run locally
uvicorn src.api:app --reload --port 8000 &  # Backend
streamlit run src/app.py --server.port 7860  # Frontend
Docker Deployment
bash
# Build image
docker build -t ipard-rag .

# Run container
docker run -p 7860:7860 -e GROQ_API_KEY=your_key_here ipard-rag
🌐 Hugging Face Space Configuration
Required Secrets
Secret	Description	Where to Get
GROQ_API_KEY	Groq API key for LLM access	Groq Console
Environment Variables
Variable	Default	Description
EMBED_MODEL	ytu-ce-cosmos/turkish-e5-large	Embedding model
RERANK_MODEL	BAAI/bge-reranker-base	Reranking model
LLM_MODEL	openai/gpt-oss-120b	LLM for generation
📊 Performance
Chunks: 7,717 document segments

Embedding Dimension: 1,024

Retrieval: 20 BM25 + 20 Semantic → 7 RRF → 5 Reranked

Response Time: ~2-5 seconds (varies by query complexity)

🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

📝 License
This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

⚠️ Disclaimer
This is an independent project. Responses are AI-generated and do not constitute official guidance. Always refer to tkdk.gov.tr for authoritative information.

🙏 Acknowledgments
TKDK for IPARD III documentation

Groq for LLM API access

Hugging Face for model hosting and Spaces
