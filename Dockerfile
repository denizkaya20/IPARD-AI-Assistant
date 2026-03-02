FROM python:3.11-slim

WORKDIR /app

# 1. System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl git git-lfs && rm -rf /var/lib/apt/lists/*

# 2. Python dependencies (CPU-only PyTorch)
RUN pip install --no-cache-dir \
    torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir \
    transformers==4.38.2 \
    sentence-transformers==2.5.1 \
    streamlit==1.30.0 \
    fastapi==0.109.0 \
    uvicorn==0.27.0 \
    chromadb==0.4.24 \
    rank-bm25==0.2.2 \
    openai httpx python-dotenv requests numpy

# 3. Copy project files
COPY . .

# 4. Build ChromaDB from chunks + embeddings
RUN mkdir -p /app/ipard_rag/data/chromadb && \
    cd /app/ipard_rag/src && \
    python build_db.py

# 5. Copy start script and set permissions
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# 6. User permissions (required by HF Spaces)
RUN useradd -m -u 1000 user && chown -R user:user /app
USER user

# 7. Environment
ENV PYTHONPATH="/app/ipard_rag/src:/app"
EXPOSE 7860

CMD ["bash", "/app/start.sh"]
