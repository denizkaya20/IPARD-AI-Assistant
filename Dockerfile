FROM python:3.11-slim

EXPOSE 7860

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# Sistem bağımlılıkları
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python bağımlılıkları (önce — layer cache için)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Uygulama kodu
COPY src/ ./src/

# Data (all_chunks.json + embeddings.npy repo'da olacak)
COPY data/all_chunks.json ./data/all_chunks.json
COPY data/embeddings.npy  ./data/embeddings.npy

# Startup script
COPY start.sh .
RUN chmod +x start.sh

# HuggingFace Spaces: user 1000
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

CMD ["./start.sh"]
