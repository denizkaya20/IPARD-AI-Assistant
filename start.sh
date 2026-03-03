#!/bin/bash
set -e

echo "========================================="
echo "  IPARD III RAG — Başlatılıyor"
echo "========================================="

DATA_DIR="/app/data"
CHROMA_DIR="$DATA_DIR/chromadb"

# ── 1. ChromaDB yoksa oluştur ─────────────────────────────────
if [ ! -d "$CHROMA_DIR" ] || [ -z "$(ls -A $CHROMA_DIR 2>/dev/null)" ]; then
    echo "→ ChromaDB oluşturuluyor (ilk başlatma ~1-2 dk)..."
    cd /app/src && python build_db.py
    echo "→ ChromaDB hazır."
else
    echo "→ ChromaDB mevcut, atlanıyor."
fi

# ── 2. Embedding modelini önbelleğe al ───────────────────────
echo "→ Embedding modeli kontrol ediliyor..."
python -c "
from sentence_transformers import SentenceTransformer
SentenceTransformer('ytu-ce-cosmos/turkish-e5-large')
print('Embedding modeli hazır.')
"

# ── 3. FastAPI arka planda başlat ────────────────────────────
echo "→ FastAPI başlatılıyor (port 8000)..."
cd /app/src && uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1 &

# API hazır olana kadar bekle
echo "→ API bekleniyor..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "→ API hazır."
        break
    fi
    sleep 2
done

# ── 4. Streamlit ön planda başlat ───────────────────────────
echo "→ Streamlit başlatılıyor (port 7860)..."
cd /app/src && streamlit run app.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.fileWatcherType none \
    --browser.gatherUsageStats false
