FROM python:3.11-slim

WORKDIR /app

# 1. Sistem paketleri
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl netcat-openbsd && rm -rf /var/lib/apt/lists/*

# 2. Uyumlu kütüphaneler (CPU sürümü)
RUN pip install --no-cache-dir \
    torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu \
    transformers==4.38.2 \
    sentence-transformers==2.5.1 \
    streamlit==1.30.0 \
    fastapi==0.109.0 \
    uvicorn==0.27.0 \
    chromadb==0.4.24 \
    rank-bm25==0.2.2 \
    openai httpx

# 3. Dosyaları kopyala
COPY . .

# 4. Veri klasörünü düzenle (Büyük dosyaları taşıyoruz)
RUN mkdir -p /app/ipard_rag/data && \
    cp chroma.sqlite3 /app/ipard_rag/data/ 2>/dev/null || true && \
    cp embeddings.npy /app/ipard_rag/data/ 2>/dev/null || true

# 5. Kullanıcı yetkileri
RUN useradd -m -u 1000 user && chown -R user:user /app
USER user

# 6. PYTHONPATH ve Başlatma
ENV PYTHONPATH="/app/ipard_rag/src:/app"
EXPOSE 7860

# API başlar, 60 saniye boyunca modellerin yüklenmesini bekler, sonra Streamlit açılır
CMD ["sh", "-c", "python3 /app/ipard_rag/src/api.py & sleep 60 && streamlit run app.py --server.port 7860 --server.address 0.0.0.0"]