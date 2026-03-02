#!/bin/bash
set -e

echo "=== Starting FastAPI backend ==="
cd /app/ipard_rag/src
uvicorn api:app --host 127.0.0.1 --port 8000 &
API_PID=$!

echo "=== Waiting for API to be ready ==="
for i in $(seq 1 120); do
    if curl -s http://127.0.0.1:8000/measures > /dev/null 2>&1; then
        echo "API is ready! (took ${i}s)"
        break
    fi
    if [ $i -eq 120 ]; then
        echo "WARNING: API did not respond in 120s, starting Streamlit anyway..."
    fi
    sleep 1
done

echo "=== Starting Streamlit frontend ==="
cd /app/ipard_rag/src
streamlit run app.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false
