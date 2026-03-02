import json
import os
import numpy as np
import chromadb
from pathlib import Path

# --- Path Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# --- Initialize ChromaDB Client ---
client = chromadb.PersistentClient(path=os.path.join(DATA_DIR, "chromadb"))

try:
    client.delete_collection("ipard_docs")
    print("Previous collection deleted successfully.")
except Exception:
    pass

print("Loading document chunks...")
with open(os.path.join(DATA_DIR, "all_chunks.json"), encoding="utf-8") as f:
    chunks = json.load(f)

print("Loading pre-computed embeddings...")
embeddings = np.load(os.path.join(DATA_DIR, "embeddings.npy")).tolist()

print(f"Chunks: {len(chunks)}, Embeddings: {len(embeddings)}")
assert len(chunks) == len(embeddings), "ERROR: Chunk count and embedding count do not match!"

collection = client.get_or_create_collection(
    name="ipard_docs",
    metadata={"hnsw:space": "cosine"}
)

print("Upserting data into ChromaDB...")

BATCH_SIZE = 500
for i in range(0, len(chunks), BATCH_SIZE):
    batch_chunks = chunks[i: i + BATCH_SIZE]
    batch_embeds = embeddings[i: i + BATCH_SIZE]

    ids = [c["chunk_id"] for c in batch_chunks]
    documents = [c["text"] for c in batch_chunks]

    metadatas = [{
        "doc_type": str(c.get("doc_type") or ""),
        "measure": str(c.get("tedbir") or ""),
        "sector": str(c.get("sektor") or ""),
        "program": str(c.get("program") or ""),
        "version": str(c.get("versiyon") or ""),
        "is_active": str(c.get("aktif") or ""),
        "heading": str(c.get("heading") or "")[:200],
        "source_file": str(c.get("source_file") or ""),
        "start_page": str(c.get("start_page") or ""),
    } for c in batch_chunks]

    collection.upsert(
        ids=ids,
        embeddings=batch_embeds,
        documents=documents,
        metadatas=metadatas,
    )
    print(f"  Progress: {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)} records indexed")

print(f"\nFinal record count: {collection.count()}")
print(f"ChromaDB is ready at: {os.path.join(DATA_DIR, 'chromadb')}")
