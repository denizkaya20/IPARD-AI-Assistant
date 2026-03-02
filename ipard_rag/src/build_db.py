import json
import numpy as np
import chromadb
from pathlib import Path

# --- Initialize ChromaDB Client ---
# Persistence ensures the database is saved to the disk
client = chromadb.PersistentClient(path="data/chromadb")

try:
    # Remove existing collection to ensure a fresh index
    client.delete_collection("ipard_docs")
    print("Previous collection deleted successfully.")
except Exception:
    # If collection doesn't exist, proceed silently
    pass

print("Loading document chunks...")
with open("data/all_chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)

print("Loading pre-computed embeddings...")
# Convert numpy array back to list for ChromaDB compatibility
embeddings = np.load("data/embeddings.npy").tolist()

print(f"Chunks: {len(chunks)}, Embeddings: {len(embeddings)}")

# Validation check to ensure data integrity
assert len(chunks) == len(embeddings), "ERROR: Chunk count and embedding count do not match!"

# Create or retrieve collection with Cosine Similarity metric
collection = client.get_or_create_collection(
    name="ipard_docs",
    metadata={"hnsw:space": "cosine"}
)

print("Upserting data into ChromaDB...")

# --- Batch Processing ---
# Using batches to optimize memory usage and insertion speed
BATCH_SIZE = 500
for i in range(0, len(chunks), BATCH_SIZE):
    batch_chunks = chunks[i: i + BATCH_SIZE]
    batch_embeds = embeddings[i: i + BATCH_SIZE]

    ids = [c["chunk_id"] for c in batch_chunks]
    documents = [c["text"] for c in batch_chunks]  # Raw text for keyword matching (BM25)

    # Prepare metadata for efficient filtering
    metadatas = [{
        "doc_type": str(c.get("doc_type") or ""),
        "measure": str(c.get("tedbir") or ""),  # Renamed to 'measure' for international standard
        "sector": str(c.get("sektor") or ""),  # Renamed to 'sector'
        "program": str(c.get("program") or ""),
        "version": str(c.get("versiyon") or ""),
        "is_active": str(c.get("aktif") or ""),
        "heading": str(c.get("heading") or "")[:200],
        "source_file": str(c.get("source_file") or ""),
        "start_page": str(c.get("start_page") or ""),
    } for c in batch_chunks]

    # Perform upsert (Insert if new, Update if exists)
    collection.upsert(
        ids=ids,
        embeddings=batch_embeds,
        documents=documents,
        metadatas=metadatas,
    )
    print(f"  Progress: {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)} records indexed")

print(f"\nFinal record count: {collection.count()}")
print("ChromaDB is ready at: data/chromadb/")