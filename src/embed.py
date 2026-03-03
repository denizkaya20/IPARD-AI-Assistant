import json
import time
import numpy as np
from sentence_transformers import SentenceTransformer

# Recommended model for Turkish language processing
MODEL_NAME = "ytu-ce-cosmos/turkish-e5-large"

print(f"Loading model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)
print("Model is ready.")

# Load chunks from the preprocessing stage
with open("data/all_chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Total chunks to process: {len(chunks)}")

# Prepare text for embedding: heading + text
# Limiting to 512 tokens as per standard transformer context windows
texts = []
for c in chunks:
    heading = c.get("heading", "")
    text = c.get("text", "")
    combined = f"{heading}\n{text}" if heading else text
    texts.append(combined[:512])

# Batch embedding process
print("Starting embedding generation...")
t0 = time.time()

# normalize_embeddings is set to True for cosine similarity compatibility
embeddings = model.encode(
    texts,
    batch_size=16,
    show_progress_bar=True,
    normalize_embeddings=True
)

elapsed = time.time() - t0
print(f"Embedding completed in {elapsed:.1f} seconds.")
print(f"Embedding matrix shape: {embeddings.shape}")

# Save the resulting vector matrix for fast retrieval
np.save("data/embeddings.npy", embeddings)
print("Saved: data/embeddings.npy")