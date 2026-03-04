"""
IPARD III - RAG Pipeline
Hybrid Search (BM25 + Semantic) -> RRF -> Cross-Encoder Reranking -> LLM Generation

Execution: Run from the src/ directory
  python rag_pipeline.py
"""

import json
import os
import re
import time
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
import openai  # OpenAI 0.28.0 için import değişikliği
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"), override=False)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Config ────────────────────────────────────────────────────────────────────

GROQ_API_KEY       = os.getenv("GROQ_API_KEY", "")  # Set via HF Spaces secrets
EMBED_MODEL        = "ytu-ce-cosmos/turkish-e5-large"
RERANK_MODEL       = "BAAI/bge-reranker-base"
CHROMA_PATH = os.path.join(BASE_DIR, "data", "chromadb")
CHUNKS_FILE = os.path.join(BASE_DIR, "data", "all_chunks.json")
LLM_MODEL          = "openai/gpt-oss-120b"

BM25_TOP_N      = 20
SEMANTIC_TOP_N  = 20
FINAL_TOP_K     = 7
RERANK_TOP_K    = 5

# ── Model & DB Loading ────────────────────────────────────────────────────────

print("Loading Embedding model...")
embed_model = SentenceTransformer(EMBED_MODEL)

print("Loading Reranker model...")
reranker = CrossEncoder(RERANK_MODEL)

print("Connecting to ChromaDB...")
chroma_client     = chromadb.PersistentClient(path=CHROMA_PATH)
chroma_collection = chroma_client.get_or_create_collection("ipard_docs")

print("Loading data chunks...")
with open(CHUNKS_FILE, encoding="utf-8") as f:
    all_chunks: list[dict] = json.load(f)

print("Building BM25 index...")

def tokenize_tr(text: str) -> list[str]:
    """
    Simple Turkish tokenizer that converts to lowercase and removes punctuation.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()

bm25_corpus  = [tokenize_tr(c.get("text", "")) for c in all_chunks]
bm25_index   = BM25Okapi(bm25_corpus)
chunk_id_map = {c["chunk_id"]: i for i, c in enumerate(all_chunks)}

print(f"Ready. {len(all_chunks)} chunks loaded.\n")

# OpenRouter client setup (OpenAI 0.28.0 için güncellendi)
if GROQ_API_KEY:
    openai.api_base = "https://api.groq.com/openai/v1"
    openai.api_key = GROQ_API_KEY
    llm_client = openai  # Uyumluluk için llm_client olarak atıyoruz
else:
    llm_client = None
    print("WARNING: GROQ_API_KEY is not set.")


# ── Retrieval ─────────────────────────────────────────────────────────────────

def semantic_search(query: str, n: int, tedbir: str = None, doc_type: str = None) -> list[dict]:
    """
    Performs vector similarity search using ChromaDB with optional metadata filtering.
    Genel belgeler (measure="") her zaman dahil edilir.
    """
    # turkish-e5-large asimetrik modeldir: sorgular "query:" prefix gerektirir
    embedding = embed_model.encode([f"query: {query}"], normalize_embeddings=True).tolist()

    where = {}
    if tedbir and doc_type:
        # Eşleşen tedbir VEYA genel belge (measure=""), VE doc_type
        where = {
            "$and": [
                {"$or": [{"measure": tedbir}, {"measure": ""}]},
                {"doc_type": doc_type},
            ]
        }
    elif tedbir:
        # Eşleşen tedbir VEYA genel belge
        where = {"$or": [{"measure": tedbir}, {"measure": ""}]}
    elif doc_type:
        where = {"doc_type": doc_type}

    kwargs = {
        "query_embeddings": embedding,
        "n_results": n,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    results = chroma_collection.query(**kwargs)

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({
            "chunk_id": meta.get("source_file", "") + "_" + meta.get("heading", "")[:30],
            "text":     doc,
            "meta":     meta,
            "score":    1 - dist,
        })
    return hits


def bm25_search(query: str, n: int, tedbir: str = None, doc_type: str = None) -> list[dict]:
    """
    Performs keyword-based search using the BM25 algorithm with optional metadata filtering.
    Genel belgeler (measure="") her zaman dahil edilir.
    """
    tokens = tokenize_tr(query)
    scores = bm25_index.get_scores(tokens)

    filtered = []
    for i, chunk in enumerate(all_chunks):
        chunk_measure = chunk.get("measure", "")
        chunk_doc_type = chunk.get("doc_type", "")

        # Tedbir filtresi: eşleşen tedbir VEYA genel belge (measure boş)
        if tedbir and chunk_measure != tedbir and chunk_measure != "":
            continue
        if doc_type and chunk_doc_type != doc_type:
            continue
        filtered.append((i, scores[i]))

    if not filtered:
        filtered = list(enumerate(scores))

    filtered.sort(key=lambda x: x[1], reverse=True)
    top = filtered[:n]

    hits = []
    for idx, score in top:
        if score == 0:
            break
        chunk = all_chunks[idx]
        hits.append({
            "chunk_id": chunk["chunk_id"],
            "text":     chunk.get("text", ""),
            "meta": {
                "doc_type":    chunk.get("doc_type", ""),
                "measure":     chunk.get("measure", ""),
                "sector":      chunk.get("sector", ""),
                "heading":     chunk.get("heading", ""),
                "source_file": chunk.get("source_file", ""),
                "start_page":  str(chunk.get("start_page", "")),
            },
            "score": float(score),
        })
    return hits


def reciprocal_rank_fusion(semantic_hits, bm25_hits, k=60):
    """
    Merges semantic and keyword search results using Reciprocal Rank Fusion (RRF).
    """
    rrf_scores = {}
    chunk_data = {}

    for rank, hit in enumerate(semantic_hits):
        cid = hit["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (k + rank + 1)
        chunk_data[cid] = hit

    for rank, hit in enumerate(bm25_hits):
        cid = hit["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (k + rank + 1)
        if cid not in chunk_data:
            chunk_data[cid] = hit

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [{**chunk_data[cid], "rrf_score": score} for cid, score in ranked]


def rerank_chunks(query, chunks, top_k):
    """
    Re-scores the retrieved chunks using a Cross-Encoder for better semantic precision.
    """
    if not chunks:
        return []
    pairs = [(query, c["text"]) for c in chunks]
    scores = reranker.predict(pairs)
    for c, s in zip(chunks, scores):
        c["rerank_score"] = float(s)
    chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
    return chunks[:top_k]


def hybrid_search(query, top_k=FINAL_TOP_K, rerank_k=RERANK_TOP_K, tedbir=None, doc_type=None):
    """
    Orchestrates the hybrid retrieval process: Search -> Merge -> Rerank.
    """
    sem_hits  = semantic_search(query, SEMANTIC_TOP_N, tedbir, doc_type)
    bm25_hits = bm25_search(query, BM25_TOP_N, tedbir, doc_type)
    merged    = reciprocal_rank_fusion(sem_hits, bm25_hits)
    candidates = merged[:top_k]
    return rerank_chunks(query, candidates, rerank_k)


# ── Prompt & LLM ─────────────────────────────────────────────────────────────

import os as _os

def _load_system_prompt():
    prompt_path = _os.path.join(_os.path.dirname(__file__), "system_prompt.txt")
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "Sen IPARD III konusunda uzman bir asistansın. Türkçe yanıt ver."

SYSTEM_PROMPT = _load_system_prompt()

# ── Identity Question Handler ─────────────────────────────────────────────────

IDENTITY_KEYWORDS = [
    "sen kimsin", "kim siniz", "kim sin", "siz kimsiniz",
    "ne yapıyorsun", "ne yapıyorsunuz", "nasıl çalışıyorsun",
    "bu sistem ne", "bu uygulama ne", "nasıl yardımcı olabilirsin",
    "hangi konularda yardımcı", "neler yapabilirsin", "ne işe yarıyor",
    "hangi destekler var", "ipard nedir", "ipard ne",
    "hangi tedbirler var", "hangi sektörler var",
]

def is_identity_question(query: str) -> bool:
    q = query.lower().strip()
    return any(kw in q for kw in IDENTITY_KEYWORDS)

def answer_identity_question():
    """Returns a direct answer from system prompt without RAG retrieval."""
    if not GROQ_API_KEY:
        return "GROQ_API_KEY is not set."
    try:
        response = openai.ChatCompletion.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Sen kimsin ve hangi konularda yardımcı olabilirsin?"},
            ],
            temperature=0.3,
            max_tokens=512,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def answer_identity_question_stream():
    """Streams a direct answer from system prompt without RAG retrieval."""
    if not GROQ_API_KEY:
        yield {"type": "error", "detail": "GROQ_API_KEY is not set."}
        return
    try:
        stream = openai.ChatCompletion.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Sen kimsin ve hangi konularda yardımcı olabilirsin?"},
            ],
            temperature=0.3,
            max_tokens=512,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.get("content", "")
            if delta:
                yield {"type": "token", "text": delta}
    except Exception as e:
        yield {"type": "error", "detail": str(e)}

def build_context(chunks):
    """
    Formats retrieved chunks into a structured text context for the LLM.
    """
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("meta", {})
        header = f"[{i}] {meta.get('doc_type','')} | Measure {meta.get('measure','')}-{meta.get('sector','')} | {meta.get('heading','')[:60]}"
        if meta.get("source_file"):
            header += f" | {meta['source_file']}"
        parts.append(f"{header}\n{chunk['text'][:1200]}")
    return "\n\n---\n\n".join(parts)


def ask_qwen(query: str, chunks: list[dict]) -> str:
    """
    Sends the prompt and context to the LLM and returns the full response.
    OpenAI 0.28.0 için güncellendi.
    """
    if not GROQ_API_KEY:  # llm_client yerine GROQ_API_KEY kontrolü
        return "GROQ_API_KEY is not set."

    context = build_context(chunks)
    try:
        response = openai.ChatCompletion.create(  # OpenAI 0.28.0 syntax
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"DOCUMENTS:\n{context}\n\nQUESTION: {query}"},
            ],
            temperature=0.2,
            max_tokens=1024,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"


def ask_qwen_stream(query: str, chunks: list[dict]):
    """
    Sends the prompt and context to the LLM and yields tokens in a stream.
    OpenAI 0.28.0 için güncellendi.
    """
    if not GROQ_API_KEY:  # llm_client yerine GROQ_API_KEY kontrolü
        yield {"type": "error", "detail": "GROQ_API_KEY is not set."}
        return

    context = build_context(chunks)
    try:
        stream = openai.ChatCompletion.create(  # OpenAI 0.28.0 syntax
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"DOCUMENTS:\n{context}\n\nQUESTION: {query}"},
            ],
            temperature=0.2,
            max_tokens=1024,
            stream=True,
        )
        for chunk in stream:
            # OpenAI 0.28.0'da delta.content yerine choices[0].delta.get("content")
            delta = chunk.choices[0].delta.get("content", "")
            if delta:
                yield {"type": "token", "text": delta}
    except Exception as e:
        yield {"type": "error", "detail": str(e)}


# ── Main RAG Functions ────────────────────────────────────────────────────────

def rag_query(query, tedbir=None, doc_type=None, verbose=False):
    """
    Executes the full RAG cycle and returns the answer with metadata.
    """
    if is_identity_question(query):
        answer = answer_identity_question()
        return {"query": query, "answer": answer, "chunks": [], "sources": []}

    chunks = hybrid_search(query, FINAL_TOP_K, RERANK_TOP_K, tedbir, doc_type)

    if verbose:
        print(f"\nQUESTION: {query}")
        for i, c in enumerate(chunks, 1):
            meta = c.get("meta", {})
            print(f"  [{i}] RERANK={c.get('rerank_score',0):.4f} | {meta.get('heading','')[:50]}")

    answer = ask_qwen(query, chunks)

    sources = []
    for c in chunks:
        meta = c.get("meta", {})
        sources.append({
            "source_file":  meta.get("source_file", ""),
            "doc_type":     meta.get("doc_type", ""),
            "measure":      meta.get("measure", ""),
            "sector":       meta.get("sector", ""),
            "heading":      meta.get("heading", "")[:80],
            "rrf_score":    round(c.get("rrf_score", 0), 4),
            "rerank_score": round(c.get("rerank_score", 0), 4),
        })

    return {"query": query, "answer": answer, "chunks": chunks, "sources": sources}


def rag_query_stream(query, tedbir=None, doc_type=None):
    """
    Executes the full RAG cycle and yields results via a stream.
    """
    if is_identity_question(query):
        yield {"type": "sources", "sources": []}
        yield from answer_identity_question_stream()
        return

    chunks = hybrid_search(query, FINAL_TOP_K, RERANK_TOP_K, tedbir, doc_type)

    sources = []
    for c in chunks:
        meta = c.get("meta", {})
        sources.append({
            "source_file":  meta.get("source_file", ""),
            "doc_type":     meta.get("doc_type", ""),
            "measure":      meta.get("measure", ""),
            "sector":       meta.get("sector", ""),
            "heading":      meta.get("heading", "")[:80],
            "rrf_score":    round(c.get("rrf_score", 0), 4),
            "rerank_score": round(c.get("rerank_score", 0), 4),
        })
    yield {"type": "sources", "sources": sources}

    yield from ask_qwen_stream(query, chunks)


# ── Execution ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    TEST_QUERIES = [
        ("ÖTP nedir ve nasıl hazırlanır?", None, None),
        ("IPARD kapsamında canlı hayvan alımı destekleniyor mu?", None, None),
        ("101 tedbiri kapsamında maksimum ve minimum uygun harcama tutarları neler?", None, None),
        ("Yerel Eylem Grubu nedir?", None, None),
        ("ÖTP (Ödeme Talep Paketi) içerisinde bulunması gereken beyan dokümanlarını ve ilgili belgeleri nerden temin edebilirim?", None, None),
        ("Ödeme öncesi yerinde kontrolde eksiklikler tespit edildiğinde,bu eksiklikleri tamamlamak için ek süre verilmekte midir?",
        None, None),
        ("Ödeme öncesi yerinde kontroller neden yapılmaktadır?",None, None),
        ]
    for query, tedbir, doc_type in TEST_QUERIES:
        result = rag_query(query, tedbir=tedbir, doc_type=doc_type, verbose=True)
        print(f"\nANSWER:\n{result['answer']}")
        print("="*60)