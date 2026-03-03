---
title: IPARD III RAG
emoji: 🌾
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
app_port: 7860
---

# IPARD III Soru-Cevap Sistemi

IPARD III programına ait rehber belgeler üzerinde çalışan hibrit RAG sistemi.

## Özellikler
- Hibrit arama: BM25 + Semantik (Turkish-E5-Large)
- Cross-Encoder reranking
- Tedbir/belge tipi bazlı filtreleme
- Gerçek zamanlı streaming yanıt

## Secrets (HF Space Settings)
| Değişken | Açıklama |
|----------|----------|
| `GROQ_API_KEY` | Groq API anahtarı |
