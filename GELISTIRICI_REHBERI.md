# IPARD III Belge Asistanı — Geliştirici Rehberi

## Çalışma Dizini
Tüm değişiklikleri bu klasörde yap:
```
C:\Users\deniz\Desktop\IPARD_QA_RAG\IPARD-AI-Assistant
```

## Proje Yapısı ve Dosyaların Görevi

```
IPARD-AI-Assistant/
│
├── src/
│   ├── app.py              → Streamlit arayüzü (kullanıcının gördüğü ekran)
│   ├── api.py              → FastAPI backend (streaming endpoint'ler)
│   ├── rag_pipeline.py     → RAG mantığı (arama, reranking, LLM çağrısı)
│   ├── system_prompt.txt   → ⭐ LLM'in kişiliği, bilgisi ve kuralları (buradan güncelle)
│   ├── build_db.py         → ChromaDB'yi sıfırdan kurar (container başlarken çalışır)
│   ├── embed.py            → Embedding üretimi (data/embeddings.npy'yi oluşturur)
│   └── parser.py           → PDF parsing
│
├── data/
│   ├── all_chunks.json     → 7.700+ belge chunk'ı (LFS ile saklanır, değiştirme)
│   └── embeddings.npy      → Önceden hesaplanmış embedding vektörleri (LFS)
│
├── .github/
│   └── workflows/
│       └── huggingface.yml → GitHub → HuggingFace otomatik sync
│
├── Dockerfile              → Container yapılandırması
├── start.sh                → Container başlangıç scripti
├── requirements.txt        → Python bağımlılıkları
└── README.md               → Proje açıklaması
```

---

## En Sık Yapılan Değişiklik: Prompt Güncelleme

Modelin davranışını, bilgisini veya tonunu değiştirmek istediğinde:

1. `src/system_prompt.txt` dosyasını düzenle
2. PowerShell'i aç ve şunu çalıştır:

```powershell
cd C:\Users\deniz\Desktop\IPARD_QA_RAG\IPARD-AI-Assistant
git add src\system_prompt.txt
git commit -m "feat: system prompt guncellendi"
git push origin main
```

✅ GitHub Actions otomatik olarak HuggingFace'e deploy eder (~30 saniye).

---

## Diğer Değişiklikler

### Arayüzde değişiklik (app.py)
```powershell
git add src\app.py
git commit -m "feat: arayuz guncellendi"
git push origin main
```

### RAG mantığında değişiklik (rag_pipeline.py)
```powershell
git add src\rag_pipeline.py
git commit -m "feat: rag pipeline guncellendi"
git push origin main
```

### Birden fazla dosya değiştiyse
```powershell
git add src\
git commit -m "feat: kaynak kodlar guncellendi"
git push origin main
```

### Tüm değişiklikleri tek seferde push et
```powershell
git add .
git commit -m "feat: guncelleme"
git push origin main
```

---

## GitHub → HuggingFace Otomatik Sync Nasıl Çalışır?

```
Sen kodu düzenlersin
    ↓
git push origin main  →  GitHub'a gider
    ↓
GitHub Actions tetiklenir (.github/workflows/huggingface.yml)
    ↓
Kod HuggingFace'e push edilir (data/ klasörü hariç)
    ↓
HuggingFace Docker container'ı yeniden build eder
    ↓
~3-5 dakika sonra canlıya geçer
```

**NOT:** `data/` klasörü (all_chunks.json, embeddings.npy) otomatik sync'e dahil değildir.
Bu dosyaları değiştirmen gerekirse HuggingFace Files sekmesinden manuel yükle.

---

## Sık Kullanılan Git Komutları

```powershell
# Değişen dosyaları gör
git status

# Son commit'leri gör
git log --oneline -10

# Yanlış bir şey yaptıysan son commit'i geri al
git revert HEAD --no-edit
git push origin main

# Remote'ları kontrol et
git remote -v
```

---

## HuggingFace Secrets (API Key'ler)
Ayarlar için:
```
huggingface.co/spaces/denizkaya2022/IPARD-AI-Assistant/settings
```
- `GROQ_API_KEY` → Groq API anahtarı (LLM için)

---

## Build Hatasında Ne Yapmalı?

1. HuggingFace → Logs sekmesini aç
2. Hata mesajını oku
3. Genellikle şunlardan biri:
   - `data/` klasöründe eksik dosya → Manuel yükle
   - Python paketi sorunu → `requirements.txt` güncelle
   - Kod hatası → `src/` dosyalarını kontrol et

---

## Önemli Linkler
- **Uygulama:** https://huggingface.co/spaces/denizkaya2022/IPARD-AI-Assistant
- **GitHub:** https://github.com/denizkaya20/IPARD-AI-Assistant
- **GitHub Actions:** https://github.com/denizkaya20/IPARD-AI-Assistant/actions
- **HF Ayarlar:** https://huggingface.co/spaces/denizkaya2022/IPARD-AI-Assistant/settings
