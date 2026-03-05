"""
IPARD III - Streamlit English Interface (Engaging Face & Locally Compatible)
"""

import os
import json
import streamlit as st
import requests

# In Hugging Face, 127.0.0.1 is used because the services are in the same container.
# Scanning from environmental variables for AWS or Docker Compose.
# app.py (Arayüz)
import os
import streamlit as st

# 127.0.0.1 yerine 0.0.0.0 kullanarak Docker içindeki iç ağ sorununu çözüyoruz
API_URL = "http://127.0.0.1:8000"

# ─────────────────────────────────────────────────────────────
# Hard-coded answers for example questions (API token tasarrufu)
# ─────────────────────────────────────────────────────────────
HARDCODED_ANSWERS = {
    "Arıcılık projesi için minimum kovan sayısı kaçtır?": {
        "answer": """Tedbir 101 kapsamında arıcılık yatırımları için minimum **50 adet kovan** şartı aranmaktadır.

Proje sonunda ulaşılması gereken minimum kapasite de 50 kovandır. Başvuru yapabilmek için bu kapasiteye sahip olunması veya yatırım sonunda bu kapasiteye ulaşılması gerekmektedir.

**Kaynak:** Tedbir 101 - Başvuru Çağrı Rehberi""",
        "sources": []
    },
    "Başvuru için gerekli belgeler nelerdir?": {
        "answer": """IPARD III başvurusu için genel olarak aşağıdaki belgeler gerekmektedir:

**Zorunlu Belgeler:**
- Başvuru formu (e-devlet üzerinden doldurulur)
- Kimlik belgesi / imza sirküleri
- Tapu belgesi veya kira sözleşmesi (yatırım yeri için)
- Vergi levhası
- SGK borcu yoktur yazısı
- Ön Teklif Paketi (ÖTP) kapsamındaki teklifler
- İşletme belgesi (mevcut işletmeler için)
- Organik tarım sertifikası (varsa, ek puan için)

**Proje Türüne Göre Ek Belgeler:**
- Yapı ruhsatı veya izin belgesi
- Kapasite raporu
- Hayvan varlığını gösterir belgeler (hayvancılık projeleri için)

Belge listesi başvurulan tedbir ve sektöre göre değişmektedir. Güncel ve kesin liste için ilgili çağrı rehberini incelemenizi tavsiye ederiz.

**Kaynak:** Tedbir 101/103/302 - Başvuru Çağrı Rehberleri""",
        "sources": []
    },
    "Ödeme öncesi yerinde kontroller neden yapılmaktadır?": {
        "answer": """Ödeme öncesi yerinde kontroller aşağıdaki amaçlarla yapılmaktadır:

1. **Fiziki Doğrulama:** Yatırımın fiilen gerçekleştirildiğinin ve beyan edilen harcamaların uygunluğunun teyit edilmesi

2. **Miktar ve Kalite Kontrolü:** Satın alınan makine, ekipman veya yapıların teknik şartnameye uygunluğunun kontrol edilmesi

3. **AB Fonları Güvencesi:** AB katkısı içeren desteklerde fonların doğru kullanıldığının denetlenmesi

4. **Usulsüzlük Önleme:** Sahte veya abartılmış harcama beyanlarının önüne geçilmesi

Yerinde kontrol sırasında eksiklik tespit edilmesi halinde ek süre verilebilir veya eksik kısım uygun harcama dışında tutulabilir.

**Kaynak:** IPARD III - Ödeme ve Kontrol Prosedürleri""",
        "sources": []
    },
}

# ─────────────────────────────────────────────────────────────
# Page Settings
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IPARD III Belge Asistanı",
    page_icon="🌾",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────
# Legal Notice (First Use)
# ─────────────────────────────────────────────────────────────
if "legal_accept" not in st.session_state:
    st.session_state.legal_accept = False

if not st.session_state.legal_accept:
    st.markdown("""
    ## ⚠️ Kullanım Bilgilendirmesi

    Bu uygulama IPARD III belgeleri üzerinde çalışan **yapay zeka destekli bir bilgi sistemidir**.

    - Verilen yanıtlar otomatik oluşturulur.
    - Resmi kurum görüşü değildir.
    - Nihai ve bağlayıcı bilgi TKDK resmi belgeleridir.
    - Bu sistem bağımsız geliştirilmiştir ve kurumla resmi bir bağı yoktur.
    - Yalnızca kamuya açık belgeler kullanılır.

    Devam ederek bu koşulları kabul etmiş olursunuz.
    """)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("✅ Kabul Ediyorum", use_container_width=True):
            st.session_state.legal_accept = True
            st.rerun()
    with c2:
        st.stop()

# ─────────────────────────────────────────────────────────────
# CSS Design
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
.main-header { background: linear-gradient(135deg,#1a5c2a,#2d8a45); padding:1.5rem 2rem; border-radius:12px; color:white; margin-bottom:0.8rem; }
.disclaimer{ background:#fffbea; border:1px solid #f0c040; border-radius:8px; padding:0.7rem 1rem; font-size:0.85rem; margin-bottom:1rem; }
.answer-box{ background:#f8fffe; border-left:4px solid #2d8a45; padding:1.2rem 1.5rem; border-radius:8px; line-height:1.7; }
.source-card{ background:#f5f5f5; padding:0.6rem 1rem; border-radius:8px; margin:0.4rem 0; font-size:0.85rem; }
.score-badge{ background:#2d8a45; color:white; padding:2px 8px; border-radius:12px; font-size:0.75rem; font-weight:bold; }
.footer{ margin-top:3rem; border-top:1px solid #e0e0e0; padding-top:1rem; text-align:center; font-size:0.8rem; color:#888; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Title and Warning
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="main-header"><h1>🌾 IPARD III Belge Asistanı</h1><p>🤖 Yapay zeka destekli belge arama ve özetleme sistemi</p></div>', unsafe_allow_html=True)

st.markdown('<div class="disclaimer">⚠️ <b>Sorumluluk Reddi:</b> Bu uygulama bağımsız bir sistemdir. Yanıtlar otomatiktir ve <b>resmi kurum görüşü değildir</b>. Nihai bilgi için <a href="https://www.tkdk.gov.tr" target="_blank">tkdk.gov.tr</a> esas alınmalıdır.</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Side Menu (Filters)
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Filtreler")
    tedbir_options = {"Tümü": None, "101 — Tarımsal İşletmelerin Fiziki Varlıklarına Yönelik Yatırımlar": "101", "103 —Tarım ve Balıkçılık Ürünlerinin İşlenmesi ve Pazarlanması İle İlgili Fiziki Varlıklara Yönelik Yatırımlar": "103", "201 — Tarım-Çevre - İklim ve Organik Tarım": "201", "202 — Yerel Kalkınma Stratejilerinin Uygulanması-LEADER Yaklaşımı": "202", "302 — Çiftlik Faaliyetlerinin Çeşitlendirilmesi ve İş Geliştirme": "302"}
    selected_tedbir = tedbir_options[st.selectbox("Tedbir", list(tedbir_options.keys()))]

    doc_type_options = {"Tümü": None, "Çağrı Rehberi (Aktif)": "call_guide_active", "Çağrı Rehberi (Arşiv)": "call_guide_archive", "Başvuru Paketi": "application_package", "Bilgilendirme Notu": "information_notes", "SSS": "faq"}
    selected_doc_type = doc_type_options[st.selectbox("Belge Türü", list(doc_type_options.keys()))]

    st.divider()
    st.markdown("### 💡 Örnek Sorular")
    for ex in ["Arıcılık projesi için minimum kovan sayısı kaçtır?", "Başvuru için gerekli belgeler nelerdir?","Ödeme öncesi yerinde kontroller neden yapılmaktadır?"]:
        if st.button(ex, use_container_width=True):
            if ex in HARDCODED_ANSWERS:
                # Direkt hardcoded cevabı göster, API çağrısı yapma
                hardcoded = HARDCODED_ANSWERS[ex]
                st.session_state.history.insert(0, {
                    "query": ex,
                    "answer": hardcoded["answer"],
                    "sources": hardcoded["sources"],
                    "elapsed": 0
                })
                st.session_state["query_input"] = ""
                st.rerun()
            else:
                st.session_state["query_input"] = ex

# ─────────────────────────────────────────────────────────────
# Question Input and Processing
# ─────────────────────────────────────────────────────────────
if "history" not in st.session_state: st.session_state.history = []

query = st.text_area("Sorunuz", value=st.session_state.get("query_input", ""), height=90, placeholder="IPARD III hakkında merak ettiğiniz her şeyi sorabilirsiniz...")
submit = st.button("🔍 Yapay Zekaya Sor", use_container_width=True)

if submit and query.strip():
    st.session_state["query_input"] = ""
    try:
        # If the model in the backend expects a 'measure', we send it accordingly.
        with requests.post(f"{API_URL}/query/stream", json={"query": query, "measure": selected_tedbir, "doc_type": selected_doc_type}, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            st.markdown(f"**🙋 Soru:** {query}")
            answer_placeholder = st.empty()
            answer_text = ""
            sources_received = []

            for line in resp.iter_lines():
                if not line: continue
                line = line.decode("utf-8")
                if not line.startswith("data: "): continue
                event = json.loads(line[6:])

                if event["type"] == "sources": sources_received = event["sources"]
                elif event["type"] == "token":
                    answer_text += event["text"]
                    answer_placeholder.markdown(f'<div class="answer-box">{answer_text}▌</div>', unsafe_allow_html=True)
                elif event["type"] == "done":
                    answer_placeholder.markdown(f'<div class="answer-box">{answer_text}</div>', unsafe_allow_html=True)
                    st.session_state.history.insert(0, {"query": query, "answer": answer_text, "sources": sources_received, "elapsed": event.get("elapsed_ms", 0)})
                    st.rerun()
    except Exception as e:
        st.error(f"Bağlantı hatası: {e}. Lütfen servislerin çalıştığından emin olun.")

# ─────────────────────────────────────────────────────────────
# Past and Resources
# ─────────────────────────────────────────────────────────────
for item in st.session_state.history:
    with st.container():
        st.markdown(f"**🙋 {item['query']}**")
        st.markdown(f'<div class="answer-box">{item["answer"]}</div>', unsafe_allow_html=True)
        with st.expander(f"📄 Kaynaklar ({len(item['sources'])})"):
            for s in item["sources"]:
                st.markdown(f'<div class="source-card"><span class="score-badge">{s.get("rerank_score", 0):.2f}</span> <b>{s.get("doc_type","")}</b> | {s.get("heading","")}</div>', unsafe_allow_html=True)

st.markdown('<div class="footer">IPARD III Belge Asistanı — Nihai bilgi için resmi kanalları takip ediniz.</div>', unsafe_allow_html=True)