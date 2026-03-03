import re
import json
import fitz  # PyMuPDF
from pathlib import Path
from collections import Counter

# --- Directory Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent  # src -> ipard_rag
UNIQUE_DIR = BASE_DIR / "data" / "raw_pdfs"
CHUNKS_DIR = BASE_DIR / "data" / "chunks"
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

# --- Chunking Constraints (Characters) ---
TARGET_CHUNK_CHARS = 600  # Ideal target length
MAX_CHUNK_CHARS = 900  # Hard limit for splitting
OVERLAP_CHARS = 120  # Context overlap (~80 tokens)
MIN_CHUNK_CHARS = 80  # Chunks below this threshold are discarded

# --- Heading Detection Patterns (Hierarchical) ---
HEADING_PATTERNS = [
    # Level 3: e.g., 302.2.2  Honey and Other Apiculture
    re.compile(r'^(\d+\.\d+\.\d+)\.?\s+\S'),
    # Level 2: e.g., 2.3  SPECIFIC ELIGIBILITY
    re.compile(r'^(\d+\.\d+)\.?\s+[A-ZÇĞİÖŞÜa-zçğışöüA-Z\(]'),
    # Level 1: e.g., 2.  GENERAL INFORMATION
    re.compile(r'^(\d+)\.\s{1,5}[A-ZÇĞİÖŞÜ]'),
    # Annexes/Chapters: e.g., ANNEX 1, CHAPTER 3
    re.compile(r'^(EK|BÖLÜM|ANNEX|CHAPTER)\s+\d+', re.IGNORECASE),
]


def is_heading(line: str) -> bool:
    """Checks if a given line matches defined heading patterns."""
    line = line.strip()
    if len(line) < 4 or len(line) > 180:
        return False
    return any(p.match(line) for p in HEADING_PATTERNS)


def heading_level(line: str) -> int:
    """Returns the hierarchy level of the heading (1=highest, 3=lowest)."""
    line = line.strip()
    if re.match(r'^\d+\.\d+\.\d+', line): return 3
    if re.match(r'^\d+\.\d+', line):      return 2
    if re.match(r'^\d+\.', line):         return 1
    return 1


# --- Metadata Extraction from Filename ---
def get_doc_type(filename: str) -> str:
    """Categorizes the document based on keywords in the filename."""
    f = filename.lower()
    if "basvuru_cagri_rehberi" in f:        return "call_guide"
    if "ab_standartlari" in f:              return "eu_standards"
    if "otp_hazirlama" in f:               return "otp_guide"
    if "otp_hazırlama" in f:               return "otp_guide"
    if "yatirim_kapsaminda" in f:           return "investment_conditions"
    if "uyarici_levha" in f:               return "warning_notice"
    if "mesleki_yeterlilik" in f:           return "qualification_list"
    if "mesleki_yeterli" in f:             return "qualification_list"
    if "sektor_bilgi" in f:                return "sector_info"
    if "is_plani" in f:                    return "application_package"
    if "yhp_kullanim" in f:               return "application_package"
    if "basvuru_paketi" in f:             return "application_package"
    if "basvuru_formu" in f:              return "application_package"
    if "basvuru_yol" in f:                return "application_package"
    if "katma_degerli" in f:              return "value_added_products"
    if "meslek_gruplari" in f:            return "professional_groups"
    if "teknik_proje" in f:               return "technical_project"
    if "is_plani_hazirlama" in f:         return "application_package"
    if "gorsel" in f or "gorunum" in f:   return "visual_guide"
    if "gorunurluk" in f:                 return "visual_guide"
    if "leader" in f:                     return "leader"
    if "sss" in f or "sikca_sorulan" in f: return "faq"
    if "bilgilendirme" in f:              return "information_notes"
    if "hazirlama" in f:                  return "preparation_guide"
    return "general"


def get_metadata_from_filename(filename: str) -> dict:
    """
    Parses Measure, Sector info from the new standardized filename format.

    Supported patterns:
      101-1_Basvuru_Cagri_Rehberi.pdf   → measure=101, sector=1
      103_Yatirim_Kapsaminda_...pdf     → measure=103, sector=""
      302-7_Basvuru_Cagri_Rehberi.pdf  → measure=302, sector=7
      Leader_OTP_Hazirlama_Rehberi.pdf → measure=202, sector=""
      OTP_Hazirlama_Rehberi.pdf        → measure="",  sector=""  (general)
    """
    stem = Path(filename).stem
    meta = {"measure": "", "sector": "", "version": "", "program": "IPARD III"}

    # Pattern 1: "101-1_..." → tedbir + sektör
    ts_match = re.match(r'^(\d{3})-(\d+)_', stem)
    if ts_match:
        meta["measure"] = ts_match.group(1)
        meta["sector"]  = ts_match.group(2)
        return meta

    # Pattern 2: "101_..." → sadece tedbir
    t_match = re.match(r'^(\d{3})_', stem)
    if t_match:
        meta["measure"] = t_match.group(1)
        return meta

    # Pattern 3: Leader belgesi → tedbir 202
    if stem.lower().startswith("leader"):
        meta["measure"] = "202"
        return meta

    # Pattern 4: Prefix yok → genel belge, measure boş kalır
    # Bu belgeler tüm tedbir sorgularında dahil edilecek
    return meta


# --- PDF Processing ---
def extract_pages(pdf_path: Path) -> list:
    """Extracts text content from each page of the PDF."""
    try:
        doc = fitz.open(pdf_path)
        pages = []
        for i, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                pages.append({"page_num": i + 1, "text": text})
        doc.close()
        return pages
    except Exception as e:
        print(f"    Error reading PDF {pdf_path.name}: {e}")
        return []


# --- Content Enrichment ---
def build_chunk_text(measure: str, sector: str, doc_type: str,
                     heading: str, body: str) -> str:
    """
    Enriches the text for the Embedding model to provide better context.
    Format example:
        Measure: 302 | Sector: 2 | call_guide_active
        Chapter: 2.3 Specific Eligibility Criteria

        <body>
    """
    lines = []
    meta_parts = []
    if measure:  meta_parts.append(f"Measure: {measure}")
    if sector:   meta_parts.append(f"Sector: {sector}")
    if doc_type: meta_parts.append(doc_type)

    if meta_parts:
        lines.append(" | ".join(meta_parts))
    if heading:
        lines.append(f"Chapter: {heading}")

    lines.append("")  # Spacer
    lines.append(body)
    return "\n".join(lines)


# --- Text Splitting Strategy ---
def sliding_split(paragraphs: list, max_chars: int, overlap_chars: int) -> list[str]:
    """
    Splits a list of paragraphs into chunks while maintaining a character overlap.
    """
    result = []
    current = []
    current_len = 0
    last_para = ""

    for para in paragraphs:
        if current_len + len(para) > max_chars and current:
            result.append("\n".join(current))
            # Create overlap from the end of the last paragraph
            overlap = last_para[-overlap_chars:] if len(last_para) > overlap_chars else last_para
            current = [overlap, para] if overlap else [para]
            current_len = len(overlap) + len(para)
        else:
            current.append(para)
            current_len += len(para)
            last_para = para

    if current:
        result.append("\n".join(current))
    return result


# --- Core Chunking Logic ---
def chunk_document(pages: list, metadata: dict, chunk_prefix: str) -> list:
    """Segments a document into chunks based on headings and length limits."""
    measure = metadata.get("measure", "")
    sector = metadata.get("sector", "")
    doc_type = metadata.get("doc_type", "")

    chunks = []
    current_heading = "INTRODUCTION"
    current_body = []
    current_page = 1

    def flush(heading, body_lines, page):
        """Helper to finalize a chunk and apply sliding window if necessary."""
        body = "\n".join(body_lines).strip()
        if len(body) < MIN_CHUNK_CHARS:
            return

        # Use sliding window if the section exceeds MAX_CHUNK_CHARS
        paragraphs = body.split("\n")
        if len(body) <= MAX_CHUNK_CHARS:
            parts = [body]
        else:
            parts = sliding_split(paragraphs, MAX_CHUNK_CHARS, OVERLAP_CHARS)

        for idx, part in enumerate(parts):
            part = part.strip()
            if len(part) < MIN_CHUNK_CHARS:
                continue

            enriched = build_chunk_text(measure, sector, doc_type, heading, part)
            chunk_id = f"{chunk_prefix}_{len(chunks):04d}"
            if len(parts) > 1:
                chunk_id += f"_p{idx}"

            chunks.append({
                "chunk_id": chunk_id,
                "heading": heading,
                "text": part,  # Raw text for BM25 (keyword search)
                "enriched_text": enriched,  # Enriched text for Vector Embedding
                "context_prefix": f"Measure {measure} Sector {sector} | {doc_type} | {heading[:80]}",
                "start_page": page,
                "char_count": len(part),
                **metadata,
            })

    for page in pages:
        lines = page["text"].split("\n")
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if is_heading(stripped):
                flush(current_heading, current_body, current_page)
                current_heading = stripped
                current_body = []
                current_page = page["page_num"]
            else:
                current_body.append(stripped)

    flush(current_heading, current_body, current_page)
    return chunks


# --- Main Processing Loop ---
def process_all():
    """Main entry point to process all PDFs in the input directory."""
    pdf_files = sorted(UNIQUE_DIR.glob("*.pdf"))
    print(f"Looking in: {UNIQUE_DIR.resolve()}")
    print(f"PDF files to process: {len(pdf_files)}")

    all_chunks = []
    stats = Counter()

    for pdf_path in pdf_files:
        doc_category = get_doc_type(pdf_path.name)
        meta = get_metadata_from_filename(pdf_path.name)
        meta["doc_type"] = doc_category
        meta["source_file"] = pdf_path.name
        # Genel belgeler (measure="") tüm tedbir sorgularına dahil edilir
        meta["is_active"] = "True"

        pages = extract_pages(pdf_path)
        if not pages:
            print(f"  EMPTY/ERROR: {pdf_path.name}")
            continue

        chunks = chunk_document(pages, meta, pdf_path.stem)
        stats[doc_category] += len(chunks)
        all_chunks.extend(chunks)

        # Save individual chunk file
        out = CHUNKS_DIR / f"{pdf_path.stem}_chunks.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        print(f"  {pdf_path.name[:55]:55} → {len(pages):3d} pages, {len(chunks):4d} chunks")

    # Add FAQ (SSS) chunks if available
    faq_path = CHUNKS_DIR / "sss_chunks.json"
    if faq_path.exists():
        with open(faq_path, encoding="utf-8") as f:
            faq_chunks = json.load(f)

        for c in faq_chunks:
            if "enriched_text" not in c:
                c["enriched_text"] = build_chunk_text(
                    c.get("measure", ""), c.get("sector", ""),
                    "faq", c.get("heading", ""), c.get("text", "")
                )
        all_chunks.extend(faq_chunks)
        stats["faq"] = len(faq_chunks)
        print(f"\nFAQ chunks added: {len(faq_chunks)}")

    # Save master chunk file
    out_path = BASE_DIR / "data" / "all_chunks.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    # Output Statistics Summary
    print(f"\n{'=' * 60}")
    print(f"Total Chunks Generated: {len(all_chunks)}")

    if all_chunks:
        for t, c in sorted(stats.items()):
            print(f"  {t:30}: {c}")

        char_counts = [c["char_count"] for c in all_chunks]
        print(f"Avg Chars: {sum(char_counts) // len(all_chunks)} | Min: {min(char_counts)} | Max: {max(char_counts)}")

        # Length Distribution
        buckets = Counter()
        for x in char_counts:
            if x < 300:
                buckets["<300"] += 1
            elif x < 600:
                buckets["300-600"] += 1
            elif x < 900:
                buckets["600-900"] += 1
            else:
                buckets[">900"] += 1

        print("Distribution:", dict(buckets))
    else:
        print("No chunks generated – check that PDFs exist in:", UNIQUE_DIR.resolve())

    print(f"Master file saved: {out_path}")


if __name__ == "__main__":
    process_all()