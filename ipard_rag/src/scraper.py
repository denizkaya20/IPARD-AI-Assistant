"""
IPARD III - Master Scraper
Downloads all document types in a single execution:
  - Application Call Guides (PDF)
  - Information Notes (PDF)
  - Application Packages (PDF)
  - Preparation Documents (PDF)
  - FAQ - Frequently Asked Questions (HTML → JSON)

Usage: From the src/ directory
  python scrape_all.py
"""

import json
import re
import time
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any

import httpx
from bs4 import BeautifulSoup
import urllib3

# Suppress insecure request warnings for legacy SSL support
urllib3.disable_warnings()

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

# --- Configuration & Constants ---
BASE_URL = "https://www.tkdk.gov.tr"
BASE_DIR = Path(__file__).resolve().parent.parent
PDF_DIR = BASE_DIR / "data" / "raw_pdfs"
CHUNK_DIR = Path("data/chunks")
META_FILE = Path("data/documents_metadata.json")

# Ensure required directories exist
PDF_DIR.mkdir(parents=True, exist_ok=True)
CHUNK_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "tr-TR,tr;q=0.9",
}

# --- Target Data Definitions ---

# URLs for Application Call Guides categorized by Measure (Tedbir) and Sector
CALL_GUIDE_PAGES = [
    {"url": "/BasvuruCagriRehberi/101-1-sut-ureten-tarimsal-isletmeler-30", "measure": "101", "sector": "1"},
    {"url": "/BasvuruCagriRehberi/101-2-kirmizi-et-ureten-tarimsal-isletmeler-31", "measure": "101", "sector": "2"},
    {"url": "/BasvuruCagriRehberi/101-3-kanatli-eti-ureten-tarimsal-isletmeler-32", "measure": "101", "sector": "3"},
    {"url": "/BasvuruCagriRehberi/101-4-yumurta-ureten-tarimsal-isletmeler-33", "measure": "101", "sector": "4"},
    {"url": "/BasvuruCagriRehberi/103-1-sut-ve-sut-urunlerinin-islenmesi-ve-pazarlanmasi-34", "measure": "103",
     "sector": "1"},
    {"url": "/BasvuruCagriRehberi/103-2-kirmizi-et-ve-et-urunlerinin-islenmesi-ve-pazarlanmasi-35", "measure": "103",
     "sector": "2"},
    {"url": "/BasvuruCagriRehberi/103-3-kanatli-eti-ve-et-urunlerinin-islenmesi-ve-pazarlanmasi-36", "measure": "103",
     "sector": "3"},
    {"url": "/BasvuruCagriRehberi/103-4-su-urunlerinin-islenmesi-ve-pazarlanmasi-37", "measure": "103", "sector": "4"},
    {"url": "/BasvuruCagriRehberi/103-5-meyve-ve-sebze-urunlerinin-islenmesi-ve-pazarlanmasi-38", "measure": "103",
     "sector": "5"},
    {"url": "/BasvuruCagriRehberi/103-6-yumurtanin-islenmesi-ve-pazarlanmasi-39", "measure": "103", "sector": "6"},
    {"url": "/BasvuruCagriRehberi/201-1-toprak-ortusu-yonetimi-ve-toprak-erozyonu-kontrolu-27", "measure": "201",
     "sector": "1"},
    {"url": "/BasvuruCagriRehberi/201-2-biyocesitlilik-toy-kusu-populasyonunun-gelistirilmesi-47", "measure": "201",
     "sector": "2"},
    {"url": "/BasvuruCagriRehberi/202-0-yerel-kalkinma-stratejilerinin-uygulanmasi-leader-yaklasimi-48",
     "measure": "202", "sector": "0"},
    {
        "url": "/BasvuruCagriRehberi/302-1-bitkisel-uretimin-cesitlendirilmesi-ve-bitkisel-urunlerin-islenmesi-ve-paketlenmesi-40",
        "measure": "302", "sector": "1"},
    {"url": "/BasvuruCagriRehberi/302-2-aricilik-ve-ari-urunlerinin-uretimi-islenmesi-ve-paketlenmesi-41",
     "measure": "302", "sector": "2"},
    {"url": "/BasvuruCagriRehberi/302-3-zanaatkarlik-ve-yerel-urun-isletmeleri-42", "measure": "302", "sector": "3"},
    {"url": "/BasvuruCagriRehberi/302-4-kirsal-turizm-ve-rekreasyon-faaliyetleri-43", "measure": "302", "sector": "4"},
    {"url": "/BasvuruCagriRehberi/302-5-su-urunleri-yetistiriciligi-44", "measure": "302", "sector": "5"},
    {"url": "/BasvuruCagriRehberi/302-6-makine-parklari-45", "measure": "302", "sector": "6"},
    {"url": "/BasvuruCagriRehberi/302-7-yenilenebilir-enerji-yatirimlari-46", "measure": "302", "sector": "7"},
]

# Mapping for Project Operation pages (Information packs, preparation docs, etc.)
SECTORS = [
    {"id": 1, "measure": "101", "sector": "1"},
    {"id": 2, "measure": "101", "sector": "2"},
    {"id": 11, "measure": "101", "sector": "3"},
    {"id": 12, "measure": "101", "sector": "4"},
    {"id": 3, "measure": "103", "sector": "1"},
    {"id": 4, "measure": "103", "sector": "2"},
    {"id": 5, "measure": "103", "sector": "3"},
    {"id": 6, "measure": "103", "sector": "4"},
    {"id": 13, "measure": "103", "sector": "5"},
    {"id": 29, "measure": "103", "sector": "6"},
    {"id": 28, "measure": "202", "sector": "0"},
    {"id": 7, "measure": "302", "sector": "1"},
    {"id": 8, "measure": "302", "sector": "2"},
    {"id": 9, "measure": "302", "sector": "3"},
    {"id": 10, "measure": "302", "sector": "4"},
    {"id": 14, "measure": "302", "sector": "5"},
    {"id": 15, "measure": "302", "sector": "6"},
    {"id": 16, "measure": "302", "sector": "7"},
]

PROJECT_DOC_TYPES = [
    ("info_notes", "/ProjeIslemleri/BilgiKartlari/"),
    ("application_package", "/ProjeIslemleri/BasvuruPaketi/"),
    ("preparation_docs", "/ProjeIslemleri/BasvuruPaketiHazirlamaDokumanlari/"),
]

# FAQ Category mapping based on collapse ID ranges
FAQ_CATEGORIES = {
    range(1, 15): ("on_site_control", "On-site Controls"),
    range(15, 30): ("accrual_unit", "Accrual > Accrual Unit"),
    range(30, 46): ("accrual_on_site", "Accrual > On-site Control Unit"),
}


# --- HTTP Helper Functions ---

def fetch_soup(url: str) -> Optional[BeautifulSoup]:
    """Fetches HTML content from a URL and returns a BeautifulSoup object."""
    full_url = url if url.startswith("http") else BASE_URL + url
    for attempt in range(3):
        try:
            with httpx.Client(verify=False, headers=HEADERS, timeout=30, http2=True) as client:
                response = client.get(full_url)
                response.raise_for_status()
                return BeautifulSoup(response.content, "html.parser")
        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed for {full_url}: {e}")
            time.sleep(3)
    return None


def download_pdf(url: str, filepath: Path) -> bool:
    """Downloads a PDF from a URL and saves it to the specified path."""
    if filepath.exists():
        return True  # Skip if already exists
    try:
        with httpx.Client(verify=False, headers=HEADERS, timeout=60) as client:
            response = client.get(url)
            response.raise_for_status()
            filepath.write_bytes(response.content)
        logging.info(f"Downloaded: {filepath.name} ({len(response.content) // 1024} KB)")
        return True
    except Exception as e:
        logging.error(f"Failed to download {url}: {e}")
        return False


def get_md5(path: Path) -> str:
    """Calculates the MD5 hash of a file for duplicate detection."""
    return hashlib.md5(path.read_bytes()).hexdigest()


# --- Main Scraping Functions ---

def scrape_call_guides() -> List[Dict[str, Any]]:
    """Scrapes IPARD III PDF links from measure-sector pages and downloads them."""
    logging.info("Step 1: Scraping Application Call Guides")
    docs_metadata = []

    for page in CALL_GUIDE_PAGES:
        logging.info(f"Processing Measure {page['measure']}-{page['sector']}")
        soup = fetch_soup(page["url"])
        if not soup:
            continue

        found_items = []
        seen_urls = set()
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "/IPARDIII/" not in href:
                continue

            full_url = BASE_URL + "/" + href if not href.startswith("http") else href
            if full_url in seen_urls:
                continue
            seen_urls.add(full_url)

            version_match = re.search(r"/IPARDIII/([\d.]+)/", href)
            version = version_match.group(1) if version_match else "0"
            found_items.append({"url": full_url, "version": version})

        if not found_items:
            logging.warning(f"No PDFs found for {page['url']}")
            continue

        # Determine the highest version to mark as active
        max_version = max(found_items, key=lambda x: float(x["version"]))["version"]

        for item in found_items:
            is_active = item["version"] == max_version
            status_suffix = "ACTIVE" if is_active else "ARCHIVE"
            filename = f"IPARDIII_T{page['measure']}_S{page['sector']}_v{item['version']}_{status_suffix}.pdf"
            filepath = PDF_DIR / filename

            download_pdf(item["url"], filepath)

            docs_metadata.append({
                "local_filename": filename,
                "url": item["url"],
                "program": "IPARD III",
                "measure": page["measure"],
                "sector": page["sector"],
                "doc_type": "application_call_guide",
                "version": item["version"],
                "is_active": is_active,
            })
        time.sleep(1)

    return docs_metadata


def scrape_project_documents() -> List[Dict[str, Any]]:
    """Downloads Information Notes, Application Packages, and Preparation docs."""
    logging.info("Step 2: Scraping Project Operation Documents")
    docs_metadata = []

    for sector in SECTORS:
        sid = sector["id"]
        measure = sector["measure"]
        sec = sector["sector"]
        logging.info(f"Processing Measure {measure}-{sec} (ID: {sid})")

        for doc_type, url_prefix in PROJECT_DOC_TYPES:
            url = f"{BASE_URL}{url_prefix}{sid}"
            soup = fetch_soup(url)
            if not soup:
                continue

            pdf_links = []
            seen = set()
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if "/Dokuman/" not in href and not href.lower().endswith(".pdf"):
                    continue
                full_path = BASE_URL + href if href.startswith("/") else href
                if full_path in seen:
                    continue
                seen.add(full_path)
                pdf_links.append(full_path)

            for i, pdf_url in enumerate(pdf_links):
                filename = f"IPARDIII_T{measure}_S{sec}_{doc_type}_{i:02d}.pdf"
                filepath = PDF_DIR / filename
                download_pdf(pdf_url, filepath)

                docs_metadata.append({
                    "local_filename": filename,
                    "url": pdf_url,
                    "program": "IPARD III",
                    "measure": measure,
                    "sector": sec,
                    "doc_type": doc_type,
                    "version": None,
                    "is_active": True,
                })
            time.sleep(0.5)
        time.sleep(1)

    return docs_metadata


def scrape_faq() -> List[Dict[str, Any]]:
    """Parses the FAQ page and generates JSON chunks for processing."""
    logging.info("Step 3: Scraping Frequently Asked Questions (FAQ)")
    soup = fetch_soup("/SikcaSorulanSorular")
    if not soup:
        logging.error("Could not retrieve FAQ page.")
        return []

    faq_chunks = []
    seen_questions = set()

    for q_tag in soup.find_all("a", href=lambda h: h and h.startswith("#collapse")):
        question_text = q_tag.get_text(strip=True)
        if not question_text:
            continue

        collapse_id = q_tag["href"].lstrip("#")
        try:
            collapse_num = int(collapse_id.replace("collapse", ""))
        except ValueError:
            continue

        answer_div = soup.find(id=collapse_id)
        answer_text = answer_div.get_text(strip=True) if answer_div else ""
        if not answer_text:
            continue

        # Deduplication check
        norm_q = question_text.lower().strip()
        if norm_q in seen_questions:
            continue
        seen_questions.add(norm_q)

        # Categorize based on ID range
        cat_code, cat_label = "general", "General"
        for r, (code, label) in FAQ_CATEGORIES.items():
            if collapse_num in r:
                cat_code, cat_label = code, label
                break

        faq_chunks.append({
            "chunk_id": f"FAQ_{collapse_num:04d}",
            "heading": question_text,
            "text": f"QUESTION: {question_text}\nANSWER: {answer_text}",
            "question": question_text,
            "answer": answer_text,
            "doc_type": "faq",
            "category_code": cat_code,
            "category_label": cat_label,
            "program": "IPARD",
            "char_count": len(question_text) + len(answer_text),
            "is_active": True
        })

    output_path = CHUNK_DIR / "faq_chunks.json"
    output_path.write_text(json.dumps(faq_chunks, ensure_ascii=False, indent=2), encoding="utf-8")

    logging.info(f"Generated {len(faq_chunks)} FAQ chunks -> {output_path}")
    return faq_chunks


def deduplicate_pdfs(metadata_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Identifies and removes duplicate PDF files using MD5 checksums."""
    logging.info("Step 4: Cleaning Duplicate Files")
    seen_hashes: Dict[str, str] = {}
    unique_metadata = []
    removed_count = 0

    for entry in metadata_list:
        file_path = PDF_DIR / entry["local_filename"]
        if not file_path.exists():
            continue

        file_hash = get_md5(file_path)
        if file_hash in seen_hashes:
            file_path.unlink()  # Delete the duplicate file
            removed_count += 1
        else:
            seen_hashes[file_hash] = entry["local_filename"]
            unique_metadata.append(entry)

    logging.info(f"Total: {len(metadata_list)} | Unique: {len(unique_metadata)} | Removed: {removed_count}")
    return unique_metadata


def save_metadata(metadata: List[Dict[str, Any]]) -> None:
    """Saves the final metadata list to a JSON file."""
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    logging.info(f"Metadata saved to {META_FILE}")


# --- Execution Entry Point ---

def main():
    start_time = time.time()
    logging.info("Starting IPARD III Master Scraper")

    # 1. Scrape Call Guides
    call_guides = scrape_call_guides()

    # 2. Scrape Project Documents
    project_docs = scrape_project_documents()

    # 3. Scrape FAQ
    scrape_faq()

    # 4. Consolidate and Deduplicate
    all_pdfs = call_guides + project_docs
    unique_pdfs = deduplicate_pdfs(all_pdfs)

    # 5. Save Final Metadata
    save_metadata(unique_pdfs)

    duration = time.time() - start_time
    logging.info("=" * 30)
    logging.info(f"PROCESS COMPLETE in {duration:.2f}s")
    logging.info(f"Unique PDFs: {len(unique_pdfs)}")
    logging.info("=" * 30)


if __name__ == "__main__":
    main()