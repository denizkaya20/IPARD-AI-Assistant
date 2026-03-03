"""
IPARD III - FAQ (Sıkça Sorulan Sorular) Scraper
Downloads and parses FAQ from tkdk.gov.tr/SikcaSorulanSorular

Usage: From the src/ directory
  python scrape_faq.py
"""

import json
import time
import logging
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
import urllib3

urllib3.disable_warnings()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

BASE_URL = "https://www.tkdk.gov.tr"
BASE_DIR = Path(__file__).resolve().parent.parent
CHUNK_DIR = BASE_DIR / "data" / "chunks"
CHUNK_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept-Language": "tr-TR,tr;q=0.9",
}

FAQ_CATEGORIES = {
    range(1, 15): ("on_site_control", "On-site Controls"),
    range(15, 30): ("accrual_unit", "Accrual > Accrual Unit"),
    range(30, 46): ("accrual_on_site", "Accrual > On-site Control Unit"),
}


def main():
    logging.info("Scraping FAQ from tkdk.gov.tr/SikcaSorulanSorular")

    for attempt in range(3):
        try:
            with httpx.Client(verify=False, headers=HEADERS, timeout=30) as client:
                resp = client.get(f"{BASE_URL}/SikcaSorulanSorular")
                resp.raise_for_status()
                soup = BeautifulSoup(resp.content, "html.parser")
                break
        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(3)
    else:
        logging.error("Could not retrieve FAQ page.")
        return

    faq_chunks = []
    seen = set()

    for q_tag in soup.find_all("a", href=lambda h: h and h.startswith("#collapse")):
        question = q_tag.get_text(strip=True)
        if not question:
            continue

        collapse_id = q_tag["href"].lstrip("#")
        try:
            collapse_num = int(collapse_id.replace("collapse", ""))
        except ValueError:
            continue

        answer_div = soup.find(id=collapse_id)
        answer = answer_div.get_text(strip=True) if answer_div else ""
        if not answer:
            continue

        norm_q = question.lower().strip()
        if norm_q in seen:
            continue
        seen.add(norm_q)

        cat_code, cat_label = "general", "General"
        for r, (code, label) in FAQ_CATEGORIES.items():
            if collapse_num in r:
                cat_code, cat_label = code, label
                break

        faq_chunks.append({
            "chunk_id": f"FAQ_{collapse_num:04d}",
            "heading": question,
            "text": f"SORU: {question}\nCEVAP: {answer}",
            "question": question,
            "answer": answer,
            "doc_type": "faq",
            "category_code": cat_code,
            "category_label": cat_label,
            "program": "IPARD",
            "char_count": len(question) + len(answer),
            "is_active": True
        })

    output_path = CHUNK_DIR / "faq_chunks.json"
    output_path.write_text(json.dumps(faq_chunks, ensure_ascii=False, indent=2), encoding="utf-8")

    logging.info(f"Done! {len(faq_chunks)} FAQ chunks saved to {output_path}")


if __name__ == "__main__":
    main()