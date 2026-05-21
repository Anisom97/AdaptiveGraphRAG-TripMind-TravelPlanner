"""
Module 1 — PDF Ingestion Pipeline
Handles both Africa safari guide and Europe travel guide.
Detects doc type, extracts text per page, cleans and segments by country/city.
"""

import fitz  # PyMuPDF
import re
import json
import os
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional


# ── Document registry ──────────────────────────────────────────────────────────

DOCUMENT_REGISTRY = [
    {
        "source_id": "africa_safari_001",
        "title": "The Ultimate Guide to a Safari in Africa",
        "author": "The Explorer Society",
        "region": "africa",
        "doc_type": "safari_guide",
        "filename": "travel_guide_africa.pdf",
    },
    {
        "source_id": "europe_guide_001",
        "title": "Ultimate Quick Travel Guide — European Edition",
        "author": "Wandering Lewis",
        "region": "europe",
        "doc_type": "budget_guide",
        "filename": "travel_guide_europe.pdf",
    },
]

# Country/city markers for section detection
AFRICA_COUNTRIES = [
    "KENYA", "TANZANIA", "UGANDA", "RWANDA",
    "SOUTH AFRICA", "NAMIBIA", "BOTSWANA", "ZAMBIA", "ZIMBABWE",
]
EUROPE_CITIES = ["Rome", "Bucharest", "Pula", "Paris", "London", "Berlin"]

VIBE_KEYWORDS = [
    "nightlife", "beach", "history", "adventure", "budget",
    "luxury", "coastal", "cultural", "hidden gem", "safari",
    "wildlife", "gorilla", "desert", "scenic", "isolation",
    "crowded", "quiet", "family", "romantic", "backpacker",
]

BUDGET_SIGNALS = {
    "budget": ["cheap", "affordable", "free", "hostel", "backpacker", "low-cost"],
    "mid":    ["moderate", "mid-range", "comfortable", "standard"],
    "luxury": ["luxury", "premium", "high-end", "boutique", "private"],
}


@dataclass
class PageData:
    page_num: int
    text: str
    doc_id: str


@dataclass
class DocumentChunk:
    chunk_id: str
    text: str
    source_id: str
    source_title: str
    region: str
    doc_type: str
    page_num: int
    destination: str = "general"
    content_type: str = "general"
    vibes: list = field(default_factory=list)
    budget_tier: Optional[str] = None
    has_best_for: bool = False


# ── Core extraction ─────────────────────────────────────────────────────────────

def extract_pages(pdf_path: str, doc_meta: dict) -> list[PageData]:
    """Extract raw text from each page of a PDF."""
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if len(text) > 80:  # skip near-blank pages
            pages.append(PageData(
                page_num=i + 1,
                text=text,
                doc_id=doc_meta["source_id"],
            ))
    doc.close()
    print(f"  Extracted {len(pages)} pages from {doc_meta['filename']}")
    return pages


def clean_text(text: str) -> str:
    """Remove noise: headers, footers, URLs, extra whitespace."""
    text = re.sub(r"theexplorersociety\.com\s*\d*", "", text)
    text = re.sub(r"wanderinglewis\.com\s*\d*", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    text = text.strip()
    return text


def detect_destination(text: str, doc_type: str) -> str:
    """Detect which destination a text block is primarily about."""
    if doc_type == "safari_guide":
        text_upper = text.upper()
        for country in AFRICA_COUNTRIES:
            if country in text_upper:
                return country.title()
    else:
        for city in EUROPE_CITIES:
            if city in text:
                return city
    return "general"


def detect_vibes(text: str) -> list[str]:
    text_lower = text.lower()
    return [v for v in VIBE_KEYWORDS if v in text_lower]


def detect_budget_tier(text: str) -> Optional[str]:
    text_lower = text.lower()
    for tier, signals in BUDGET_SIGNALS.items():
        if any(s in text_lower for s in signals):
            return tier
    return None


def detect_content_type(text: str) -> str:
    text_lower = text.lower()
    if any(w in text_lower for w in ["review", "recommend", "loved", "amazing", "must-visit"]):
        return "review"
    if any(w in text_lower for w in ["best for", "highlights", "popular"]):
        return "highlights"
    if any(w in text_lower for w in ["budget", "cost", "price", "afford", "hostel"]):
        return "budget_tips"
    if any(w in text_lower for w in ["itinerary", "day 1", "days", "duration"]):
        return "itinerary"
    if any(w in text_lower for w in ["weather", "season", "rain", "dry", "wet"]):
        return "seasonal"
    return "guide"


# ── Chunking strategies ──────────────────────────────────────────────────────────

def chunk_by_tokens(text: str, max_tokens: int = 400, overlap: int = 50) -> list[str]:
    """Generic token-approximate chunking with overlap."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_tokens, len(words))
        chunk = " ".join(words[start:end])
        if len(chunk.strip()) > 80:
            chunks.append(chunk.strip())
        start += max_tokens - overlap
    return chunks


def chunk_africa_guide(pages: list[PageData], doc_meta: dict) -> list[DocumentChunk]:
    """
    Africa guide has structured per-country sections (KENYA, TANZANIA etc).
    Split on country headers, then further chunk long sections.
    """
    full_text = "\n\n".join(clean_text(p.text) for p in pages)
    page_map = {p.page_num: p.text for p in pages}

    pattern = r"\b(" + "|".join(AFRICA_COUNTRIES) + r")\b"
    parts = re.split(pattern, full_text)

    chunks = []
    current_country = "general"
    chunk_idx = 0

    for part in parts:
        part = part.strip()
        if part.upper() in AFRICA_COUNTRIES:
            current_country = part.title()
            continue
        if len(part) < 100:
            continue

        sub_chunks = chunk_by_tokens(part, max_tokens=400, overlap=50)
        for sub in sub_chunks:
            chunk_idx += 1
            chunks.append(DocumentChunk(
                chunk_id=f"{doc_meta['source_id']}_{current_country.lower().replace(' ', '_')}_{chunk_idx:04d}",
                text=sub,
                source_id=doc_meta["source_id"],
                source_title=doc_meta["title"],
                region=doc_meta["region"],
                doc_type=doc_meta["doc_type"],
                page_num=0,
                destination=current_country,
                content_type=detect_content_type(sub),
                vibes=detect_vibes(sub),
                budget_tier=detect_budget_tier(sub),
                has_best_for="BEST FOR" in sub.upper(),
            ))

    print(f"  Chunked Africa guide → {len(chunks)} chunks across {len(set(c.destination for c in chunks))} destinations")
    return chunks


def chunk_europe_guide(pages: list[PageData], doc_meta: dict) -> list[DocumentChunk]:
    """
    Europe guide is more free-form. Use page-level chunking
    with city detection per chunk.
    """
    chunks = []
    chunk_idx = 0

    for page in pages:
        text = clean_text(page.text)
        if not text:
            continue

        sub_chunks = chunk_by_tokens(text, max_tokens=400, overlap=50)
        for sub in sub_chunks:
            chunk_idx += 1
            dest = detect_destination(sub, doc_meta["doc_type"])
            chunks.append(DocumentChunk(
                chunk_id=f"{doc_meta['source_id']}_p{page.page_num}_{chunk_idx:04d}",
                text=sub,
                source_id=doc_meta["source_id"],
                source_title=doc_meta["title"],
                region=doc_meta["region"],
                doc_type=doc_meta["doc_type"],
                page_num=page.page_num,
                destination=dest,
                content_type=detect_content_type(sub),
                vibes=detect_vibes(sub),
                budget_tier=detect_budget_tier(sub),
                has_best_for=False,
            ))

    print(f"  Chunked Europe guide → {len(chunks)} chunks across {len(set(c.destination for c in chunks))} destinations")
    return chunks


# ── Main entry point ─────────────────────────────────────────────────────────────

def ingest_all(data_dir: str = "data/raw") -> list[DocumentChunk]:
    """
    Process all registered PDFs and return combined chunk list.
    Call this from m2 and m3 to get chunks for KG and vector store.
    """
    all_chunks = []
    sources_log = []

    for doc_meta in DOCUMENT_REGISTRY:
        pdf_path = os.path.join(data_dir, doc_meta["filename"])

        if not os.path.exists(pdf_path):
            print(f"  [SKIP] {doc_meta['filename']} not found at {pdf_path}")
            continue

        print(f"\nIngesting: {doc_meta['title']}")
        pages = extract_pages(pdf_path, doc_meta)

        if doc_meta["doc_type"] == "safari_guide":
            chunks = chunk_africa_guide(pages, doc_meta)
        else:
            chunks = chunk_europe_guide(pages, doc_meta)

        all_chunks.extend(chunks)
        sources_log.append({
            **doc_meta,
            "pages_extracted": len(pages),
            "chunks_created": len(chunks),
        })

    # Save sources manifest
    sources_path = os.path.join(data_dir, "..", "sources.json")
    with open(sources_path, "w") as f:
        json.dump(sources_log, f, indent=2)
    print(f"\nTotal chunks: {len(all_chunks)}")
    print(f"Sources manifest saved to {sources_path}")

    return all_chunks


if __name__ == "__main__":
    chunks = ingest_all(data_dir="data/raw")
    print("\nSample chunk:")
    if chunks:
        c = chunks[0]
        print(f"  ID: {c.chunk_id}")
        print(f"  Destination: {c.destination}")
        print(f"  Vibes: {c.vibes}")
        print(f"  Text[:100]: {c.text[:100]}")
