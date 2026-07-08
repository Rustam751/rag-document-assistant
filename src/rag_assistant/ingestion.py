"""PDF text extraction and chunking."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit of text with provenance metadata."""

    chunk_id: str
    text: str
    source: str  # original file name
    page: int  # 1-indexed page number


def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Extract text per page. Returns (page_number, text) pairs, skipping empty pages."""
    reader = PdfReader(str(pdf_path))
    pages: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((i, text))
    return pages


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks, preferring sentence boundaries.

    Guarantees forward progress even for pathological inputs (no separators).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            # Prefer breaking at a sentence end, then any whitespace,
            # but only if the break point keeps the chunk reasonably large.
            for candidate in (text.rfind(". ", start, end), text.rfind(" ", start, end)):
                if candidate > start + chunk_size // 2:
                    end = candidate + 1
                    break
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def chunk_pages(
    pages: list[tuple[int, str]],
    source: str,
    chunk_size: int,
    overlap: int,
) -> list[Chunk]:
    """Chunk extracted pages, preserving page-level provenance for citations."""
    chunks: list[Chunk] = []
    for page_number, page_text in pages:
        for i, piece in enumerate(split_text(page_text, chunk_size, overlap)):
            digest = hashlib.sha1(f"{source}:{page_number}:{i}:{piece[:64]}".encode()).hexdigest()
            chunks.append(
                Chunk(chunk_id=digest[:16], text=piece, source=source, page=page_number)
            )
    return chunks


def ingest_pdf(pdf_path: Path, chunk_size: int, overlap: int) -> list[Chunk]:
    """Full ingestion: extract pages from a PDF and chunk them."""
    pages = extract_pages(pdf_path)
    return chunk_pages(pages, source=pdf_path.name, chunk_size=chunk_size, overlap=overlap)
