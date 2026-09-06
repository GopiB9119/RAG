"""Ingest PDFs and web pages into a Chroma vector database.

Sources can be provided three ways:
  1. PDF files in a folder (--pdf-dir, default: documents/)
  2. URLs listed in a text file (--urls-file, default: urls.txt)
  3. Directly on the command line (--pdf file.pdf --url https://...)

HTML pages are scanned for linked PDF files, which are downloaded and
indexed automatically. Disable with --no-discover-pdfs.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import chromadb
import pymupdf
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

from rag_core import DEFAULT_COLLECTION, DEFAULT_DATABASE, MODEL_NAME

logger = logging.getLogger(__name__)

USER_AGENT = "RAG-ingest/1.0"
REQUEST_TIMEOUT = 30
CHUNK_SIZE = 900
CHUNK_OVERLAP = 1
UPSERT_BATCH_SIZE = 500
PDF_MAGIC = b"%PDF"


def split_into_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text into sentence-aware chunks of about `chunk_size` characters."""
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", " ".join(text.split()))
        if sentence.strip()
    ]
    chunks: list[str] = []
    current: list[str] = []
    current_length = 0

    for sentence in sentences:
        added_length = len(sentence) + (1 if current else 0)
        if current and current_length + added_length > chunk_size:
            chunks.append(" ".join(current))
            current = current[-overlap:] if overlap else []
            current_length = len(" ".join(current))

        current.append(sentence)
        current_length += len(sentence) + (1 if len(current) > 1 else 0)

    if current:
        chunks.append(" ".join(current))
    return chunks


def title_from_source(source: str) -> str:
    """Derive a display title from a URL or a file path."""
    without_query = source.split("?", 1)[0].rstrip("/\\")
    return re.split(r"[\\/]", without_query)[-1] or source


def extract_pdf(content: bytes, source: str) -> list[dict[str, Any]]:
    """Extract one text record per PDF page from raw PDF bytes."""
    pdf = pymupdf.open(stream=content, filetype="pdf")
    records = []
    for page_number, page in enumerate(pdf, start=1):
        text = " ".join(page.get_text().split())
        if text:
            records.append({
                "text": text,
                "metadata": {
                    "source": source,
                    "title": title_from_source(source),
                    "type": "pdf",
                    "page": page_number,
                },
            })
    if not records:
        raise ValueError("PDF contains no readable text; use OCR for scanned pages")
    return records


def load_pdf(path: Path, source: str | None = None) -> list[dict[str, Any]]:
    """Extract one text record per page from a local PDF file."""
    resolved = Path(path).resolve()
    return extract_pdf(resolved.read_bytes(), source or str(resolved))


def fetch_url(url: str) -> requests.Response:
    """Fetch a URL and raise on HTTP errors."""
    response = requests.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response


def is_pdf_response(response: requests.Response, url: str) -> bool:
    """Detect PDFs by content type, URL extension, or magic bytes."""
    content_type = response.headers.get("content-type", "").lower()
    url_path = url.split("?", 1)[0].lower()
    return (
        "application/pdf" in content_type
        or url_path.endswith(".pdf")
        or response.content.startswith(PDF_MAGIC)
    )


def extract_html(html: str, url: str) -> list[dict[str, Any]]:
    """Extract the readable text of an HTML page as a single record."""
    soup = BeautifulSoup(html, "html.parser")
    for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
        element.decompose()
    title = soup.title.get_text(" ", strip=True) if soup.title else url
    text = " ".join(soup.get_text(" ", strip=True).split())
    if not text:
        raise ValueError("No readable text found on the page")
    return [{
        "text": text,
        "metadata": {"source": url, "title": title, "type": "web", "page": "web"},
    }]


def load_url(url: str) -> list[dict[str, Any]]:
    """Load one URL: a PDF is parsed page by page, HTML as a single record."""
    response = fetch_url(url)
    if is_pdf_response(response, url):
        return extract_pdf(response.content, url)
    return extract_html(response.text, url)


def discover_pdf_links(url: str, html: str) -> list[str]:
    """Return absolute URLs of .pdf files linked from an HTML page."""
    soup = BeautifulSoup(html, "html.parser")
    links = (urljoin(url, link["href"]) for link in soup.find_all("a", href=True))
    return list(dict.fromkeys(
        link for link in links if link.lower().split("?", 1)[0].endswith(".pdf")
    ))


def load_linked_pdfs(url: str, html: str) -> list[dict[str, Any]]:
    """Download and parse every PDF linked from an HTML page."""
    records: list[dict[str, Any]] = []
    for pdf_url in discover_pdf_links(url, html):
        logger.info("  [linked PDF] %s", pdf_url)
        try:
            records.extend(extract_pdf(fetch_url(pdf_url).content, pdf_url))
        except Exception as error:
            logger.warning("    Skipped linked PDF: %s", error)
    return records


def load_url_list(urls: list[str], discover_pdfs: bool) -> list[dict[str, Any]]:
    """Load each URL once; optionally follow linked PDFs from HTML pages."""
    records: list[dict[str, Any]] = []
    for number, url in enumerate(urls, start=1):
        logger.info("[web %d/%d] Reading %s", number, len(urls), url)
        try:
            response = fetch_url(url)
            if is_pdf_response(response, url):
                records.extend(extract_pdf(response.content, url))
                continue
            records.extend(extract_html(response.text, url))
            if discover_pdfs:
                records.extend(load_linked_pdfs(url, response.text))
        except Exception as error:
            logger.warning("  Skipped: %s", error)
    return records


def load_pdf_paths(paths: list[str]) -> list[dict[str, Any]]:
    """Load explicitly provided PDF files, skipping ones that fail."""
    records: list[dict[str, Any]] = []
    for path in paths:
        logger.info("Reading %s", path)
        try:
            records.extend(load_pdf(Path(path)))
        except Exception as error:
            logger.warning("  Skipped %s: %s", path, error)
    return records


def load_pdf_dir(pdf_dir: Path) -> list[dict[str, Any]]:
    """Load every PDF under pdf_dir, skipping ones that fail."""
    if not pdf_dir.exists():
        logger.warning("PDF folder not found, continuing: %s", pdf_dir)
        return []
    pdf_paths = sorted(pdf_dir.rglob("*.pdf"))
    records: list[dict[str, Any]] = []
    for number, path in enumerate(pdf_paths, start=1):
        logger.info("[%d/%d] Reading %s", number, len(pdf_paths), path)
        try:
            records.extend(load_pdf(path))
        except Exception as error:
            logger.warning("  Skipped: %s", error)
    return records


def read_urls_file(urls_file: Path) -> list[str]:
    """Read one URL per line; blank lines and # comments are ignored."""
    if not urls_file.exists():
        logger.warning("URL file not found, continuing: %s", urls_file)
        return []
    return [
        line.strip()
        for line in urls_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def chunk_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Split each source record into sentence-aware chunks."""
    chunks: list[dict[str, Any]] = []
    for record in records:
        for chunk_number, text in enumerate(split_into_chunks(record["text"])):
            chunks.append({
                "text": text,
                "metadata": {**record["metadata"], "chunk": chunk_number},
            })
    return chunks


def make_record_id(metadata: dict[str, Any], text: str) -> str:
    """Stable content hash so re-ingesting updates instead of duplicating."""
    value = f"{metadata['source']}|{metadata['page']}|{metadata['chunk']}|{text}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_index(
    chunks: list[dict[str, Any]],
    database: str,
    collection_name: str,
    reset: bool,
) -> int:
    """Embed chunks and upsert them into Chroma; return the number stored."""
    logger.info("Creating embeddings for %d chunks...", len(chunks))
    model = SentenceTransformer(MODEL_NAME)
    vectors = model.encode(
        [chunk["text"] for chunk in chunks],
        show_progress_bar=True,
        normalize_embeddings=True,
    ).tolist()

    client = chromadb.PersistentClient(path=database)
    if reset:
        try:
            client.delete_collection(collection_name)
        except Exception:
            logger.info("No existing collection '%s' to delete.", collection_name)
    collection = client.get_or_create_collection(name=collection_name)

    for start in range(0, len(chunks), UPSERT_BATCH_SIZE):
        end = start + UPSERT_BATCH_SIZE
        batch = chunks[start:end]
        collection.upsert(
            ids=[make_record_id(chunk["metadata"], chunk["text"]) for chunk in batch],
            documents=[chunk["text"] for chunk in batch],
            embeddings=vectors[start:end],
            metadatas=[chunk["metadata"] for chunk in batch],
        )
        logger.info("Stored chunks %d-%d of %d", start + 1, min(end, len(chunks)), len(chunks))
    return len(chunks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index PDF files and web pages into a Chroma vector database.",
    )
    parser.add_argument("--pdf-dir", default="documents", help="Folder of local PDF files")
    parser.add_argument("--urls-file", default="urls.txt", help="Text file with one URL per line")
    parser.add_argument("--pdf", action="append", default=[], metavar="FILE",
                        help="Single PDF file to index (repeatable)")
    parser.add_argument("--url", action="append", default=[], metavar="URL",
                        help="Single web page or PDF URL to index (repeatable)")
    parser.add_argument("--database", default=DEFAULT_DATABASE, help="Chroma database folder")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="Chroma collection name")
    parser.add_argument("--reset", action="store_true", help="Delete the collection first")
    parser.add_argument(
        "--discover-pdfs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download .pdf files linked from HTML pages (default: enabled)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    records = [
        *load_pdf_paths(args.pdf),
        *load_pdf_dir(Path(args.pdf_dir)),
        *load_url_list(
            [*args.url, *read_urls_file(Path(args.urls_file))],
            args.discover_pdfs,
        ),
    ]
    chunks = chunk_records(records)
    if not chunks:
        raise SystemExit(
            "No sources were loaded. Add PDFs to documents/, URLs to urls.txt, "
            "or pass --pdf/--url on the command line."
        )

    stored = build_index(chunks, args.database, args.collection, args.reset)
    logger.info("Indexed %d chunks in collection '%s'.", stored, args.collection)


if __name__ == "__main__":
    main()