"""Ingest PDFs and web pages into a Chroma vector database.

Sources can be provided three ways:
    1. PDF files in a folder (--pdf-dir, default: data/input/)
  2. URLs listed in a text file (--urls-file, default: urls.txt)
  3. Directly on the command line (--pdf file.pdf --url https://...)

HTML pages are scanned for linked PDF files, which are downloaded and
indexed automatically. Disable with --no-discover-pdfs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from rag_core import (
    COLLECTION_METADATA, DEFAULT_COLLECTION, DEFAULT_DATABASE, MODEL_NAME, validate_collection,
)

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from pdf_pipeline.models import ExtractionOptions

logger = logging.getLogger(__name__)

USER_AGENT = "RAG-ingest/1.0"
REQUEST_TIMEOUT = 30
CHUNK_SIZE = 900
# Chunk size is in characters; overlap is a number of sentences, not characters.
# Neither setting is the embedding model's tokenizer limit.
CHUNK_OVERLAP = 1
UPSERT_BATCH_SIZE = 500
PDF_MAGIC = b"%PDF"


def split_into_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text into sentence-aware chunks of about `chunk_size` characters."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    if overlap < 0:
        raise ValueError("overlap must not be negative")
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", " ".join(text.split()))
        if sentence.strip()
    ]
    chunks: list[str] = []
    current: list[str] = []
    current_length = 0

    for sentence in sentences:
        if len(sentence) > chunk_size:
            # A long sentence must still fit. Prefer spaces; hard-split a token
            # only when there is no whitespace before the size limit.
            if current:
                chunks.append(" ".join(current))
                current = []
                current_length = 0
            while len(sentence) > chunk_size:
                boundary = sentence.rfind(" ", 0, chunk_size + 1)
                if boundary <= 0:
                    boundary = chunk_size
                chunks.append(sentence[:boundary])
                sentence = sentence[boundary:].lstrip()
            if not sentence:
                continue
        added_length = len(sentence) + (1 if current else 0)
        if current and current_length + added_length > chunk_size:
            chunks.append(" ".join(current))
            current = current[-overlap:] if overlap else []
            # Overlap preserves nearby context only if it fits with the new text.
            while current and len(" ".join([*current, sentence])) > chunk_size:
                current.pop(0)
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


def extract_pdf(content: bytes, source: str, *, workers: int = 4,
                extraction: ExtractionOptions | None = None) -> list[dict[str, Any]]:
    """Extract one text record per PDF page from raw PDF bytes."""
    with tempfile.TemporaryDirectory(prefix="rag-download-") as directory:
        path = Path(directory) / "download.pdf"
        path.write_bytes(content)
        return load_pdf(path, source=source, workers=workers, extraction=extraction)


def load_pdf(
    path: Path, source: str | None = None, *, workers: int = 4,
    checkpoint_root: str | None = None, pages_per_task: int = 1,
    extraction: ExtractionOptions | None = None,
) -> list[dict[str, Any]]:
    """Extract one text record per page from a local PDF file."""
    from pdf_pipeline.main import run_pipeline

    resolved = Path(path).resolve()
    source = source or str(resolved)
    summary = run_pipeline(
        pdf_path=str(resolved), document_id="rag-document", workers=workers,
        output_root="", write_outputs=False,
        checkpoint_root=checkpoint_root, pages_per_task=pages_per_task,
        extraction=extraction,
    )
    if summary["status"] != "complete":
        raise RuntimeError(f"PDF extraction failed on pages: {summary['failed_pages']}")
    records = []
    for page in summary["pages"]:
        text = " ".join(page["text"].split())
        if text:
            records.append({
                "text": text,
                "metadata": {"source": source, "title": title_from_source(source),
                             "type": "pdf", "page": page["page_number"],
                             "extraction_method": page["extraction_method"]},
            })
    if not records:
        raise ValueError("PDF contains no readable text; use OCR for scanned pages")
    return records


def fetch_url(url: str) -> requests.Response:
    """Fetch a URL and raise on HTTP errors."""
    # Trusted-operator input only: this timeout is not an SSRF or download-size guard.
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


def load_url(url: str, *, workers: int = 4, extraction: ExtractionOptions | None = None) -> list[dict[str, Any]]:
    """Load one URL: a PDF is parsed page by page, HTML as a single record."""
    response = fetch_url(url)
    if is_pdf_response(response, url):
        return extract_pdf(response.content, url, workers=workers, extraction=extraction)
    return extract_html(response.text, url)


def discover_pdf_links(url: str, html: str) -> list[str]:
    """Return absolute URLs of .pdf files linked from an HTML page."""
    soup = BeautifulSoup(html, "html.parser")
    links = (urljoin(url, link["href"]) for link in soup.find_all("a", href=True))
    return list(dict.fromkeys(
        link for link in links if link.lower().split("?", 1)[0].endswith(".pdf")
    ))


def load_linked_pdfs(url: str, html: str, *, workers: int = 4,
                    extraction: ExtractionOptions | None = None) -> list[dict[str, Any]]:
    """Download and parse every PDF linked from an HTML page."""
    records: list[dict[str, Any]] = []
    for pdf_url in discover_pdf_links(url, html):
        logger.info("  [linked PDF] %s", pdf_url)
        try:
            records.extend(extract_pdf(fetch_url(pdf_url).content, pdf_url, workers=workers, extraction=extraction))
        except Exception as error:
            logger.warning("    Skipped linked PDF: %s", error)
    return records


def load_url_list(
    urls: list[str], discover_pdfs: bool, *, workers: int = 4,
    extraction: ExtractionOptions | None = None,
) -> list[dict[str, Any]]:
    """Load each URL once; optionally follow linked PDFs from HTML pages."""
    records: list[dict[str, Any]] = []
    for number, url in enumerate(urls, start=1):
        logger.info("[web %d/%d] Reading %s", number, len(urls), url)
        try:
            response = fetch_url(url)
            if is_pdf_response(response, url):
                records.extend(extract_pdf(response.content, url, workers=workers, extraction=extraction))
                continue
            records.extend(extract_html(response.text, url))
            if discover_pdfs:
                records.extend(load_linked_pdfs(url, response.text, workers=workers, extraction=extraction))
        except Exception as error:
            logger.warning("  Skipped: %s", error)
    return records


def load_pdf_paths(paths: list[str], *, workers: int = 4,
                   extraction: ExtractionOptions | None = None) -> list[dict[str, Any]]:
    """Load explicitly provided PDF files, skipping ones that fail."""
    records: list[dict[str, Any]] = []
    for path in paths:
        logger.info("Reading %s", path)
        try:
            records.extend(load_pdf(Path(path), workers=workers, extraction=extraction))
        except Exception as error:
            logger.warning("  Skipped %s: %s", path, error)
    return records


def load_pdf_dir(pdf_dir: Path, *, workers: int = 4,
                 extraction: ExtractionOptions | None = None) -> list[dict[str, Any]]:
    """Load every PDF under pdf_dir, skipping ones that fail."""
    if not pdf_dir.exists():
        logger.warning("PDF folder not found, continuing: %s", pdf_dir)
        return []
    pdf_paths = sorted(pdf_dir.rglob("*.pdf"))
    records: list[dict[str, Any]] = []
    for number, path in enumerate(pdf_paths, start=1):
        logger.info("[%d/%d] Reading %s", number, len(pdf_paths), path)
        try:
            records.extend(load_pdf(path, workers=workers, extraction=extraction))
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
                # ** copies source/page metadata into a new dict, then adds the
                # within-page chunk number. The original record stays unchanged.
                "metadata": {**record["metadata"], "chunk": chunk_number},
            })
    return chunks


def make_record_id(metadata: dict[str, Any], text: str) -> str:
    """Stable content hash so re-ingesting updates instead of duplicating."""
    # Content IDs deduplicate input. Stored vector IDs also include an immutable
    # publication revision so staging never overwrites a visible document.
    value = f"{metadata['source']}|{metadata['page']}|{metadata['chunk']}|{text}"
    if "token_part" in metadata:
        value += f"|token_part:{metadata['token_part']}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_index(
    chunks: list[dict[str, Any]],
    database: str,
    collection_name: str,
    reset: bool,
    *,
    model: Any = None,
    publication_generation: str | None = None,
) -> int | dict:
    """Stage complete immutable revisions and atomically publish their pointers."""
    import chromadb
    from chromadb.errors import NotFoundError
    from sentence_transformers import SentenceTransformer
    from index_publication import PublicationStore, REVISION_FIELD

    # Dictionary keys remove identical input chunks before passing IDs to Chroma.
    chunks = list({make_record_id(chunk["metadata"], chunk["text"]): chunk for chunk in chunks}.values())
    if not chunks:
        raise ValueError("Cannot index an empty set of chunks")
    if publication_generation is not None:
        if not re.fullmatch(r"[0-9a-f]{32}", publication_generation):
            raise ValueError("publication_generation must be a 32-character generation ID")
        if len({chunk["metadata"]["source"] for chunk in chunks}) != 1:
            raise ValueError("A publication receipt must describe exactly one source")
    logger.info("Creating embeddings for %d chunks...", len(chunks))
    if model is None:
        model = SentenceTransformer(MODEL_NAME)

    from embedding_chunks import fit_embedding_chunks

    # Enforce the actual tokenizer budget before touching collection/publication
    # state, including before a requested destructive reset.
    chunks = fit_embedding_chunks(chunks, model)
    input_hash = hashlib.sha256(json.dumps(chunks, sort_keys=True, ensure_ascii=False,
                                          separators=(",", ":")).encode("utf-8")).hexdigest()

    client = chromadb.PersistentClient(path=database)
    publications = PublicationStore(database)
    # Register the empty collection durably before staging any vectors. If the
    # first attempt crashes, its uncommitted vectors can be safely ignored on retry.
    with publications.writer() as connection:
        if reset:
            # Reset remains destructive maintenance: all readers must be stopped.
            try:
                client.delete_collection(collection_name)
            except NotFoundError:
                logger.info("No existing collection '%s' to delete.", collection_name)
        collection = client.get_or_create_collection(name=collection_name, metadata=COLLECTION_METADATA)
        validate_collection(collection)
        publications.bind(connection, collection_name, collection, reset)
    grouped: dict[str, list[dict]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk["metadata"]["source"], []).append(chunk)
    with publications.writer() as connection:
        publications.bind(connection, collection_name, collection, False)
        if publication_generation is not None:
            existing = publications.receipt(connection, collection_name, str(collection.id),
                                             publication_generation, input_hash)
            if existing is not None:
                # A committed generation must never be republished over a newer
                # source revision. Return the historical receipt, not a new write.
                return existing
        for source, document_chunks in grouped.items():
            revision = uuid.uuid4().hex
            expected_ids = set()
            for start in range(0, len(document_chunks), UPSERT_BATCH_SIZE):
                batch = document_chunks[start:start + UPSERT_BATCH_SIZE]
                vectors = model.encode([chunk["text"] for chunk in batch],
                                       show_progress_bar=True, normalize_embeddings=True).tolist()
                ids = [f"{revision}:{make_record_id(chunk['metadata'], chunk['text'])}" for chunk in batch]
                expected_ids.update(ids)
                collection.upsert(
                    ids=ids, documents=[chunk["text"] for chunk in batch], embeddings=vectors,
                    metadatas=[{**chunk["metadata"], REVISION_FIELD: revision} for chunk in batch],
                )
            actual_ids = collection.get(where={REVISION_FIELD: revision}, include=[])["ids"]
            if len(actual_ids) != len(expected_ids) or set(actual_ids) != expected_ids:
                raise RuntimeError("Staged revision does not contain every expected chunk")
            publications.publish(connection, collection_name, source, revision, len(expected_ids))
            if publication_generation is not None:
                receipt = publications.record_receipt(connection, collection_name, str(collection.id),
                                                       publication_generation, input_hash, source, revision,
                                                       len(expected_ids))
        # Context exit commits all pointers together. Old and failed revisions stay
        # stored so readers that pinned an earlier view can finish safely.
    logger.info("Published %d chunks across %d documents.", len(chunks), len(grouped))
    return receipt if publication_generation is not None else len(chunks)


class DocumentPublisher:
    """Shared local/Azure publication service; input adapters supply page records."""

    def __init__(self, database: str, collection: str):
        self.database = str(Path(database).resolve())
        self.collection = collection
        self.model = None

    def publish(self, records: list[dict], generation: str) -> dict:
        if not isinstance(generation, str) or not re.fullmatch(r"[0-9a-f]{32}", generation):
            raise ValueError("A valid processing generation is required")
        if not records or len({record["metadata"]["source"] for record in records}) != 1:
            raise ValueError("Publication requires nonempty records for exactly one source")
        if self.model is None:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(MODEL_NAME)
        # One model, chunking path, token budget, and atomic receipt protocol for
        # both adapters. No queue or cloud SDK belongs in this service.
        receipt = build_index(chunk_records(records), self.database, self.collection, False,
                              model=self.model, publication_generation=generation)
        from index_publication import verify_publication_receipt

        verify_publication_receipt(receipt, database=self.database, collection=self.collection,
                                   source=records[0]["metadata"]["source"], generation=generation)
        return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index PDF files and web pages into a Chroma vector database.",
    )
    parser.add_argument("--pdf-dir", default="data/input", help="Folder of local PDF files")
    parser.add_argument("--workers", type=int, default=4, help="PDF extraction worker processes")
    ExtractionOptions.add_arguments(parser)
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
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    try:
        args.extraction = ExtractionOptions.from_namespace(args)
    except ValueError as error:
        parser.error(str(error))
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    records = [
        *load_pdf_paths(args.pdf, workers=args.workers, extraction=args.extraction),
        *load_pdf_dir(Path(args.pdf_dir), workers=args.workers, extraction=args.extraction),
        *load_url_list(
            [*args.url, *read_urls_file(Path(args.urls_file))],
            args.discover_pdfs,
            workers=args.workers,
            extraction=args.extraction,
        ),
    ]
    chunks = chunk_records(records)
    if not chunks:
        raise SystemExit(
            "No sources were loaded. Add PDFs to data/input/, URLs to urls.txt, "
            "or pass --pdf/--url on the command line."
        )

    stored = build_index(chunks, args.database, args.collection, args.reset)
    logger.info("Indexed %d chunks in collection '%s'.", stored, args.collection)


if __name__ == "__main__":
    main()