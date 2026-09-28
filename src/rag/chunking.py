"""Sentence-aware page chunks that always fit the embedding model's token limit."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Callable

CHUNK_CHARACTERS = 900
OVERLAP_SENTENCES = 1


@dataclass(frozen=True)
class Chunk:
    text: str
    page: int
    index: int
    part: int = 0
    method: str = "native"


def split_into_chunks(text: str, chunk_size: int = CHUNK_CHARACTERS, overlap: int = OVERLAP_SENTENCES) -> list[str]:
    if chunk_size < 1 or overlap < 0:
        raise ValueError("chunk_size must be positive and overlap non-negative")
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", " ".join(text.split()))
                 if sentence.strip()]
    chunks: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        if len(sentence) > chunk_size:
            if current:
                chunks.append(" ".join(current))
                current = []
            while len(sentence) > chunk_size:
                boundary = sentence.rfind(" ", 0, chunk_size + 1)
                if boundary <= 0:
                    boundary = chunk_size
                chunks.append(sentence[:boundary])
                sentence = sentence[boundary:].lstrip()
            if not sentence:
                continue
        if current and len(" ".join([*current, sentence])) > chunk_size:
            chunks.append(" ".join(current))
            current = current[-overlap:] if overlap else []
            # Overlap keeps neighbouring context only when it still fits.
            while current and len(" ".join([*current, sentence])) > chunk_size:
                current.pop(0)
        current.append(sentence)
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_pages(pages: list[dict]) -> list[Chunk]:
    """pages: [{"page": 1-based number, "text": str, "method": str}] in reading order."""
    return [Chunk(text, page["page"], number, 0, page.get("method", "native"))
            for page in pages for number, text in enumerate(split_into_chunks(page["text"]))]


def fit_to_token_limit(chunks: list[Chunk], count_tokens: Callable[[str], int], limit: int) -> list[Chunk]:
    """Split oversized chunks near whitespace; joined parts reproduce the text exactly."""
    if limit < 1:
        raise ValueError("Token limit must be positive")
    if count_tokens("") > limit:
        raise ValueError("Token limit cannot accommodate special tokens")
    fitted: list[Chunk] = []
    for chunk in chunks:
        if not chunk.text.strip():
            raise ValueError("Chunks must contain text")
        pending, parts = [chunk.text], []
        while pending:
            part = pending.pop()
            if count_tokens(part) <= limit:
                parts.append(part)
                continue
            if len(part) < 2:
                raise ValueError("A single character exceeds the token limit")
            middle = len(part) // 2
            boundary = part.rfind(" ", 0, middle + 1) + 1
            if boundary <= 0 or boundary >= len(part) or boundary < middle // 2:
                boundary = middle
            pending.extend([part[boundary:], part[:boundary]])
        if "".join(parts) != chunk.text:
            raise RuntimeError("Token splitting did not preserve source text")
        fitted.extend(replace(chunk, text=part, part=number) for number, part in enumerate(parts))
    return fitted
