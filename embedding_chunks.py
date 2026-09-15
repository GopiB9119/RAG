"""Fit text to the loaded embedding tokenizer without silently truncating it."""

from __future__ import annotations

from typing import Any


def fit_embedding_chunks(chunks: list[dict], model: Any) -> list[dict]:
    tokenizer = model.tokenizer
    limit = model.max_seq_length
    if type(limit) is not int or limit < 1:
        raise ValueError("Embedding model must declare a positive max_seq_length")
    tokenizer_limit = getattr(tokenizer, "model_max_length", limit)
    if isinstance(tokenizer_limit, int) and tokenizer_limit > 0:
        limit = min(limit, tokenizer_limit)
    if getattr(model, "default_prompt_name", None):
        raise ValueError("Prompted embedding models require an explicit token-budget adapter")

    def fits(text: str) -> bool:
        # Include special tokens and explicitly disable tokenizer truncation.
        return len(tokenizer.encode(text, add_special_tokens=True, truncation=False)) <= limit

    if not fits(""):
        raise ValueError("Embedding token limit cannot accommodate special tokens")
    prepared = []
    for chunk in chunks:
        text = chunk["text"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Embedding chunks must contain nonempty text")
        pending = [text]
        parts = []
        while pending:
            part = pending.pop()
            if fits(part):
                parts.append(part)
                continue
            if len(part) < 2:
                raise ValueError("A text character cannot fit the embedding token limit")
            # Preserve exact Unicode text instead of decoding token IDs, which
            # can normalize or lose original characters. Every leaf is rechecked.
            midpoint = len(part) // 2
            boundary = part.rfind(" ", 0, midpoint + 1) + 1
            if boundary <= 0 or boundary >= len(part) or boundary < midpoint // 2:
                boundary = midpoint
            pending.extend([part[boundary:], part[:boundary]])
        if "".join(parts) != text:
            raise RuntimeError("Token splitting did not preserve source text")
        for part_number, part in enumerate(parts):
            metadata = dict(chunk["metadata"])
            # Always assign this field, so re-ingesting no longer oversized text
            # cannot retain a stale token-part identity supplied by a caller.
            metadata["token_part"] = part_number
            prepared.append({"text": part, "metadata": metadata})
    return prepared