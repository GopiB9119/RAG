from types import SimpleNamespace

import pytest

from embedding_chunks import fit_embedding_chunks


class CharacterTokenizer:
    model_max_length = 8

    def encode(self, text, *, add_special_tokens, truncation):
        assert add_special_tokens is True
        assert truncation is False
        return ["start", *text, "end"]


def model(limit=8):
    return SimpleNamespace(tokenizer=CharacterTokenizer(), max_seq_length=limit)


@pytest.mark.parametrize("text", ["short", "a long sentence with many words", "x" * 41,
                                 "\u4e2d\u6587\u6d4b\u8bd5" * 5, " spaced   words "])
def test_token_parts_fit_and_preserve_exact_text_and_citations(text):
    metadata = {"source": "report.pdf", "page": 7, "chunk": 3}
    chunks = [{"text": text, "metadata": metadata}]
    parts = fit_embedding_chunks(chunks, model())
    assert "".join(part["text"] for part in parts) == text
    assert all(len(CharacterTokenizer().encode(part["text"], add_special_tokens=True, truncation=False)) <= 8 for part in parts)
    assert [part["metadata"]["token_part"] for part in parts] == list(range(len(parts)))
    assert all(all(part["metadata"][key] == value for key, value in metadata.items()) for part in parts)
    assert "token_part" not in metadata


def test_respects_smaller_model_limit_and_special_tokens():
    parts = fit_embedding_chunks([{"text": "abcdefgh", "metadata": {}}], model(4))
    assert all(len(part["text"]) <= 2 for part in parts)


@pytest.mark.parametrize("limit", [0, 1, 2])
def test_impossible_budget_fails_instead_of_truncating(limit):
    with pytest.raises(ValueError):
        fit_embedding_chunks([{"text": "x", "metadata": {}}], model(limit))


def test_prompted_model_requires_explicit_adapter():
    prompted = model()
    prompted.default_prompt_name = "document"
    with pytest.raises(ValueError, match="Prompted"):
        fit_embedding_chunks([{"text": "text", "metadata": {}}], prompted)