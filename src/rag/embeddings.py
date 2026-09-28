"""Local ONNX sentence embeddings with pinned, checksum-verified model files."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    repo: str
    revision: str
    onnx_file: str
    files: dict[str, str]
    dimensions: int
    max_tokens: int
    pooling: str = "mean"
    query_prefix: str = ""

    @property
    def identity(self) -> str:
        # Stored in the index: vectors from different models are never comparable.
        return f"{self.repo}@{self.revision}/{self.onnx_file}:{self.pooling}:{self.dimensions}"


MODELS = {
    "all-MiniLM-L6-v2": ModelSpec(
        repo="sentence-transformers/all-MiniLM-L6-v2",
        revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        onnx_file="onnx/model.onnx",
        files={
            "onnx/model.onnx": "6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452",
            "tokenizer.json": "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037",
        },
        dimensions=384,
        max_tokens=256,
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _verified(path: Path, expected: str) -> bool:
    if not path.is_file():
        return False
    stat = path.stat()
    marker = path.with_name(path.name + ".verified")
    stamp = f"{expected}:{stat.st_size}:{stat.st_mtime_ns}"
    if marker.is_file() and marker.read_text(encoding="utf-8") == stamp:
        return True
    if _sha256(path) != expected:
        return False
    marker.write_text(stamp, encoding="utf-8")
    return True


def _download(url: str, target: Path, expected: str, attempts: int = 20) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    for attempt in range(1, attempts + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(url, headers={"User-Agent": "rag-model-fetch/1",
                                                       **({"Range": f"bytes={offset}-"} if offset else {})})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                if offset and response.status != 206:
                    offset = 0
                with partial.open("ab" if offset else "wb") as output:
                    last_report = time.monotonic()
                    while block := response.read(1 << 20):
                        output.write(block)
                        if time.monotonic() - last_report > 30:
                            logger.info("Downloading %s: %.1f MB", target.name, output.tell() / 1e6)
                            last_report = time.monotonic()
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as error:
            logger.warning("Download interrupted (%s), attempt %d/%d; resuming", type(error).__name__, attempt, attempts)
            time.sleep(min(30, 2 * attempt))
            continue
        if _sha256(partial) != expected:
            partial.unlink()
            raise RuntimeError(f"Checksum mismatch for {target.name}; the download was discarded")
        os.replace(partial, target)
        return
    raise RuntimeError(f"Could not download {target.name} after {attempts} attempts")


def ensure_model(spec: ModelSpec, model_dir: Path, *, allow_download: bool = True) -> Path:
    root = model_dir / spec.repo.replace("/", "--") / spec.revision
    for relative, expected in spec.files.items():
        path = root / relative
        if _verified(path, expected):
            continue
        if not allow_download:
            raise RuntimeError(f"Model file {relative} is missing; run: rag models")
        logger.warning("One-time download of embedding model file %s into %s", relative, model_dir)
        _download(f"https://huggingface.co/{spec.repo}/resolve/{spec.revision}/{relative}", path, expected)
        _verified(path, expected)
    return root


def get_spec(name: str) -> ModelSpec:
    try:
        return MODELS[name]
    except KeyError:
        raise ValueError(f"Unknown embedding model {name!r}; choose one of: {', '.join(MODELS)}") from None


class Embedder:
    """Thread-safe ONNX encoder; normalized vectors make dot product equal cosine similarity."""

    def __init__(self, spec: ModelSpec, model_dir: Path, *, allow_download: bool = True, threads: int = 0):
        import onnxruntime
        from tokenizers import Tokenizer

        self.spec = spec
        self.identity = spec.identity
        self.dimensions = spec.dimensions
        self.max_tokens = spec.max_tokens
        root = ensure_model(spec, model_dir, allow_download=allow_download)
        self._counter = Tokenizer.from_file(str(root / "tokenizer.json"))
        self._counter.no_truncation()
        self._counter.no_padding()
        self._encoder = Tokenizer.from_file(str(root / "tokenizer.json"))
        self._encoder.enable_truncation(max_length=spec.max_tokens)
        self._encoder.no_padding()
        options = onnxruntime.SessionOptions()
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = threads
        self._session = onnxruntime.InferenceSession(str(root / spec.onnx_file), options,
                                                     providers=["CPUExecutionProvider"])
        self._inputs = {item.name for item in self._session.get_inputs()}
        self._lock = threading.Lock()

    def count_tokens(self, text: str) -> int:
        return len(self._counter.encode(text, add_special_tokens=True).ids)

    def _run(self, texts: list[str]) -> np.ndarray:
        encodings = self._encoder.encode_batch(texts, add_special_tokens=True)
        length = max(len(encoding.ids) for encoding in encodings)
        ids = np.zeros((len(texts), length), dtype=np.int64)
        mask = np.zeros_like(ids)
        types = np.zeros_like(ids)
        for row, encoding in enumerate(encodings):
            size = len(encoding.ids)
            ids[row, :size] = encoding.ids
            mask[row, :size] = encoding.attention_mask
            types[row, :size] = encoding.type_ids
        feeds = {"input_ids": ids, "attention_mask": mask, "token_type_ids": types}
        hidden = self._session.run(None, {name: value for name, value in feeds.items() if name in self._inputs})[0]
        if self.spec.pooling == "cls":
            pooled = hidden[:, 0]
        else:
            weights = mask[..., None].astype(np.float32)
            pooled = (hidden * weights).sum(axis=1) / np.clip(weights.sum(axis=1), 1e-9, None)
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        return (pooled / np.clip(norms, 1e-12, None)).astype(np.float32)

    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)
        # Similar lengths share a batch, so short chunks are not padded to long ones.
        order = sorted(range(len(texts)), key=lambda index: len(texts[index]))
        output = np.empty((len(texts), self.dimensions), dtype=np.float32)
        for start in range(0, len(order), batch_size):
            selected = order[start:start + batch_size]
            output[selected] = self._run([texts[index] for index in selected])
        return output

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([self.spec.query_prefix + text])[0]


_cache: dict[tuple[str, str], Embedder] = {}
_cache_lock = threading.Lock()


def load_embedder(name: str, model_dir: Path, *, allow_download: bool = True) -> Embedder:
    key = (name, str(model_dir))
    with _cache_lock:
        if key not in _cache:
            _cache[key] = Embedder(get_spec(name), model_dir, allow_download=allow_download)
        return _cache[key]


def describe(spec: ModelSpec) -> str:
    return json.dumps({"repo": spec.repo, "revision": spec.revision, "file": spec.onnx_file,
                       "dimensions": spec.dimensions, "max_tokens": spec.max_tokens})
