"""One place for every runtime setting: environment variables, optionally loaded from .env."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, fields, replace
from pathlib import Path


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(override=False)


def _integer(name: str, default: int, low: int, high: int) -> int:
    raw = os.environ.get(name)
    try:
        value = default if raw in (None, "") else int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer") from None
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _number(name: str, default: float, low: float, high: float) -> float:
    raw = os.environ.get(name)
    try:
        value = default if raw in (None, "") else float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number") from None
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def default_workers() -> int:
    return max(1, min(4, (os.cpu_count() or 2) // 2))


def default_model_dir() -> Path:
    # Shared per user: every data directory reuses one verified model download.
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / "rag" / "models"
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "rag" / "models"


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    input_dir: Path
    model_dir: Path
    embedding_model: str = "all-MiniLM-L6-v2"
    workers: int = field(default_factory=default_workers)
    pages_per_task: int = 10
    task_timeout: float = 300.0
    batch_pages: int = 400
    max_upload_mb: int = 100
    max_attempts: int = 3
    ocr: str = "off"
    ocr_language: str = "eng"
    ocr_dpi: int = 300
    top_k: int = 8
    candidates: int = 50
    min_similarity: float = 0.2
    fusion: str = "convex"
    hybrid_alpha: float = 0.5
    llm_endpoint: str = ""
    llm_model: str = ""
    llm_api_version: str = "2025-04-01-preview"
    llm_timeout: float = 120.0
    llm_api_key: str = field(default="", repr=False)
    api_key: str = field(default="", repr=False)

    @property
    def index_path(self) -> Path:
        return self.data_dir / "index.sqlite3"

    @property
    def queue_dir(self) -> Path:
        return self.data_dir / "queue"

    def extraction(self):
        from .extraction.models import ExtractionOptions

        return ExtractionOptions(ocr=self.ocr, language=self.ocr_language, dpi=self.ocr_dpi)

    def with_overrides(self, **values) -> "Settings":
        known = {item.name for item in fields(self)}
        return replace(self, **{key: value for key, value in values.items() if key in known and value is not None})


def load_settings(**overrides) -> Settings:
    _load_dotenv()
    environ = os.environ
    data_dir = Path(overrides.pop("data_dir", None) or environ.get("RAG_DATA_DIR") or "data").resolve()
    settings = Settings(
        data_dir=data_dir,
        input_dir=Path(environ.get("RAG_INPUT_DIR") or data_dir / "input").resolve(),
        model_dir=Path(environ.get("RAG_MODEL_DIR") or default_model_dir()).resolve(),
        embedding_model=environ.get("RAG_EMBEDDING_MODEL") or "all-MiniLM-L6-v2",
        workers=_integer("RAG_WORKERS", default_workers(), 1, 64),
        pages_per_task=_integer("RAG_PAGES_PER_TASK", 10, 1, 100),
        task_timeout=_number("RAG_TASK_TIMEOUT", 300, 1, 3600),
        batch_pages=_integer("RAG_BATCH_PAGES", 400, 1, 100000),
        max_upload_mb=_integer("RAG_MAX_UPLOAD_MB", 100, 1, 2048),
        max_attempts=_integer("RAG_MAX_ATTEMPTS", 3, 1, 100),
        ocr=environ.get("RAG_OCR") or "off",
        ocr_language=environ.get("RAG_OCR_LANGUAGE") or "eng",
        ocr_dpi=_integer("RAG_OCR_DPI", 300, 72, 600),
        top_k=_integer("RAG_TOP_K", 8, 1, 50),
        candidates=_integer("RAG_CANDIDATES", 50, 1, 500),
        min_similarity=_number("RAG_MIN_SIMILARITY", 0.2, -1, 1),
        fusion=environ.get("RAG_FUSION") or "convex",
        hybrid_alpha=_number("RAG_HYBRID_ALPHA", 0.5, 0, 1),
        llm_endpoint=(environ.get("AZURE_OPENAI_ENDPOINT") or environ.get("OPENAI_BASE_URL") or "").rstrip("/"),
        llm_model=environ.get("AZURE_OPENAI_DEPLOYMENT") or environ.get("RAG_LLM_MODEL") or "",
        llm_api_version=environ.get("AZURE_OPENAI_API_VERSION") or "2025-04-01-preview",
        llm_timeout=_number("RAG_LLM_TIMEOUT", 120, 1, 900),
        llm_api_key=environ.get("AZURE_OPENAI_API_KEY") or environ.get("OPENAI_API_KEY") or "",
        api_key=environ.get("RAG_API_KEY") or "",
    )
    settings = settings.with_overrides(**overrides)
    if settings.fusion not in ("convex", "rrf"):
        raise ValueError("RAG_FUSION must be convex or rrf")
    settings.extraction()
    return settings
