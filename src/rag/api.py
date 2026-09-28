"""HTTP API and a minimal web page; background thread keeps indexing uploaded PDFs."""

from __future__ import annotations

import hmac
import logging
import re
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

logger = logging.getLogger("rag.api")


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=50)


class Query(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=8, ge=1, le=50)
    mode: str = Field(default="hybrid", pattern="^(hybrid|dense|lexical)$")


def _safe_name(filename: str | None) -> str:
    name = Path(filename or "").name
    stem = re.sub(r"[^\w.\- ]+", "_", Path(name).stem).strip(" ._") or f"upload-{uuid.uuid4().hex[:8]}"
    return f"{stem[:120]}.pdf"


def create_app(rag, *, watch: bool = False) -> FastAPI:
    settings = rag.settings
    stop = threading.Event()
    uploads = settings.input_dir / "uploads"

    def log_event(event: dict) -> None:
        logger.info("%s", event)

    def worker() -> None:
        scanner = None
        if watch:
            from .watcher import FolderScanner

            scanner = FolderScanner(settings.input_dir, rag.queue, log_event,
                                    max_bytes=settings.max_upload_mb * 1024 * 1024, max_attempts=settings.max_attempts)
        try:
            rag.process(log_event, stop=stop, scanner=scanner)
        except Exception:
            logger.exception("Indexing worker stopped")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        thread = threading.Thread(target=worker, name="rag-indexer", daemon=True)
        thread.start()
        yield
        stop.set()
        thread.join(timeout=60)
        rag.close()

    app = FastAPI(title="RAG", version="2.0.0", lifespan=lifespan)

    def authorize(request: Request) -> None:
        if not settings.api_key:
            return
        supplied = request.headers.get("x-api-key") or request.headers.get("authorization", "").removeprefix("Bearer ")
        if not hmac.compare_digest(supplied.encode(), settings.api_key.encode()):
            raise HTTPException(status_code=401, detail="Missing or invalid API key")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/status", dependencies=[Depends(authorize)])
    def status() -> dict:
        return rag.status()

    @app.get("/documents", dependencies=[Depends(authorize)])
    def documents(limit: int = 100) -> list[dict]:
        return [{key: value for key, value in document.items() if key != "source"}
                for document in rag.index.documents(max(1, min(limit, 1000)))]

    @app.post("/documents", dependencies=[Depends(authorize)])
    def upload(files: list[UploadFile] = File(...)) -> dict:
        uploads.mkdir(parents=True, exist_ok=True)
        limit = settings.max_upload_mb * 1024 * 1024
        accepted, rejected = [], []
        for item in files:
            target = uploads / _safe_name(item.filename)
            partial = target.with_name(f".{uuid.uuid4().hex}.partial")
            try:
                size = 0
                with partial.open("wb") as output:
                    while block := item.file.read(1 << 20):
                        if size == 0 and not block.startswith(b"%PDF-"):
                            raise ValueError("not a PDF file")
                        size += len(block)
                        if size > limit:
                            raise ValueError(f"larger than {settings.max_upload_mb} MB")
                        output.write(block)
                if size == 0:
                    raise ValueError("empty file")
                partial.replace(target)
                job = rag.queue.enqueue(target, limit, settings.max_attempts)
                accepted.append({"file": target.name, "job_id": job["id"], "state": job["state"]})
            except (ValueError, OSError) as error:
                rejected.append({"file": item.filename, "error": str(error)})
            finally:
                partial.unlink(missing_ok=True)
        return {"accepted": accepted, "rejected": rejected}

    @app.post("/search", dependencies=[Depends(authorize)])
    def find(body: Query) -> list[dict]:
        return [vars(item) | {"source": None} for item in rag.search(body.query, top_k=body.top_k, mode=body.mode)]

    @app.post("/ask", dependencies=[Depends(authorize)])
    def ask(body: Question) -> dict:
        try:
            result = rag.ask(body.question, top_k=body.top_k)
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from None
        return {"answer": result["answer"], "timings": result["timings"],
                "sources": [{key: value for key, value in source.items() if key != "source"}
                            for source in result["sources"]]}

    @app.get("/", response_class=HTMLResponse)
    def page() -> str:
        return _PAGE

    return app


_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>RAG</title>
<style>
body{font-family:system-ui,sans-serif;max-width:860px;margin:2rem auto;padding:0 1rem;color:#1d1d1f}
h1{font-size:1.4rem} section{border:1px solid #ddd;border-radius:8px;padding:1rem;margin:1rem 0}
textarea,input[type=text],input[type=password]{width:100%;box-sizing:border-box;padding:.5rem;font:inherit}
button{padding:.5rem 1rem;font:inherit;cursor:pointer} #answer{white-space:pre-wrap;line-height:1.5}
.muted{color:#666;font-size:.9rem} li{margin:.3rem 0}
</style></head><body>
<h1>Ask your documents</h1>
<section><label class="muted">API key (only if the server requires one)
<input type="password" id="key" autocomplete="off"></label></section>
<section><strong>Add PDFs</strong><p><input type="file" id="files" accept="application/pdf" multiple>
<button id="upload">Upload</button></p><div id="uploadStatus" class="muted"></div>
<div id="status" class="muted"></div></section>
<section><textarea id="question" rows="3" placeholder="Ask a question about your PDFs"></textarea>
<p><button id="ask">Ask</button></p><div id="answer"></div><ol id="sources" class="muted"></ol></section>
<script>
const $ = (id) => document.getElementById(id);
$("key").value = sessionStorage.getItem("ragKey") || "";
function headers(json) {
  const key = $("key").value.trim(); sessionStorage.setItem("ragKey", key);
  const h = json ? {"Content-Type": "application/json"} : {};
  if (key) h["X-API-Key"] = key;
  return h;
}
async function call(path, options) {
  const response = await fetch(path, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || response.statusText);
  return body;
}
async function refresh() {
  try {
    const s = await call("/status", {headers: headers(false)});
    $("status").textContent = `${s.index.documents} documents, ${s.index.chunks} passages indexed | queue: ` +
      JSON.stringify(s.queue);
  } catch (e) { $("status").textContent = e.message; }
}
$("upload").onclick = async () => {
  const form = new FormData();
  for (const file of $("files").files) form.append("files", file);
  $("uploadStatus").textContent = "Uploading...";
  try {
    const r = await call("/documents", {method: "POST", headers: headers(false), body: form});
    $("uploadStatus").textContent = `${r.accepted.length} accepted, ${r.rejected.length} rejected ` +
      r.rejected.map((x) => `(${x.file}: ${x.error})`).join(" ");
  } catch (e) { $("uploadStatus").textContent = e.message; }
  refresh();
};
$("ask").onclick = async () => {
  $("answer").textContent = "Thinking..."; $("sources").replaceChildren();
  try {
    const r = await call("/ask", {method: "POST", headers: headers(true),
      body: JSON.stringify({question: $("question").value})});
    $("answer").textContent = r.answer;
    for (const s of r.sources) {
      const li = document.createElement("li");
      li.textContent = `[${s.n}] ${s.title}, page ${s.page}`;
      $("sources").append(li);
    }
  } catch (e) { $("answer").textContent = e.message; }
};
refresh(); setInterval(refresh, 5000);
</script></body></html>
"""
