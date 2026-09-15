# RAG with PDF Worker Pool

Based on https://github.com/GopiB9119/RAG.git (commit
`4f7cde575052a6e449c80cf5916ce9c908ccda8b`). The cloned RAG application is in
this folder; the PDF worker pool lives in `src/pdf_pipeline/`.

## Read this first

**[Complete system guide: architecture, code, storage, operations, and production gaps](docs/CODE_WALKTHROUGH.md)**

That is the main detailed document. This README contains only the overview and
starting commands. Existing specialist notes are retained for reference; the
complete guide distinguishes current capabilities from future plans.

## What the application does

Local and Azure ingestion share `DocumentPublisher` in [ingest_sources.py](ingest_sources.py):
one path loads/reuses the embedding model, prepares chunks, and publishes a complete
document with a durable receipt. Both adapters use the same receipt verifier before
marking a job ready. Local jobs use their durable job ID as the processing generation;
Azure jobs use their observed-upload generation. Storage, discovery and message
delivery remain adapter responsibilities, not separate RAG implementations.

If publication succeeds but saving job completion fails, retrying the same generation
reuses its receipt without another vector write. It still verifies the input and
may repeat extraction/tokenization. Existing local job rows migrate with a nullable
receipt field; old ready jobs are not falsely upgraded or automatically reprocessed.
Full orchestration is not yet unified: local/cloud retry scheduling and source
identities differ, and a cross-backend move requires deliberate re-ingestion.

PDF extraction now uses one shared PyMuPDF implementation for local and Azure
workers, with optional Tesseract OCR. `--ocr auto` recognizes flagged scan-like or
unusable-text pages; `--ocr off` rejects them instead of silently indexing partial
documents. The default is `off`. Native/OCR provenance is retained with citations.
This is not a 99.9% accuracy guarantee; see the
[extraction quality and OCR guide](docs/CODE_WALKTHROUGH.md#extraction-quality-and-ocr).

Before embedding, document chunks are checked against the loaded model's actual
tokenizer limit, including special tokens. Oversized chunks are split without
discarding source text; each part retains its source/page/chunk citation and a
`token_part` identifier. This improves input coverage, not a guarantee of better
answers. Existing indexed sources must be re-ingested to benefit. See the
[complete guide](docs/CODE_WALKTHROUGH.md#local-processing-step-by-step).

```text
Local PDF folder -> Stable-file detection -> Durable document queue
                 -> Page-range worker pool -> Validated page text
                 -> Chunks -> MiniLM embeddings -> Chroma index

Your question -> Same embedding model -> Relevant indexed text and citations
              -> Azure OpenAI -> Answer
```

The local watcher automates ingestion, not conversations. It does not upload PDFs
to Azure. Optional Azure extraction commands use Blob Storage and Service Bus,
and an opt-in polling coordinator now automates cloud dispatch and complete-document
indexing. The coordinator is single-instance with durable local state/index, not
a deployed highly available service. It does not change the local watcher's behavior.

## Status

This is a **single-host MVP with additional Azure worker code**, not a deployed
production system.

- Implemented: folder watching, persistent jobs, range extraction, checkpointed
  retries, duplicate detection, chunking, indexing, retrieval, and CLI answers.
- Implemented cleanup: successful unshared local retry artifacts are removed;
  original PDFs and the search index remain. Normal ingestion creates no extra
  full-text or JSONL export files.
- Latest recorded consolidation check, 2026-09-15: **171 non-live tests passed,
  15 baseline/live tests excluded**. This is not a real-document live result.
- Last checked blockers: PyMuPDF, Chroma, SentenceTransformers and Azure SDKs were
  missing; no input PDFs or populated local index were available. The Docker image
  build was blocked by the stopped Linux engine. These checks are historical,
  not a new environment check performed by editing this documentation.
- Implemented next milestone: immutable document revisions with atomic local
  publication pointers and a consistent retrieval snapshot per question. This
  requires rebuilding older indexes and using the application's published read view.
- Implemented local revision cleanup: preview by default, explicit apply, with
  active-revision and persistent reader-lease protection. No cleanup has been run
  against your real index. Abandoned reader leases conservatively retain data.
- Still needed: real-PDF evaluation, cloud retention, authenticated
  conversation API, HA cloud coordination, shared vector storage and load testing.

## Local setup

Use Python 3.11 on Windows. Run from this folder. Create the environment only if
it does not already exist:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

If installation fails, stop there. In this environment previous downloads from
`files.pythonhosted.org` failed with an SSL handshake error. Keep TLS verification
enabled; an installation failure is not a successful setup.

## Automatic ingestion

After dependencies are installed, start once and leave it running:

```powershell
.\.venv\Scripts\python.exe jobs.py watch data/input --workers 2 --pages-per-task 10
```

Drop your real PDFs into `data/input/` or its subfolders. For reliable completion
detection, copy as `.partial`, then rename to `.pdf` once the copy finishes.
Default detection uses a 2-second scan interval and 10-second quiet period.
Progress is printed automatically. Ctrl+C stops the watcher; reboot/logon startup
is not installed. Missing packages cause a `blocked` exit before processing.

For five 60-page PDFs, this means five document jobs and 30 ten-page range tasks,
not 300 processes. Documents are processed sequentially; ranges within a document
run in parallel. This is local multiprocessing, not multiple Azure machines.

For scanned PDFs, after configuring Tesseract language data on the worker machine,
start with a small worker count and the shared OCR options:

```powershell
.\.venv\Scripts\python.exe jobs.py watch data/input --workers 2 --pages-per-task 10 --ocr auto --ocr-language eng --ocr-dpi 300
```

Changing extraction policy requires a new job-state namespace and deliberate
re-ingestion, not silently reusing old ready receipts. Existing processed queues
without a recorded extraction policy are rejected. Preserve them, stop their
consumers, and use `--state-dir data/jobs/extraction-v2` before `watch` for a new
namespace; do not run two consumers against the same index. Originals and old
artifacts are not deleted by this change. Azure workers/manifests must be upgraded
together to task schema 2 before sending the new tasks.

## Status and questions

```powershell
.\.venv\Scripts\python.exe jobs.py status
.\.venv\Scripts\python.exe chat_rag.py
```

Configure Azure answer settings locally using the names in [.env.example](.env.example)
before starting chat. Never commit credentials. Chat sends the question and
retrieved text to Azure, and may incur charges. It needs a populated index.
Questions are currently independent; conversation history and streaming are not
implemented. Normal ingestion now stages document revisions and atomically publishes
them for application readers. Old revisions are retained so in-flight questions can
finish. Direct/raw Chroma queries bypass this protection. Stop readers and writers
for `--reset`, index deletion, or restore; these are not online publication operations.

## Publication migration

Indexes created before lease-aware publication schema 2 are rejected, not silently migrated.
Stop the watcher/chat, preserve your original PDFs and index backup, then rebuild
all intended sources with the legacy ingestion CLI's explicit `--reset` option.
Review `urls.txt` first: that CLI also loads configured URLs. Do not assume ready
queue jobs will automatically rerun after a reset. No reset is performed for you.

Publication pointers are stored in `chroma_data/rag_publications.sqlite3` beside
the vectors. Reader leases use `chroma_data/rag_readers.sqlite3`. All application
readers must be upgraded/restarted before enabling cleanup; old reader binaries
and direct Chroma access do not participate in lease protection. Missing or
mismatched publication state fails closed. Stop readers/writers for backup or restore.

## Safe revision cleanup

For intermediate PDF/checkpoint/Blob retention, read the
[storage lifecycle plan](docs/MVP.md#storage-lifecycle-plan) first. It separates
existing local cleanup from proposed Azure retirement, explains shared references
and late-worker races, and defines the checks required before any cloud deletion.
The plan is not an enabled cleanup feature. The commands below apply only to
inactive local vector revisions.

Preview only (no vectors deleted):

```powershell
.\.venv\Scripts\python.exe index_publication.py
```

After reviewing the report, explicitly delete eligible old/staged vectors:

```powershell
.\.venv\Scripts\python.exe index_publication.py --apply --batch-size 500
```

Active revisions and any revision pinned by a reader are excluded. Cleanup holds
local publication/reader-registration locks and may briefly block new requests;
use a maintenance window for large indexes. It does not delete PDFs, checkpoints,
Azure blobs, publication pointers, or legacy chunks lacking revision metadata.
Reader leases do not expire automatically: crashed readers can retain data and
require operator investigation with all readers stopped. This is deliberate safety,
not a guarantee of bounded disk use. Removing vectors may not immediately shrink
Chroma files; physical compaction is backend-specific and unverified.

## Automatic Azure coordination

After configuring existing Azure Blob/Service Bus resources and installing the
coordinator dependencies, start one coordinator plus the extraction workers:

```powershell
.\.venv\Scripts\python.exe azure_pipeline.py coordinate --incoming-prefix incoming/
# Separate processes/machines:
.\.venv\Scripts\python.exe azure_pipeline.py worker
# Check durable coordinator progress:
.\.venv\Scripts\python.exe azure_pipeline.py coordinate-status
```

Upload complete PDFs to the configured Blob container's `incoming/` prefix. The
coordinator polls committed blob ETags, dispatches ranges, waits/reconciles, then
indexes only complete documents. No local per-PDF dispatch/collect commands are
needed. No unsolicited Azure answer requests are made. Native SDK and cloud behavior
remain unverified; nothing has been uploaded or deployed during implementation.

See the [coordinator operations section](docs/AZURE_WORKERS.md#automatic-coordinator)
for retry deadlines, persistence, overwrite races, and the coordinator image.

### Cleanup preparation: history and publication receipts

The coordinator now preserves per-generation transition history and verifies an
index receipt committed atomically with its published revision. A retry after
index success but before coordinator receipt storage reuses that receipt instead
of writing another revision. Local `work` and `watch` now use that same publisher
and require a verified receipt too; `jobs.py status` includes `has_publication_receipt`
without exposing the full receipt or document path. Inspect cloud evidence with:

```powershell
.\.venv\Scripts\python.exe azure_pipeline.py coordinate-status --history --retirement-preview
```

This preview does not delete anything or call Azure. Legacy ready rows without
receipts remain distinguishable. Worker fencing and cloud retirement records are
still missing, so **all Azure deletion remains disabled**. See
[Stage A progress](docs/MVP.md#stage-a-progress-local-evidence-implemented).

## Important boundaries

- Original PDFs: `data/input/`; never removed by automatic cleanup.
- Jobs and retry artifacts: `data/jobs/`; unfinished/failed jobs keep recovery data.
- Search text, vectors and citations: `chroma_data/`; vectors alone cannot supply evidence.
- Azure blobs: retained until a coordinated cloud retention policy is implemented.
- Removing an input PDF does not remove its old indexed content automatically.
- The full guide contains manual retry commands, Azure commands, error diagnosis,
  monitoring, storage rules, and the staged production roadmap.

No runtime code, demo data, cloud resources, or live API calls are created by
this documentation update.