import sqlite3
from types import SimpleNamespace

import pytest

from index_publication import PublicationStore, PublishedCollection, REVISION_FIELD


def test_reader_sees_only_committed_revision_and_pins_old_view(tmp_path):
    store = PublicationStore(str(tmp_path))
    queries = []
    raw = SimpleNamespace(id="collection-id", metadata={}, count=lambda: 0,
                          query=lambda **kwargs: queries.append(kwargs))
    with store.writer() as writer:
        store.bind(writer, "documents", raw, False)
        store.publish(writer, "documents", "report.pdf", "old", 2)
    view = PublishedCollection(raw, str(tmp_path), "documents")
    pinned = view.snapshot()
    with store.writer() as writer:
        store.publish(writer, "documents", "report.pdf", "new", 3)
        assert view.count() == 2
        view.query(n_results=8)
        assert queries[-1]["where"] == {REVISION_FIELD: {"$in": ["old"]}}
    assert view.count() == 3
    pinned.query(n_results=8)
    assert queries[-1]["where"] == {REVISION_FIELD: {"$in": ["old"]}}
    view.query(n_results=8)
    assert queries[-1]["where"] == {REVISION_FIELD: {"$in": ["new"]}}


def test_publication_rolls_back_on_failure(tmp_path):
    store = PublicationStore(str(tmp_path))
    raw = SimpleNamespace(id="identity", count=lambda: 0)
    with store.writer() as writer:
        store.bind(writer, "documents", raw, False)
        store.publish(writer, "documents", "report.pdf", "old", 2)
    with pytest.raises(RuntimeError):
        with store.writer() as writer:
            store.publish(writer, "documents", "report.pdf", "partial", 3)
            raise RuntimeError("Interrupted publication")
    assert store.snapshot("documents", "identity") == (["old"], 2)


def test_missing_or_replaced_publication_state_fails_closed(tmp_path):
    store = PublicationStore(str(tmp_path))
    with pytest.raises(sqlite3.OperationalError):
        store.snapshot("documents", "identity")
    assert not store.path.exists()
    with store.writer() as writer:
        with pytest.raises(ValueError, match="Publication state"):
            store.bind(writer, "documents", SimpleNamespace(id="unknown", count=lambda: 5), False)
        store.bind(writer, "documents", SimpleNamespace(id="identity", count=lambda: 0), False)
    with pytest.raises(ValueError, match="identity"):
        store.snapshot("documents", "other-identity")


def test_collection_reset_invalidates_unpinned_old_reader(tmp_path):
    store = PublicationStore(str(tmp_path))
    original = SimpleNamespace(id="original", count=lambda: 0)
    replacement = SimpleNamespace(id="replacement", count=lambda: 0)
    with store.writer() as writer:
        store.bind(writer, "documents", original, False)
        store.publish(writer, "documents", "report.pdf", "old", 1)
    view = PublishedCollection(original, str(tmp_path), "documents")
    with store.writer() as writer:
        store.bind(writer, "documents", replacement, True)
    with pytest.raises(ValueError, match="identity"):
        view.count()
    assert store.snapshot("documents", "replacement") == ([], 0)


def test_failed_multi_source_publication_rolls_back_all_pointers(tmp_path):
    store = PublicationStore(str(tmp_path))
    raw = SimpleNamespace(id="identity", count=lambda: 0)
    with store.writer() as writer:
        store.bind(writer, "documents", raw, False)
        store.publish(writer, "documents", "first.pdf", "old-first", 2)
        store.publish(writer, "documents", "second.pdf", "old-second", 3)
    with pytest.raises(RuntimeError):
        with store.writer() as writer:
            store.publish(writer, "documents", "first.pdf", "new-first", 1)
            store.publish(writer, "documents", "second.pdf", "new-second", 1)
            raise RuntimeError("Batch interrupted before commit")
    assert store.snapshot("documents", "identity") == (["old-first", "old-second"], 5)


def test_reader_lease_is_durable_and_released_on_close(tmp_path):
    store = PublicationStore(str(tmp_path))
    raw = SimpleNamespace(id="identity", count=lambda: 0)
    with store.writer() as writer:
        store.bind(writer, "documents", raw, False)
        store.publish(writer, "documents", "report.pdf", "old", 1)
    pinned = PublishedCollection(raw, str(tmp_path), "documents").snapshot()
    with store.reader_registry() as registry:
        assert registry.execute("SELECT COUNT(*) FROM readers").fetchone()[0] == 1
    pinned.close()
    with store.reader_registry() as registry:
        assert registry.execute("SELECT COUNT(*) FROM readers").fetchone()[0] == 0
    with pytest.raises(ValueError, match="closed"):
        pinned.count()


def test_query_error_releases_temporary_lease(tmp_path):
    store = PublicationStore(str(tmp_path))

    def broken_query(**kwargs):
        raise RuntimeError("Query failed")

    raw = SimpleNamespace(id="identity", count=lambda: 0, query=broken_query)
    with store.writer() as writer:
        store.bind(writer, "documents", raw, False)
        store.publish(writer, "documents", "report.pdf", "old", 1)
    with pytest.raises(RuntimeError):
        PublishedCollection(raw, str(tmp_path), "documents").query(n_results=1)
    with store.reader_registry() as registry:
        assert registry.execute("SELECT COUNT(*) FROM readers").fetchone()[0] == 0


class CleanupCollection:
    def __init__(self):
        from rag_core import COLLECTION_METADATA

        self.metadata = dict(COLLECTION_METADATA)
        self.id = "cleanup-collection"
        self.records = {}
        self.deletions = []

    def count(self):
        return len(self.records)

    def get(self, where=None, ids=None, offset=0, limit=500, include=None):
        rows = list(self.records.items())
        if where:
            rows = [(key, metadata) for key, metadata in rows if all(metadata.get(field) == value for field, value in where.items())]
        if ids is not None:
            rows = [(key, metadata) for key, metadata in rows if key in ids]
        rows = rows[offset:offset + limit]
        return {"ids": [key for key, _ in rows], "metadatas": [metadata for _, metadata in rows]}

    def delete(self, ids):
        self.deletions.append(ids)
        for key in ids:
            self.records.pop(key, None)


def test_cleanup_preserves_active_and_pinned_then_deletes_after_close(tmp_path):
    from index_publication import cleanup_revisions

    store = PublicationStore(str(tmp_path))
    raw = CleanupCollection()
    with store.writer() as writer:
        store.bind(writer, "documents", raw, False)
        store.publish(writer, "documents", "report.pdf", "old", 2)
    reader = PublishedCollection(raw, str(tmp_path), "documents").snapshot()
    with store.writer() as writer:
        store.publish(writer, "documents", "report.pdf", "new", 1)
    raw.records = {"old-1": {REVISION_FIELD: "old"}, "old-2": {REVISION_FIELD: "old"},
                   "new-1": {REVISION_FIELD: "new"}, "abandoned": {REVISION_FIELD: "failed"},
                   "legacy": {"source": "legacy.pdf"}}
    preview = cleanup_revisions(raw, str(tmp_path), "documents")
    assert preview["candidate_chunks"] == 1 and preview["deleted_chunks"] == 0
    assert preview["reader_leases"] == 1 and raw.deletions == []
    applied = cleanup_revisions(raw, str(tmp_path), "documents", apply=True, batch_size=1)
    assert applied["deleted_chunks"] == 1
    assert "old-1" in raw.records and "old-2" in raw.records and "new-1" in raw.records
    reader.close()
    applied = cleanup_revisions(raw, str(tmp_path), "documents", apply=True, batch_size=1)
    assert applied["deleted_chunks"] == 2
    assert set(raw.records) == {"new-1", "legacy"}
    assert all(len(batch) == 1 for batch in raw.deletions)


def test_cleanup_failure_does_not_change_active_pointers(tmp_path):
    from index_publication import cleanup_revisions

    store = PublicationStore(str(tmp_path))
    raw = CleanupCollection()
    with store.writer() as writer:
        store.bind(writer, "documents", raw, False)
        store.publish(writer, "documents", "report.pdf", "active", 1)
    raw.records = {"active": {REVISION_FIELD: "active"}, "old": {REVISION_FIELD: "old"}}
    raw.delete = lambda ids: None
    with pytest.raises(RuntimeError, match="not confirmed"):
        cleanup_revisions(raw, str(tmp_path), "documents", apply=True)
    assert store.snapshot("documents", raw.id) == (["active"], 1)
    assert "active" in raw.records


def test_cleanup_refuses_missing_state(tmp_path):
    from index_publication import cleanup_revisions

    raw = CleanupCollection()
    with pytest.raises(sqlite3.OperationalError):
        cleanup_revisions(raw, str(tmp_path), "documents", apply=True)
    assert raw.deletions == []
    assert not PublicationStore(str(tmp_path)).path.exists()


def test_crashed_reader_lease_retains_old_revision(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    from index_publication import cleanup_revisions

    store = PublicationStore(str(tmp_path))
    raw = CleanupCollection()
    with store.writer() as writer:
        store.bind(writer, "documents", raw, False)
        store.publish(writer, "documents", "report.pdf", "old", 1)
    script = (
        "import sys; from threading import Event; sys.path.insert(0,sys.argv[1]); "
        "from index_publication import PublicationStore; "
        "store=PublicationStore(sys.argv[2]); "
        "state,lease=store.acquire_snapshot('documents','cleanup-collection'); "
        "print('lease-created',flush=True); Event().wait()"
    )
    with pytest.raises(subprocess.TimeoutExpired) as error:
        subprocess.run([sys.executable, "-c", script, str(Path(__file__).resolve().parents[1]), str(tmp_path)],
                       capture_output=True, timeout=3)
    assert b"lease-created" in error.value.stdout
    with store.writer() as writer:
        store.publish(writer, "documents", "report.pdf", "new", 1)
    raw.records = {"old": {REVISION_FIELD: "old"}, "new": {REVISION_FIELD: "new"}}
    report = cleanup_revisions(raw, str(tmp_path), "documents", apply=True)
    assert report["reader_leases"] == 1 and report["deleted_chunks"] == 0
    assert set(raw.records) == {"old", "new"}


def test_cleanup_refuses_indexes_without_reader_protocol(tmp_path):
    from index_publication import cleanup_revisions

    raw = CleanupCollection()
    raw.metadata["publication_schema"] = 1
    with pytest.raises(ValueError, match="configuration"):
        cleanup_revisions(raw, str(tmp_path), "documents", apply=True)
    assert raw.deletions == []


def test_retrieval_error_releases_full_question_lease(tmp_path):
    from rag_core import retrieve, open_published_collection

    raw = CleanupCollection()
    store = PublicationStore(str(tmp_path))
    with store.writer() as writer:
        store.bind(writer, "documents", raw, False)
        store.publish(writer, "documents", "report.pdf", "active", 1)

    def broken_encode(*args, **kwargs):
        raise RuntimeError("Embedding failed")

    view = open_published_collection(raw, str(tmp_path), "documents")
    with pytest.raises(RuntimeError, match="Embedding failed"):
        retrieve("Question", view, SimpleNamespace(encode=broken_encode), 1)
    with store.reader_registry() as connection:
        assert connection.execute("SELECT COUNT(*) FROM readers").fetchone()[0] == 0


def test_malformed_reader_state_prevents_any_deletion(tmp_path):
    from index_publication import cleanup_revisions

    raw = CleanupCollection()
    store = PublicationStore(str(tmp_path))
    with store.writer() as writer:
        store.bind(writer, "documents", raw, False)
    raw.records = {"orphan": {REVISION_FIELD: "orphan"}}
    with store.reader_registry() as registry:
        registry.execute("INSERT INTO readers VALUES (?,?,?,?)", ("bad", "documents", raw.id, '{}'))
    with pytest.raises(ValueError, match="Malformed reader"):
        cleanup_revisions(raw, str(tmp_path), "documents", apply=True)
    assert raw.deletions == []