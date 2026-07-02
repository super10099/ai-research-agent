"""
Tests for src/ingestion/store.py.

get_client() branching is tested with mocked chromadb constructors (we only
care that the right constructor gets the right args). Everything downstream
of that — store_chunks/get_parent_by_ids/get_collections — is tested against
a real chromadb.EphemeralClient(): an in-memory, no-disk-I/O instance, so
there's no need to mock ChromaDB's actual query/upsert behavior.
"""

from unittest.mock import MagicMock

import chromadb
import numpy as np
import pytest

from src.config import settings
from src.ingestion import store
from src.ingestion.chunker import Chunk


@pytest.fixture(autouse=True)
def _reset_client_singleton():
    store._client = None
    yield
    store._client = None


class TestGetClientBranching:
    def test_uses_persistent_client_by_default(self, monkeypatch, tmp_path):
        monkeypatch.setattr(settings, "chroma_use_http", False)
        monkeypatch.setattr(settings, "chroma_persist_dir", str(tmp_path))
        mock_persistent = MagicMock()
        monkeypatch.setattr(chromadb, "PersistentClient", mock_persistent)

        store.get_client()

        mock_persistent.assert_called_once()
        assert mock_persistent.call_args.kwargs["path"] == str(tmp_path)

    def test_uses_http_client_when_configured(self, monkeypatch):
        monkeypatch.setattr(settings, "chroma_use_http", True)
        monkeypatch.setattr(settings, "chroma_host", "chromadb")
        monkeypatch.setattr(settings, "chroma_port", 8000)
        mock_http = MagicMock()
        monkeypatch.setattr(chromadb, "HttpClient", mock_http)

        store.get_client()

        mock_http.assert_called_once()
        assert mock_http.call_args.kwargs["host"] == "chromadb"
        assert mock_http.call_args.kwargs["port"] == 8000

    def test_client_is_a_singleton(self, monkeypatch, tmp_path):
        monkeypatch.setattr(settings, "chroma_use_http", False)
        monkeypatch.setattr(settings, "chroma_persist_dir", str(tmp_path))
        mock_persistent = MagicMock()
        monkeypatch.setattr(chromadb, "PersistentClient", mock_persistent)

        first = store.get_client()
        second = store.get_client()

        assert first is second
        mock_persistent.assert_called_once()


@pytest.fixture
def ephemeral_store(monkeypatch):
    """Route store.get_client() to a real in-memory ChromaDB client.

    chromadb.EphemeralClient() instances with identical default settings
    share underlying storage via chromadb's internal system cache — without
    clearing it, data written by one test would leak into the next.
    """
    chromadb.api.client.SharedSystemClient.clear_system_cache()
    client = chromadb.EphemeralClient()
    monkeypatch.setattr(store, "get_client", lambda: client)
    return client


class TestCollections:
    def test_get_collections_uses_cosine_space(self, ephemeral_store):
        child_col, parent_col = store.get_collections()

        assert child_col.metadata["hnsw:space"] == "cosine"
        assert parent_col.metadata["hnsw:space"] == "cosine"
        assert child_col.name == store.CHILD_COLLECTION
        assert parent_col.name == store.PARENT_COLLECTION


class TestStoreAndFetch:
    def _sample_chunks(self):
        parent = Chunk(
            chunk_id="parent-1",
            text="Full parent context about retrieval augmented generation.",
            token_count=10,
            source_url="http://example.test/paper",
            parent_id=None,
        )
        child = Chunk(
            chunk_id="child-1",
            text="retrieval augmented generation",
            token_count=3,
            source_url="http://example.test/paper",
            parent_id="parent-1",
        )
        embeddings = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
        return [parent], [child], embeddings

    def test_store_chunks_persists_parents_and_children(self, ephemeral_store):
        parents, children, embeddings = self._sample_chunks()

        store.store_chunks(parents, children, embeddings)

        child_col, parent_col = store.get_collections()
        assert child_col.count() == 1
        assert parent_col.count() == 1

    def test_get_parent_by_ids_roundtrip(self, ephemeral_store):
        parents, children, embeddings = self._sample_chunks()
        store.store_chunks(parents, children, embeddings)

        fetched = store.get_parent_by_ids(["parent-1"])

        assert len(fetched) == 1
        assert fetched[0]["chunk_id"] == "parent-1"
        assert fetched[0]["text"] == parents[0].text
        assert fetched[0]["source_url"] == "http://example.test/paper"

    def test_store_chunks_is_idempotent_on_rerun(self, ephemeral_store):
        parents, children, embeddings = self._sample_chunks()

        store.store_chunks(parents, children, embeddings)
        store.store_chunks(parents, children, embeddings)  # re-run, same IDs

        child_col, parent_col = store.get_collections()
        assert child_col.count() == 1  # upserted, not duplicated
        assert parent_col.count() == 1

    def test_child_embedding_is_searchable(self, ephemeral_store):
        parents, children, embeddings = self._sample_chunks()
        store.store_chunks(parents, children, embeddings)

        child_col, _ = store.get_collections()
        result = child_col.query(query_embeddings=embeddings.tolist(), n_results=1)

        assert result["ids"][0][0] == "child-1"
        assert result["metadatas"][0][0]["parent_id"] == "parent-1"
