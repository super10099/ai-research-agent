"""
Tests for src/memory/session_memory.py — episodic memory.

embed_batch/embed_query are mocked with hand-picked vectors so cosine
distances are exact and predictable. ChromaDB itself is real (EphemeralClient)
so the 0.45 distance-threshold filtering is exercised against real HNSW
query results, not a mocked approximation of them.
"""

from unittest.mock import AsyncMock, MagicMock

import chromadb
import numpy as np
import pytest

from src.memory import session_memory


@pytest.fixture
def ephemeral_memory(monkeypatch):
    # chromadb.EphemeralClient() instances with identical default settings
    # share underlying storage via chromadb's internal system cache — without
    # clearing it, data written by one test leaks into the next.
    chromadb.api.client.SharedSystemClient.clear_system_cache()
    client = chromadb.EphemeralClient()
    monkeypatch.setattr(session_memory, "get_client", lambda: client)
    return client


class TestStoreSession:
    def test_stores_with_generated_session_id(self, ephemeral_memory, monkeypatch):
        monkeypatch.setattr(session_memory, "embed_batch", lambda texts: np.array([[1.0, 0.0]]))

        session_id = session_memory.store_session(topic="RAG systems", summary="A summary.")

        assert session_id  # non-empty, auto-generated UUID
        col = session_memory._get_memory_collection()
        assert col.count() == 1
        stored = col.get(ids=[session_id], include=["documents", "metadatas"])
        assert stored["documents"][0] == "A summary."
        assert stored["metadatas"][0]["topic"] == "RAG systems"
        assert stored["metadatas"][0]["session_id"] == session_id
        assert "stored_at" in stored["metadatas"][0]

    def test_stores_with_provided_session_id(self, ephemeral_memory, monkeypatch):
        monkeypatch.setattr(session_memory, "embed_batch", lambda texts: np.array([[1.0, 0.0]]))

        session_id = session_memory.store_session(
            topic="RAG systems", summary="A summary.", session_id="fixed-id-123",
        )

        assert session_id == "fixed-id-123"


class TestRetrieveRelevantSessions:
    def test_empty_collection_short_circuits(self, ephemeral_memory, monkeypatch):
        embed_query_mock = MagicMock()  # embed_query is sync; should never be called
        monkeypatch.setattr(session_memory, "embed_query", embed_query_mock)

        result = session_memory.retrieve_relevant_sessions("any topic")

        assert result == []
        embed_query_mock.assert_not_called()

    def test_filters_by_cosine_distance_threshold(self, ephemeral_memory, monkeypatch):
        # Store two sessions with orthogonal-ish embeddings so distances are
        # exact and predictable under ChromaDB's cosine metric.
        embeddings_by_summary = {
            "close summary": [1.0, 0.0],
            "far summary": [0.0, 1.0],  # cosine distance from [1,0] is 1.0
        }
        monkeypatch.setattr(
            session_memory, "embed_batch",
            lambda texts: np.array([embeddings_by_summary[t] for t in texts]),
        )
        session_memory.store_session(topic="topic A", summary="close summary")
        session_memory.store_session(topic="topic B", summary="far summary")

        # Query embedding identical to "close summary" -> distance ~0 (< 0.45)
        # and far from "far summary" -> distance ~1.0 (>= 0.45).
        monkeypatch.setattr(session_memory, "embed_query", lambda topic: [1.0, 0.0])

        results = session_memory.retrieve_relevant_sessions("query topic", top_k=3)

        assert len(results) == 1
        assert results[0]["summary"] == "close summary"
        assert results[0]["topic"] == "topic A"

    def test_respects_top_k(self, ephemeral_memory, monkeypatch):
        monkeypatch.setattr(
            session_memory, "embed_batch", lambda texts: np.array([[1.0, 0.0]] * len(texts)),
        )
        for i in range(5):
            session_memory.store_session(topic=f"topic {i}", summary=f"summary {i}")

        monkeypatch.setattr(session_memory, "embed_query", lambda topic: [1.0, 0.0])

        results = session_memory.retrieve_relevant_sessions("query", top_k=2)

        assert len(results) == 2


class TestFormatPriorContext:
    def test_empty_list_returns_empty_string(self):
        assert session_memory.format_prior_context([]) == ""

    def test_formats_sessions_numbered(self):
        sessions = [
            {"topic": "Topic A", "summary": "Summary A"},
            {"topic": "Topic B", "summary": "Summary B"},
        ]

        output = session_memory.format_prior_context(sessions)

        assert "Session 1 — topic: Topic A" in output
        assert "Summary A" in output
        assert "Session 2 — topic: Topic B" in output
        assert "Summary B" in output
        assert output.index("Session 1") < output.index("Session 2")


class TestSummarizeSession:
    async def test_truncates_report_to_6000_chars(self, monkeypatch):
        fake_response = AsyncMock()
        fake_response.content = [type("Block", (), {"text": "A concise summary."})()]
        mock_create = AsyncMock(return_value=fake_response)
        monkeypatch.setattr(session_memory._anthropic.messages, "create", mock_create)

        long_report = "x" * 10_000
        result = await session_memory.summarize_session("A topic", long_report)

        assert result == "A concise summary."
        _, kwargs = mock_create.call_args
        sent_content = kwargs["messages"][0]["content"]
        assert ("x" * 6000) in sent_content
        assert ("x" * 6001) not in sent_content
