"""
Tests for src/tools/retrieval.py — two-stage retrieve-then-rerank.

get_collections, get_parent_by_ids, embed_query, and the Cohere client are all
mocked: this module's job is to orchestrate those calls correctly (dedup,
call args, formatting), not to re-test ChromaDB or Cohere themselves.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.config import settings
from src.tools import retrieval


@pytest.fixture(autouse=True)
def _reset_cohere_singleton():
    retrieval._cohere_client = None
    yield
    retrieval._cohere_client = None


def _fake_ann_result(parent_ids: list[str]) -> dict:
    """Shape matches chromadb's Collection.query(...) return value."""
    return {"metadatas": [[{"parent_id": pid} for pid in parent_ids]]}


def _fake_rerank_response(pairs: list[tuple[int, float]]):
    """pairs = [(parent_index, relevance_score), ...] in rank order."""
    results = [SimpleNamespace(index=i, relevance_score=s) for i, s in pairs]
    return SimpleNamespace(results=results)


class TestRetrieveDocuments:
    def test_dedup_preserves_first_seen_order(self, monkeypatch):
        monkeypatch.setattr(retrieval, "embed_query", lambda q: [0.1, 0.2])
        fake_child_col = MagicMock()
        fake_child_col.query.return_value = _fake_ann_result(["p2", "p1", "p2", "p3"])
        monkeypatch.setattr(retrieval, "get_collections", lambda: (fake_child_col, None))

        captured_parent_ids = {}

        def fake_get_parents(parent_ids):
            captured_parent_ids["ids"] = parent_ids
            return [
                {"chunk_id": pid, "text": f"text-{pid}", "source_url": "http://x"}
                for pid in parent_ids
            ]
        monkeypatch.setattr(retrieval, "get_parent_by_ids", fake_get_parents)

        fake_cohere = MagicMock()
        fake_cohere.rerank.return_value = _fake_rerank_response([(0, 0.9), (1, 0.5), (2, 0.3)])
        monkeypatch.setattr(retrieval, "_get_cohere", lambda: fake_cohere)

        retrieval.retrieve_documents("what is RAG?")

        assert captured_parent_ids["ids"] == ["p2", "p1", "p3"]

    def test_cohere_called_with_correct_args(self, monkeypatch):
        monkeypatch.setattr(retrieval, "embed_query", lambda q: [0.1])
        fake_child_col = MagicMock()
        fake_child_col.query.return_value = _fake_ann_result(["p1", "p2"])
        monkeypatch.setattr(retrieval, "get_collections", lambda: (fake_child_col, None))
        monkeypatch.setattr(
            retrieval,
            "get_parent_by_ids",
            lambda ids: [
                {"chunk_id": pid, "text": f"text-{pid}", "source_url": "http://x"} for pid in ids
            ],
        )
        monkeypatch.setattr(settings, "rerank_top_n", 5)
        fake_cohere = MagicMock()
        fake_cohere.rerank.return_value = _fake_rerank_response([(0, 0.9), (1, 0.4)])
        monkeypatch.setattr(retrieval, "_get_cohere", lambda: fake_cohere)

        retrieval.retrieve_documents("a focused query")

        _, kwargs = fake_cohere.rerank.call_args
        assert kwargs["query"] == "a focused query"
        assert kwargs["documents"] == ["text-p1", "text-p2"]
        assert kwargs["top_n"] == 2  # min(rerank_top_n=5, len(parents)=2)
        assert kwargs["return_documents"] is False

    def test_output_formatted_in_rerank_rank_order(self, monkeypatch):
        monkeypatch.setattr(retrieval, "embed_query", lambda q: [0.1])
        fake_child_col = MagicMock()
        fake_child_col.query.return_value = _fake_ann_result(["p1", "p2"])
        monkeypatch.setattr(retrieval, "get_collections", lambda: (fake_child_col, None))
        monkeypatch.setattr(
            retrieval,
            "get_parent_by_ids",
            lambda ids: [
                {"chunk_id": pid, "text": f"text-{pid}", "source_url": f"http://{pid}"}
                for pid in ids
            ],
        )
        fake_cohere = MagicMock()
        # rerank reorders: parent index 1 (p2) ranks above index 0 (p1)
        fake_cohere.rerank.return_value = _fake_rerank_response([(1, 0.95), (0, 0.42)])
        monkeypatch.setattr(retrieval, "_get_cohere", lambda: fake_cohere)

        output = retrieval.retrieve_documents("query")

        first_doc_pos = output.index("[Document 1]")
        second_doc_pos = output.index("[Document 2]")
        assert first_doc_pos < second_doc_pos
        assert "source=http://p2" in output.split("[Document 2]")[0]
        assert "relevance=0.950" in output
        assert "text-p2" in output.split("[Document 2]")[0]

    def test_no_ann_results_returns_early_without_reranking(self, monkeypatch):
        monkeypatch.setattr(retrieval, "embed_query", lambda q: [0.1])
        fake_child_col = MagicMock()
        fake_child_col.query.return_value = _fake_ann_result([])
        monkeypatch.setattr(retrieval, "get_collections", lambda: (fake_child_col, None))
        get_parents_mock = MagicMock()
        monkeypatch.setattr(retrieval, "get_parent_by_ids", get_parents_mock)
        cohere_mock = MagicMock()
        monkeypatch.setattr(retrieval, "_get_cohere", lambda: cohere_mock)

        result = retrieval.retrieve_documents("obscure query")

        assert result == "No relevant documents found in the vector database."
        get_parents_mock.assert_not_called()
        cohere_mock.rerank.assert_not_called()

    def test_parents_could_not_be_retrieved(self, monkeypatch):
        monkeypatch.setattr(retrieval, "embed_query", lambda q: [0.1])
        fake_child_col = MagicMock()
        fake_child_col.query.return_value = _fake_ann_result(["p1"])
        monkeypatch.setattr(retrieval, "get_collections", lambda: (fake_child_col, None))
        monkeypatch.setattr(retrieval, "get_parent_by_ids", lambda ids: [])

        result = retrieval.retrieve_documents("query")

        assert result == "Parent chunks could not be retrieved."
