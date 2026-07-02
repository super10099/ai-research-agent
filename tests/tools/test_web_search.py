"""Tests for src/tools/web_search.py — Tavily-backed web search tool."""

import sys
from unittest.mock import MagicMock

import pytest

import src.tools.web_search  # noqa: F401 — ensures the module is imported/cached

# src/tools/__init__.py does `from src.tools.web_search import web_search`,
# which rebinds the package attribute `src.tools.web_search` to the function
# of the same name. Both `from src.tools import web_search` and
# `import src.tools.web_search as web_search` resolve through that attribute
# and would silently give us the function instead of the module. Pulling the
# module directly out of sys.modules sidesteps the shadowing entirely.
web_search = sys.modules["src.tools.web_search"]


@pytest.fixture(autouse=True)
def _reset_tavily_singleton():
    web_search._tavily_client = None
    yield
    web_search._tavily_client = None


def test_formats_answer_and_results(monkeypatch):
    fake_client = MagicMock()
    fake_client.search.return_value = {
        "answer": "RAG combines retrieval with generation.",
        "results": [
            {"title": "Paper A", "url": "http://a.test", "content": "Content A"},
            {"title": "Paper B", "url": "http://b.test", "content": "Content B"},
        ],
    }
    monkeypatch.setattr(web_search, "_get_tavily", lambda: fake_client)

    output = web_search.web_search("what is RAG?")

    assert output.startswith("[Synthesized Answer]\nRAG combines retrieval with generation.")
    assert "[Web Result 1] Paper A" in output
    assert "URL: http://a.test" in output
    assert "Content A" in output
    assert "[Web Result 2] Paper B" in output
    assert output.index("[Synthesized Answer]") < output.index("[Web Result 1]")


def test_skips_answer_section_when_absent(monkeypatch):
    fake_client = MagicMock()
    fake_client.search.return_value = {
        "results": [{"title": "Paper A", "url": "http://a.test", "content": "Content A"}],
    }
    monkeypatch.setattr(web_search, "_get_tavily", lambda: fake_client)

    output = web_search.web_search("query")

    assert "[Synthesized Answer]" not in output
    assert "[Web Result 1] Paper A" in output


def test_no_results_and_no_answer(monkeypatch):
    fake_client = MagicMock()
    fake_client.search.return_value = {"results": []}
    monkeypatch.setattr(web_search, "_get_tavily", lambda: fake_client)

    output = web_search.web_search("obscure query")

    assert output == "No web results found."


def test_calls_tavily_with_correct_kwargs(monkeypatch):
    fake_client = MagicMock()
    fake_client.search.return_value = {"results": []}
    monkeypatch.setattr(web_search, "_get_tavily", lambda: fake_client)

    web_search.web_search("a query", max_results=3)

    _, kwargs = fake_client.search.call_args
    assert kwargs["query"] == "a query"
    assert kwargs["search_depth"] == "advanced"
    assert kwargs["max_results"] == 3
    assert kwargs["include_answer"] is True
    assert kwargs["include_raw_content"] is False


def test_handles_missing_content_field_gracefully(monkeypatch):
    fake_client = MagicMock()
    fake_client.search.return_value = {
        "results": [{"title": "No Content Paper", "url": "http://x.test"}],
    }
    monkeypatch.setattr(web_search, "_get_tavily", lambda: fake_client)

    output = web_search.web_search("query")

    assert "[Web Result 1] No Content Paper" in output
