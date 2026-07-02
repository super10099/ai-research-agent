"""Tests for src/config.py — typed settings loading."""

import pytest
from pydantic import ValidationError

from src.config import Settings


def test_settings_loads_with_required_env_vars(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
    monkeypatch.setenv("COHERE_API_KEY", "co-fake")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-fake")

    s = Settings(_env_file=None)  # bypass the real .env file for this test

    assert s.anthropic_api_key == "sk-ant-fake"
    assert s.cohere_api_key == "co-fake"
    assert s.tavily_api_key == "tvly-fake"


def test_settings_defaults(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("COHERE_API_KEY", "x")
    monkeypatch.setenv("TAVILY_API_KEY", "x")

    s = Settings(_env_file=None)

    assert s.llm_model == "claude-sonnet-4-6"
    assert s.max_tool_turns == 5
    assert s.max_research_iterations == 2
    assert s.chroma_use_http is False
    assert s.chroma_port == 8000
    assert s.retrieval_top_k == 20
    assert s.rerank_top_n == 5


def test_settings_missing_required_field_raises(monkeypatch):
    # Explicitly remove all three required keys and bypass the .env file,
    # so there is genuinely nowhere for the required fields to come from.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None)

    missing_fields = {e["loc"][0] for e in exc_info.value.errors()}
    assert missing_fields == {"anthropic_api_key", "cohere_api_key", "tavily_api_key"}
