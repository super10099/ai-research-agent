"""Tests for src/tracing.py — LangSmith configuration and client wrapping."""

import anthropic
import pytest

from src.config import settings
from src.tracing import configure_langsmith, make_traced_async_client


@pytest.fixture(autouse=True)
def _clean_tracing_env(monkeypatch):
    """configure_langsmith uses os.environ.setdefault — each test needs a
    clean slate to actually observe that behavior rather than leftover state
    from a previous test or the real shell environment."""
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)


def test_configure_langsmith_sets_env_vars(monkeypatch):
    monkeypatch.setattr(settings, "langsmith_api_key", "ls-fake-key")
    monkeypatch.setattr(settings, "langsmith_project", "my-project")
    monkeypatch.setattr(settings, "langchain_tracing_v2", True)

    configure_langsmith()

    import os
    assert os.environ["LANGSMITH_API_KEY"] == "ls-fake-key"
    assert os.environ["LANGSMITH_PROJECT"] == "my-project"
    assert os.environ["LANGCHAIN_TRACING_V2"] == "true"


def test_configure_langsmith_does_not_override_existing_env(monkeypatch):
    # os.environ.setdefault means a value already present in the environment
    # (e.g. set by the deployment platform) must win over settings.
    monkeypatch.setenv("LANGSMITH_PROJECT", "already-set-by-deployment")
    monkeypatch.setattr(settings, "langsmith_project", "settings-value")

    configure_langsmith()

    import os
    assert os.environ["LANGSMITH_PROJECT"] == "already-set-by-deployment"


def test_configure_langsmith_skips_empty_api_key(monkeypatch):
    monkeypatch.setattr(settings, "langsmith_api_key", "")

    configure_langsmith()

    import os
    assert "LANGSMITH_API_KEY" not in os.environ


def test_make_traced_async_client_returns_plain_client_when_tracing_disabled(monkeypatch):
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)

    client = make_traced_async_client()

    assert isinstance(client, anthropic.AsyncAnthropic)


def test_make_traced_async_client_wraps_when_tracing_enabled(monkeypatch):
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")

    sentinel = object()
    monkeypatch.setattr(
        "src.tracing.wrappers.wrap_anthropic",
        lambda raw: sentinel,
    )

    client = make_traced_async_client()

    assert client is sentinel
