"""
Tests for src/ingestion/embedder.py.

SentenceTransformer is mocked throughout — loading the real bge-large-en-v1.5
model takes ~2s and ~800MB RAM and requires downloading weights on first use,
none of which unit tests should depend on.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.ingestion import embedder


@pytest.fixture(autouse=True)
def _reset_model_singleton():
    """embedder._model is a process-global singleton; each test must start
    with it unset so mocking SentenceTransformer actually takes effect."""
    embedder._model = None
    yield
    embedder._model = None


def test_get_model_is_a_singleton(monkeypatch):
    mock_ctor = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(embedder, "SentenceTransformer", mock_ctor)

    first = embedder.get_model()
    second = embedder.get_model()

    assert first is second
    mock_ctor.assert_called_once()


def test_embed_batch_calls_encode_with_correct_kwargs(monkeypatch):
    fake_model = MagicMock()
    fake_model.encode.return_value = np.ones((2, 1024), dtype=np.float32)
    monkeypatch.setattr(embedder, "SentenceTransformer", MagicMock(return_value=fake_model))

    result = embedder.embed_batch(["hello", "world"], batch_size=32)

    fake_model.encode.assert_called_once_with(
        ["hello", "world"],
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    assert result.shape == (2, 1024)


def test_embed_batch_shows_progress_bar_for_large_jobs(monkeypatch):
    fake_model = MagicMock()
    fake_model.encode.return_value = np.ones((150, 1024), dtype=np.float32)
    monkeypatch.setattr(embedder, "SentenceTransformer", MagicMock(return_value=fake_model))

    embedder.embed_batch(["x"] * 150)

    _, kwargs = fake_model.encode.call_args
    assert kwargs["show_progress_bar"] is True


def test_embed_query_returns_plain_list(monkeypatch):
    fake_model = MagicMock()
    fake_model.encode.return_value = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
    monkeypatch.setattr(embedder, "SentenceTransformer", MagicMock(return_value=fake_model))

    result = embedder.embed_query("a query")

    assert isinstance(result, list)
    assert result == pytest.approx([0.1, 0.2, 0.3])
