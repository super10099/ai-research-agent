import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import settings

# Module-level singleton — loading a transformer model takes ~2s and ~800 MB RAM.
# Keeping it alive avoids re-loading on every call.
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"[embedder] Loading {settings.embedding_model} (first call only)...")
        _model = SentenceTransformer(settings.embedding_model)
    return _model


def embed_batch(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """
    Embed all texts in a single batched forward pass.

    Why batch and not one-at-a-time?
    A transformer encoder has a fixed per-call overhead (kernel launch, memory
    transfer, attention mask setup) regardless of how many sequences are in the
    batch.  Encoding N texts one-at-a-time pays that overhead N times; encoding
    them together pays it once. For 1 000 child chunks with batch_size=64 that's
    ~16 forward passes instead of 1 000 — same reasoning as why you fuse
    CUDA kernels or use collective allreduce instead of N point-to-point sends.

    normalize_embeddings=True → each vector is L2-normalized before returning.
    This means cosine similarity reduces to a dot product, which is what
    ChromaDB's HNSW index computes natively (faster and no sqrt needed).

    Returns: float32 array of shape (len(texts), embedding_dim).
    embedding_dim == 1024 for bge-large-en-v1.5.
    """
    model = get_model()
    embeddings: np.ndarray = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 100,  # only show bar for large jobs
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings


def embed_query(query: str) -> list[float]:
    """
    Embed a single query string.  Returns a plain list so ChromaDB can consume it
    without an extra .tolist() call at the call site.
    """
    vec = embed_batch([query])
    return vec[0].tolist()
