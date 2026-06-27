import numpy as np
import chromadb
from chromadb.config import Settings as ChromaSettings

from src.config import settings
from src.ingestion.chunker import Chunk

# Collection names are stable constants — if they ever change, a re-index is
# required because the existing data is keyed under the old names.
CHILD_COLLECTION = "child_chunks"
PARENT_COLLECTION = "parent_chunks"

_client: chromadb.ClientAPI | None = None


def get_client() -> chromadb.ClientAPI:
    """
    Return a ChromaDB client in the correct mode for the current environment.

    Local dev  (CHROMA_USE_HTTP=false): PersistentClient — runs embedded in the
    same process, writes directly to disk.  No server needed.

    Docker     (CHROMA_USE_HTTP=true):  HttpClient — connects to the chromadb
    container over HTTP.  The client API is identical; only the transport changes.
    This is the same pattern as switching a database driver from sqlite3 to
    psycopg2: the calling code is untouched, only the connection string changes.
    """
    global _client
    if _client is None:
        if settings.chroma_use_http:
            _client = chromadb.HttpClient(
                host=settings.chroma_host,
                port=settings.chroma_port,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        else:
            _client = chromadb.PersistentClient(
                path=settings.chroma_persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
    return _client


def get_collections() -> tuple[chromadb.Collection, chromadb.Collection]:
    client = get_client()
    # hnsw:space=cosine tells ChromaDB's HNSW index which distance metric to use.
    # Our embeddings are L2-normalized, so cosine distance == 1 - dot_product.
    child_col = client.get_or_create_collection(
        CHILD_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    parent_col = client.get_or_create_collection(
        PARENT_COLLECTION,
        # Parents are retrieved by ID, not by ANN search, so the distance
        # metric doesn't matter — but we set it for consistency.
        metadata={"hnsw:space": "cosine"},
    )
    return child_col, parent_col


def store_chunks(
    parent_chunks: list[Chunk],
    child_chunks: list[Chunk],
    child_embeddings: np.ndarray,
) -> None:
    """
    Persist parent and child chunks in ChromaDB.

    Two-collection design:
    - parent_col  stores the full text of parent chunks, keyed by chunk_id.
                  No embedding stored — we look these up by ID, not by ANN.
    - child_col   stores child text + embedding + metadata (including parent_id).
                  These are what the HNSW index searches over.

    Upsert semantics: safe to re-run the pipeline on the same docs — existing
    chunks are updated rather than duplicated.  Same idempotency guarantee as a
    Raft log entry replayed on restart.
    """
    child_col, parent_col = get_collections()

    # Batch size for upsert — ChromaDB handles this internally but we chunk
    # explicitly to avoid hitting SQLite's variable limit on very large ingests.
    BATCH = 512

    # ── Parent chunks (no embedding) ──────────────────────────────────────────
    for i in range(0, len(parent_chunks), BATCH):
        batch = parent_chunks[i : i + BATCH]
        parent_col.upsert(
            ids=[c.chunk_id for c in batch],
            documents=[c.text for c in batch],
            metadatas=[{"source_url": c.source_url} for c in batch],
        )

    # ── Child chunks (with embedding) ─────────────────────────────────────────
    for i in range(0, len(child_chunks), BATCH):
        batch = child_chunks[i : i + BATCH]
        emb_batch = child_embeddings[i : i + BATCH]
        child_col.upsert(
            ids=[c.chunk_id for c in batch],
            documents=[c.text for c in batch],
            embeddings=emb_batch.tolist(),
            metadatas=[{
                "source_url": c.source_url,
                "parent_id": c.parent_id,
                "token_count": c.token_count,
            } for c in batch],
        )

    print(f"[store] Persisted {len(parent_chunks)} parents, {len(child_chunks)} children.")


def get_parent_by_ids(parent_ids: list[str]) -> list[dict]:
    """
    Bulk-fetch parent chunks by their IDs.
    Returns list of {"chunk_id", "text", "source_url"} dicts.
    """
    _, parent_col = get_collections()
    result = parent_col.get(ids=parent_ids, include=["documents", "metadatas"])
    return [
        {
            "chunk_id": cid,
            "text": doc,
            "source_url": meta["source_url"],
        }
        for cid, doc, meta in zip(
            result["ids"], result["documents"], result["metadatas"]
        )
    ]
