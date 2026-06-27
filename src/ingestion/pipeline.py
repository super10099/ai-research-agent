"""
Entry point for the document ingestion pipeline.

Run directly to seed ChromaDB:
    python -m src.ingestion.pipeline
"""
import asyncio

from src.ingestion.chunker import Chunk, chunk_document
from src.ingestion.embedder import embed_batch
from src.ingestion.fetcher import Document, fetch_all
from src.ingestion.store import store_chunks

# These papers were chosen because they cover the key ideas this agent is built on:
# RAG, tool-use reasoning, query rewriting, and multi-step planning.
# All are freely accessible HTML pages on ArXiv.
SAMPLE_PAPER_URLS = [
    "https://arxiv.org/abs/2005.11401",  # RAG (Lewis et al., 2020)
    "https://arxiv.org/abs/2210.11610",  # ReAct: Synergizing Reasoning and Acting
    "https://arxiv.org/abs/2303.17580",  # HyDE: Hypothetical Document Embeddings
    "https://arxiv.org/abs/2305.10601",  # Tree of Thoughts
    "https://arxiv.org/abs/2304.09797",  # Judging LLM-as-a-Judge
]


async def run_pipeline(
    urls: list[str] | None = None,
    parent_size: int = 512,
    child_size: int = 256,
    overlap: float = 0.20,
    embed_batch_size: int = 64,
) -> dict:
    """
    Full ingestion pipeline: fetch → chunk → embed → store.

    Returns a summary dict with counts for quick sanity-checking.
    """
    urls = urls or SAMPLE_PAPER_URLS

    # ── 1. Fetch ──────────────────────────────────────────────────────────────
    print(f"[pipeline] Fetching {len(urls)} documents concurrently...")
    documents: list[Document] = await fetch_all(urls)
    print(f"[pipeline] Fetched {len(documents)} documents.")

    # ── 2. Chunk ──────────────────────────────────────────────────────────────
    all_parents: list[Chunk] = []
    all_children: list[Chunk] = []

    for doc in documents:
        parents, children = chunk_document(
            doc.text,
            source_url=doc.url,
            parent_size=parent_size,
            child_size=child_size,
            overlap=overlap,
        )
        all_parents.extend(parents)
        all_children.extend(children)
        print(f"[pipeline]   {doc.url}: {len(parents)} parents, {len(children)} children")

    print(
        f"[pipeline] Total: {len(all_parents)} parent chunks, "
        f"{len(all_children)} child chunks to embed."
    )

    # ── 3. Embed (batch) ──────────────────────────────────────────────────────
    # All child texts are embedded in a single batched call.  This is the step
    # where the HPC instinct pays off: 1 batched GPU call vs N serial ones.
    print(f"[pipeline] Embedding {len(all_children)} child chunks...")
    child_texts = [c.text for c in all_children]
    child_embeddings = embed_batch(child_texts, batch_size=embed_batch_size)
    print(f"[pipeline] Embeddings shape: {child_embeddings.shape}")

    # ── 4. Store ──────────────────────────────────────────────────────────────
    print("[pipeline] Storing in ChromaDB...")
    store_chunks(all_parents, all_children, child_embeddings)

    return {
        "documents_fetched": len(documents),
        "parent_chunks": len(all_parents),
        "child_chunks": len(all_children),
        "embedding_dim": int(child_embeddings.shape[1]),
    }


if __name__ == "__main__":
    import os

    os.makedirs("./data/chroma", exist_ok=True)
    result = asyncio.run(run_pipeline())
    print(f"\n[pipeline] Done: {result}")
