"""
RAG retrieval tool: two-stage retrieve-then-rerank.

Stage 1 — ANN search over child chunks (fast, approximate, high recall)
Stage 2 — Cohere cross-encoder reranking over parent chunks (slow, exact, high precision)
"""

import cohere
from langsmith import traceable

from src.config import settings
from src.ingestion.embedder import embed_query
from src.ingestion.store import get_collections, get_parent_by_ids

# Singleton Cohere client — creating a client spins up an HTTP session,
# so we reuse it across calls rather than rebuilding per request.
_cohere_client: cohere.Client | None = None


def _get_cohere() -> cohere.Client:
    global _cohere_client
    if _cohere_client is None:
        _cohere_client = cohere.Client(api_key=settings.cohere_api_key)
    return _cohere_client


# ── Anthropic tool schema ─────────────────────────────────────────────────────
# This dict is passed verbatim to the Anthropic API in the `tools` list.
# The model reads the description to decide when to call this tool;
# the input_schema tells it what arguments to emit in its tool_use block.
RETRIEVAL_TOOL_SCHEMA: dict = {
    "name": "retrieve_documents",
    "description": (
        "Search a vector database of AI research papers for passages relevant to a query. "
        "Returns the most relevant excerpts, ranked by semantic similarity and cross-encoder score. "
        "Use this for factual questions about AI techniques, architectures, and experimental results."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "A focused, self-contained search query. "
                    "More specific is better — 'how does HNSW handle deletions' outperforms 'vector search'."
                ),
            }
        },
        "required": ["query"],
    },
}


@traceable(run_type="tool", name="retrieve_documents")
def retrieve_documents(query: str) -> str:
    """
    Execute the retrieval tool synchronously.

    Returns a formatted string ready to paste into a tool_result message.
    String format is chosen over a structured dict because the model needs to
    read and reason about the content directly — structured data would require
    an extra parsing step that adds no value here.
    """
    child_col, _ = get_collections()

    # ── Stage 1: ANN search over child chunk embeddings ───────────────────────
    query_vec = embed_query(query)  # shape (1024,) float32, L2-normalized

    raw = child_col.query(
        query_embeddings=[query_vec],
        n_results=settings.retrieval_top_k,
        include=["metadatas", "distances"],
    )

    # Deduplicate: multiple child chunks may share the same parent.
    # We only want each parent once — the parent is what we send to the model.
    # Preserve insertion order so the first-seen (closest) child determines rank.
    seen_parent_ids: set[str] = set()
    ordered_parent_ids: list[str] = []
    for meta in raw["metadatas"][0]:
        pid: str = meta["parent_id"]
        if pid not in seen_parent_ids:
            seen_parent_ids.add(pid)
            ordered_parent_ids.append(pid)

    if not ordered_parent_ids:
        return "No relevant documents found in the vector database."

    # ── Stage 2: fetch parent chunks (the larger context windows) ────────────
    parents = get_parent_by_ids(ordered_parent_ids)
    if not parents:
        return "Parent chunks could not be retrieved."

    # ── Stage 3: Cohere cross-encoder reranking ───────────────────────────────
    # Cohere's rerank model is a cross-encoder: it jointly encodes (query, doc)
    # pairs and produces a calibrated relevance score.  This is more accurate
    # than cosine similarity (which encodes query and doc independently) but
    # too slow to run over the full corpus — hence the two-stage design.
    co = _get_cohere()
    rerank_result = co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=[p["text"] for p in parents],
        top_n=min(settings.rerank_top_n, len(parents)),
        return_documents=False,  # we already have the text; skip redundant payload
    )

    # ── Format for model consumption ─────────────────────────────────────────
    sections: list[str] = []
    for rank, r in enumerate(rerank_result.results, start=1):
        parent = parents[r.index]
        sections.append(
            f"[Document {rank}] relevance={r.relevance_score:.3f} | source={parent['source_url']}\n"
            f"{parent['text']}"
        )

    return "\n\n---\n\n".join(sections)
