"""
External episodic memory: stores and retrieves summaries of past research sessions.

This is one of four memory types in the system:

  1. In-context (ephemeral)  — the messages list inside a single LLM call.
                               Lives in RAM, gone when the call returns.
  2. Working memory          — AgentState flowing through the LangGraph.
                               Lives for the duration of one research session.
                               Persisted per-node via the SqliteSaver checkpointer.
  3. Semantic memory         — the paper corpus in ChromaDB (child_chunks /
                               parent_chunks collections, built in Step 2).
                               Long-lived, read-only during agent execution.
  4. Episodic memory (this)  — summaries of completed past sessions, stored
                               in ChromaDB and retrieved at the start of new
                               sessions relevant to the same topic.
                               Lets the agent say "I've seen this before"
                               without replaying the full prior session.
"""

import uuid
from datetime import datetime, timezone

import anthropic

from src.config import settings
from src.ingestion.embedder import embed_batch, embed_query
from src.ingestion.store import get_client

SESSION_MEMORY_COLLECTION = "session_memory"

# Reuse the same Anthropic client pattern from nodes.py
_anthropic = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)


def _get_memory_collection():
    """Session memory lives in its own ChromaDB collection, separate from papers."""
    return get_client().get_or_create_collection(
        SESSION_MEMORY_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


# ── Store ─────────────────────────────────────────────────────────────────────

def store_session(topic: str, summary: str, session_id: str | None = None) -> str:
    """
    Persist a session summary as a vector in the episodic memory collection.

    The summary is embedded and stored with the original topic as metadata.
    Future sessions on similar topics will retrieve this entry via ANN search.

    Returns the session_id (useful for updating or deleting the entry later).
    """
    session_id = session_id or str(uuid.uuid4())
    col = _get_memory_collection()

    embedding = embed_batch([summary])[0].tolist()

    col.upsert(
        ids=[session_id],
        documents=[summary],
        embeddings=[embedding],
        metadatas=[{
            "topic": topic,
            "session_id": session_id,
            # ISO 8601 UTC — sortable and timezone-unambiguous
            "stored_at": datetime.now(timezone.utc).isoformat(),
        }],
    )

    print(f"[memory] Stored session {session_id[:8]}... for: {topic[:60]}")
    return session_id


# ── Retrieve ──────────────────────────────────────────────────────────────────

def retrieve_relevant_sessions(topic: str, top_k: int = 3) -> list[dict]:
    """
    Retrieve past sessions whose topic is semantically close to the new topic.

    Distance threshold of 0.45 (cosine) is intentionally tight — we only want
    sessions that are genuinely on the same subject, not loosely related ones.
    Too-permissive retrieval injects irrelevant context that confuses the planner.
    """
    col = _get_memory_collection()
    count = col.count()
    if count == 0:
        return []

    query_vec = embed_query(topic)
    results = col.query(
        query_embeddings=[query_vec],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )

    sessions = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        if dist < 0.45:  # cosine distance; 0.0 = identical, 1.0 = orthogonal
            sessions.append({
                "topic": meta["topic"],
                "summary": doc,
                "stored_at": meta["stored_at"],
            })

    return sessions


def format_prior_context(sessions: list[dict]) -> str:
    """
    Format retrieved sessions as a text block for injection into system prompts.
    Returns an empty string if no sessions were retrieved (callers must check).
    """
    if not sessions:
        return ""

    lines = ["[Prior research context — use this to avoid repeating known work]"]
    for i, s in enumerate(sessions, 1):
        lines.append(f"\nSession {i} — topic: {s['topic']}\n{s['summary']}")
    return "\n".join(lines)


# ── Summarize ─────────────────────────────────────────────────────────────────

async def summarize_session(topic: str, report: str) -> str:
    """
    Use Claude to distill a completed report into a compact memory entry.

    We summarize rather than storing the full report because:
    - The full report can be thousands of tokens — too expensive to inject
      into future prompts in its entirety.
    - A 3-5 sentence summary captures the key findings without the structure.
    - The summary embedding reflects the *content* of findings, not the report
      format, making retrieval more accurate.

    The report is trimmed to 6000 chars before sending to avoid hitting
    max_tokens on very long reports.
    """
    response = await _anthropic.messages.create(
        model=settings.llm_model,
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": (
                "Summarize the key findings from this research report in 4-6 sentences. "
                "Focus on facts, conclusions, and named techniques — not report structure.\n\n"
                f"Topic: {topic}\n\n"
                f"Report excerpt:\n{report[:6000]}"
            ),
        }],
    )
    return response.content[0].text
