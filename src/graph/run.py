"""
Minimal CLI driver for the research graph.

Usage:
    python -m src.graph.run "Retrieval-Augmented Generation"

This runs the full graph:
  [memory retrieval] → planner → researcher → critic → [interrupt] → human review → synthesizer → [memory store]
"""

import asyncio
import sys
import uuid

from src.graph.builder import build_graph, make_checkpointer, make_initial_state
from src.tracing import configure_langsmith
from src.memory.session_memory import (
    format_prior_context,
    retrieve_relevant_sessions,
    store_session,
    summarize_session,
)


async def run(topic: str) -> None:
    configure_langsmith()
    checkpointer = make_checkpointer()
    graph = build_graph(checkpointer=checkpointer)

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # ── Memory retrieval (before graph) ───────────────────────────────────────
    # Look up past sessions on similar topics and format them as prior_context.
    # This happens outside the graph so it doesn't appear in the checkpoint —
    # it's a one-time lookup, not a node in the research workflow.
    print("[memory] Searching for relevant prior sessions...")
    prior_sessions = retrieve_relevant_sessions(topic)
    prior_context = format_prior_context(prior_sessions)
    if prior_context:
        print(f"[memory] Found {len(prior_sessions)} relevant prior session(s).")
    else:
        print("[memory] No relevant prior sessions found.")

    initial_state = make_initial_state(topic, prior_context=prior_context)

    print(f"\n{'='*60}")
    print(f"Research topic: {topic}")
    print(f"Thread ID:      {thread_id}")
    print(f"{'='*60}\n")

    # ── Phase 1: planner → researcher → critic (pauses at synthesizer) ────────
    print("[graph] Phase 1: Planning and research...\n")
    state = await graph.ainvoke(initial_state, config=config)

    print("\n" + "="*60)
    print("RESEARCH COMPLETE — HUMAN REVIEW")
    print("="*60)

    print("\nSub-questions investigated:")
    for i, q in enumerate(state.get("sub_questions", []), 1):
        print(f"  {i}. {q}")

    print(f"\nCritic's assessment:\n  {state.get('critique', 'N/A')}")

    if state.get("gaps"):
        print("\nIdentified gaps:")
        for g in state["gaps"]:
            print(f"  • {g}")

    print(f"\nResearch iterations completed: {state.get('iteration', 0)}")

    answer = input("\nProceed to synthesis? [y/n]: ").strip().lower()
    if answer != "y":
        print("[graph] Synthesis skipped. State saved.")
        print(f"[graph] Resume with: thread_id={thread_id}")
        return

    # ── Phase 2: synthesizer (resume from checkpoint) ─────────────────────────
    print("\n[graph] Phase 2: Synthesizing report...\n")
    final_state = await graph.ainvoke(None, config=config)

    report = final_state.get("final_report", "")
    print("\n" + "="*60)
    print("FINAL REPORT")
    print("="*60)
    print(report)

    # ── Memory storage (after graph) ──────────────────────────────────────────
    # Summarize and store this session so future runs on related topics can
    # retrieve it as prior context.
    if report:
        print("\n[memory] Summarizing session for long-term memory...")
        summary = await summarize_session(topic, report)
        store_session(topic=topic, summary=summary, session_id=thread_id)
        print("[memory] Session stored.")


if __name__ == "__main__":
    topic = sys.argv[1] if len(sys.argv) > 1 else "Retrieval-Augmented Generation for LLMs"
    asyncio.run(run(topic))
