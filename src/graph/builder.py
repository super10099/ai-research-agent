"""
Assembles the LangGraph StateGraph: nodes, edges, checkpointer, interrupt.

Keeping graph construction in a separate module from the node functions lets
tests inject a MemorySaver checkpointer without touching production code, and
lets the FastAPI app inject a long-lived SqliteSaver without circular imports.
"""

import os

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from src.config import settings
from src.graph.nodes import critic_node, planner_node, researcher_node, synthesizer_node
from src.graph.state import AgentState


# ── Routing function ─────────────────────────────────────────────────────────

def _route_after_critic(state: AgentState) -> str:
    """
    Conditional edge evaluated after every critic run.

    Returns the name of the next node.  LangGraph uses the returned string
    as a key into the dict passed to add_conditional_edges to resolve the
    actual target node — the extra indirection lets you rename nodes without
    touching the routing logic.
    """
    return "researcher" if state.get("needs_more_research") else "synthesizer"


# ── Graph factory ─────────────────────────────────────────────────────────────

def build_graph(checkpointer=None):
    """
    Construct and compile the research agent StateGraph.

    Graph topology:

        START
          │
        planner          ← decomposes topic into sub-questions
          │
        researcher       ← agentic tool loop per sub-question (parallel)
          │
        critic           ← evaluates coverage, sets needs_more_research
          │
        ┌─┴──────────────────────────────┐
        │ needs_more_research=True        │ needs_more_research=False
        ▼                                 ▼
      researcher  (loop)            synthesizer   ← [INTERRUPT HERE]
                                         │
                                        END

    The interrupt_before=["synthesizer"] pause lets a human inspect the
    research_results and critique before the final report is written.
    Resuming with graph.invoke(None, config=config) continues from the pause
    without re-running any prior nodes — the checkpointer replays state.

    Parameters
    ----------
    checkpointer:
        SqliteSaver for production, MemorySaver for tests, None to disable.
    """
    g = StateGraph(AgentState)

    # ── Register nodes ────────────────────────────────────────────────────────
    g.add_node("planner", planner_node)
    g.add_node("researcher", researcher_node)
    g.add_node("critic", critic_node)
    g.add_node("synthesizer", synthesizer_node)

    # ── Wire edges ────────────────────────────────────────────────────────────
    g.add_edge(START, "planner")
    g.add_edge("planner", "researcher")
    g.add_edge("researcher", "critic")

    # The conditional edge replaces the simple "critic → synthesizer" edge.
    # _route_after_critic is called with the post-critic state and returns
    # one of the keys in the mapping dict below.
    g.add_conditional_edges(
        "critic",
        _route_after_critic,
        {
            "researcher": "researcher",   # loop: fill identified gaps
            "synthesizer": "synthesizer", # proceed: coverage is sufficient
        },
    )

    g.add_edge("synthesizer", END)

    # ── Compile ───────────────────────────────────────────────────────────────
    return g.compile(
        checkpointer=checkpointer,
        # Pause execution immediately before the synthesizer runs.
        # At this point all research is complete and the human can review it.
        interrupt_before=["synthesizer"],
    )


# ── Checkpointer factory ──────────────────────────────────────────────────────

def make_checkpointer() -> SqliteSaver:
    """
    Create the on-disk SQLite checkpointer used in production.

    SqliteSaver writes one row per (thread_id, checkpoint_id) to a local
    SQLite database.  thread_id maps to a research session — different users
    get different thread_ids and never see each other's state.

    The checkpointer is meant to be created once at application startup and
    kept alive for the process lifetime (it holds a database connection).
    """
    db_path = settings.langgraph_checkpoint_db
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    return SqliteSaver.from_conn_string(db_path)


# ── Initial state helper ──────────────────────────────────────────────────────

def make_initial_state(
    topic: str,
    prior_context: str = "",
    max_iterations: int | None = None,
) -> dict:
    """
    Return a fully-populated initial state dict for a new research session.

    All list fields must be pre-initialized to [] so LangGraph's operator.add
    reducer has a valid list to append to on the first node execution.
    Omitting a field (or leaving it as None) causes a KeyError inside the
    node when it tries to read it.
    """
    return {
        "topic": topic,
        "sub_questions": [],
        "research_results": [],
        "critique": "",
        "gaps": [],
        "needs_more_research": False,
        "final_report": "",
        "prior_context": prior_context,
        "iteration": 0,
        "max_iterations": max_iterations or settings.max_research_iterations,
    }
