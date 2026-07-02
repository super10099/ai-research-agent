"""
Assembles the LangGraph StateGraph: nodes, edges, checkpointer, interrupt.

Keeping graph construction in a separate module from the node functions lets
tests inject a MemorySaver checkpointer without touching production code, and
lets the FastAPI app inject a long-lived SqliteSaver without circular imports.
"""

import os

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from src.config import settings
from src.graph.nodes import critic_node, planner_node, research_one_node, synthesizer_node
from src.graph.state import AgentState

# ── Routing functions ─────────────────────────────────────────────────────────
#
# Both of these return `Send` objects instead of plain node-name strings.
# Each `Send("research_one", {...})` spawns its own instance of research_one
# with an isolated input — LangGraph runs all of them as parallel branches,
# gives each its own checkpoint, and joins them back into shared AgentState
# (via the research_results reducer) before the next node in the edge list
# (critic) runs. This replaces the old single "researcher" node that used
# asyncio.gather internally — that gave concurrency but not real parallel
# graph branches (no per-question checkpoint/resumability, no visibility
# from the graph itself).

def _fan_out_from_planner(state: AgentState) -> list[Send]:
    """First research wave: one Send branch per planner sub-question."""
    return [Send("research_one", {"question": q}) for q in state["sub_questions"]]


def _route_after_critic(state: AgentState) -> list[Send] | str:
    """
    Conditional edge evaluated after every critic run.

    If more research is needed, fan out one Send branch per gap the critic
    identified (a second research wave). Otherwise proceed to synthesizer.
    Which question set to use (sub_questions vs gaps) is now encoded by
    *which routing function fires* rather than an iteration-count check.
    """
    if state.get("needs_more_research"):
        return [Send("research_one", {"question": q}) for q in state.get("gaps", [])]
    return "synthesizer"


# ── Graph factory ─────────────────────────────────────────────────────────────

def build_graph(checkpointer=None):
    """
    Construct and compile the research agent StateGraph.

    Graph topology:

        START
          │
        planner                     ← decomposes topic into sub-questions
          │
        [Send research_one × N]     ← fan-out: one branch per sub-question,
          │                            each its own checkpointed node instance
        research_one  (parallel)
          │
        critic                      ← join point; evaluates coverage,
          │                            sets needs_more_research, bumps iteration
          │
        ┌─┴────────────────────────────────┐
        │ needs_more_research=True          │ needs_more_research=False
        ▼                                   ▼
      [Send research_one × N]  (loop)  synthesizer   ← [INTERRUPT HERE]
      → research_one → critic               │
                                            END

    Sub-question fan-out uses LangGraph's `Send` API (see _fan_out_from_planner
    and _route_after_critic) rather than a single node doing asyncio.gather
    internally. Each Send spawns an independent instance of research_one with
    its own checkpoint; all instances join back into shared AgentState via the
    research_results reducer before critic runs.

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
    g.add_node("research_one", research_one_node)
    g.add_node("critic", critic_node)
    g.add_node("synthesizer", synthesizer_node)

    # ── Wire edges ────────────────────────────────────────────────────────────
    g.add_edge(START, "planner")

    # Fan-out: planner emits one Send per sub-question instead of a plain edge.
    g.add_conditional_edges("planner", _fan_out_from_planner, ["research_one"])

    # Join: every research_one branch (however many Sends were fired) routes
    # to critic. LangGraph waits for all parallel branches from the same
    # superstep before running critic.
    g.add_edge("research_one", "critic")

    # Second conditional edge: loop back with a fresh fan-out (gaps) or
    # proceed to synthesizer. _route_after_critic returns either a list of
    # Send objects or the string "synthesizer".
    g.add_conditional_edges(
        "critic",
        _route_after_critic,
        ["research_one", "synthesizer"],
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
