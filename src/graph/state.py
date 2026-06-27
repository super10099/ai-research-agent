import operator
from typing import Annotated, TypedDict


class AgentState(TypedDict):
    # ── Input ─────────────────────────────────────────────────────────────────
    topic: str               # the research question supplied by the user

    # ── Planner output ────────────────────────────────────────────────────────
    sub_questions: list[str]  # 3-5 focused sub-questions decomposed from the topic

    # ── Researcher output ─────────────────────────────────────────────────────
    # Annotated[list, operator.add] is a LangGraph reducer annotation.
    # When a node returns {"research_results": [new_item]}, LangGraph calls
    # operator.add(existing_list, [new_item]) — i.e., it appends rather than
    # replaces.  This lets results accumulate across multiple research iterations
    # without each node needing to re-read and re-emit the full list.
    research_results: Annotated[list[dict], operator.add]

    # ── Critic output ─────────────────────────────────────────────────────────
    critique: str             # prose evaluation of research coverage
    gaps: list[str]           # specific topics that still need investigation
    needs_more_research: bool  # whether the graph should loop back to researcher

    # ── Synthesizer output ────────────────────────────────────────────────────
    final_report: str         # the completed, structured research report

    # ── Memory ───────────────────────────────────────────────────────────────
    # Summaries of past sessions on related topics, injected at graph start.
    # Read by planner and synthesizer; never written to by any node.
    prior_context: str

    # ── Control flow ──────────────────────────────────────────────────────────
    iteration: int            # how many researcher → critic loops have completed
    max_iterations: int       # ceiling set at graph invocation time
