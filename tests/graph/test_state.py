"""
Tests for src/graph/state.py.

The one property genuinely worth protecting here is the operator.add reducer
annotation on research_results — LangGraph reads this via typing metadata to
decide whether a node's returned value replaces or appends to existing state.
If this annotation were accidentally removed, every research_one Send branch
would silently overwrite research_results instead of accumulating it, and
nothing would fail loudly — the graph would just quietly lose research
results from all but the last-completing branch.
"""

import operator
import typing

from src.graph.state import AgentState


def test_research_results_has_operator_add_reducer():
    hints = typing.get_type_hints(AgentState, include_extras=True)
    annotated_type, reducer = typing.get_args(hints["research_results"])

    assert annotated_type == list[dict]
    assert reducer is operator.add


def test_other_fields_are_not_annotated():
    """Only research_results should carry a reducer — every other field is
    replaced outright by whichever node last wrote it, which is the correct
    semantics for single-writer fields like critique, gaps, final_report."""
    hints = typing.get_type_hints(AgentState, include_extras=True)

    for field in ("topic", "sub_questions", "critique", "gaps", "final_report", "iteration"):
        assert typing.get_origin(hints[field]) is not typing.Annotated
