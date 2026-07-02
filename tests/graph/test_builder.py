"""
Tests for src/graph/builder.py.

Covers the routing functions in isolation (_fan_out_from_planner,
_route_after_critic), the graph's node wiring, make_initial_state defaults,
and a fully stubbed end-to-end run that exercises the real Send fan-out/join
mechanics — two research waves, iteration counting owned by critic, and the
interrupt_before=["synthesizer"] pause — without any real Anthropic API calls.
"""

from langgraph.types import Send

from src.graph.builder import (
    _fan_out_from_planner,
    _route_after_critic,
    build_graph,
    make_initial_state,
)


class TestRoutingFunctions:
    def test_fan_out_from_planner_emits_one_send_per_sub_question(self):
        state = {"sub_questions": ["Q1", "Q2", "Q3"]}

        result = _fan_out_from_planner(state)

        assert result == [
            Send("research_one", {"question": "Q1"}),
            Send("research_one", {"question": "Q2"}),
            Send("research_one", {"question": "Q3"}),
        ]

    def test_route_after_critic_fans_out_over_gaps_when_needs_more(self):
        state = {"needs_more_research": True, "gaps": ["Gap 1", "Gap 2"]}

        result = _route_after_critic(state)

        assert result == [
            Send("research_one", {"question": "Gap 1"}),
            Send("research_one", {"question": "Gap 2"}),
        ]

    def test_route_after_critic_proceeds_to_synthesizer_when_satisfied(self):
        state = {"needs_more_research": False, "gaps": []}

        result = _route_after_critic(state)

        assert result == "synthesizer"

    def test_route_after_critic_with_no_gaps_key_defaults_empty(self):
        state = {"needs_more_research": True}

        result = _route_after_critic(state)

        assert result == []


class TestMakeInitialState:
    def test_defaults(self):
        state = make_initial_state("A research topic")

        assert state["topic"] == "A research topic"
        assert state["sub_questions"] == []
        assert state["research_results"] == []
        assert state["critique"] == ""
        assert state["gaps"] == []
        assert state["needs_more_research"] is False
        assert state["final_report"] == ""
        assert state["prior_context"] == ""
        assert state["iteration"] == 0
        assert state["max_iterations"] == 2  # settings.max_research_iterations default

    def test_overrides(self):
        state = make_initial_state("Topic", prior_context="prior stuff", max_iterations=4)

        assert state["prior_context"] == "prior stuff"
        assert state["max_iterations"] == 4


class TestBuildGraph:
    def test_node_set(self):
        graph = build_graph()

        nodes = set(graph.get_graph().nodes.keys())

        assert nodes == {"__start__", "planner", "research_one", "critic", "synthesizer", "__end__"}


class TestEndToEndStubbedRun:
    """Runs the compiled graph with every LLM call stubbed out, verifying the
    Send-based fan-out/join actually behaves like real parallel graph
    branches: two research waves accumulate into a shared research_results
    list, iteration increments once per wave (owned by critic, the join
    point), and the graph halts exactly at the synthesizer interrupt."""

    async def test_two_wave_fan_out_and_join(self, monkeypatch):
        async def fake_planner(state):
            return {"sub_questions": ["Q1", "Q2", "Q3"]}

        call_count = {"n": 0}

        async def fake_critic(state):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return {
                    "critique": "Needs more on evaluation.",
                    "gaps": ["Q4"],
                    "needs_more_research": True,
                    "iteration": state.get("iteration", 0) + 1,
                }
            return {
                "critique": "Coverage sufficient.",
                "gaps": [],
                "needs_more_research": False,
                "iteration": state.get("iteration", 0) + 1,
            }

        monkeypatch.setattr("src.graph.builder.planner_node", fake_planner)
        monkeypatch.setattr("src.graph.builder.critic_node", fake_critic)
        monkeypatch.setattr(
            "src.graph.nodes._research_one_question",
            lambda q: _fake_research(q),
        )

        graph = build_graph()
        state = make_initial_state("Test topic")
        config = {"configurable": {"thread_id": "test-thread-builder"}}

        result = await graph.ainvoke(state, config=config)

        assert result["sub_questions"] == ["Q1", "Q2", "Q3"]
        assert {r["question"] for r in result["research_results"]} == {"Q1", "Q2", "Q3", "Q4"}
        assert result["iteration"] == 2
        assert result["needs_more_research"] is False
        assert result["gaps"] == []
        # Graph must have paused at the synthesizer interrupt, not run it.
        assert result["final_report"] == ""


async def _fake_research(question: str) -> dict:
    return {"question": question, "answer": f"Answer for: {question}"}
