"""
Tests for src/graph/nodes.py.

The module-level `_client` singleton is monkeypatched per-test with a fake
Anthropic client exposing just the two methods actually used
(messages.create, messages.stream), so no real network calls happen and no
real API key is needed.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.graph import nodes


def _text_response(text: str):
    """Mimics an Anthropic Message response whose first content block is text."""
    return SimpleNamespace(content=[SimpleNamespace(text=text)])


class FakeStream:
    """Mimics the async context manager returned by client.messages.stream(...)."""

    def __init__(self, tokens: list[str]):
        self._tokens = tokens

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    @property
    def text_stream(self):
        return self._agen()

    async def _agen(self):
        for t in self._tokens:
            yield t


@pytest.fixture
def fake_client(monkeypatch):
    client = SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(), stream=MagicMock()))
    monkeypatch.setattr(nodes, "_client", client)
    return client


class TestPlannerNode:
    async def test_parses_sub_questions_from_json(self, fake_client):
        fake_client.messages.create.return_value = _text_response(
            '{"sub_questions": ["Q1", "Q2", "Q3"]}'
        )

        result = await nodes.planner_node({"topic": "RAG systems", "prior_context": ""})

        assert result == {"sub_questions": ["Q1", "Q2", "Q3"]}

    async def test_raises_on_malformed_json(self, fake_client):
        fake_client.messages.create.return_value = _text_response("not valid json")

        with pytest.raises(ValueError, match="malformed JSON"):
            await nodes.planner_node({"topic": "RAG systems", "prior_context": ""})

    async def test_injects_prior_context_as_dynamic_suffix(self, fake_client):
        fake_client.messages.create.return_value = _text_response('{"sub_questions": ["Q1"]}')

        await nodes.planner_node({"topic": "RAG systems", "prior_context": "past session notes"})

        _, kwargs = fake_client.messages.create.call_args
        system_blocks = kwargs["system"]
        assert system_blocks[0]["cache_control"] == {"type": "ephemeral"}
        assert system_blocks[1]["text"] == "past session notes"
        assert "cache_control" not in system_blocks[1]


class TestResearchOneNode:
    async def test_wraps_research_one_question(self, monkeypatch):
        async def fake_research(question):
            return {"question": question, "answer": f"Answer for {question}"}
        monkeypatch.setattr(nodes, "_research_one_question", fake_research)

        result = await nodes.research_one_node({"question": "What is RAG?"})

        assert result == {
            "research_results": [{"question": "What is RAG?", "answer": "Answer for What is RAG?"}]
        }


class TestCriticNode:
    def _base_state(self, **overrides):
        state = {
            "topic": "RAG systems",
            "sub_questions": ["Q1"],
            "research_results": [{"question": "Q1", "answer": "A1"}],
            "iteration": 0,
            "max_iterations": 2,
        }
        state.update(overrides)
        return state

    async def test_increments_iteration(self, fake_client):
        fake_client.messages.create.return_value = _text_response(
            '{"critique": "ok", "gaps": [], "needs_more_research": false}'
        )

        result = await nodes.critic_node(self._base_state(iteration=0))

        assert result["iteration"] == 1

    async def test_needs_more_research_when_below_ceiling(self, fake_client):
        fake_client.messages.create.return_value = _text_response(
            '{"critique": "gaps found", "gaps": ["G1"], "needs_more_research": true}'
        )

        result = await nodes.critic_node(self._base_state(iteration=0, max_iterations=2))

        assert result["needs_more_research"] is True
        assert result["gaps"] == ["G1"]
        assert result["iteration"] == 1

    async def test_forces_stop_at_max_iterations_ceiling(self, fake_client):
        # Critic says "needs more" but we're already at the iteration ceiling —
        # the ceiling must win regardless of what the critic's JSON says.
        fake_client.messages.create.return_value = _text_response(
            '{"critique": "still gaps", "gaps": ["G1"], "needs_more_research": true}'
        )

        result = await nodes.critic_node(self._base_state(iteration=1, max_iterations=2))

        assert result["iteration"] == 2
        assert result["needs_more_research"] is False

    async def test_raises_on_malformed_json(self, fake_client):
        fake_client.messages.create.return_value = _text_response("not json")

        with pytest.raises(ValueError, match="malformed JSON"):
            await nodes.critic_node(self._base_state())


class TestSynthesizerNode:
    async def test_assembles_full_report_from_stream(self, fake_client, monkeypatch):
        fake_client.messages.stream.return_value = FakeStream(["Hel", "lo ", "world"])
        dispatched = []

        async def fake_dispatch(name, data):
            dispatched.append((name, data))
        monkeypatch.setattr(nodes, "adispatch_custom_event", fake_dispatch)

        result = await nodes.synthesizer_node({
            "topic": "RAG systems",
            "research_results": [{"question": "Q1", "answer": "A1"}],
            "critique": "Looks good.",
            "prior_context": "",
        })

        assert result == {"final_report": "Hello world"}
        assert dispatched == [
            ("synthesis_token", {"token": "Hel"}),
            ("synthesis_token", {"token": "lo "}),
            ("synthesis_token", {"token": "world"}),
        ]

    async def test_works_with_no_prior_context(self, fake_client, monkeypatch):
        fake_client.messages.stream.return_value = FakeStream(["Report."])
        monkeypatch.setattr(nodes, "adispatch_custom_event", AsyncMock())

        result = await nodes.synthesizer_node({
            "topic": "RAG systems",
            "research_results": [],
            "critique": "",
            "prior_context": "",
        })

        assert result == {"final_report": "Report."}
