"""
Tests for src/api/routes.py.

A minimal FastAPI app mounts just the router with a fake app.state.graph and
app.state.sessions — the real lifespan (build_graph + SqliteSaver) is
intentionally bypassed so these are true unit tests of the route handlers,
not integration tests of the whole app.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.api.routes import router


def make_app(graph, sessions=None):
    app = FastAPI()
    app.include_router(router)
    app.state.graph = graph
    app.state.sessions = sessions if sessions is not None else {}
    return app


def make_client(app):
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


class TestStartResearch:
    async def test_happy_path(self, monkeypatch):
        monkeypatch.setattr("src.api.routes.retrieve_relevant_sessions", lambda topic: [])
        monkeypatch.setattr("src.api.routes.format_prior_context", lambda sessions: "")

        fake_state = {
            "sub_questions": ["Q1", "Q2"],
            "research_results": [{"question": "Q1", "answer": "A1"}],
            "critique": "Looks reasonable.",
            "gaps": [],
            "iteration": 1,
        }
        fake_graph = SimpleNamespace(ainvoke=AsyncMock(return_value=fake_state))
        sessions = {}
        app = make_app(fake_graph, sessions)

        async with make_client(app) as client:
            resp = await client.post(
                "/api/research", json={"topic": "A perfectly valid research topic"},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["topic"] == "A perfectly valid research topic"
        assert body["sub_questions"] == ["Q1", "Q2"]
        assert body["research_results"] == [{"question": "Q1", "answer": "A1"}]
        assert body["critique"] == "Looks reasonable."
        assert body["iteration"] == 1
        session_id = body["session_id"]
        assert sessions[session_id]["status"] == "awaiting_synthesis"

    async def test_rejects_too_short_topic(self, monkeypatch):
        monkeypatch.setattr("src.api.routes.retrieve_relevant_sessions", lambda topic: [])
        monkeypatch.setattr("src.api.routes.format_prior_context", lambda sessions: "")
        fake_graph = SimpleNamespace(ainvoke=AsyncMock())
        app = make_app(fake_graph)

        async with make_client(app) as client:
            resp = await client.post("/api/research", json={"topic": "hi"})

        assert resp.status_code == 422

    async def test_passes_max_iterations_through(self, monkeypatch):
        monkeypatch.setattr("src.api.routes.retrieve_relevant_sessions", lambda topic: [])
        monkeypatch.setattr("src.api.routes.format_prior_context", lambda sessions: "")
        fake_graph = SimpleNamespace(ainvoke=AsyncMock(return_value={
            "sub_questions": [], "research_results": [], "critique": "", "gaps": [], "iteration": 0,
        }))
        app = make_app(fake_graph)

        async with make_client(app) as client:
            await client.post(
                "/api/research",
                json={"topic": "A perfectly valid research topic", "max_iterations": 3},
            )

        initial_state_arg = fake_graph.ainvoke.call_args[0][0]
        assert initial_state_arg["max_iterations"] == 3


class TestSessionState:
    async def test_unknown_session_returns_404(self):
        app = make_app(SimpleNamespace(), sessions={})

        async with make_client(app) as client:
            resp = await client.get("/api/research/unknown-id/state")

        assert resp.status_code == 404

    async def test_known_session_returns_status(self):
        sessions = {"sid-1": {"topic": "t", "status": "awaiting_synthesis"}}
        app = make_app(SimpleNamespace(), sessions=sessions)

        async with make_client(app) as client:
            resp = await client.get("/api/research/sid-1/state")

        assert resp.status_code == 200
        assert resp.json() == {"session_id": "sid-1", "status": "awaiting_synthesis"}


class TestStreamSynthesis:
    async def test_unknown_session_returns_404(self):
        app = make_app(SimpleNamespace(), sessions={})

        async with make_client(app) as client:
            resp = await client.get("/api/research/unknown-id/stream")

        assert resp.status_code == 404

    async def test_session_not_ready_returns_409(self):
        sessions = {"sid-1": {"topic": "t", "status": "researching"}}
        app = make_app(SimpleNamespace(), sessions=sessions)

        async with make_client(app) as client:
            resp = await client.get("/api/research/sid-1/stream")

        assert resp.status_code == 409

    async def test_streams_tokens_and_completes(self, monkeypatch):
        async def fake_astream_events(*args, **kwargs):
            events = [
                {"event": "on_chain_start", "name": "critic", "data": {}},
                {"event": "on_custom_event", "name": "synthesis_token", "data": {"token": "Hel"}},
                {"event": "on_custom_event", "name": "synthesis_token", "data": {"token": "lo"}},
                {"event": "on_chain_end", "name": "synthesizer", "data": {}},
            ]
            for ev in events:
                yield ev

        fake_graph = SimpleNamespace(astream_events=fake_astream_events)
        sessions = {"sid-1": {"topic": "RAG systems", "status": "awaiting_synthesis"}}
        app = make_app(fake_graph, sessions)

        monkeypatch.setattr(
            "src.api.routes.summarize_session", AsyncMock(return_value="A summary.")
        )
        store_session_mock = MagicMock()  # store_session is sync in production
        monkeypatch.setattr("src.api.routes.store_session", store_session_mock)

        async with make_client(app) as client:
            resp = await client.get("/api/research/sid-1/stream")

        assert resp.status_code == 200
        lines = [
            json.loads(line[len("data: "):])
            for line in resp.text.splitlines()
            if line.startswith("data: ")
        ]
        token_events = [e for e in lines if e["type"] == "token"]
        done_events = [e for e in lines if e["type"] == "done"]

        assert [e["content"] for e in token_events] == ["Hel", "lo"]
        assert len(done_events) == 1
        assert done_events[0]["session_id"] == "sid-1"
        assert sessions["sid-1"]["status"] == "complete"
        store_session_mock.assert_called_once()

    async def test_stream_error_sets_error_status(self, monkeypatch):
        async def failing_astream_events(*args, **kwargs):
            raise RuntimeError("graph blew up")
            yield  # pragma: no cover — makes this an async generator function

        fake_graph = SimpleNamespace(astream_events=failing_astream_events)
        sessions = {"sid-1": {"topic": "t", "status": "awaiting_synthesis"}}
        app = make_app(fake_graph, sessions)

        async with make_client(app) as client:
            resp = await client.get("/api/research/sid-1/stream")

        assert sessions["sid-1"]["status"] == "error"
        lines = [
            json.loads(line[len("data: "):])
            for line in resp.text.splitlines()
            if line.startswith("data: ")
        ]
        assert any(e["type"] == "error" for e in lines)
