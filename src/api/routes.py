"""
API routes for the research agent.

Two-phase design matching the LangGraph interrupt:
  Phase 1 — POST /api/research
    Runs planner → researcher → critic, pauses at the synthesizer interrupt.
    Returns the research state so the frontend can display it for human review.

  Phase 2 — GET /api/research/{session_id}/stream
    Client opens an EventSource to this URL.
    Server resumes the graph (runs synthesizer) and streams tokens via SSE.
    When synthesis finishes, sends a terminal "done" event and stores memory.
"""

import json
import uuid

from fastapi import APIRouter, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from src.api.models import ResearchRequest, ResearchResult, ResearchSessionResponse
from src.graph.builder import make_initial_state
from src.memory.session_memory import (
    format_prior_context,
    retrieve_relevant_sessions,
    store_session,
    summarize_session,
)

router = APIRouter(prefix="/api")


# ── Phase 1: research ─────────────────────────────────────────────────────────

@router.post("/research", response_model=ResearchSessionResponse)
async def start_research(body: ResearchRequest, request: Request):
    """
    Run the research phase (planner → researcher → critic) synchronously.

    The graph pauses at the synthesizer interrupt and this endpoint returns,
    giving the frontend the research summary for human review.
    The session is keyed by session_id in the checkpointer and the app's
    in-memory session registry.
    """
    graph = request.app.state.graph
    sessions = request.app.state.sessions  # {session_id: {"topic": ..., "status": ...}}

    session_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}

    # Retrieve episodic memories for this topic before invoking the graph.
    prior_sessions = retrieve_relevant_sessions(body.topic)
    prior_context = format_prior_context(prior_sessions)

    initial_state = make_initial_state(
        topic=body.topic,
        prior_context=prior_context,
        max_iterations=body.max_iterations,
    )

    # Register the session before invoking in case the client polls state quickly.
    sessions[session_id] = {"topic": body.topic, "status": "researching"}

    # ainvoke runs the graph until the synthesizer interrupt, then returns state.
    state = await graph.ainvoke(initial_state, config=config)

    sessions[session_id]["status"] = "awaiting_synthesis"

    return ResearchSessionResponse(
        session_id=session_id,
        topic=body.topic,
        sub_questions=state.get("sub_questions", []),
        research_results=[
            ResearchResult(question=r["question"], answer=r["answer"])
            for r in state.get("research_results", [])
        ],
        critique=state.get("critique", ""),
        gaps=state.get("gaps", []),
        iteration=state.get("iteration", 0),
    )


# ── Phase 2: streaming synthesis ──────────────────────────────────────────────

@router.get("/research/{session_id}/stream")
async def stream_synthesis(session_id: str, request: Request):
    """
    SSE endpoint: resume the graph from the synthesizer interrupt and stream tokens.

    Why SSE and not WebSockets?
    - SSE is unidirectional (server → client) — exactly what we need here.
      The client does not need to send messages during synthesis.
    - SSE uses plain HTTP/1.1 GET; no upgrade handshake, works through
      standard proxies and CDNs without configuration.
    - SSE has built-in reconnect (the browser re-sends the Last-Event-ID header
      on disconnect so you can resume from the last token), whereas WebSockets
      require application-level reconnect logic.
    - WebSockets are worth the complexity when you need bidirectional real-time
      communication (chat, collaborative editing).  For token streaming, SSE
      is the simpler correct choice.

    Event types emitted:
      {"type": "token",  "content": "..."}   — one per synthesizer token
      {"type": "done",   "session_id": "..."}  — terminal event
      {"type": "error",  "message": "..."}    — on failure
    """
    sessions = request.app.state.sessions
    graph = request.app.state.graph

    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session["status"] not in ("awaiting_synthesis", "synthesizing"):
        raise HTTPException(
            status_code=409,
            detail=f"Session is not ready for synthesis (status={session['status']})",
        )

    # Mark as synthesizing to prevent duplicate stream connections.
    session["status"] = "synthesizing"
    topic = session["topic"]
    config = {"configurable": {"thread_id": session_id}}

    async def event_generator():
        """
        Async generator consumed by EventSourceResponse.

        Yields dicts that sse_starlette formats as:
            data: <json_string>\n\n

        How tool_use and streaming interact:
        Anthropic's streaming API emits deltas only during text generation.
        When stop_reason is "tool_use", the model has switched to emitting a
        structured tool_use block — these arrive as input_json_delta events,
        not text_delta events.  Our synthesizer never calls tools (it only
        writes prose), so every delta it produces is a text token.
        The researcher node uses tools but runs non-streaming (messages.create),
        so there are no tool_use streaming events in the SSE flow at all.
        """
        final_report = ""

        try:
            # astream_events(None, ...) resumes from the checkpointed interrupt.
            # version="v2" is required for on_custom_event to be surfaced.
            async for event in graph.astream_events(None, config=config, version="v2"):
                # Disconnect gracefully if the client closes the connection.
                if await request.is_disconnected():
                    break

                if (
                    event["event"] == "on_custom_event"
                    and event["name"] == "synthesis_token"
                ):
                    token: str = event["data"]["token"]
                    final_report += token
                    yield {"data": json.dumps({"type": "token", "content": token})}

        except Exception as exc:
            session["status"] = "error"
            yield {"data": json.dumps({"type": "error", "message": str(exc)})}
            return

        # ── Post-synthesis: store episodic memory ────────────────────────────
        if final_report:
            try:
                summary = await summarize_session(topic, final_report)
                store_session(topic=topic, summary=summary, session_id=session_id)
            except Exception as exc:
                # Memory storage failure is non-fatal — log and continue.
                print(f"[routes] WARNING: failed to store session memory: {exc}")

        session["status"] = "complete"
        yield {"data": json.dumps({"type": "done", "session_id": session_id})}

    # ping keeps the HTTP connection alive through proxies that close idle
    # connections (common default is 60s timeout on many load balancers).
    # Named ping_interval in older sse-starlette releases; renamed to ping.
    return EventSourceResponse(event_generator(), ping=15)


# ── Session state ─────────────────────────────────────────────────────────────

@router.get("/research/{session_id}/state")
async def get_session_state(session_id: str, request: Request):
    """
    Lightweight status check.  The frontend polls this while the stream is active
    to detect completion without relying solely on the SSE "done" event
    (useful when the user refreshes mid-synthesis and reconnects).
    """
    session = request.app.state.sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session_id, "status": session["status"]}
