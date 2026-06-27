"""
FastAPI application entry point.

Run with:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.graph.builder import build_graph, make_checkpointer
from src.tracing import configure_langsmith


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager — replaces the deprecated @app.on_event.

    We create exactly one checkpointer and one compiled graph for the entire
    process lifetime.  Both are stored on app.state so route handlers can
    access them via request.app.state.

    Why not create these per-request?
    - make_checkpointer() opens a SQLite connection.  Opening/closing a DB
      connection per request wastes ~5ms and thrashes the file descriptor table.
    - build_graph() compiles the StateGraph into an executable object.
      Compilation is ~50ms and the result is stateless (thread-safe) — there
      is no reason to redo it per request.
    - The sessions dict is an in-memory registry mapping session_id to status.
      It is intentionally not persisted — session status can be reconstructed
      from the checkpointer if needed.  For a multi-process deployment, move
      this to Redis.
    """
    configure_langsmith()
    checkpointer = make_checkpointer()
    app.state.graph = build_graph(checkpointer=checkpointer)
    app.state.sessions: dict = {}  # session_id → {"topic": str, "status": str}

    yield  # application runs here

    # SqliteSaver holds a connection; closing it explicitly is good practice
    # even though CPython's GC would collect it eventually.
    if hasattr(checkpointer, "conn"):
        checkpointer.conn.close()


app = FastAPI(
    title="AI Research Agent",
    version="0.1.0",
    description="Multi-agent research assistant with RAG, reranking, and streaming synthesis.",
    lifespan=lifespan,
)

# CORS — allow the React dev server (port 5173 for Vite, 3000 for CRA) and
# any localhost origin.  In production, replace ["*"] with the actual domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health")
async def health():
    return {"status": "ok"}
