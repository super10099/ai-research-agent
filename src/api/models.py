from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    topic: str = Field(..., min_length=5, max_length=500)
    max_iterations: int = Field(default=2, ge=1, le=4)


class ResearchResult(BaseModel):
    question: str
    answer: str


class ResearchSessionResponse(BaseModel):
    """Returned by POST /api/research after the research phase completes."""
    session_id: str
    topic: str
    sub_questions: list[str]
    research_results: list[ResearchResult]
    critique: str
    gaps: list[str]
    iteration: int
    # Tells the client whether they can open the SSE stream.
    ready_for_synthesis: bool = True


class SessionStateResponse(BaseModel):
    """Returned by GET /api/research/{session_id}/state."""
    session_id: str
    status: str   # "researching" | "awaiting_synthesis" | "complete" | "error"
    final_report: str = ""
