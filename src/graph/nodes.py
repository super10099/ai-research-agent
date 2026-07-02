"""
LangGraph nodes — one async function per agent role.

Each node receives the full AgentState and returns a dict of fields to update.
LangGraph merges the returned dict into the state using per-field reducers
(defined in state.py).  Nodes never mutate state in place.
"""

import json
from typing import TypedDict

from langchain_core.callbacks import adispatch_custom_event

from src.config import settings
from src.graph.state import AgentState
from src.tools.executor import ALL_TOOL_SCHEMAS, execute_tools_parallel
from src.tracing import make_traced_async_client

# Traced async client — emits LangSmith "llm" spans for every messages.create()
# and messages.stream() call.  Interface is identical to AsyncAnthropic.
_client = make_traced_async_client()


# ── System prompts ────────────────────────────────────────────────────────────
# These static strings are passed with cache_control so Anthropic caches the
# KV computation for the prompt prefix.
#
# Cache eligibility: Anthropic requires ≥1024 tokens for claude-sonnet-4-6.
# In development these short prompts won't hit the cache (too few tokens),
# but the pattern is correct — in production you'd pad with detailed
# instructions, few-shot examples, or domain-specific guidelines.
# The cache miss is silent; the flag is still valid to include.

_PLANNER_SYSTEM = """\
You are a research planning assistant. Decompose the given topic into 3-5 focused,
non-overlapping sub-questions that together give comprehensive coverage.

If prior research context is provided, use it to avoid re-investigating already
known sub-topics and to focus on genuinely open questions.

Respond with ONLY valid JSON — no markdown fences, no explanation:
{"sub_questions": ["question 1", "question 2", ...]}\
"""

_RESEARCHER_SYSTEM = """\
You are a rigorous research assistant with access to a vector database of AI papers
and a web search tool. For the given research question:
1. Call retrieve_documents with a focused query.
2. If the results are insufficient, call web_search for supplementary information.
3. Synthesize what you found into a clear, factual answer.
4. Cite sources inline as (source: <url>).

Be thorough but stay on topic. Do not hallucinate facts not present in the sources.\
"""

_CRITIC_SYSTEM = """\
You are a research critic. Given a topic, its sub-questions, and the research
findings so far, evaluate coverage quality.

Respond with ONLY valid JSON — no markdown fences, no explanation:
{
  "critique": "prose evaluation of what was found and what is lacking",
  "gaps": ["specific missing topic 1", "specific missing topic 2"],
  "needs_more_research": true
}

Set needs_more_research to false if coverage is sufficient or gaps are minor.\
"""

_SYNTHESIZER_SYSTEM = """\
You are a research synthesis expert. Write a comprehensive report based on the
provided research findings. Structure:

## Executive Summary
## Key Findings
  (one subsection per major theme — do not use sub-question labels)
## Critical Analysis
## Conclusion

Cite sources inline as (source: <url>). Write in clear, academic prose.
If prior research context is provided, you may reference it but do not repeat it
verbatim — the report should synthesize new findings with prior knowledge.\
"""


# ── Prompt caching helper ─────────────────────────────────────────────────────

def _cached_system(static_text: str, dynamic_suffix: str = "") -> list[dict] | str:
    """
    Build a system parameter suitable for Anthropic's prompt caching API.

    Returns a list of content blocks when there is content to cache, or a
    plain string if there is no dynamic suffix and we're in a context where
    a list isn't needed.

    Layout:
        Block 1 (cached):   static instructions — same on every call
        Block 2 (uncached): dynamic content (prior_context) — varies per topic

    The cache is keyed on (model, block_content, block_position).  Block 1
    will be served from cache on every subsequent call for the same model,
    saving input token processing cost for the static portion.
    """
    blocks: list[dict] = [
        {
            "type": "text",
            "text": static_text,
            # Marks this block as a cache breakpoint.
            # Anthropic will cache everything up to and including this block.
            "cache_control": {"type": "ephemeral"},
        }
    ]
    if dynamic_suffix:
        # Dynamic content goes AFTER the cache breakpoint — it is not cached
        # because it changes per request.
        blocks.append({"type": "text", "text": dynamic_suffix})
    return blocks


# ── Helper: single-question agentic research loop ─────────────────────────────

async def _research_one_question(question: str) -> dict:
    """
    Run a mini agentic tool loop for a single sub-question.

    The model drives which tools to call and when to stop.  We just execute
    whatever tool_use blocks it emits and feed results back until stop_reason
    is "end_turn".

    max_tool_turns is a safety cap — without it, a confused model could
    loop indefinitely calling the same tool.
    """
    messages: list[dict] = [
        {"role": "user", "content": f"Research this question thoroughly:\n\n{question}"}
    ]

    for _ in range(settings.max_tool_turns):
        response = await _client.messages.create(
            model=settings.llm_model,
            max_tokens=4096,
            system=_RESEARCHER_SYSTEM,
            tools=ALL_TOOL_SCHEMAS,
            messages=messages,
        )

        # Append the full assistant turn (may contain text + tool_use blocks).
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            # Extract the final prose answer from the assistant's last message.
            answer = next(
                (block.text for block in response.content if hasattr(block, "text")),
                "",
            )
            return {"question": question, "answer": answer}

        if response.stop_reason == "tool_use":
            tool_uses = [b for b in response.content if b.type == "tool_use"]
            # All tool calls from this turn run concurrently (scatter-gather).
            tool_results = await execute_tools_parallel(tool_uses)
            # Tool results go back as a user message — Anthropic API requirement.
            messages.append({"role": "user", "content": tool_results})

    # Fell out of the loop — return whatever the last answer text was.
    last_answer = next(
        (
            block.text
            for block in reversed(messages)
            if isinstance(block, dict) and block.get("role") == "assistant"
            for block in (block.get("content") or [])
            if hasattr(block, "text")
        ),
        f"Research incomplete after {settings.max_tool_turns} tool turns.",
    )
    return {"question": question, "answer": last_answer}


# ── Node: Planner ─────────────────────────────────────────────────────────────

async def planner_node(state: AgentState) -> dict:
    """
    Decompose the topic into focused sub-questions.

    Prior context (summaries of related past sessions) is injected as a
    dynamic suffix to the cached system prompt so the planner can skip
    already-answered questions and focus on genuine unknowns.
    """
    response = await _client.messages.create(
        model=settings.llm_model,
        max_tokens=512,
        system=_cached_system(_PLANNER_SYSTEM, state.get("prior_context", "")),
        messages=[{"role": "user", "content": f"Research topic: {state['topic']}"}],
    )

    try:
        parsed = json.loads(response.content[0].text)
        sub_questions: list[str] = parsed["sub_questions"]
    except (json.JSONDecodeError, KeyError) as exc:
        raise ValueError(f"Planner returned malformed JSON: {exc}\n{response.content[0].text}")

    print(f"[planner] Decomposed into {len(sub_questions)} sub-questions.")
    return {"sub_questions": sub_questions}


# ── Node: research_one (parallel LangGraph branch, one per question) ──────────

class ResearchTask(TypedDict):
    """Input schema for a single `Send("research_one", ...)` branch invocation."""
    question: str


async def research_one_node(task: ResearchTask) -> dict:
    """
    Research exactly one question. Each sub-question becomes its own LangGraph
    node invocation via the `Send` API (see builder.py fan-out functions), so
    this receives an isolated {"question": ...} input rather than the full
    AgentState — LangGraph runs one instance of this node per Send, each with
    its own checkpoint, and joins the results back into AgentState via the
    research_results reducer (operator.add).
    """
    result = await _research_one_question(task["question"])
    print(f"[research_one] Done: {task['question'][:60]}")
    return {"research_results": [result]}


# ── Node: Critic ──────────────────────────────────────────────────────────────

async def critic_node(state: AgentState) -> dict:
    """
    Evaluate research coverage and decide whether another research loop is needed.

    The critic sees all accumulated research_results, not just the latest batch,
    so it can reason about overall coverage rather than per-iteration quality.
    """
    research_summary = "\n\n".join(
        f"Q: {r['question']}\nA: {r['answer']}"
        for r in state["research_results"]
    )

    response = await _client.messages.create(
        model=settings.llm_model,
        max_tokens=1024,
        system=_CRITIC_SYSTEM,
        messages=[{
            "role": "user",
            "content": (
                f"Topic: {state['topic']}\n\n"
                f"Sub-questions:\n{json.dumps(state['sub_questions'], indent=2)}\n\n"
                f"Research findings:\n{research_summary}"
            ),
        }],
    )

    try:
        parsed = json.loads(response.content[0].text)
    except (json.JSONDecodeError, KeyError) as exc:
        raise ValueError(f"Critic returned malformed JSON: {exc}\n{response.content[0].text}")

    # `critic` is the join point after every research wave (all parallel
    # `research_one` Send branches converge here), so it owns the iteration
    # counter — no single research node runs "once per wave" anymore.
    new_iteration = state.get("iteration", 0) + 1

    # Respect the max_iterations ceiling regardless of what the critic says.
    at_limit = new_iteration >= state.get("max_iterations", settings.max_research_iterations)
    needs_more = parsed.get("needs_more_research", False) and not at_limit

    print(f"[critic] needs_more_research={needs_more}, gaps={parsed.get('gaps', [])}")
    return {
        "critique": parsed.get("critique", ""),
        "gaps": parsed.get("gaps", []),
        "needs_more_research": needs_more,
        "iteration": new_iteration,
    }


# ── Node: Synthesizer ─────────────────────────────────────────────────────────

async def synthesizer_node(state: AgentState) -> dict:
    """
    Write the final structured report, streaming tokens via LangGraph custom events.

    Uses _client.messages.stream() (the streaming context manager) instead of
    messages.create().  For each text delta, we dispatch a "synthesis_token"
    custom event.

    adispatch_custom_event only requires an active parent run (which LangGraph
    always sets up for a node, whether invoked via ainvoke or astream_events);
    it has no effect when there's no handler listening for custom events
    (e.g., running via graph.ainvoke in run.py).  So this node works correctly
    in both streaming and non-streaming execution paths without any
    conditional branching. Imported from langchain_core.callbacks directly —
    LangGraph re-exported this from langgraph.config prior to v1.0, but that
    re-export was removed; the underlying function is unchanged.
    """
    research_block = "\n\n".join(
        f"**Question:** {r['question']}\n**Findings:** {r['answer']}"
        for r in state["research_results"]
    )

    full_text = ""

    async with _client.messages.stream(
        model=settings.llm_model,
        max_tokens=8192,
        system=_cached_system(_SYNTHESIZER_SYSTEM, state.get("prior_context", "")),
        messages=[{
            "role": "user",
            "content": (
                f"Topic: {state['topic']}\n\n"
                f"Research findings:\n{research_block}\n\n"
                f"Critic's notes:\n{state.get('critique', 'No critique.')}"
            ),
        }],
    ) as stream:
        async for text in stream.text_stream:
            full_text += text
            # Surface each token to astream_events consumers (the SSE endpoint).
            # No-op when there is no active astream_events call.
            await adispatch_custom_event("synthesis_token", {"token": text})

    print(f"[synthesizer] Report written ({len(full_text)} chars).")
    return {"final_report": full_text}
