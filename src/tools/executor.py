"""
Tool executor: routes Claude's tool_use blocks to Python functions
and supports parallel execution of multiple tool calls in one turn.

Why is this a separate module?
Because the dispatch logic (name → function) and the parallelism logic
(run N tools concurrently) are both needed in multiple graph nodes.
Centralizing them here means the researcher and critic nodes don't each
re-implement the fan-out pattern.
"""

import asyncio
from typing import Any

from anthropic.types import ToolUseBlock

from src.tools.retrieval import RETRIEVAL_TOOL_SCHEMA, retrieve_documents
from src.tools.web_search import WEB_SEARCH_TOOL_SCHEMA, web_search

# ── Tool registry ─────────────────────────────────────────────────────────────
# Single source of truth for which tools exist and what their schemas are.
# Passed directly to the Anthropic API as the `tools` parameter.
ALL_TOOL_SCHEMAS: list[dict] = [
    RETRIEVAL_TOOL_SCHEMA,
    WEB_SEARCH_TOOL_SCHEMA,
]

# Maps tool name → sync callable.  All tool functions are synchronous because
# ChromaDB and Cohere's Python clients are sync.  We lift them into async
# via asyncio.to_thread so the event loop stays unblocked during I/O.
_TOOL_REGISTRY: dict[str, Any] = {
    "retrieve_documents": retrieve_documents,
    "web_search": web_search,
}


async def _run_one(tool_use: ToolUseBlock) -> dict:
    """
    Execute a single tool call in a thread pool and return a tool_result dict.

    asyncio.to_thread() runs the sync function in the default ThreadPoolExecutor,
    yielding control back to the event loop while the HTTP call or disk read
    blocks.  This is the Python equivalent of offloading a blocking syscall
    to a worker thread in a reactor-pattern server.
    """
    fn = _TOOL_REGISTRY.get(tool_use.name)
    if fn is None:
        result_text = f"Error: unknown tool '{tool_use.name}'"
    else:
        try:
            # All current tools take their kwargs directly from input dict.
            result_text = await asyncio.to_thread(fn, **tool_use.input)
        except Exception as exc:
            result_text = f"Tool execution error: {exc}"

    # Anthropic API format for tool results in a subsequent user message.
    return {
        "type": "tool_result",
        "tool_use_id": tool_use.id,
        "content": result_text,
    }


async def execute_tools_parallel(tool_uses: list[ToolUseBlock]) -> list[dict]:
    """
    Execute all tool calls from a single model turn concurrently.

    When Claude emits multiple tool_use blocks in one response, the Anthropic
    API expects ALL their results to be returned together in the next user
    message before the model continues.  Running them in parallel (via
    asyncio.gather) cuts wall-clock latency from sum(tool_times) to
    max(tool_times) — the same win you get from parallel disk reads vs
    sequential reads on an HPC storage system.

    Returns a list of tool_result dicts ready to embed in the messages list.
    """
    return list(
        await asyncio.gather(*[_run_one(tu) for tu in tool_uses])
    )
