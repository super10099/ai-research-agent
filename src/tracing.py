"""
LangSmith tracing configuration.

Three layers of instrumentation work together:

  1. LangGraph auto-tracing
     LangGraph uses langchain-core's callback system internally.  When
     LANGCHAIN_TRACING_V2=true and LANGSMITH_API_KEY are set, every node
     execution is automatically recorded as a "chain" span in LangSmith,
     including inputs (state dict) and outputs (returned dict).
     No code changes required for this layer.

  2. Anthropic SDK tracing (wrap_anthropic)
     The direct Anthropic SDK is not auto-instrumented.  wrap_anthropic()
     patches the client's messages.create() and messages.stream() methods
     to emit an "llm" span for every call, recording the model, messages,
     token counts, and latency.

  3. Tool tracing (@traceable)
     Each tool function is decorated with @traceable(run_type="tool") so
     its inputs and outputs are recorded as child spans under the researcher
     node span that called it.

Together these three layers give you a complete causal trace tree:
  session → planner (chain)
               └─ claude-sonnet-4-6 call (llm)
          → researcher (chain)
               ├─ claude-sonnet-4-6 call (llm)
               │    ├─ retrieve_documents (tool)   ← stage 1: ANN search
               │    │    └─ cohere rerank (inside tool)
               │    └─ web_search (tool)
               └─ claude-sonnet-4-6 call (llm)    ← synthesis after tools
          → critic (chain)
               └─ claude-sonnet-4-6 call (llm)
          → synthesizer (chain)
               └─ claude-sonnet-4-6 call (llm, streaming)
"""

import os

import anthropic
from langsmith import wrappers

from src.config import settings


def configure_langsmith() -> None:
    """
    Ensure the LangSmith environment is properly configured.

    LangGraph reads LANGCHAIN_TRACING_V2 and LANGSMITH_API_KEY directly
    from the environment via langchain-core.  We set them here (from our
    typed Settings object) as a single authoritative configuration point
    so callers don't need to manually manage env vars.

    Idempotent — safe to call multiple times.
    """
    if settings.langsmith_api_key:
        os.environ.setdefault("LANGSMITH_API_KEY", settings.langsmith_api_key)
    os.environ.setdefault("LANGSMITH_PROJECT", settings.langsmith_project)
    # This is the master switch for LangGraph and LangChain auto-instrumentation.
    os.environ.setdefault(
        "LANGCHAIN_TRACING_V2",
        "true" if settings.langchain_tracing_v2 else "false",
    )


def make_traced_async_client() -> anthropic.AsyncAnthropic:
    """
    Return an AsyncAnthropic client that emits LangSmith spans on every call.

    wrap_anthropic() intercepts messages.create() and messages.stream() to:
    - Record the model name, system prompt, and messages as span inputs
    - Record the response content and usage (input/output tokens) as outputs
    - Measure wall-clock latency of the HTTP call
    - Link the span to the parent LangGraph node span via the active run context

    The returned object is interface-identical to AsyncAnthropic — all
    existing code works without modification.

    Falls back to a plain client if tracing is disabled (e.g., in unit tests
    that set LANGCHAIN_TRACING_V2=false to avoid network calls).
    """
    raw = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

    if os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true":
        return wrappers.wrap_anthropic(raw)

    return raw
