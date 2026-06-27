"""
Web search tool backed by Tavily.

Tavily is designed specifically for LLM agents: it crawls and extracts clean
text from pages rather than returning raw HTML snippets, and it optionally
synthesizes a brief answer from the top results.
"""

from langsmith import traceable
from tavily import TavilyClient

from src.config import settings

_tavily_client: TavilyClient | None = None


def _get_tavily() -> TavilyClient:
    global _tavily_client
    if _tavily_client is None:
        _tavily_client = TavilyClient(api_key=settings.tavily_api_key)
    return _tavily_client


# ── Anthropic tool schema ─────────────────────────────────────────────────────
WEB_SEARCH_TOOL_SCHEMA: dict = {
    "name": "web_search",
    "description": (
        "Search the live web for current information about a topic. "
        "Use this when: (1) the topic is too recent to be in the paper database, "
        "(2) you need general background context, or "
        "(3) the retrieve_documents tool returned insufficient results."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query. Use natural language — Tavily handles query expansion.",
            }
        },
        "required": ["query"],
    },
}


@traceable(run_type="tool", name="web_search")
def web_search(query: str, max_results: int = 5) -> str:
    """
    Execute a Tavily web search and return formatted results.

    search_depth="advanced" instructs Tavily to crawl result pages for full
    text rather than using only the search snippet.  More tokens but higher
    quality signal for the model.
    """
    client = _get_tavily()
    response = client.search(
        query=query,
        search_depth="advanced",
        max_results=max_results,
        include_answer=True,   # Tavily's synthesized 1-2 sentence answer
        include_raw_content=False,  # raw HTML not needed; clean text is enough
    )

    sections: list[str] = []

    # Tavily's pre-synthesized answer is often the most useful first line.
    if response.get("answer"):
        sections.append(f"[Synthesized Answer]\n{response['answer']}")

    for i, result in enumerate(response.get("results", []), start=1):
        sections.append(
            f"[Web Result {i}] {result['title']}\n"
            f"URL: {result['url']}\n"
            f"{result.get('content', '').strip()}"
        )

    return "\n\n---\n\n".join(sections) if sections else "No web results found."
