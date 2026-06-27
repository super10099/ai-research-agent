import asyncio
from dataclasses import dataclass

import httpx
from bs4 import BeautifulSoup


@dataclass
class Document:
    url: str
    title: str
    text: str


async def _fetch_one(client: httpx.AsyncClient, url: str) -> Document:
    response = await client.get(url, follow_redirects=True)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "lxml")

    # Strip chrome that adds noise without information content.
    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()

    title = soup.title.string.strip() if soup.title else url
    # separator="\n" preserves paragraph boundaries; strip=True collapses whitespace.
    text = soup.get_text(separator="\n", strip=True)

    return Document(url=url, title=title, text=text)


async def fetch_all(urls: list[str]) -> list[Document]:
    """
    Fetch all URLs concurrently inside a single connection pool.

    We use asyncio.gather(return_exceptions=True) rather than TaskGroup here
    because we want graceful degradation — one bad URL should not abort the
    entire batch, analogous to how a failed worker rank in MPI_Gather shouldn't
    force every other rank to abort.
    """
    async with httpx.AsyncClient(
        timeout=30.0,
        headers={"User-Agent": "ai-research-agent/0.1 (educational project)"},
        limits=httpx.Limits(max_connections=10),
    ) as client:
        raw = await asyncio.gather(
            *[_fetch_one(client, url) for url in urls],
            return_exceptions=True,
        )

    results: list[Document] = []
    for url, outcome in zip(urls, raw):
        if isinstance(outcome, Exception):
            print(f"[fetcher] WARNING: skipping {url}: {outcome}")
        else:
            results.append(outcome)

    if not results:
        raise RuntimeError("Every URL failed to fetch — check network or URLs.")

    return results
