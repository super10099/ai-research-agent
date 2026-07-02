"""
Tests for src/ingestion/fetcher.py.

httpx.AsyncClient.get is monkeypatched at the class level so fetch_all's own
httpx.AsyncClient(...) construction is untouched — but real httpx.Response
objects are returned, so raise_for_status() exercises real httpx behavior
rather than a hand-rolled mock of it.
"""

import httpx
import pytest

from src.ingestion.fetcher import fetch_all

SAMPLE_HTML = """
<html>
<head><title>  A Paper About RAG  </title></head>
<body>
<nav>Skip this nav</nav>
<header>Skip this header</header>
<article>
<p>Retrieval-augmented generation improves factual accuracy.</p>
</article>
<footer>Skip this footer</footer>
<script>console.log("skip this too")</script>
</body>
</html>
"""


def _make_get(responses: dict[str, httpx.Response]):
    async def fake_get(self, url, follow_redirects=True):
        return responses[url]
    return fake_get


class TestFetchAll:
    async def test_extracts_title_and_strips_chrome(self, monkeypatch):
        url = "http://example.test/paper"
        responses = {
            url: httpx.Response(200, text=SAMPLE_HTML, request=httpx.Request("GET", url)),
        }
        monkeypatch.setattr(httpx.AsyncClient, "get", _make_get(responses))

        docs = await fetch_all([url])

        assert len(docs) == 1
        assert docs[0].url == url
        assert docs[0].title == "A Paper About RAG"
        assert "Retrieval-augmented generation" in docs[0].text
        assert "Skip this nav" not in docs[0].text
        assert "Skip this header" not in docs[0].text
        assert "Skip this footer" not in docs[0].text
        assert "console.log" not in docs[0].text

    async def test_falls_back_to_url_when_no_title_tag(self, monkeypatch):
        url = "http://example.test/no-title"
        html = "<html><body><p>No title here.</p></body></html>"
        responses = {url: httpx.Response(200, text=html, request=httpx.Request("GET", url))}
        monkeypatch.setattr(httpx.AsyncClient, "get", _make_get(responses))

        docs = await fetch_all([url])

        assert docs[0].title == url

    async def test_partial_failure_skips_bad_url_gracefully(self, monkeypatch):
        good_url = "http://example.test/good"
        bad_url = "http://example.test/bad"
        responses = {
            good_url: httpx.Response(
                200, text=SAMPLE_HTML, request=httpx.Request("GET", good_url)
            ),
            bad_url: httpx.Response(
                500, text="server error", request=httpx.Request("GET", bad_url)
            ),
        }
        monkeypatch.setattr(httpx.AsyncClient, "get", _make_get(responses))

        docs = await fetch_all([good_url, bad_url])

        assert len(docs) == 1
        assert docs[0].url == good_url

    async def test_raises_when_every_url_fails(self, monkeypatch):
        url = "http://example.test/bad"
        responses = {
            url: httpx.Response(500, text="server error", request=httpx.Request("GET", url)),
        }
        monkeypatch.setattr(httpx.AsyncClient, "get", _make_get(responses))

        with pytest.raises(RuntimeError, match="Every URL failed to fetch"):
            await fetch_all([url])
