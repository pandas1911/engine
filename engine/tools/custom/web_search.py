"""Web search tool using DuckDuckGo HTML endpoint.

Provides web search capability without external API keys by scraping
the DuckDuckGo HTML search page and parsing results with stdlib.
Includes retry-with-backoff for HTTP 202 rate-limit responses, User-Agent
rotation, cookie persistence via shared httpx client, and browser-realistic
headers to improve resilience against DuckDuckGo rate-limiting.
"""

import asyncio
import random
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import List
from urllib.parse import parse_qs, unquote, urlparse

import httpx

from engine.logging import get_logger
from engine.safety import ResultTruncator
from engine.tools.base import Tool

_USER_AGENTS: List[str] = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
]

_ClientType = httpx.AsyncClient | None
_shared_client: _ClientType = None
_client_lock: asyncio.Lock = asyncio.Lock()


async def _get_client() -> httpx.AsyncClient:
    """Lazy-init a shared httpx client with cookie persistence."""
    global _shared_client
    if _shared_client is None or _shared_client.is_closed:
        async with _client_lock:
            if _shared_client is None or _shared_client.is_closed:
                _shared_client = httpx.AsyncClient(
                    timeout=httpx.Timeout(15.0),
                    follow_redirects=True,
                    headers={
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                        "Accept-Language": "en-US,en;q=0.5",
                    },
                )
    return _shared_client


@dataclass
class SearchResult:
    title: str
    link: str
    snippet: str
    position: int


class _DuckDuckGoHTMLParser(HTMLParser):
    """Parse DDG HTML search results.

    Extracts results from ``<div class="result">`` blocks:
    title/URL from ``<a class="result__a">``, snippet from ``result__snippet``.
    Also detects captcha/challenge pages.
    """

    def __init__(self) -> None:
        super().__init__()
        self._results: List[SearchResult] = []
        self._position = 0
        self._captcha_detected = False
        self._in_result = False
        self._result_div_depth: int = 0
        self._in_title = False
        self._in_snippet = False
        self._current_href = ""
        self._current_title = ""
        self._current_snippet = ""

    def parse(self, html: str) -> List[SearchResult]:
        self._results = []
        self._position = 0
        self._captcha_detected = False
        self._result_div_depth = 0
        self.feed(html)
        return self._results

    def handle_starttag(self, tag: str, attrs: List[tuple]) -> None:
        attr_dict = dict(attrs)
        classes = attr_dict.get("class", "").split()

        if tag == "div" and "challenge-form" in classes:
            self._captcha_detected = True
            return

        if tag == "input" and attr_dict.get("name") == "captcha":
            self._captcha_detected = True
            return

        if tag == "div" and "result" in classes:
            if not self._in_result:
                self._in_result = True
                self._result_div_depth = 1
                self._current_href = ""
                self._current_title = ""
                self._current_snippet = ""
            else:
                self._result_div_depth += 1
            return

        if self._in_result and tag == "div":
            self._result_div_depth += 1
            return

        if tag == "a" and "result__a" in classes and self._in_result:
            self._in_title = True
            self._current_href = attr_dict.get("href", "")
            return

        if self._in_result and "result__snippet" in classes:
            self._in_snippet = True

    def handle_endtag(self, tag: str) -> None:
        if self._in_title and tag == "a":
            self._in_title = False
            return

        if self._in_snippet and tag in ("td", "div", "span", "a", "p"):
            self._in_snippet = False

        if self._in_result and tag == "div":
            self._result_div_depth -= 1
            if self._result_div_depth == 0:
                if self._current_title:
                    link = self._clean_url(self._current_href)
                    if link and not self._is_ad(link):
                        self._position += 1
                        self._results.append(
                            SearchResult(
                                title=self._current_title.strip(),
                                link=link,
                                snippet=self._current_snippet.strip(),
                                position=self._position,
                            )
                        )
                self._in_result = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._current_title += data
        elif self._in_snippet:
            self._current_snippet += data

    @staticmethod
    def _is_ad(link: str) -> bool:
        return "y.js" in link

    @staticmethod
    def _clean_url(href: str) -> str:
        if not href:
            return ""
        if href.startswith("//"):
            href = "https:" + href
        parsed = urlparse(href)
        # DDG redirect: //duckduckgo.com/l/?uddg=<encoded_url>&rut=...
        if "duckduckgo.com" in parsed.netloc and "/l/" in parsed.path:
            qs = parse_qs(parsed.query)
            uddg_list = qs.get("uddg")
            if uddg_list:
                return unquote(uddg_list[0])
        return href


class _RateLimiter:
    """Minimum-interval gate between requests."""

    _last_request_time: float = 0.0
    MIN_INTERVAL: float = 4.0
    _lock: asyncio.Lock = asyncio.Lock()

    @classmethod
    async def acquire(cls) -> None:
        async with cls._lock:
            now = asyncio.get_event_loop().time()
            elapsed = now - cls._last_request_time
            if elapsed < cls.MIN_INTERVAL:
                await asyncio.sleep(cls.MIN_INTERVAL - elapsed)
            cls._last_request_time = asyncio.get_event_loop().time()


class WebSearchTool(Tool):
    """Web search tool powered by DuckDuckGo HTML endpoint.

    Retries HTTP 202 (rate-limit/captcha) responses with exponential
    backoff and random jitter for improved resilience.
    """

    name = "web_search"
    description = (
        "Search the web for up-to-date information. Use this tool when you need "
        "to find current facts, news, recent events, or any topic that requires "
        "real-time internet data. Returns a list of web results with titles, URLs, "
        "and snippets. To retrieve the full content of any URL found in the results, "
        "use the web_fetch tool."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Precise keywords for the search query. Use specific terms "
                    "rather than natural language questions "
                    "(e.g., 'Python asyncio tutorial' not 'How do I use asyncio in Python?')."
                ),
            },
            "freshness": {
                "type": "string",
                "description": (
                    "Time range filter. Options: 'd' (past day), 'w' (past week), "
                    "'m' (past month), 'y' (past year). Leave empty for no time limit."
                ),
            },
        },
        "required": ["query"],
    }

    _MAX_RESULTS: int = 5
    _MAX_FIELD_LENGTH: int = 1000
    _BASE_URL: str = "https://html.duckduckgo.com/html"
    _REGION: str = "wt-wt"
    _SAFE_SEARCH: str = "-2"
    _MAX_RETRIES: int = 3
    _RETRY_BASE_DELAYS: List[float] = [5.0, 12.0, 20.0]

    async def execute(self, arguments: dict, context: dict) -> str:
        query = arguments.get("query", "")
        if not query or not isinstance(query, str) or not query.strip():
            return "Web search error: empty query provided"
        query = query.strip()

        await _RateLimiter.acquire()

        form_data: dict = {
            "q": query,
            "b": "",
            "kl": self._REGION,
            "kp": self._SAFE_SEARCH,
        }
        freshness = arguments.get("freshness")
        if freshness and isinstance(freshness, str) and freshness.strip():
            form_data["df"] = freshness.strip()

        headers = {
            "User-Agent": random.choice(_USER_AGENTS),
            "Referer": "https://duckduckgo.com/",
        }

        for attempt in range(self._MAX_RETRIES + 1):
            try:
                client = await _get_client()
                response = await client.post(
                    self._BASE_URL, data=form_data, headers=headers
                )
            except httpx.TimeoutException:
                return "Web search error: request timed out"
            except httpx.RequestError:
                return "Web search error: network request failed"
            except Exception as exc:
                return f"Web search error: unexpected failure - {exc}"

            if response.status_code == 200:
                break

            if attempt < self._MAX_RETRIES:
                if response.status_code == 202:
                    delay = self._RETRY_BASE_DELAYS[attempt] + (random.random() * 1.0 - 0.5)
                    get_logger().warning(
                        "WebSearch",
                        "Retry {}/{} after HTTP 202 | query={:.50s} | delay={:.1f}s".format(
                            attempt + 1, self._MAX_RETRIES, query, delay
                        ),
                        event_type="tool_retry",
                        data={"attempt": attempt + 1, "status": 202, "query": query, "delay": delay},
                    )
                    await asyncio.sleep(delay)
                elif response.status_code == 403:
                    headers["User-Agent"] = random.choice(_USER_AGENTS)
                    await asyncio.sleep(1.0 + random.random())
                else:
                    return f"Web search error: HTTP {response.status_code}"
                continue

            return f"Web search error: HTTP {response.status_code}"

        try:
            parser = _DuckDuckGoHTMLParser()
            results = parser.parse(response.text)
        except Exception:
            return "Web search error: failed to parse search results"

        if parser._captcha_detected:
            return (
                "Web search error: DuckDuckGo returned a challenge page, "
                "please try again later"
            )

        if not results:
            return f'No results found for "{query}"'

        results = results[: self._MAX_RESULTS]
        return self._format_results(query, results)

    def _format_results(self, query: str, results: List[SearchResult]) -> str:
        lines: List[str] = [f'## Search Results for "{query}"', ""]

        for result in results:
            title = ResultTruncator.truncate(result.title, self._MAX_FIELD_LENGTH)
            link = ResultTruncator.truncate(result.link, self._MAX_FIELD_LENGTH)
            snippet = ResultTruncator.truncate(result.snippet, self._MAX_FIELD_LENGTH)

            lines.append(f"[{result.position}] **Title:** {title}")
            lines.append(f"    **URL:** {link}")
            lines.append(f"    **Snippet:** {snippet}")
            lines.append("")

        return "\n".join(lines)
