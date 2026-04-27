"""Web search tool using ddgs metasearch library.

Provides web search capability via the ddgs library, which aggregates
results from multiple search engines (DuckDuckGo, Bing, Brave, Google, etc.)
with automatic failover for improved stability.
"""

import asyncio
from typing import Any, Dict, List

from ddgs import DDGS
from ddgs.exceptions import DDGSException, TimeoutException

from engine.safety import ResultTruncator
from engine.tools.base import Tool

_ddgs_client: DDGS | None = None


def _get_ddgs() -> DDGS:
    """Lazy-init a singleton DDGS instance for reuse."""
    global _ddgs_client
    if _ddgs_client is None:
        _ddgs_client = DDGS(timeout=10)
    return _ddgs_client


class WebSearchTool(Tool):
    """Web search tool powered by ddgs metasearch library.

    Aggregates results from multiple search engines with automatic
    failover via backend="auto" for improved stability.
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

    _BACKEND: str = "auto"
    _MAX_RESULTS: int = 5
    _MAX_SNIPPET_LENGTH: int = 1000
    _REGION: str = "wt-wt"

    async def execute(self, arguments: dict, context: dict) -> str:
        query = arguments.get("query", "")
        if not query or not isinstance(query, str) or not query.strip():
            return "Web search error: empty query provided"
        query = query.strip()

        freshness = arguments.get("freshness")
        timelimit = None
        if freshness and isinstance(freshness, str) and freshness.strip():
            timelimit = freshness.strip()

        try:
            ddgs = _get_ddgs()
            results: List[Dict[str, Any]] = await asyncio.to_thread(
                ddgs.text,
                query,
                region=self._REGION,
                timelimit=timelimit,
                max_results=self._MAX_RESULTS,
                backend=self._BACKEND,
            )
        except TimeoutException:
            return "Web search error: request timed out"
        except DDGSException:
            return "Web search error: search request failed"
        except Exception as exc:
            return f"Web search error: unexpected failure - {exc}"

        if not results:
            return f'No results found for "{query}"'

        return self._format_results(query, results)

    def _format_results(self, query: str, results: List[Dict[str, Any]]) -> str:
        lines: List[str] = [f'## Search Results for "{query}"', ""]

        for position, result in enumerate(results, start=1):
            title = result.get("title", "No title")
            link = result.get("href", "")
            snippet = ResultTruncator.truncate(result.get("body", ""), self._MAX_SNIPPET_LENGTH)

            lines.append(f"[{position}] **Title:** {title}")
            lines.append(f"    **URL:** {link}")
            lines.append(f"    **Snippet:** {snippet}")
            lines.append("")

        return "\n".join(lines)
