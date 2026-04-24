"""Bocha AI web search tool for the Agent system."""

import os

import httpx

from engine.safety import ResultTruncator
from engine.tools.base import Tool

MAX_FIELD_LENGTH = 500


class WebSearchTool(Tool):
    """Web search tool using Bocha AI API."""

    name = "web_search"
    description = (
        "Search the web for up-to-date information. Use this tool when you need "
        "to find current facts, news, recent events, or any topic that requires "
        "real-time internet data. Returns a list of web results with titles, URLs, "
        "summaries, and source metadata."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search keywords or natural language question",
            },
            "freshness": {
                "type": "string",
                "description": (
                    "Time range filter. Options: noLimit (default), oneDay, "
                    "oneWeek, oneMonth, oneYear, or a date (YYYY-MM-DD), or a "
                    "date range (YYYY-MM-DD..YYYY-MM-DD)"
                ),
            },
        },
        "required": ["query"],
    }

    async def execute(self, arguments: dict, context: dict) -> str:
        """Execute web search via Bocha AI API.

        Args:
            arguments: LLM-provided arguments (query, freshness)
            context: Execution context (unused by this tool)

        Returns:
            Formatted search results or error message
        """
        api_key = os.getenv("BOCHA_API_KEY")
        if not api_key:
            return "Web search unavailable: BOCHA_API_KEY not configured in .env"

        query = arguments.get("query", "")
        if not query or not query.strip():
            return "Web search error: empty query provided"

        freshness = arguments.get("freshness", "noLimit")

        payload = {
            "query": query,
            "freshness": freshness,
            "summary": True,
            "count": 3,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
                response = await client.post(
                    "https://api.bochaai.com/v1/web-search",
                    json=payload,
                    headers=headers,
                )
        except httpx.RequestError as exc:
            return f"Web search error: network request failed - {exc}"
        except Exception as exc:
            return f"Web search error: unexpected failure - {exc}"

        if response.status_code != 200:
            return (
                f"Web search error: API returned status {response.status_code}"
            )

        try:
            data = response.json()
            results = data["data"]["webPages"]["value"]
        except (KeyError, TypeError, ValueError):
            return "Web search error: unexpected response format"

        if not results:
            return f'No results found for "{query}"'

        # Build formatted output with truncated fields
        output_lines = [f'Found {len(results)} results for "{query}":\n']

        for i, item in enumerate(results, 1):
            result_id = ResultTruncator.truncate(
                str(item.get("id", "")) or "", MAX_FIELD_LENGTH
            )
            name = ResultTruncator.truncate(
                str(item.get("name", "")) or "", MAX_FIELD_LENGTH
            )
            url = ResultTruncator.truncate(
                str(item.get("url", "")) or "", MAX_FIELD_LENGTH
            )
            snippet = ResultTruncator.truncate(
                str(item.get("snippet", "")) or "", MAX_FIELD_LENGTH
            )
            summary = ResultTruncator.truncate(
                str(item.get("summary", "")) or "", MAX_FIELD_LENGTH
            )
            site_name = ResultTruncator.truncate(
                str(item.get("siteName", "")) or "", MAX_FIELD_LENGTH
            )
            date_published = ResultTruncator.truncate(
                str(item.get("datePublished", "")) or "", MAX_FIELD_LENGTH
            )

            output_lines.append(f"[{i}] {name}")
            output_lines.append(f"ID: {result_id}")
            output_lines.append(f"URL: {url}")
            output_lines.append(f"Snippet: {snippet}")
            output_lines.append(f"Summary: {summary}")
            output_lines.append(
                f"Source: {site_name} | Date: {date_published}"
            )
            output_lines.append("")

        return "\n".join(output_lines)
