"""Web search tool using Alibaba Cloud IQS service.

Provides web search capability via the Alibaba Cloud Intelligent Search SDK
with support for enhanced summaries, rerank scoring, and full text extraction.
"""

import asyncio
import hashlib
import time
from typing import List, Optional

from alibabacloud_iqs20241111 import models
from alibabacloud_iqs20241111.client import Client
from alibabacloud_tea_openapi import models as open_api_models
from Tea.exceptions import TeaException

from engine.config import get_config
from engine.safety import ResultTruncator
from engine.tools.base import Tool

_iqs_client: Optional[Client] = None


def _get_iqs_client() -> Optional[Client]:
    """Lazy-init a singleton IQS client. Returns None if not configured."""
    global _iqs_client
    if _iqs_client is None:
        config = get_config()
        if not config.aliyun_search:
            return None
        ak_config = open_api_models.Config(
            access_key_id=config.aliyun_search["access_key_id"],
            access_key_secret=config.aliyun_search["access_key_secret"],
        )
        ak_config.endpoint = "iqs.cn-zhangjiakou.aliyuncs.com"
        _iqs_client = Client(ak_config)
    return _iqs_client


# Time range mapping: short format -> IQS format
_TIME_RANGE_MAP = {
    "d": "OneDay",
    "w": "OneWeek",
    "m": "OneMonth",
    "y": "OneYear",
    "OneDay": "OneDay",
    "OneWeek": "OneWeek",
    "OneMonth": "OneMonth",
    "OneYear": "OneYear",
}


class WebSearchTool(Tool):
    """Web search tool powered by Alibaba Cloud IQS."""

    name = "web_search"
    short_description = "Search the web using Alibaba Cloud IQS"
    description = (
        "Search the web for up-to-date information. Use this tool when you need "
        "to find current facts, news, recent events, or any topic that requires "
        "real-time internet data. Returns a list of web results with titles, URLs, "
        "snippets, and optionally full page content saved to files."
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
            "time_range": {
                "type": "string",
                "description": (
                    "Time range filter. Options: 'd' (past day), 'w' (past week), "
                    "'m' (past month), 'y' (past year). Leave empty for no time limit."
                ),
            },
        },
        "required": ["query"],
    }

    _ENABLE_SUMMARY: bool = True
    _ENABLE_RERANK_SCORE: bool = True
    _ENABLE_MAIN_TEXT: bool = True
    _MAX_RESULTS: int = 5
    _ENGINE_TYPE: str = "Generic"
    _MAX_SNIPPET_LENGTH: int = 1000
    _MAX_MAINTEXT_LENGTH: int = 3000

    async def execute(self, arguments: dict, context: dict) -> str:
        query = arguments.get("query", "")
        if not query or not isinstance(query, str) or not query.strip():
            return "Web search error: empty query provided"
        query = query.strip()

        time_range_raw = arguments.get("time_range")
        time_range_iqs = "NoLimit"
        if time_range_raw and isinstance(time_range_raw, str) and time_range_raw.strip():
            time_range_iqs = _TIME_RANGE_MAP.get(time_range_raw.strip(), "NoLimit")

        client = _get_iqs_client()
        if client is None:
            return "Web search error: IQS credentials not configured"

        try:
            contents = models.RequestContents(
                summary=self._ENABLE_SUMMARY,
                rerank_score=self._ENABLE_RERANK_SCORE,
                main_text=self._ENABLE_MAIN_TEXT,
            )
            search_input = models.UnifiedSearchInput(
                query=query,
                engine_type=self._ENGINE_TYPE,
                time_range=time_range_iqs,
                contents=contents,
            )
            request = models.UnifiedSearchRequest(body=search_input)
            response = await asyncio.to_thread(client.unified_search, request)
        except TeaException as exc:
            return f"Web search error: IQS request failed - {exc.message}"
        except Exception as exc:
            return f"Web search error: unexpected failure - {exc}"

        page_items = response.body.page_items if response.body else None
        if not page_items:
            return f'No results found for "{query}"'

        results = sorted(
            page_items,
            key=lambda r: getattr(r, "rerank_score", None) or 0.0,
            reverse=True,
        )
        results = results[: self._MAX_RESULTS]

        # Save mainText files and collect refs
        file_refs: List[Optional[str]] = []
        for i, result in enumerate(results):
            main_text = getattr(result, "main_text", None)
            ref = self._save_maintext(main_text, query, i)
            file_refs.append(ref)

        return self._format_results(query, results, file_refs)

    def _save_maintext(
        self, main_text: Optional[str], query: str, index: int
    ) -> Optional[str]:
        """Save mainText to disk. Returns relative file path or None."""
        if not main_text or not main_text.strip():
            return None
        config = get_config()
        cache_dir = config.get_workspace_path() / "search_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
        timestamp = int(time.time())
        filename = f"{timestamp}_{query_hash}_{index}.md"
        filepath = cache_dir / filename
        content = ResultTruncator.truncate(main_text, self._MAX_MAINTEXT_LENGTH)
        filepath.write_text(content, encoding="utf-8")
        return f"search_cache/{filename}"

    def _format_results(
        self,
        query: str,
        results: list,
        file_refs: List[Optional[str]],
    ) -> str:
        lines = [f'## Search Results for "{query}"', ""]
        for i, (result, file_ref) in enumerate(zip(results, file_refs), start=1):
            title = getattr(result, "title", "No title") or "No title"
            link = getattr(result, "link", "") or ""
            snippet = getattr(result, "snippet", "") or ""
            summary = getattr(result, "summary", "") or ""
            if summary:
                snippet = summary
            snippet = ResultTruncator.truncate(snippet, self._MAX_SNIPPET_LENGTH)

            lines.append(f"[{i}] **Title:** {title}")
            lines.append(f"    **URL:** {link}")
            lines.append(f"    **Snippet:** {snippet}")
            if file_ref:
                lines.append(
                    f"    **Full Text:** Full page content saved to `{file_ref}`. "
                    f"You can use the `read` tool to view it if you need more details."
                )
            lines.append("")
        return "\n".join(lines)
