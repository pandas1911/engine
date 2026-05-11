"""Content search tool using regex with ripgrep/Python auto-fallback."""

import os
from typing import Any, Dict

from engine.config import get_config
from engine.tools.base import Tool
from engine.tools.builtin._utils.search import MAX_RESULTS, get_search_engine


class GrepTool(Tool):
    name = "grep"
    description = (
        "Fast content search tool that works with any regex pattern. "
        "Supports file type filtering via include parameter. "
        "Returns matches in file:line:content format, up to 100 results.\n\n"
        "Usage notes:\n"
        "- Pattern is a regular expression (Python re syntax).\n"
        "- Search scope defaults to workspace directory; specify path to override.\n"
        "- Use include to filter file types (e.g. '*.py', '*.{ts,tsx}').\n"
        "- This tool is read-only and does not modify any files.\n"
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Regular expression pattern to search for (Python re syntax).",
            },
            "path": {
                "type": "string",
                "description": "Directory or file to search in. Defaults to workspace directory (configurable in engine.json).",
            },
            "include": {
                "type": "string",
                "description": "File glob filter to narrow search scope (e.g. '*.py', '*.{ts,tsx}').",
            },
        },
        "required": ["pattern"],
    }

    async def execute(self, arguments: Dict[str, Any], context: Dict[str, Any]) -> str:
        pattern = arguments.get("pattern", "")
        if not pattern:
            return "Error: pattern is required"

        path = arguments.get("path") or str(get_config().get_workspace_path())
        path = os.path.abspath(path)
        include = arguments.get("include")

        searcher = get_search_engine()
        try:
            results = searcher.search_content(pattern, path, include)
        except ValueError as e:
            return f"Error: {e}"

        if not results:
            return (
                f"<pattern>{pattern}</pattern>\n"
                f"<path>{path}</path>\n"
                f"<matches>\n</matches>\n"
                f"<summary>No matches found</summary>"
            )

        lines = [f"{r.file}:{r.line_number}: {r.content}" for r in results]
        matches = "\n".join(lines)
        files_count = len(set(r.file for r in results))

        summary = f"Found {len(results)} matches in {files_count} file(s)"
        if len(results) >= MAX_RESULTS:
            summary += f" (showing first {MAX_RESULTS})"

        return (
            f"<pattern>{pattern}</pattern>\n"
            f"<path>{path}</path>\n"
            f"<matches>\n{matches}\n</matches>\n"
            f"<summary>{summary}</summary>"
        )
