"""Content search tool using regex with ripgrep/Python auto-fallback."""

import os
from typing import Any, Dict

from engine.tools.base import Tool
from engine.tools.builtin.search import MAX_RESULTS, get_search_engine
from engine.tools.builtin.security import PathGuard


class GrepTool(Tool):
    name = "grep"
    description = (
        "Fast content search tool that works with any regex pattern. "
        "Supports file type filtering via include parameter. "
        "Returns matches in file:line:content format, up to 100 results.\n\n"
        "Usage notes:\n"
        "- Pattern is a regular expression (Python re syntax).\n"
        "- Search scope defaults to current directory; specify path to override.\n"
        "- Use include to filter file types (e.g. '*.py', '*.{ts,tsx}').\n"
        "- Results are capped at 100 matches.\n"
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
                "description": "Directory or file to search in. Defaults to current working directory.",
            },
            "include": {
                "type": "string",
                "description": "File glob filter to narrow search scope (e.g. '*.py', '*.{ts,tsx}').",
            },
        },
        "required": ["pattern"],
    }

    def __init__(self, path_guard: PathGuard | None = None):
        self._guard = path_guard or PathGuard()

    async def execute(self, arguments: Dict[str, Any], context: Dict[str, Any]) -> str:
        pattern = arguments.get("pattern", "")
        if not pattern:
            return "Error: pattern is required"

        path = arguments.get("path") or os.getcwd()
        path = os.path.abspath(path)
        include = arguments.get("include")

        # Security check
        denial = self._guard.check_path(path)
        if denial:
            return f"Error: {denial}"

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
