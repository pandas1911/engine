"""File pattern matching tool with ripgrep/Python auto-fallback."""

import os
from typing import Any, Dict

from engine.config import get_config
from engine.tools.base import Tool
from engine.tools.builtin._utils.search import get_search_engine, MAX_RESULTS


class GlobTool(Tool):
    name = "glob"
    short_description = "Find files matching glob patterns"
    description = (
        "Fast file pattern matching tool that returns file paths. "
        "Supports glob patterns like '**/*.py' or 'src/**/*.ts'.\n\n"
        "Usage notes:\n"
        "- Pattern uses standard glob syntax (** for recursive, * for wildcard).\n"
        "- Search scope defaults to workspace directory; specify path to override.\n"
        "- This tool is read-only and does not modify any files.\n"
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern to match files (e.g. '**/*.py', 'src/**/*.ts').",
            },
            "path": {
                "type": "string",
                "description": "Directory to search in. Defaults to workspace directory (configurable in engine.json).",
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

        searcher = get_search_engine()
        files = searcher.search_files(pattern, path)

        if not files:
            return (
                f"<pattern>{pattern}</pattern>\n"
                f"<path>{path}</path>\n"
                f"<files>\n</files>\n"
                f"<summary>Found 0 files</summary>"
            )

        file_list = "\n".join(files)
        summary = f"Found {len(files)} file(s)"
        if len(files) >= MAX_RESULTS:
            summary += f" (showing first {MAX_RESULTS})"

        return (
            f"<pattern>{pattern}</pattern>\n"
            f"<path>{path}</path>\n"
            f"<files>\n{file_list}\n</files>\n"
            f"<summary>{summary}</summary>"
        )
