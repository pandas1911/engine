"""File and directory reading tool for the agent system."""

import os
from typing import Any, Dict

from engine.tools.base import Tool
from engine.tools.builtin._utils.binary import BinaryDetector


class ReadTool(Tool):
    name = "read"
    description = (
        "Read a file or directory from the local filesystem. "
        "For files: returns content with line numbers, paginated by offset/limit. "
        "For directories: lists entries with / suffix for subdirectories. "
        "Usage notes:\n"
        "- The filePath parameter accepts an absolute path.\n"
        "- Each line is prefixed with its line number (1-indexed).\n"
        "- Use offset to skip to later portions of a large file.\n"
        "- If the file is too large, the output will be truncated with guidance.\n"
        "- This tool is read-only and does not modify any files.\n"
    )
    parameters = {
        "type": "object",
        "properties": {
            "filePath": {
                "type": "string",
                "description": "Absolute path to file or directory.",
            },
            "offset": {
                "type": "integer",
                "description": "1-indexed line number to start reading from. Default: 1.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of lines to read. Default: 5000.",
            },
        },
        "required": ["filePath"],
    }

    DEFAULT_READ_LIMIT = 5000
    MAX_LINE_LENGTH = 5000
    MAX_BYTES = 70 * 1024

    async def execute(self, arguments: Dict[str, Any], context: Dict[str, Any]) -> str:
        file_path = arguments.get("filePath", "")
        if not file_path or not isinstance(file_path, str):
            return "Error: filePath is required"
        file_path = os.path.abspath(file_path)

        if not os.path.exists(file_path):
            return self._format_not_found(file_path)

        if os.path.isdir(file_path):
            return self._read_directory(file_path)

        return self._read_file(file_path, arguments)

    def _format_not_found(self, file_path: str) -> str:
        parent = os.path.dirname(file_path)
        siblings: list[str] = []
        try:
            siblings = sorted(os.listdir(parent))[:20]
        except OSError:
            pass
        suggestion = ""
        if siblings:
            suggestion = "\nSibling files:\n" + "\n".join(siblings)
        return f"Error: Path not found: {file_path}{suggestion}"

    def _read_directory(self, dir_path: str) -> str:
        try:
            entries = sorted(os.listdir(dir_path))
        except PermissionError:
            return f"Error: Permission denied: {dir_path}"

        formatted: list[str] = []
        for entry in entries:
            full = os.path.join(dir_path, entry)
            suffix = "/" if os.path.isdir(full) else ""
            formatted.append(f"{entry}{suffix}")

        body = "\n".join(formatted)
        return f"<path>{dir_path}</path>\n<type>directory</type>\n<entries>\n{body}\n</entries>"

    def _read_file(self, file_path: str, arguments: Dict[str, Any]) -> str:
        if BinaryDetector.is_binary(file_path):
            return f"Error: Cannot read binary file: {file_path}"

        offset = max(1, arguments.get("offset", 1) or 1)
        limit = arguments.get("limit", self.DEFAULT_READ_LIMIT) or self.DEFAULT_READ_LIMIT

        lines_out: list[str] = []
        total_bytes = 0
        total_lines = 0
        reached_end = False

        try:
            with open(file_path, "r", errors="replace") as f:
                for line_num, raw_line in enumerate(f, 1):
                    total_lines += 1
                    if line_num < offset:
                        continue
                    if len(lines_out) >= limit:
                        break

                    stripped = raw_line.rstrip("\n\r")
                    if len(stripped) > self.MAX_LINE_LENGTH:
                        stripped = stripped[: self.MAX_LINE_LENGTH] + "... [truncated]"

                    entry = f"{line_num}: {stripped}"
                    entry_bytes = len(entry.encode("utf-8"))
                    if total_bytes + entry_bytes > self.MAX_BYTES:
                        break
                    total_bytes += entry_bytes
                    lines_out.append(entry)
                else:
                    reached_end = True
        except PermissionError:
            return f"Error: Permission denied: {file_path}"
        except OSError as e:
            return f"Error: Cannot read file: {e}"

        content = "\n".join(lines_out)
        footer = ""

        shown_end = offset + len(lines_out) - 1
        if total_lines > shown_end or not reached_end:
            remaining = total_lines - shown_end if reached_end else None
            footer = (
                f"\n\n[Showing lines {offset}-{shown_end} of {total_lines}. "
                f"Use offset={shown_end + 1} to continue. "
                f"The file is large — consider using the grep tool to search "
                f"for specific content, or spawn a sub-agent to read the full file.]"
            )

        return (
            f"<path>{file_path}</path>\n"
            f"<type>file</type>\n"
            f"<content>\n{content}\n</content>"
            f"{footer}"
        )
