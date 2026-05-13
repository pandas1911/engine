"""ProcessTool for managing background shell processes.

Provides list, poll, log, and kill operations for background
processes started by BashTool.
"""

from __future__ import annotations

import time
from typing import Any, Dict

from engine.tools.base import Tool
from engine.tools.builtin._bash.background import ProcessRegistry
from engine.tools.builtin._bash.schemas import PROCESS_TOOL_SCHEMA


class ProcessTool(Tool):
    name = "process"
    short_description = "Manage background shell processes"
    description = "Manage background processes (list, poll, log, kill)."
    parameters = PROCESS_TOOL_SCHEMA

    def __init__(self, registry: ProcessRegistry | None = None) -> None:
        self._registry = registry or self._get_shared_registry()

    @staticmethod
    def _get_shared_registry() -> ProcessRegistry:
        """Get or create the shared registry singleton."""
        if not hasattr(ProcessTool, "_shared_registry"):
            ProcessTool._shared_registry = ProcessRegistry()
        return ProcessTool._shared_registry

    async def execute(self, arguments: Dict[str, Any], context: Dict[str, Any]) -> str:
        action = arguments.get("action", "")
        if action == "list":
            return self._list_processes()
        elif action == "poll":
            return self._poll_process(arguments.get("session_id", ""))
        elif action == "log":
            return self._log_process(arguments.get("session_id", ""))
        elif action == "kill":
            return self._kill_process(arguments.get("session_id", ""))
        else:
            return f"Error: Unknown action '{action}'. Valid actions: list, poll, log, kill."

    def _list_processes(self) -> str:
        processes = self._registry.list_all()
        if not processes:
            return "No background processes."
        lines = []
        for p in processes:
            elapsed = time.time() - p.start_time
            lines.append(
                f"  {p.session_id}  {p.command[:50]:<50}  {p.status:<12}  {elapsed:.0f}s"
            )
        header = f"  {'Session ID':<14}  {'Command':<50}  {'Status':<12}  {'Runtime'}\n"
        header += "  " + "-" * 90 + "\n"
        return header + "\n".join(lines)

    def _poll_process(self, session_id: str) -> str:
        if not session_id:
            return "Error: session_id is required for poll action."
        proc = self._registry.get(session_id)
        if not proc:
            return f"Error: No process found with session_id '{session_id}'."
        elapsed = time.time() - proc.start_time
        lines = [
            f"Session ID: {proc.session_id}",
            f"Command: {proc.command}",
            f"Status: {proc.status}",
            f"Runtime: {elapsed:.1f}s",
        ]
        if proc.exit_code is not None:
            lines.append(f"Exit code: {proc.exit_code}")
        return "\n".join(lines)

    def _log_process(self, session_id: str) -> str:
        if not session_id:
            return "Error: session_id is required for log action."
        proc = self._registry.get(session_id)
        if not proc:
            return f"Error: No process found with session_id '{session_id}'."
        parts = []
        stdout = getattr(proc, "stdout", "")
        stderr = getattr(proc, "stderr", "")
        if stdout:
            parts.append(stdout)
        if stderr:
            parts.append(f"\n[stderr]\n{stderr}")
        if not parts:
            return "No output available yet." if proc.status == "running" else "Process produced no output."
        return "".join(parts)

    def _kill_process(self, session_id: str) -> str:
        if not session_id:
            return "Error: session_id is required for kill action."
        proc = self._registry.get(session_id)
        if not proc:
            return f"Error: No process found with session_id '{session_id}'."
        if proc.status in ("completed", "killed", "timeout"):
            return f"Process '{session_id}' is already {proc.status}."
        proc.status = "killed"
        proc.exit_code = -9
        return f"Process '{session_id}' killed."
