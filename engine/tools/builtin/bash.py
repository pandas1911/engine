"""BashTool -- shell command execution for the Engine agent system.

Integrates: AST parsing -> security check -> env sanitization -> execution -> truncation.
Registered as a builtin tool in engine/tools/builtin/__init__.py.
"""

from __future__ import annotations

import os
from typing import Any, Dict

from engine.tools.base import Tool
from engine.tools.builtin._bash.ast_parser import BashASTParser
from engine.tools.builtin._bash.executor import ExecutionResult, ProcessExecutor
from engine.tools.builtin._bash.security import SecurityChecker, sanitize_env
from engine.tools.builtin._bash.schemas import BASH_TOOL_SCHEMA, DEFAULT_TIMEOUT_MS, YIELD_THRESHOLD_MS
from engine.tools.builtin._bash.prompt import BASH_TOOL_DESCRIPTION


class BashTool(Tool):
    name = "bash"
    short_description = "Execute bash shell commands"
    description = BASH_TOOL_DESCRIPTION
    parameters = BASH_TOOL_SCHEMA

    def __init__(self) -> None:
        self._ast_parser = BashASTParser()
        self._security = SecurityChecker()

    async def execute(self, arguments: Dict[str, Any], context: Dict[str, Any]) -> str:
        command = arguments.get("command", "")
        if not command or not isinstance(command, str) or not command.strip():
            return "Error: empty command provided"

        _description = arguments.get("description", "")
        timeout_ms = int(arguments.get("timeout") or DEFAULT_TIMEOUT_MS)
        workdir = arguments.get("workdir") or os.getcwd()
        user_env = arguments.get("env") or {}
        background = bool(arguments.get("background", False))

        parsed = self._ast_parser.parse(command)
        sec_result = self._security.check_command(parsed, os.getcwd())
        if not sec_result.allowed:
            return f"Error: Command blocked -- {sec_result.reason}"

        warnings_prefix = ""
        if sec_result.warnings:
            warnings_prefix = "[WARNING] " + "; ".join(sec_result.warnings) + "\n\n"

        env = sanitize_env(user_env, dict(os.environ))

        if background:
            return await self._execute_background(command, workdir, timeout_ms, env)
        return await self._execute_foreground(command, workdir, timeout_ms, env, warnings_prefix)

    async def _execute_foreground(
        self,
        command: str,
        workdir: str,
        timeout_ms: int,
        env: Dict[str, str],
        warnings_prefix: str,
    ) -> str:
        executor = ProcessExecutor()
        result = await executor.execute(command=command, workdir=workdir, timeout_ms=timeout_ms, env=env)
        return self._format_result(result, warnings_prefix)

    async def _execute_background(
        self,
        command: str,
        workdir: str,
        timeout_ms: int,
        env: Dict[str, str],
    ) -> str:
        from engine.tools.builtin._bash.background import BackgroundExecutor
        bg = BackgroundExecutor()
        result = await bg.execute_background(
            command=command, workdir=workdir, env=env, yield_ms=YIELD_THRESHOLD_MS,
        )
        if result.backgrounded:
            return (
                f"Command running in background.\n"
                f"Session ID: {result.session_id}\n"
                f"Use 'process' tool to manage."
            )
        if result.direct_result:
            return self._format_result(result.direct_result, "")
        return "Background execution returned no result."

    @staticmethod
    def _format_result(result: ExecutionResult, prefix: str) -> str:
        parts: list[str] = []
        if prefix:
            parts.append(prefix)
        if result.stdout:
            parts.append(result.stdout)
        if result.stderr:
            parts.append(f"\n[stderr]\n{result.stderr}")
        meta: list[str] = []
        if result.timed_out:
            meta.append("Command timed out. Retry with larger timeout if needed.")
        if result.aborted:
            meta.append("Command was aborted.")
        if result.truncated and result.full_output_path:
            meta.append(f"Full output saved to: {result.full_output_path}")
        if meta:
            parts.append("\n\n" + "\n".join(meta))
        if result.exit_code is not None:
            parts.append(f"\n[exit code: {result.exit_code}]")
        return "".join(parts) if parts else "[Tool returned empty output]"
