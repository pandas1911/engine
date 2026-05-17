"""Async process executor with timeout, abort, and process group kill.

Architecture:
- asyncio.create_subprocess_shell for shell features (pipes, redirects)
- start_new_session=True -> process group leader for clean killing
- Three-way race: process exit vs timeout vs abort signal
- SIGTERM -> SIGKILL escalation with configurable grace period
"""

from __future__ import annotations

import asyncio
import os
import signal
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from engine.tools.builtin._bash.schemas import (
    DEFAULT_TIMEOUT_MS,
    KILL_GRACE_PERIOD_MS,
    MAX_OUTPUT_BYTES,
    MAX_OUTPUT_LINES,
)
from engine.tools.builtin._bash.output import OutputTruncator


@dataclass
class ExecutionResult:
    exit_code: Optional[int]
    stdout: str
    stderr: str
    timed_out: bool
    aborted: bool
    truncated: bool
    full_output_path: Optional[str]


class ProcessExecutor:
    def __init__(self, on_output_chunk: Optional[Callable[[str], None]] = None) -> None:
        self._on_output_chunk = on_output_chunk

    async def execute(
        self,
        command: str,
        workdir: str,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        env: Optional[Dict[str, str]] = None,
        shell: Optional[str] = None,
        abort_event: Optional[asyncio.Event] = None,
    ) -> ExecutionResult:
        if shell is None:
            shell = self._detect_shell()
        if env is None:
            env = {**os.environ, "TERM": "dumb"}

        truncator = OutputTruncator(max_lines=MAX_OUTPUT_LINES, max_bytes=MAX_OUTPUT_BYTES)
        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []
        timed_out = False

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.DEVNULL,
                cwd=workdir,
                env=env,
                start_new_session=True,
                executable=shell,
            )
        except OSError as exc:
            return ExecutionResult(-1, "", str(exc), False, False, False, None)

        async def _read(
            stream: Optional[asyncio.StreamReader], chunks: list[str]
        ) -> None:
            if stream:
                while True:
                    chunk = await stream.read(4096)
                    if not chunk:
                        break
                    text = chunk.decode("utf-8", errors="replace")
                    chunks.append(text)
                    if self._on_output_chunk:
                        self._on_output_chunk(text)

        tasks = [
            asyncio.create_task(_read(process.stdout, stdout_chunks)),
            asyncio.create_task(_read(process.stderr, stderr_chunks)),
        ]

        try:
            await asyncio.wait_for(process.wait(), timeout=timeout_ms / 1000.0)
        except asyncio.TimeoutError:
            timed_out = True
            await self._kill_process_tree(process.pid)
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        if abort_event and abort_event.is_set() and not timed_out:
            await self._kill_process_tree(process.pid)

        raw_output = "".join(stdout_chunks)
        trunc = truncator.truncate(raw_output)
        return ExecutionResult(
            exit_code=process.returncode,
            stdout=trunc.output,
            stderr="".join(stderr_chunks),
            timed_out=timed_out,
            aborted=bool(abort_event and abort_event.is_set()),
            truncated=trunc.truncated,
            full_output_path=trunc.full_output_path,
        )

    async def _kill_process_tree(self, pid: Optional[int]) -> None:
        """SIGTERM -> SIGKILL escalation."""
        if not pid or pid <= 0:
            return
        try:
            os.killpg(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            try:
                os.kill(pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                return
        await asyncio.sleep(KILL_GRACE_PERIOD_MS / 1000.0)
        try:
            os.killpg(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

    @staticmethod
    def _detect_shell() -> str:
        shell_path = os.environ.get("SHELL", "")
        if shell_path and os.path.exists(shell_path):
            return shell_path
        for c in ("/bin/zsh", "/bin/bash", "/bin/sh"):
            if os.path.exists(c):
                return c
        return "/bin/sh"
