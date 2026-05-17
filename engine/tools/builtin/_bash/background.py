"""Background execution with auto-background yield and process registry."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from engine.tools.builtin._bash.executor import ExecutionResult, ProcessExecutor
from engine.tools.builtin._bash.schemas import YIELD_THRESHOLD_MS


@dataclass
class BackgroundProcess:
    session_id: str
    pid: Optional[int]
    command: str
    workdir: str
    start_time: float
    status: str  # "running", "completed", "killed", "timeout"
    process: Optional[asyncio.subprocess.Process] = None
    exit_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    _stdout_chunks: list[str] = field(default_factory=list)
    _stderr_chunks: list[str] = field(default_factory=list)


@dataclass
class BackgroundProcessInfo:
    session_id: str
    command: str
    status: str
    start_time: float
    exit_code: Optional[int]


@dataclass
class BackgroundResult:
    backgrounded: bool
    session_id: Optional[str] = None
    direct_result: Optional[ExecutionResult] = None


class ProcessRegistry:
    """In-memory registry for background processes."""

    def __init__(self) -> None:
        self._processes: Dict[str, BackgroundProcess] = {}

    def register(self, process: BackgroundProcess) -> str:
        self._processes[process.session_id] = process
        return process.session_id

    def get(self, session_id: str) -> Optional[BackgroundProcess]:
        return self._processes.get(session_id)

    def list_all(self) -> List[BackgroundProcessInfo]:
        result = []
        for p in self._processes.values():
            result.append(BackgroundProcessInfo(
                session_id=p.session_id,
                command=p.command,
                status=p.status,
                start_time=p.start_time,
                exit_code=p.exit_code,
            ))
        return result

    def remove(self, session_id: str) -> None:
        self._processes.pop(session_id, None)

    def cleanup_stale(self, max_age_seconds: float = 3600.0) -> int:
        """Remove completed/killed/timeout processes older than max_age_seconds."""
        now = time.time()
        to_remove = []
        for sid, proc in self._processes.items():
            if proc.status in ("completed", "killed", "timeout"):
                if now - proc.start_time > max_age_seconds:
                    to_remove.append(sid)
        for sid in to_remove:
            del self._processes[sid]
        return len(to_remove)


class BackgroundExecutor:
    """Execute commands with auto-background after yield threshold."""

    def __init__(self, registry: Optional[ProcessRegistry] = None) -> None:
        self._registry = registry or ProcessRegistry()
        self._executor = ProcessExecutor()

    @property
    def registry(self) -> ProcessRegistry:
        return self._registry

    async def execute_background(
        self,
        command: str,
        workdir: str = "/tmp",
        env: Optional[Dict[str, str]] = None,
        shell: Optional[str] = None,
        yield_ms: int = YIELD_THRESHOLD_MS,
    ) -> BackgroundResult:
        """Execute with auto-background: if command completes within yield_ms,
        return result directly; otherwise background it."""
        session_id = uuid.uuid4().hex[:12]
        bg_process = BackgroundProcess(
            session_id=session_id,
            pid=None,
            command=command,
            workdir=workdir,
            start_time=time.time(),
            status="running",
        )

        try:
            result = await asyncio.wait_for(
                self._run_and_capture(bg_process, command, workdir, env, shell),
                timeout=yield_ms / 1000.0,
            )
            return BackgroundResult(backgrounded=False, direct_result=result)
        except asyncio.TimeoutError:
            self._registry.register(bg_process)
            asyncio.create_task(
                self._run_background(bg_process, command, workdir, env, shell)
            )
            return BackgroundResult(backgrounded=True, session_id=session_id)

    async def _run_and_capture(
        self,
        bg_process: BackgroundProcess,
        command: str,
        workdir: str,
        env: Optional[Dict[str, str]],
        shell: Optional[str],
    ) -> ExecutionResult:
        """Run command and update bg_process state. Used for yield-race."""
        result = await self._executor.execute(
            command=command, workdir=workdir, env=env, shell=shell,
        )
        bg_process.exit_code = result.exit_code
        bg_process.stdout = result.stdout
        bg_process.stderr = result.stderr
        if result.timed_out:
            bg_process.status = "timeout"
        else:
            bg_process.status = "completed"
        return result

    async def _run_background(
        self,
        bg_process: BackgroundProcess,
        command: str,
        workdir: str,
        env: Optional[Dict[str, str]],
        shell: Optional[str],
    ) -> None:
        """Run command in background and update registry when done."""
        result = await self._executor.execute(
            command=command, workdir=workdir, env=env, shell=shell,
        )
        bg_process.exit_code = result.exit_code
        bg_process.stdout = result.stdout
        bg_process.stderr = result.stderr
        if result.timed_out:
            bg_process.status = "timeout"
        else:
            bg_process.status = "completed"
