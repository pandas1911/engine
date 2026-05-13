"""Comprehensive end-to-end integration tests for the bash tool pipeline.

Tests the full foreground flow: AST parse -> security check -> env sanitization
-> execute -> truncate -> format.

Tests the background flow: execute -> auto-background yield -> poll -> kill.

All tests run against REAL subprocess execution -- no mocking of internals.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import time

import pytest

from engine.tools.builtin.bash import BashTool
from engine.tools.builtin.process import ProcessTool
from engine.tools.builtin._bash.background import (
    BackgroundExecutor,
    BackgroundProcess,
    ProcessRegistry,
)

EMPTY_CTX: dict = {}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def registry() -> ProcessRegistry:
    return ProcessRegistry()


@pytest.fixture()
def bash_tool() -> BashTool:
    return BashTool()


@pytest.fixture()
def process_tool(registry: ProcessRegistry) -> ProcessTool:
    return ProcessTool(registry=registry)


@pytest.fixture()
def bash_tool_bg(registry: ProcessRegistry) -> BashTool:
    """BashTool patched so background sessions land in the shared *registry*.

    Default BashTool._execute_background creates an isolated BackgroundExecutor
    each call.  We replace it so the executor uses our fixture registry, giving
    ProcessTool visibility.  yield_ms=100 ensures ``sleep 1`` is reliably
    backgrounded without long waits.
    """
    tool = BashTool()
    _reg = registry

    async def _execute_bg(
        command: str, workdir: str, timeout_ms: int, env: dict,
    ) -> str:
        bg = BackgroundExecutor(registry=_reg)
        result = await bg.execute_background(
            command=command, workdir=workdir, env=env, yield_ms=100,
        )
        if result.backgrounded:
            return (
                f"Command running in background.\n"
                f"Session ID: {result.session_id}\n"
                f"Use 'process' tool to manage."
            )
        if result.direct_result:
            return tool._format_result(result.direct_result, "")
        return "Background execution returned no result."

    tool._execute_background = _execute_bg  # type: ignore[assignment]
    return tool


@pytest.fixture(autouse=True)
def _cleanup_tool_output():
    yield
    d = ".engine/tool-output"
    if os.path.isdir(d):
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_session_id(text: str) -> str:
    m = re.search(r"Session ID:\s*(\S+)", text)
    assert m, f"Could not extract session_id from:\n{text}"
    return m.group(1)


async def _drain_background_tasks() -> None:
    """Wait for background subprocesses to finish, then cancel stragglers.

    BackgroundExecutor.execute_background fires ``asyncio.create_task`` for
    the long-running ``_run_background`` coroutine.  If the subprocess has
    not exited by the time the test ends, the pending task blocks event-loop
    teardown in pytest-asyncio strict mode.

    We wait long enough for ``sleep 1`` subprocesses to complete naturally,
    then cancel anything still running.
    """
    await asyncio.sleep(1.2)
    me = asyncio.current_task()
    remaining = [t for t in asyncio.all_tasks() if t is not me and not t.done()]
    for t in remaining:
        t.cancel()
    if remaining:
        await asyncio.gather(*remaining, return_exceptions=True)


# ===================================================================
# 1. Full foreground pipeline tests
# ===================================================================


class TestForegroundPipeline:

    @pytest.mark.asyncio
    async def test_full_pipeline_safe_command(self, bash_tool: BashTool) -> None:
        result = await bash_tool.execute(
            {"command": "echo hello", "description": "test echo"},
            EMPTY_CTX,
        )
        assert isinstance(result, str)
        assert "hello" in result
        assert not result.startswith("Error:")

    @pytest.mark.asyncio
    async def test_full_pipeline_blocked_command(self, bash_tool: BashTool) -> None:
        result = await bash_tool.execute(
            {"command": "rm -rf /", "description": "dangerous"},
            EMPTY_CTX,
        )
        assert "Error" in result or "blocked" in result.lower()

    @pytest.mark.asyncio
    async def test_full_pipeline_with_env(self, bash_tool: BashTool) -> None:
        result = await bash_tool.execute(
            {
                "command": "echo $MY_TEST_VAR_999",
                "description": "test env var",
                "env": {"MY_TEST_VAR_999": "integration_test_value"},
            },
            EMPTY_CTX,
        )
        assert "integration_test_value" in result

    @pytest.mark.asyncio
    async def test_full_pipeline_env_sanitization(self, bash_tool: BashTool) -> None:
        result = await bash_tool.execute(
            {
                "command": 'echo "NODE=$NODE_OPTIONS SAFE=$MY_SAFE_VAR"',
                "description": "test env sanitization",
                "env": {
                    "NODE_OPTIONS": "evil_injection",
                    "MY_SAFE_VAR": "safe_value_42",
                },
            },
            EMPTY_CTX,
        )
        assert "evil_injection" not in result
        assert "safe_value_42" in result

    @pytest.mark.asyncio
    async def test_full_pipeline_working_directory(self, bash_tool: BashTool) -> None:
        result = await bash_tool.execute(
            {"command": "pwd", "description": "print workdir", "workdir": "/tmp"},
            EMPTY_CTX,
        )
        assert "/tmp" in result

    @pytest.mark.asyncio
    async def test_full_pipeline_stderr(self, bash_tool: BashTool) -> None:
        result = await bash_tool.execute(
            {"command": "echo stderr_marker >&2", "description": "test stderr"},
            EMPTY_CTX,
        )
        assert "[stderr]" in result
        assert "stderr_marker" in result

    @pytest.mark.asyncio
    async def test_full_pipeline_timeout(self, bash_tool: BashTool) -> None:
        result = await bash_tool.execute(
            {"command": "sleep 60", "description": "long sleep", "timeout": 1000},
            EMPTY_CTX,
        )
        assert "timed out" in result.lower()

    @pytest.mark.asyncio
    async def test_output_truncation_integration(self, bash_tool: BashTool) -> None:
        result = await bash_tool.execute(
            {
                "command": 'for i in $(seq 1 5000); do echo "line $i"; done',
                "description": "generate large output",
                "timeout": 30000,
            },
            EMPTY_CTX,
        )
        assert "truncated" in result.lower() or "Full output saved" in result
        assert "line 4999" in result or "line 5000" in result

    @pytest.mark.asyncio
    async def test_piped_command_execution(self, bash_tool: BashTool) -> None:
        result = await bash_tool.execute(
            {"command": "echo hello | grep hello", "description": "test pipe"},
            EMPTY_CTX,
        )
        assert "hello" in result

    @pytest.mark.asyncio
    async def test_nonzero_exit_code(self, bash_tool: BashTool) -> None:
        result = await bash_tool.execute(
            {"command": "exit 1", "description": "test exit code"},
            EMPTY_CTX,
        )
        assert "exit code: 1" in result

    @pytest.mark.asyncio
    async def test_empty_command_error(self, bash_tool: BashTool) -> None:
        result = await bash_tool.execute(
            {"command": "", "description": "empty cmd"},
            EMPTY_CTX,
        )
        assert "Error" in result


# ===================================================================
# 2. Background flow integration tests
# ===================================================================


class TestBackgroundFlow:

    @pytest.mark.asyncio
    async def test_background_flow(
        self,
        bash_tool_bg: BashTool,
        process_tool: ProcessTool,
    ) -> None:
        start = await bash_tool_bg.execute(
            {
                "command": "sleep 1 && echo done",
                "description": "bg test",
                "background": True,
            },
            EMPTY_CTX,
        )
        assert "background" in start.lower() or "session" in start.lower()

        session_id = _extract_session_id(start)

        poll = await process_tool.execute(
            {"action": "poll", "session_id": session_id},
            EMPTY_CTX,
        )
        assert "running" in poll.lower()

        await process_tool.execute(
            {"action": "kill", "session_id": session_id},
            EMPTY_CTX,
        )
        await _drain_background_tasks()

    @pytest.mark.asyncio
    async def test_process_tool_list_empty(self, process_tool: ProcessTool) -> None:
        result = await process_tool.execute({"action": "list"}, EMPTY_CTX)
        assert "no background processes" in result.lower()

    @pytest.mark.asyncio
    async def test_process_tool_kill(
        self,
        bash_tool_bg: BashTool,
        process_tool: ProcessTool,
    ) -> None:
        start = await bash_tool_bg.execute(
            {
                "command": "sleep 1 && echo done",
                "description": "bg kill target",
                "background": True,
            },
            EMPTY_CTX,
        )
        session_id = _extract_session_id(start)

        kill = await process_tool.execute(
            {"action": "kill", "session_id": session_id},
            EMPTY_CTX,
        )
        assert "killed" in kill.lower()

        poll = await process_tool.execute(
            {"action": "poll", "session_id": session_id},
            EMPTY_CTX,
        )
        assert "killed" in poll.lower()
        await _drain_background_tasks()

    @pytest.mark.asyncio
    async def test_process_tool_log_running(
        self,
        bash_tool_bg: BashTool,
        process_tool: ProcessTool,
    ) -> None:
        start = await bash_tool_bg.execute(
            {
                "command": "sleep 1 && echo output_here",
                "description": "bg log test",
                "background": True,
            },
            EMPTY_CTX,
        )
        session_id = _extract_session_id(start)

        log = await process_tool.execute(
            {"action": "log", "session_id": session_id},
            EMPTY_CTX,
        )
        assert "no output available" in log.lower()

        await process_tool.execute(
            {"action": "kill", "session_id": session_id},
            EMPTY_CTX,
        )
        await _drain_background_tasks()

    @pytest.mark.asyncio
    async def test_kill_already_killed(
        self,
        bash_tool_bg: BashTool,
        process_tool: ProcessTool,
    ) -> None:
        start = await bash_tool_bg.execute(
            {
                "command": "sleep 1",
                "description": "bg double-kill",
                "background": True,
            },
            EMPTY_CTX,
        )
        session_id = _extract_session_id(start)

        await process_tool.execute(
            {"action": "kill", "session_id": session_id},
            EMPTY_CTX,
        )
        second = await process_tool.execute(
            {"action": "kill", "session_id": session_id},
            EMPTY_CTX,
        )
        assert "already" in second.lower()
        await _drain_background_tasks()

    @pytest.mark.asyncio
    async def test_poll_unknown_session(self, process_tool: ProcessTool) -> None:
        result = await process_tool.execute(
            {"action": "poll", "session_id": "nonexistent_id"},
            EMPTY_CTX,
        )
        assert "error" in result.lower() or "no process found" in result.lower()
