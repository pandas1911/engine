"""Tests for engine/tools/builtin/process.py"""

from __future__ import annotations

import time

import pytest

from engine.tools.builtin._bash.background import BackgroundProcess, ProcessRegistry
from engine.tools.builtin.process import ProcessTool


@pytest.fixture
def registry() -> ProcessRegistry:
    return ProcessRegistry()


@pytest.fixture
def tool(registry: ProcessRegistry) -> ProcessTool:
    return ProcessTool(registry=registry)


def _make_process(
    session_id: str = "test-123",
    command: str = "sleep 60",
    status: str = "running",
    exit_code: int | None = None,
    stdout: str = "",
    stderr: str = "",
) -> BackgroundProcess:
    return BackgroundProcess(
        session_id=session_id,
        pid=12345,
        command=command,
        workdir="/tmp",
        start_time=time.time(),
        status=status,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
    )


CTX: dict = {}


class TestList:
    @pytest.mark.asyncio
    async def test_list_empty(self, tool: ProcessTool) -> None:
        result = await tool.execute({"action": "list"}, CTX)
        assert result == "No background processes."

    @pytest.mark.asyncio
    async def test_list_with_processes(self, tool: ProcessTool, registry: ProcessRegistry) -> None:
        registry.register(_make_process(session_id="abc", command="echo hi"))
        result = await tool.execute({"action": "list"}, CTX)
        assert "abc" in result
        assert "echo hi" in result
        assert "running" in result


class TestPoll:
    @pytest.mark.asyncio
    async def test_poll_running_process(self, tool: ProcessTool, registry: ProcessRegistry) -> None:
        registry.register(_make_process(session_id="p1", status="running"))
        result = await tool.execute({"action": "poll", "session_id": "p1"}, CTX)
        assert "p1" in result
        assert "running" in result
        assert "sleep 60" in result

    @pytest.mark.asyncio
    async def test_poll_completed_process(self, tool: ProcessTool, registry: ProcessRegistry) -> None:
        registry.register(_make_process(session_id="p2", status="completed", exit_code=0))
        result = await tool.execute({"action": "poll", "session_id": "p2"}, CTX)
        assert "Exit code: 0" in result
        assert "completed" in result

    @pytest.mark.asyncio
    async def test_poll_missing_session(self, tool: ProcessTool) -> None:
        result = await tool.execute({"action": "poll", "session_id": "nope"}, CTX)
        assert "Error" in result
        assert "nope" in result

    @pytest.mark.asyncio
    async def test_poll_empty_session_id(self, tool: ProcessTool) -> None:
        result = await tool.execute({"action": "poll"}, CTX)
        assert "Error" in result
        assert "session_id is required" in result


class TestLog:
    @pytest.mark.asyncio
    async def test_log_process_output(self, tool: ProcessTool, registry: ProcessRegistry) -> None:
        registry.register(
            _make_process(session_id="lg1", status="completed", stdout="hello out", stderr="err msg")
        )
        result = await tool.execute({"action": "log", "session_id": "lg1"}, CTX)
        assert "hello out" in result
        assert "[stderr]" in result
        assert "err msg" in result

    @pytest.mark.asyncio
    async def test_log_no_output_running(self, tool: ProcessTool, registry: ProcessRegistry) -> None:
        registry.register(_make_process(session_id="lg2", status="running"))
        result = await tool.execute({"action": "log", "session_id": "lg2"}, CTX)
        assert "No output available yet" in result

    @pytest.mark.asyncio
    async def test_log_no_output_completed(self, tool: ProcessTool, registry: ProcessRegistry) -> None:
        registry.register(_make_process(session_id="lg3", status="completed"))
        result = await tool.execute({"action": "log", "session_id": "lg3"}, CTX)
        assert "no output" in result.lower()

    @pytest.mark.asyncio
    async def test_log_missing_session(self, tool: ProcessTool) -> None:
        result = await tool.execute({"action": "log", "session_id": "ghost"}, CTX)
        assert "Error" in result
        assert "ghost" in result


class TestKill:
    @pytest.mark.asyncio
    async def test_kill_running_process(self, tool: ProcessTool, registry: ProcessRegistry) -> None:
        proc = _make_process(session_id="k1", status="running")
        registry.register(proc)
        result = await tool.execute({"action": "kill", "session_id": "k1"}, CTX)
        assert "killed" in result
        assert proc.status == "killed"
        assert proc.exit_code == -9

    @pytest.mark.asyncio
    async def test_kill_already_completed(self, tool: ProcessTool, registry: ProcessRegistry) -> None:
        registry.register(_make_process(session_id="k2", status="completed", exit_code=0))
        result = await tool.execute({"action": "kill", "session_id": "k2"}, CTX)
        assert "already completed" in result

    @pytest.mark.asyncio
    async def test_kill_missing_session(self, tool: ProcessTool) -> None:
        result = await tool.execute({"action": "kill", "session_id": "nope"}, CTX)
        assert "Error" in result
        assert "nope" in result

    @pytest.mark.asyncio
    async def test_kill_empty_session_id(self, tool: ProcessTool) -> None:
        result = await tool.execute({"action": "kill"}, CTX)
        assert "Error" in result
        assert "session_id is required" in result


class TestMisc:
    @pytest.mark.asyncio
    async def test_unknown_action(self, tool: ProcessTool) -> None:
        result = await tool.execute({"action": "invalid"}, CTX)
        assert "Error" in result
        assert "Unknown action" in result

    def test_tool_name_is_process(self, tool: ProcessTool) -> None:
        assert tool.name == "process"

    def test_tool_has_parameters(self, tool: ProcessTool) -> None:
        assert "action" in tool.parameters.get("properties", {})
