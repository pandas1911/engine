"""Tests for engine/tools/builtin/_bash/executor.py"""

from __future__ import annotations

import asyncio
import os

import pytest

from engine.tools.builtin._bash.executor import ExecutionResult, ProcessExecutor


@pytest.fixture
def executor() -> ProcessExecutor:
    return ProcessExecutor()


class TestProcessExecutor:
    @pytest.mark.asyncio
    async def test_basic_command_execution(self, executor: ProcessExecutor) -> None:
        result = await executor.execute("echo 'hello world'", workdir="/tmp")
        assert result.exit_code == 0
        assert "hello world" in result.stdout
        assert result.timed_out is False
        assert result.aborted is False

    @pytest.mark.asyncio
    async def test_non_zero_exit_code(self, executor: ProcessExecutor) -> None:
        result = await executor.execute("exit 42", workdir="/tmp")
        assert result.exit_code == 42

    @pytest.mark.asyncio
    async def test_timeout_kills_process(self, executor: ProcessExecutor) -> None:
        result = await executor.execute("sleep 60", workdir="/tmp", timeout_ms=1000)
        assert result.timed_out is True
        assert result.exit_code != 0

    @pytest.mark.asyncio
    async def test_stderr_capture(self, executor: ProcessExecutor) -> None:
        result = await executor.execute("echo error >&2", workdir="/tmp")
        assert "error" in result.stderr

    @pytest.mark.asyncio
    async def test_working_directory_respected(self, executor: ProcessExecutor) -> None:
        result = await executor.execute("pwd", workdir="/tmp")
        assert result.exit_code == 0
        assert "/tmp" in result.stdout

    @pytest.mark.asyncio
    async def test_shell_detection(self) -> None:
        shell = ProcessExecutor._detect_shell()
        assert isinstance(shell, str)
        assert len(shell) > 0
        assert os.path.exists(shell)

    @pytest.mark.asyncio
    async def test_on_output_chunk_callback(self) -> None:
        chunks: list[str] = []
        ex = ProcessExecutor(on_output_chunk=chunks.append)
        result = await ex.execute("echo hello", workdir="/tmp")
        assert result.exit_code == 0
        assert len(chunks) >= 1
        combined = "".join(chunks)
        assert "hello" in combined

    @pytest.mark.asyncio
    async def test_invalid_command(self, executor: ProcessExecutor) -> None:
        result = await executor.execute(
            "this_command_does_not_exist_xyz_12345", workdir="/tmp"
        )
        assert result.exit_code != 0

    @pytest.mark.asyncio
    async def test_output_truncation_integration(self, executor: ProcessExecutor) -> None:
        result = await executor.execute(
            "for i in $(seq 1 5000); do echo \"line $i\"; done",
            workdir="/tmp",
            timeout_ms=30_000,
        )
        assert result.truncated is True

    @pytest.mark.asyncio
    async def test_abort_signal(self, executor: ProcessExecutor) -> None:
        abort_event = asyncio.Event()
        abort_event.set()
        result = await executor.execute(
            "sleep 10", workdir="/tmp", abort_event=abort_event, timeout_ms=30_000
        )
        assert result.aborted is True

    @pytest.mark.asyncio
    async def test_custom_env_variables(self, executor: ProcessExecutor) -> None:
        result = await executor.execute(
            "echo $MY_TEST_VAR_1234",
            workdir="/tmp",
            env={**os.environ, "MY_TEST_VAR_1234": "custom_value"},
        )
        assert result.exit_code == 0
        assert "custom_value" in result.stdout

    @pytest.mark.asyncio
    async def test_execution_result_dataclass_defaults(self) -> None:
        result = ExecutionResult(
            exit_code=0, stdout="", stderr="",
            timed_out=False, aborted=False,
            truncated=False, full_output_path=None,
        )
        assert result.exit_code == 0
        assert result.timed_out is False
        assert result.aborted is False
        assert result.truncated is False
