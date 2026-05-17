"""Tests for engine/tools/builtin/_bash/background.py"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest

from engine.tools.builtin._bash.background import (
    BackgroundExecutor,
    BackgroundProcess,
    BackgroundProcessInfo,
    BackgroundResult,
    ProcessRegistry,
)
from engine.tools.builtin._bash.executor import ExecutionResult


class TestProcessRegistry:
    def test_registry_crud(self) -> None:
        registry = ProcessRegistry()
        proc = BackgroundProcess(
            session_id="abc123",
            pid=None,
            command="echo hello",
            workdir="/tmp",
            start_time=time.time(),
            status="running",
        )

        sid = registry.register(proc)
        assert sid == "abc123"

        fetched = registry.get("abc123")
        assert fetched is proc

        all_info = registry.list_all()
        assert len(all_info) == 1
        assert all_info[0].session_id == "abc123"
        assert all_info[0].command == "echo hello"
        assert all_info[0].status == "running"

        registry.remove("abc123")
        assert registry.get("abc123") is None
        assert registry.list_all() == []

    def test_registry_cleanup_stale(self) -> None:
        registry = ProcessRegistry()

        old_completed = BackgroundProcess(
            session_id="old1", pid=None, command="true",
            workdir="/tmp", start_time=time.time() - 7200, status="completed",
        )
        old_killed = BackgroundProcess(
            session_id="old2", pid=None, command="true",
            workdir="/tmp", start_time=time.time() - 7200, status="killed",
        )
        fresh_running = BackgroundProcess(
            session_id="fresh1", pid=None, command="sleep 999",
            workdir="/tmp", start_time=time.time(), status="running",
        )
        recent_completed = BackgroundProcess(
            session_id="recent1", pid=None, command="true",
            workdir="/tmp", start_time=time.time() - 60, status="completed",
        )

        registry.register(old_completed)
        registry.register(old_killed)
        registry.register(fresh_running)
        registry.register(recent_completed)

        removed = registry.cleanup_stale(max_age_seconds=3600.0)
        assert removed == 2
        assert registry.get("old1") is None
        assert registry.get("old2") is None
        assert registry.get("fresh1") is not None
        assert registry.get("recent1") is not None

    def test_registry_get_nonexistent(self) -> None:
        registry = ProcessRegistry()
        assert registry.get("nope") is None

    def test_registry_remove_nonexistent(self) -> None:
        registry = ProcessRegistry()
        registry.remove("nope")


class TestBackgroundExecutor:
    @pytest.mark.asyncio
    async def test_short_command_completes_directly(self) -> None:
        executor = BackgroundExecutor()
        result = await executor.execute_background(
            "echo fast", workdir="/tmp", yield_ms=5000,
        )
        assert result.backgrounded is False
        assert result.direct_result is not None
        assert result.direct_result.exit_code == 0
        assert "fast" in result.direct_result.stdout
        assert result.session_id is None

    @pytest.mark.asyncio
    async def test_long_command_auto_backgrounds(self) -> None:
        executor = BackgroundExecutor()

        async def slow_execute(*args, **kwargs):
            await asyncio.sleep(10)
            return ExecutionResult(0, "", "", False, False, False, None)

        async def noop_background(*args, **kwargs):
            pass

        with patch.object(executor._executor, "execute", side_effect=slow_execute), \
             patch.object(executor, "_run_background", side_effect=noop_background):
            result = await executor.execute_background(
                "sleep 999", workdir="/tmp", yield_ms=200,
            )

        assert result.backgrounded is True
        assert result.session_id is not None
        assert len(result.session_id) == 12
        assert result.direct_result is None

        registered = executor.registry.get(result.session_id)
        assert registered is not None
        assert registered.command == "sleep 999"

    @pytest.mark.asyncio
    async def test_custom_registry_shared(self) -> None:
        shared = ProcessRegistry()
        executor = BackgroundExecutor(registry=shared)

        async def slow_execute(*args, **kwargs):
            await asyncio.sleep(10)
            return ExecutionResult(0, "", "", False, False, False, None)

        async def noop_background(*args, **kwargs):
            pass

        with patch.object(executor._executor, "execute", side_effect=slow_execute), \
             patch.object(executor, "_run_background", side_effect=noop_background):
            await executor.execute_background(
                "sleep 999", workdir="/tmp", yield_ms=100,
            )

        assert len(shared.list_all()) == 1


class TestDataclasses:
    def test_background_result_dataclass(self) -> None:
        result = BackgroundResult(backgrounded=True, session_id="abc")
        assert result.backgrounded is True
        assert result.session_id == "abc"
        assert result.direct_result is None

        result2 = BackgroundResult(backgrounded=False)
        assert result2.backgrounded is False
        assert result2.session_id is None

    def test_process_info_dataclass(self) -> None:
        info = BackgroundProcessInfo(
            session_id="xyz", command="ls", status="completed",
            start_time=1234.5, exit_code=0,
        )
        assert info.session_id == "xyz"
        assert info.command == "ls"
        assert info.status == "completed"
        assert info.start_time == 1234.5
        assert info.exit_code == 0

    def test_background_process_dataclass(self) -> None:
        proc = BackgroundProcess(
            session_id="abc", pid=123, command="echo hi",
            workdir="/tmp", start_time=100.0, status="running",
        )
        assert proc.stdout == ""
        assert proc.stderr == ""
        assert proc.exit_code is None
        assert proc._stdout_chunks == []
        assert proc._stderr_chunks == []
