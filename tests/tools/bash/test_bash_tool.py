"""Tests for engine/tools/builtin/bash.py (BashTool) and process.py (ProcessTool)."""

from __future__ import annotations

import pytest

from engine.tools.builtin import BUILTIN_TOOLS
from engine.tools.builtin.bash import BashTool
from engine.tools.builtin.process import ProcessTool


class TestBashToolRegistration:
    def test_bash_tool_name(self) -> None:
        assert BashTool().name == "bash"

    def test_process_tool_name(self) -> None:
        assert ProcessTool().name == "process"

    def test_builtin_tools_includes_bash_and_process(self) -> None:
        names = [cls().name for cls in BUILTIN_TOOLS]
        assert "bash" in names
        assert "process" in names

    def test_tool_has_required_parameters(self) -> None:
        tool = BashTool()
        assert "command" in tool.parameters.get("required", [])
        assert "description" in tool.parameters.get("required", [])


class TestBashToolExecution:
    @pytest.mark.asyncio
    async def test_allowed_command_executes(self) -> None:
        tool = BashTool()
        result = await tool.execute(
            {"command": "echo hello", "description": "test echo"}, {},
        )
        assert "hello" in result

    @pytest.mark.asyncio
    async def test_blocked_command_returns_error(self) -> None:
        tool = BashTool()
        result = await tool.execute(
            {"command": "rm -rf /", "description": "dangerous"}, {},
        )
        assert "Error" in result or "blocked" in result.lower()

    @pytest.mark.asyncio
    async def test_empty_command_returns_error(self) -> None:
        tool = BashTool()
        result = await tool.execute(
            {"command": "", "description": "empty"}, {},
        )
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_nonzero_exit_code_captured(self) -> None:
        tool = BashTool()
        result = await tool.execute(
            {"command": "exit 42", "description": "test exit"}, {},
        )
        assert "42" in result

    @pytest.mark.asyncio
    async def test_process_tool_list_empty(self) -> None:
        tool = ProcessTool()
        result = await tool.execute({"action": "list"}, {})
        assert "no background processes" in result.lower()

    @pytest.mark.asyncio
    async def test_whitespace_command_returns_error(self) -> None:
        tool = BashTool()
        result = await tool.execute(
            {"command": "   ", "description": "whitespace"}, {},
        )
        assert "Error" in result
