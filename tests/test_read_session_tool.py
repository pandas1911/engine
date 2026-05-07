"""Tests for ReadSessionTool."""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from unittest.mock import MagicMock

import pytest

from engine.runtime.agent_models import Message, Session
from engine.tools.builtin.read_session import ReadSessionTool


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_context(
    task_id: str = "parent_001",
    agent=None,
    parent_agent=None,
):
    ctx = {"task_id": task_id}
    if agent is not None:
        ctx["agent"] = agent
    if parent_agent is not None:
        ctx["parent_agent"] = parent_agent
    return ctx


def _make_messages(*pairs: tuple) -> List[Message]:
    msgs = []
    for role, content in pairs:
        msgs.append(Message(role=role, content=content))
    return msgs


def _mock_agent_with_live_session(messages: List[Message], task_id: str):
    """Create a mock agent whose task_registry returns a child with live session."""
    child_session = Session(id="sess_child", messages=messages)
    child_agent = MagicMock()
    child_agent.session = child_session

    child_task = MagicMock()
    child_task.agent = child_agent
    child_task.task_id = task_id

    task_registry = MagicMock()
    task_registry.get_task.return_value = child_task

    agent = MagicMock()
    agent.task_registry = task_registry
    return agent


def _mock_agent_with_session_store(messages: List[Message], task_id: str):
    """Create a mock agent where child has no live agent, but session_store has data."""
    child_task = MagicMock()
    child_task.agent = None
    child_task.task_id = task_id

    task_registry = MagicMock()
    task_registry.get_task.return_value = child_task

    stored_session = Session(id="sess_stored", messages=messages)
    session_store = MagicMock()
    session_store.read_child_session.return_value = stored_session

    agent = MagicMock()
    agent.task_registry = task_registry
    agent.session_store = session_store
    return agent


tool = ReadSessionTool()


class TestScopeFull:
    def test_returns_all_messages_excluding_thinking(self):
        messages = _make_messages(
            ("user", "Hello"),
            ("assistant", "Hi there"),
            ("user", "How are you?"),
            ("assistant", "I'm fine"),
        )
        agent = _mock_agent_with_live_session(messages, "child_001")
        ctx = _make_context(agent=agent)

        result = _run(tool.execute(
            {"task_id": "child_001", "scope": "full"}, ctx
        ))

        assert "[user] Hello" in result
        assert "[assistant] Hi there" in result
        assert "[user] How are you?" in result
        assert "[assistant] I'm fine" in result


class TestScopeSummary:
    def test_returns_last_assistant_message(self):
        messages = _make_messages(
            ("user", "Hello"),
            ("assistant", "Hi there"),
            ("user", "How are you?"),
            ("assistant", "Final answer here"),
        )
        agent = _mock_agent_with_live_session(messages, "child_002")
        ctx = _make_context(agent=agent)

        result = _run(tool.execute(
            {"task_id": "child_002", "scope": "summary"}, ctx
        ))

        assert result == "Final answer here"
        assert "Hi there" not in result


class TestScopeLastN:
    def test_returns_n_most_recent_messages(self):
        msgs = _make_messages(
            ("user", "m1"), ("assistant", "m2"),
            ("user", "m3"), ("assistant", "m4"),
            ("user", "m5"), ("assistant", "m6"),
            ("user", "m7"), ("assistant", "m8"),
        )
        agent = _mock_agent_with_live_session(msgs, "child_003")
        ctx = _make_context(agent=agent)

        result = _run(tool.execute(
            {"task_id": "child_003", "scope": "last_n", "count": 3}, ctx
        ))

        lines = [l for l in result.split("\n") if l.strip()]
        assert len(lines) == 3
        assert "[assistant] m6" in result
        assert "[user] m7" in result
        assert "[assistant] m8" in result


class TestThinkingFiltered:
    def test_reasoning_role_shown_with_tags(self):
        messages = _make_messages(
            ("user", "Hello"),
            ("reasoning", "Let me think..."),
            ("assistant", "Answer"),
        )
        agent = _mock_agent_with_live_session(messages, "child_004")
        ctx = _make_context(agent=agent)

        result = _run(tool.execute(
            {"task_id": "child_004", "scope": "full"}, ctx
        ))

        assert "[thinking] Let me think... [/thinking]" in result
        assert "[user] Hello" in result
        assert "[assistant] Answer" in result

    def test_assistant_think_tag_shown_with_tags(self):
        messages = _make_messages(
            ("user", "Hello"),
            ("assistant", "<think\ninternal reasoning\n</think\nActually..."),
            ("assistant", "Clean answer"),
        )
        agent = _mock_agent_with_live_session(messages, "child_005")
        ctx = _make_context(agent=agent)

        result = _run(tool.execute(
            {"task_id": "child_005", "scope": "full"}, ctx
        ))

        assert "[thinking] <think\ninternal reasoning\n</think\nActually... [/thinking]" in result
        assert "[assistant] Clean answer" in result


class TestSystemMessagesFiltered:
    def test_system_role_excluded(self):
        messages = _make_messages(
            ("system", "You are helpful"),
            ("user", "Hello"),
            ("assistant", "Hi"),
        )
        agent = _mock_agent_with_live_session(messages, "child_006")
        ctx = _make_context(agent=agent)

        result = _run(tool.execute(
            {"task_id": "child_006", "scope": "full"}, ctx
        ))

        assert "You are helpful" not in result
        assert "[system]" not in result
        assert "[user] Hello" in result


class TestRunningChildLiveSession:
    def test_reads_from_live_session(self):
        messages = _make_messages(
            ("user", "Live message"),
            ("assistant", "Live response"),
        )
        agent = _mock_agent_with_live_session(messages, "child_007")
        ctx = _make_context(agent=agent)

        result = _run(tool.execute(
            {"task_id": "child_007", "scope": "full"}, ctx
        ))

        assert "[user] Live message" in result
        assert "[assistant] Live response" in result


class TestCompletedChildPersistedFile:
    def test_reads_from_session_store(self):
        messages = _make_messages(
            ("user", "Stored message"),
            ("assistant", "Stored response"),
        )
        agent = _mock_agent_with_session_store(messages, "child_008")
        ctx = _make_context(agent=agent)

        result = _run(tool.execute(
            {"task_id": "child_008", "scope": "full"}, ctx
        ))

        assert "[user] Stored message" in result
        assert "[assistant] Stored response" in result


class TestNonExistentTaskId:
    def test_returns_error_message(self):
        task_registry = MagicMock()
        task_registry.get_task.return_value = None

        agent = MagicMock()
        agent.task_registry = task_registry
        agent.session_store = None

        ctx = _make_context(agent=agent)

        result = _run(tool.execute(
            {"task_id": "nonexistent", "scope": "full"}, ctx
        ))

        assert "Error" in result
        assert "nonexistent" in result


class TestMissingTaskIdParameter:
    def test_returns_error_when_task_id_missing(self):
        agent = MagicMock()
        ctx = _make_context(agent=agent)

        result = _run(tool.execute(
            {"scope": "full"}, ctx
        ))

        assert "Error" in result
        assert "task_id" in result


class TestBackwardCompatContextKeys:
    def test_works_with_parent_agent_key(self):
        messages = _make_messages(
            ("user", "Hello"),
            ("assistant", "World"),
        )
        agent = _mock_agent_with_live_session(messages, "child_010")
        ctx = _make_context(parent_agent=agent)

        result = _run(tool.execute(
            {"task_id": "child_010", "scope": "full"}, ctx
        ))

        assert "[user] Hello" in result
        assert "[assistant] World" in result
