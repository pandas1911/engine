"""Tests for ListChildrenTool."""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from unittest.mock import MagicMock

import pytest

from engine.tools.builtin.list_children import ListChildrenTool


@dataclass
class FakeSession:
    messages: List[Any] = field(default_factory=list)


@dataclass
class FakeAgent:
    label: Optional[str] = None
    session: Optional[FakeSession] = None
    state: Any = None
    task_registry: Any = None
    session_store: Any = None


@dataclass
class FakeTask:
    task_id: str = ""
    session_id: str = ""
    task_description: str = ""
    parent_agent: Any = None
    parent_task_id: Optional[str] = None
    result: Optional[str] = None
    depth: int = 0
    child_task_ids: Set[str] = field(default_factory=set)
    ended_at: Optional[float] = None
    agent: Any = None


class FakeRegistry:
    def __init__(self, tasks: Optional[Dict[str, FakeTask]] = None):
        self._tasks = tasks or {}

    def get_task(self, task_id: str):
        return self._tasks.get(task_id)


def _run(tool: ListChildrenTool, context: dict) -> str:
    return asyncio.get_event_loop().run_until_complete(
        tool.execute({}, context)
    )


def _make_context(agent: FakeAgent, task_id: str = "parent_task") -> dict:
    return {"agent": agent, "task_id": task_id, "session": MagicMock()}


class TestNoChildren:
    def test_empty_child_ids(self):
        parent_task = FakeTask(task_id="parent_task", child_task_ids=set())
        registry = FakeRegistry({"parent_task": parent_task})
        agent = FakeAgent(task_registry=registry)
        tool = ListChildrenTool()

        result = _run(tool, _make_context(agent))
        assert "No child agents have been spawned yet" in result

    def test_no_agent_in_context(self):
        tool = ListChildrenTool()
        result = _run(tool, {"task_id": "parent_task"})
        assert "No child agents found" in result

    def test_no_task_id_in_context(self):
        tool = ListChildrenTool()
        result = _run(tool, {"agent": FakeAgent()})
        assert "No child agents found" in result

    def test_no_registry_on_agent(self):
        agent = FakeAgent(task_registry=None)
        tool = ListChildrenTool()
        result = _run(tool, _make_context(agent))
        assert "No child agents found" in result

    def test_task_not_in_registry(self):
        registry = FakeRegistry({})
        agent = FakeAgent(task_registry=registry)
        tool = ListChildrenTool()
        result = _run(tool, _make_context(agent))
        assert "No child agents found" in result


class TestStatusClassification:
    def test_completed_child(self):
        child_task = FakeTask(
            task_id="child_1",
            task_description="Do something",
            result="Done!",
        )
        parent_task = FakeTask(
            task_id="parent_task",
            child_task_ids={"child_1"},
        )
        registry = FakeRegistry({"parent_task": parent_task, "child_1": child_task})
        agent = FakeAgent(task_registry=registry)
        tool = ListChildrenTool()

        result = _run(tool, _make_context(agent))
        assert "status=completed" in result
        assert "child_1" in result

    def test_running_child(self):
        child_agent = FakeAgent(state=MagicMock(value="running"))
        child_task = FakeTask(
            task_id="child_2",
            task_description="Running task",
            agent=child_agent,
        )
        parent_task = FakeTask(
            task_id="parent_task",
            child_task_ids={"child_2"},
        )
        registry = FakeRegistry({"parent_task": parent_task, "child_2": child_task})
        agent = FakeAgent(task_registry=registry)
        tool = ListChildrenTool()

        result = _run(tool, _make_context(agent))
        assert "status=running" in result

    def test_error_child(self):
        child_agent = FakeAgent(state=MagicMock(value="error"))
        child_task = FakeTask(
            task_id="child_3",
            task_description="Failing task",
            agent=child_agent,
        )
        parent_task = FakeTask(
            task_id="parent_task",
            child_task_ids={"child_3"},
        )
        registry = FakeRegistry({"parent_task": parent_task, "child_3": child_task})
        agent = FakeAgent(task_registry=registry)
        tool = ListChildrenTool()

        result = _run(tool, _make_context(agent))
        assert "status=error" in result

    def test_unknown_status_no_agent_no_result(self):
        child_task = FakeTask(
            task_id="child_4",
            task_description="Unknown task",
            agent=None,
            result=None,
        )
        parent_task = FakeTask(
            task_id="parent_task",
            child_task_ids={"child_4"},
        )
        registry = FakeRegistry({"parent_task": parent_task, "child_4": child_task})
        agent = FakeAgent(task_registry=registry)
        tool = ListChildrenTool()

        result = _run(tool, _make_context(agent))
        assert "status=unknown" in result


class TestLabelResolution:
    def test_uses_agent_label(self):
        child_agent = FakeAgent(label="Sub-1", state=MagicMock(value="running"))
        child_task = FakeTask(
            task_id="child_5",
            task_description="Labeled task",
            agent=child_agent,
        )
        parent_task = FakeTask(
            task_id="parent_task",
            child_task_ids={"child_5"},
        )
        registry = FakeRegistry({"parent_task": parent_task, "child_5": child_task})
        agent = FakeAgent(task_registry=registry)
        tool = ListChildrenTool()

        result = _run(tool, _make_context(agent))
        assert "[child_5]" in result

    def test_falls_back_to_task_id(self):
        child_task = FakeTask(
            task_id="child_6",
            task_description="No label task",
            agent=None,
        )
        parent_task = FakeTask(
            task_id="parent_task",
            child_task_ids={"child_6"},
        )
        registry = FakeRegistry({"parent_task": parent_task, "child_6": child_task})
        agent = FakeAgent(task_registry=registry)
        tool = ListChildrenTool()

        result = _run(tool, _make_context(agent))
        assert "[child_6]" in result


class TestMessageCount:
    def test_counts_live_session_messages(self):
        child_session = FakeSession(messages=["msg1", "msg2", "msg3"])
        child_agent = FakeAgent(
            label="Sub-2",
            state=MagicMock(value="running"),
            session=child_session,
        )
        child_task = FakeTask(
            task_id="child_7",
            task_description="Count messages",
            agent=child_agent,
        )
        parent_task = FakeTask(
            task_id="parent_task",
            child_task_ids={"child_7"},
        )
        registry = FakeRegistry({"parent_task": parent_task, "child_7": child_task})
        agent = FakeAgent(task_registry=registry)
        tool = ListChildrenTool()

        result = _run(tool, _make_context(agent))
        assert "messages=3" in result

    def test_zero_messages_when_no_agent(self):
        child_task = FakeTask(
            task_id="child_8",
            task_description="No agent",
            agent=None,
        )
        parent_task = FakeTask(
            task_id="parent_task",
            child_task_ids={"child_8"},
        )
        registry = FakeRegistry({"parent_task": parent_task, "child_8": child_task})
        agent = FakeAgent(task_registry=registry)
        tool = ListChildrenTool()

        result = _run(tool, _make_context(agent))
        assert "messages=0" in result


class TestDescriptionTruncation:
    def test_long_description_truncated(self):
        long_desc = "A" * 200
        child_task = FakeTask(
            task_id="child_9",
            task_description=long_desc,
            result="done",
        )
        parent_task = FakeTask(
            task_id="parent_task",
            child_task_ids={"child_9"},
        )
        registry = FakeRegistry({"parent_task": parent_task, "child_9": child_task})
        agent = FakeAgent(task_registry=registry)
        tool = ListChildrenTool()

        result = _run(tool, _make_context(agent))
        assert "..." in result
        assert long_desc not in result

    def test_short_description_not_truncated(self):
        short_desc = "Quick task"
        child_task = FakeTask(
            task_id="child_10",
            task_description=short_desc,
            result="done",
        )
        parent_task = FakeTask(
            task_id="parent_task",
            child_task_ids={"child_10"},
        )
        registry = FakeRegistry({"parent_task": parent_task, "child_10": child_task})
        agent = FakeAgent(task_registry=registry)
        tool = ListChildrenTool()

        result = _run(tool, _make_context(agent))
        assert short_desc in result


class TestBackwardCompat:
    def test_works_with_parent_agent_key(self):
        child_task = FakeTask(
            task_id="child_11",
            task_description="Compat test",
            result="done",
        )
        parent_task = FakeTask(
            task_id="parent_task",
            child_task_ids={"child_11"},
        )
        registry = FakeRegistry({"parent_task": parent_task, "child_11": child_task})
        agent = FakeAgent(task_registry=registry)
        tool = ListChildrenTool()

        context = {"parent_agent": agent, "task_id": "parent_task", "session": MagicMock()}
        result = _run(tool, context)
        assert "status=completed" in result
        assert "child_11" in result


class TestMultipleChildren:
    def test_lists_all_children(self):
        child_a = FakeTask(
            task_id="child_a",
            task_description="Task A",
            result="A done",
        )
        child_b_agent = FakeAgent(label="Sub-B", state=MagicMock(value="running"))
        child_b = FakeTask(
            task_id="child_b",
            task_description="Task B",
            agent=child_b_agent,
        )
        parent_task = FakeTask(
            task_id="parent_task",
            child_task_ids={"child_a", "child_b"},
        )
        registry = FakeRegistry({
            "parent_task": parent_task,
            "child_a": child_a,
            "child_b": child_b,
        })
        agent = FakeAgent(task_registry=registry)
        tool = ListChildrenTool()

        result = _run(tool, _make_context(agent))
        assert "Child agents (2 total):" in result
        assert "child_a" in result
        assert "child_b" in result
        assert "[child_b]" in result


class TestToolMetadata:
    def test_name(self):
        assert ListChildrenTool.name == "list_children"

    def test_no_required_params(self):
        assert ListChildrenTool.parameters["required"] == []

    def test_empty_properties(self):
        assert ListChildrenTool.parameters["properties"] == {}
