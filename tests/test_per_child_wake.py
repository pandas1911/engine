"""Tests for per-child immediate wake in SubAgentManager._on_child_complete().

Verifies the simplified 1-gate-2-branch model:
- Gate: only checks parent existence (no sibling/children gates)
- Branch A: single notification resumes WAITING_FOR_CHILDREN parent
- Branch B: single notification enqueued for RUNNING parent
- No collect_and_cleanup called (child tasks remain in registry)
- Notification contains correct fields

All tests use mock drainable objects — no real LLM API calls.
"""

import asyncio
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from engine.config import Config
from engine.runtime.agent_models import AgentState, Message, Session
from engine.runtime.task_registry import AgentTaskRegistry, CompleteInfo
from engine.subagent.events import ChildCompletionEvent
from engine.subagent.manager import SubAgentManager
from engine.subagent.subagent_models import AgentTask, ChildCompletionNotification


class MockDrainable:
    """Mock Drainable that tracks run() calls and exposes configurable state."""

    def __init__(self, state: AgentState = AgentState.IDLE):
        self._state = state
        self._run_calls: list[dict] = []

    @property
    def state(self) -> AgentState:
        return self._state

    @state.setter
    def state(self, value: AgentState) -> None:
        self._state = value

    async def run(self, message: Optional[str] = None, *, trigger: str = "start") -> str:
        self._run_calls.append({"message": message, "trigger": trigger})
        return "mock_run_result"


@pytest.fixture
def registry():
    return AgentTaskRegistry()


@pytest.fixture
def config():
    return Config(max_result_length=4000)


def make_manager(
    registry: AgentTaskRegistry,
    drainable: MockDrainable,
    config: Config,
    agent_task_id: str = "parent_task",
):
    event_queue: list = []
    manager = SubAgentManager(
        task_registry=registry,
        event_queue=event_queue,
        drainable=drainable,
        agent_task_id=agent_task_id,
        parent_label="TestParent",
        config=config,
    )
    return manager, event_queue


async def register_parent(
    registry: AgentTaskRegistry,
    task_id: str = "parent_task",
):
    await registry.register(
        task_id=task_id,
        session_id=f"sess_{task_id}",
        description="Parent task",
        parent_task_id=None,
        depth=0,
    )


async def register_child_with_agent(
    registry: AgentTaskRegistry,
    child_task_id: str,
    parent_task_id: str = "parent_task",
    result: Optional[str] = "child result",
    agent_state: AgentState = AgentState.COMPLETED,
    agent_label: str = "Sub-1",
    agent_summary: Optional[str] = None,
):
    """Register a child task and set a mock agent on it."""
    await registry.register(
        task_id=child_task_id,
        session_id=f"sess_{child_task_id}",
        description=f"Task for {child_task_id}",
        parent_task_id=parent_task_id,
        depth=1,
    )

    child_task = registry.get_task(child_task_id)
    child_task.result = result

    mock_agent = MagicMock()
    mock_agent.label = agent_label
    mock_agent.state = agent_state
    mock_agent._final_result = None

    if agent_summary:
        session = Session(id=f"sess_{child_task_id}")
        session.messages.append(Message(role="assistant", content=agent_summary))
        mock_agent.session = session
    else:
        mock_agent.session = None

    await registry.set_agent(child_task_id, mock_agent)
    return child_task


@pytest.mark.asyncio
async def test_branch_a_waiting_parent_resumes_with_single_notification(registry, config):
    """Branch A: child completes while parent is WAITING_FOR_CHILDREN → parent resumes."""
    drainable = MockDrainable(state=AgentState.WAITING_FOR_CHILDREN)
    manager, event_queue = make_manager(registry, drainable, config)

    await register_parent(registry)
    await register_child_with_agent(
        registry, "child_1", agent_label="Sub-1", agent_summary="I am done"
    )

    info = CompleteInfo(
        parent_task_id="parent_task",
        pending_children=0,
        pending_siblings=1,
    )

    await manager._on_child_complete("child_1", info)

    await asyncio.sleep(0)

    assert len(drainable._run_calls) == 1
    call = drainable._run_calls[0]
    assert call["trigger"] == "children_settled"
    assert call["message"] is None
    assert len(event_queue) == 1  # always enqueued


@pytest.mark.asyncio
async def test_branch_b_running_parent_enqueues_single_notification(registry, config):
    """Branch B: child completes while parent is RUNNING → notification enqueued."""
    drainable = MockDrainable(state=AgentState.RUNNING)
    manager, event_queue = make_manager(registry, drainable, config)

    await register_parent(registry)
    await register_child_with_agent(
        registry, "child_2", agent_label="Sub-2", agent_summary="Task complete"
    )

    info = CompleteInfo(
        parent_task_id="parent_task",
        pending_children=0,
        pending_siblings=0,
    )

    await manager._on_child_complete("child_2", info)

    assert len(drainable._run_calls) == 0
    assert len(event_queue) == 1
    event = event_queue[0]
    assert isinstance(event, ChildCompletionEvent)
    assert event.notification.task_id == "child_2"
    assert event.notification.status == "completed"
    assert event.notification.label == "Sub-2"
    assert event.notification.summary == "Task complete"


@pytest.mark.asyncio
async def test_multiple_children_wake_parent_independently(registry, config):
    """Multiple children wake parent independently — no sibling gate blocking."""
    drainable = MockDrainable(state=AgentState.RUNNING)
    manager, event_queue = make_manager(registry, drainable, config)

    await register_parent(registry)
    await register_child_with_agent(
        registry, "child_a", agent_label="Sub-1", agent_summary="Alpha done"
    )
    await register_child_with_agent(
        registry, "child_b", agent_label="Sub-2", agent_summary="Beta done"
    )

    info_a = CompleteInfo(parent_task_id="parent_task", pending_children=0, pending_siblings=1)
    info_b = CompleteInfo(parent_task_id="parent_task", pending_children=0, pending_siblings=0)

    await manager._on_child_complete("child_a", info_a)
    await manager._on_child_complete("child_b", info_b)

    assert len(event_queue) == 2
    assert event_queue[0].notification.task_id == "child_a"
    assert event_queue[1].notification.task_id == "child_b"

    assert event_queue[0].notification.summary == "Alpha done"
    assert event_queue[1].notification.summary == "Beta done"


@pytest.mark.asyncio
async def test_no_collect_and_cleanup_child_tasks_remain_in_registry(registry, config):
    """Child tasks remain in registry after notification — no collect_and_cleanup."""
    drainable = MockDrainable(state=AgentState.RUNNING)
    manager, event_queue = make_manager(registry, drainable, config)

    await register_parent(registry)
    await register_child_with_agent(
        registry, "child_x", agent_label="Sub-1", agent_summary="Done"
    )

    info = CompleteInfo(parent_task_id="parent_task", pending_children=0, pending_siblings=0)

    await manager._on_child_complete("child_x", info)

    assert registry.get_task("child_x") is not None, "Child task should remain in registry"
    parent = registry.get_task("parent_task")
    assert "child_x" in parent.child_task_ids, "Child should still be in parent's child_task_ids"


@pytest.mark.asyncio
async def test_notification_contains_correct_fields(registry, config):
    """Notification has correct task_id, label, status, summary, session_file."""
    drainable = MockDrainable(state=AgentState.RUNNING)
    manager, event_queue = make_manager(registry, drainable, config)

    await register_parent(registry)
    await register_child_with_agent(
        registry,
        "task_abc123",
        agent_label="Sub-3",
        result="final output",
        agent_summary="Here is my answer",
    )

    info = CompleteInfo(parent_task_id="parent_task", pending_children=0, pending_siblings=0)
    await manager._on_child_complete("task_abc123", info)

    notif = event_queue[0].notification
    assert notif.task_id == "task_abc123"
    assert notif.label == "Sub-3"
    assert notif.task == "Task for task_abc123"
    assert notif.status == "completed"
    assert notif.summary == "Here is my answer"
    assert notif.session_file == "task_abc123.jsonl"


@pytest.mark.asyncio
async def test_notification_error_status_for_failed_child(registry, config):
    """Error child gets status='error' and summary from _final_result."""
    drainable = MockDrainable(state=AgentState.RUNNING)
    manager, event_queue = make_manager(registry, drainable, config)

    await register_parent(registry)
    await register_child_with_agent(
        registry,
        "child_err",
        result=None,
        agent_state=AgentState.ERROR,
        agent_label="Sub-err",
    )

    mock_agent = registry.get_task("child_err").agent
    mock_agent._final_result = "Something went wrong"

    info = CompleteInfo(parent_task_id="parent_task", pending_children=0, pending_siblings=0)
    await manager._on_child_complete("child_err", info)

    notif = event_queue[0].notification
    assert notif.status == "error"
    assert notif.summary == "Something went wrong"


@pytest.mark.asyncio
async def test_gate_returns_when_parent_not_registered(registry, config):
    """Gate: no parent registered → handler returns without action."""
    drainable = MockDrainable(state=AgentState.WAITING_FOR_CHILDREN)
    manager, event_queue = make_manager(registry, drainable, config)

    await register_child_with_agent(registry, "orphan_child", result="done")

    info = CompleteInfo(parent_task_id=None, pending_children=0, pending_siblings=0)

    await manager._on_child_complete("orphan_child", info)

    assert len(drainable._run_calls) == 0
    assert len(event_queue) == 0


@pytest.mark.asyncio
async def test_gate_returns_when_parent_task_not_found(registry, config):
    """Gate: parent_task_id set but parent task missing → return."""
    drainable = MockDrainable(state=AgentState.RUNNING)
    manager, event_queue = make_manager(registry, drainable, config)

    await register_child_with_agent(registry, "child_no_parent", result="done")

    info = CompleteInfo(parent_task_id="nonexistent_parent", pending_children=0, pending_siblings=0)

    await manager._on_child_complete("child_no_parent", info)

    assert len(event_queue) == 0


@pytest.mark.asyncio
async def test_skip_when_parent_completed(registry, config):
    """Parent in COMPLETED state → no run() resume, but event still enqueued."""
    drainable = MockDrainable(state=AgentState.COMPLETED)
    manager, event_queue = make_manager(registry, drainable, config)

    await register_parent(registry)
    await register_child_with_agent(registry, "child_late", result="late result")

    info = CompleteInfo(parent_task_id="parent_task", pending_children=0, pending_siblings=0)
    await manager._on_child_complete("child_late", info)

    assert len(drainable._run_calls) == 0
    assert len(event_queue) == 1  # event always enqueued regardless of parent state
