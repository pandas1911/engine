"""Tests for structured child completion notification pipeline.

Validates _build_child_notification() end-to-end, covering:
- Completed child notification format (all 6 fields)
- Error child notification format
- Child with no assistant messages (only tool calls)
- Child with thinking-only assistant messages
- Child that errors before producing any output (no agent/session)
- to_prompt() output format
- Notification flows through Branch A (WAITING_FOR_CHILDREN)
- Notification flows through Branch B (RUNNING)

All tests use mock objects — no real LLM API calls.
"""

import asyncio
from typing import Optional
from unittest.mock import MagicMock

import pytest

from engine.config import Config
from engine.runtime.agent_models import AgentState, Message, Session
from engine.runtime.task_registry import AgentTaskRegistry, CompleteInfo
from engine.subagent.events import ChildCompletionEvent
from engine.subagent.manager import SubAgentManager
from engine.subagent.subagent_models import ChildCompletionNotification


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def make_mock_agent(
    label: str = "Sub-1",
    state: AgentState = AgentState.COMPLETED,
    final_result: Optional[str] = None,
    messages: Optional[list[Message]] = None,
    session_id: str = "sess_child",
) -> MagicMock:
    """Create a mock Agent with configurable attributes."""
    agent = MagicMock()
    agent.label = label
    agent.state = state
    agent._final_result = final_result

    if messages is not None:
        session = Session(id=session_id)
        session.messages = messages
        agent.session = session
    else:
        agent.session = None

    return agent


async def register_child(
    registry: AgentTaskRegistry,
    child_task_id: str,
    parent_task_id: str = "parent_task",
    result: Optional[str] = "child result",
    description: str = "Task description",
    agent: Optional[MagicMock] = None,
    depth: int = 1,
):
    """Register a child task, optionally attaching a mock agent."""
    await registry.register(
        task_id=child_task_id,
        session_id=f"sess_{child_task_id}",
        description=description,
        parent_task_id=parent_task_id,
        depth=depth,
    )

    child_task = registry.get_task(child_task_id)
    child_task.result = result

    if agent is not None:
        await registry.set_agent(child_task_id, agent)

    return child_task


# ===========================================================================
# 1. Completed child notification format — all 6 fields present and correct
# ===========================================================================


@pytest.mark.asyncio
async def test_completed_child_all_six_fields(registry, config):
    """All 6 ChildCompletionNotification fields are correctly populated."""
    drainable = MockDrainable(state=AgentState.RUNNING)
    manager, event_queue = make_manager(registry, drainable, config)

    await register_parent(registry)

    agent = make_mock_agent(
        label="Sub-3(d:1)",
        state=AgentState.COMPLETED,
        messages=[
            Message(role="user", content="Do something"),
            Message(role="assistant", content="I did it perfectly"),
        ],
    )
    await register_child(
        registry,
        "task_xyz789",
        description="Analyze the dataset",
        result="final output",
        agent=agent,
    )

    child_task = registry.get_task("task_xyz789")
    notif = manager._build_child_notification("task_xyz789", child_task)

    # task_id
    assert notif.task_id == "task_xyz789"
    # label from agent.label
    assert notif.label == "Sub-3(d:1)"
    # task from task_description
    assert notif.task == "Analyze the dataset"
    # status is "completed" (result is not None)
    assert notif.status == "completed"
    # summary is last non-thinking assistant message
    assert notif.summary == "I did it perfectly"
    # session_file is "{task_id}.jsonl"
    assert notif.session_file == "task_xyz789.jsonl"


# ===========================================================================
# 2. Error child notification format
# ===========================================================================


@pytest.mark.asyncio
async def test_error_child_status_and_summary(registry, config):
    """Error child gets status='error' and summary from _final_result."""
    drainable = MockDrainable(state=AgentState.RUNNING)
    manager, _ = make_manager(registry, drainable, config)

    await register_parent(registry)

    agent = make_mock_agent(
        label="Sub-err",
        state=AgentState.ERROR,
        final_result="Something went terribly wrong",
        messages=None,
    )
    await register_child(
        registry,
        "child_err",
        result=None,
        description="Failing task",
        agent=agent,
    )

    child_task = registry.get_task("child_err")
    notif = manager._build_child_notification("child_err", child_task)

    assert notif.status == "error"
    # No session messages → fallback to _final_result
    assert notif.summary == "Something went terribly wrong"
    assert notif.label == "Sub-err"
    assert notif.task == "Failing task"
    assert notif.session_file == "child_err.jsonl"


# ===========================================================================
# 3. Child with no assistant messages (only tool calls)
# ===========================================================================


@pytest.mark.asyncio
async def test_no_assistant_messages_falls_back_to_result(registry, config):
    """When session has only tool/user messages, summary falls back to child_task.result."""
    drainable = MockDrainable(state=AgentState.RUNNING)
    manager, _ = make_manager(registry, drainable, config)

    await register_parent(registry)

    agent = make_mock_agent(
        label="Sub-tools",
        state=AgentState.COMPLETED,
        messages=[
            Message(role="user", content="Do task"),
            Message(role="tool", content="tool output here"),
        ],
    )
    await register_child(
        registry,
        "child_tools",
        result="Fallback result from task",
        description="Tool-heavy task",
        agent=agent,
    )

    child_task = registry.get_task("child_tools")
    notif = manager._build_child_notification("child_tools", child_task)

    assert notif.status == "completed"
    # No assistant messages → summary falls back to child_task.result
    assert notif.summary == "Fallback result from task"


# ===========================================================================
# 4. Child with thinking-only assistant messages
# ===========================================================================


@pytest.mark.asyncio
async def test_thinking_only_assistant_messages_falls_back_to_result(registry, config):
    """When all assistant messages start with <think, summary falls back to result."""
    drainable = MockDrainable(state=AgentState.RUNNING)
    manager, _ = make_manager(registry, drainable, config)

    await register_parent(registry)

    agent = make_mock_agent(
        label="Sub-think",
        state=AgentState.COMPLETED,
        messages=[
            Message(role="user", content="Think hard"),
            Message(role="assistant", content="<think\nLet me reason about this...\n</think"),
            Message(role="assistant", content="<think\nMore thinking...\n</think"),
        ],
    )
    await register_child(
        registry,
        "child_think",
        result="Final result after thinking",
        description="Deep reasoning task",
        agent=agent,
    )

    child_task = registry.get_task("child_think")
    notif = manager._build_child_notification("child_think", child_task)

    assert notif.status == "completed"
    # All assistant messages start with <think → fallback to result
    assert notif.summary == "Final result after thinking"


# ===========================================================================
# 5. Child that errors before producing any output (agent is None)
# ===========================================================================


@pytest.mark.asyncio
async def test_no_agent_session_falls_back_gracefully(registry, config):
    """When child_task.agent is None, notification still builds with defaults."""
    drainable = MockDrainable(state=AgentState.RUNNING)
    manager, _ = make_manager(registry, drainable, config)

    await register_parent(registry)

    # Register child with no agent (agent=None by default)
    await register_child(
        registry,
        "child_no_agent",
        result="Some fallback result",
        description="Task with no agent",
    )

    child_task = registry.get_task("child_no_agent")
    assert child_task.agent is None

    notif = manager._build_child_notification("child_no_agent", child_task)

    # status defaults to "completed" when result is not None
    assert notif.status == "completed"
    # label falls back to task_id when no agent
    assert notif.label == "child_no_agent"
    # summary falls back to child_task.result
    assert notif.summary == "Some fallback result"
    assert notif.task == "Task with no agent"
    assert notif.session_file == "child_no_agent.jsonl"


@pytest.mark.asyncio
async def test_no_agent_no_result_status_completed(registry, config):
    """When agent is None and result is None, status still defaults to 'completed'."""
    drainable = MockDrainable(state=AgentState.RUNNING)
    manager, _ = make_manager(registry, drainable, config)

    await register_parent(registry)

    await register_child(
        registry,
        "child_empty",
        result=None,
        description="Empty task",
    )

    child_task = registry.get_task("child_empty")
    notif = manager._build_child_notification("child_empty", child_task)

    # No result, no agent → else branch defaults to "completed"
    assert notif.status == "completed"
    # summary will be empty string (no agent, no result, no _final_result)
    assert notif.summary == ""


# ===========================================================================
# 6. to_prompt() output format
# ===========================================================================


def test_to_prompt_contains_all_sections():
    """to_prompt() output contains [Child Agent Report], status, task, summary."""
    notif = ChildCompletionNotification(
        task_id="task_abc123",
        label="Sub-1(d:1)",
        task="Analyze the data",
        status="completed",
        summary="Analysis complete with 42 results",
        session_file="task_abc123.json",
    )

    prompt = notif.to_prompt()

    assert "[Child Agent Report]" in prompt
    assert "Sub-1(d:1)" in prompt
    assert "task_abc123" in prompt
    assert "Status: completed" in prompt
    assert "Task: Analyze the data" in prompt
    assert "Summary: Analysis complete with 42 results" in prompt


def test_to_prompt_error_status():
    """to_prompt() correctly formats error status notifications."""
    notif = ChildCompletionNotification(
        task_id="task_err",
        label="Sub-err",
        task="Failing task",
        status="error",
        summary="API rate limit exceeded",
        session_file="task_err.json",
    )

    prompt = notif.to_prompt()

    assert "[Child Agent Report]" in prompt
    assert "Status: error" in prompt
    assert "Summary: API rate limit exceeded" in prompt


def test_to_prompt_with_empty_summary():
    """to_prompt() handles empty summary gracefully."""
    notif = ChildCompletionNotification(
        task_id="task_empty",
        label="Sub-x",
        task="Empty task",
        status="completed",
        summary="",
        session_file="task_empty.json",
    )

    prompt = notif.to_prompt()

    assert "[Child Agent Report]" in prompt
    assert "Summary: " in prompt  # Empty summary still shows the label


def test_to_prompt_with_long_summary():
    """to_prompt() does NOT truncate the summary."""
    long_summary = "A" * 10000
    notif = ChildCompletionNotification(
        task_id="task_long",
        label="Sub-long",
        task="Long task",
        status="completed",
        summary=long_summary,
        session_file="task_long.json",
    )

    prompt = notif.to_prompt()

    assert long_summary in prompt
    assert len(prompt) > 10000


# ===========================================================================
# 7. Notification flows through Branch A correctly
# ===========================================================================


@pytest.mark.asyncio
async def test_branch_a_notification_to_prompt_in_run_call(registry, config):
    """Branch A: parent WAITING_FOR_CHILDREN → run() called with notification.to_prompt()."""
    drainable = MockDrainable(state=AgentState.WAITING_FOR_CHILDREN)
    manager, event_queue = make_manager(registry, drainable, config)

    await register_parent(registry)

    agent = make_mock_agent(
        label="Sub-A",
        state=AgentState.COMPLETED,
        messages=[
            Message(role="assistant", content="Branch A summary"),
        ],
    )
    await register_child(
        registry,
        "child_a",
        result="result_a",
        description="Branch A task",
        agent=agent,
    )

    info = CompleteInfo(parent_task_id="parent_task", pending_children=0, pending_siblings=0)
    await manager._on_child_complete("child_a", info)

    await asyncio.sleep(0)

    # Branch A: run() was called
    assert len(drainable._run_calls) == 1
    call = drainable._run_calls[0]
    assert call["trigger"] == "children_settled"
    assert call["message"] is None

    # Event is always enqueued (even in Branch A)
    assert len(event_queue) == 1
    event = event_queue[0]
    assert isinstance(event, ChildCompletionEvent)
    notif = event.notification
    assert notif.task_id == "child_a"
    assert notif.label == "Sub-A"
    assert notif.status == "completed"
    assert notif.summary == "Branch A summary"


# ===========================================================================
# 8. Notification flows through Branch B correctly
# ===========================================================================


@pytest.mark.asyncio
async def test_branch_b_child_completion_event_with_notification(registry, config):
    """Branch B: parent RUNNING → ChildCompletionEvent enqueued with correct notification."""
    drainable = MockDrainable(state=AgentState.RUNNING)
    manager, event_queue = make_manager(registry, drainable, config)

    await register_parent(registry)

    agent = make_mock_agent(
        label="Sub-B",
        state=AgentState.COMPLETED,
        messages=[
            Message(role="assistant", content="Branch B summary"),
        ],
    )
    await register_child(
        registry,
        "child_b",
        result="result_b",
        description="Branch B task",
        agent=agent,
    )

    info = CompleteInfo(parent_task_id="parent_task", pending_children=0, pending_siblings=0)
    await manager._on_child_complete("child_b", info)

    # Branch B: event enqueued, no run() call
    assert len(drainable._run_calls) == 0
    assert len(event_queue) == 1

    event = event_queue[0]
    assert isinstance(event, ChildCompletionEvent)

    notif = event.notification
    assert isinstance(notif, ChildCompletionNotification)
    assert notif.task_id == "child_b"
    assert notif.label == "Sub-B"
    assert notif.task == "Branch B task"
    assert notif.status == "completed"
    assert notif.summary == "Branch B summary"
    assert notif.session_file == "child_b.jsonl"


# ===========================================================================
# Additional edge cases
# ===========================================================================


@pytest.mark.asyncio
async def test_summary_uses_last_non_thinking_assistant_message(registry, config):
    """Summary picks the LAST (most recent) non-thinking assistant message."""
    drainable = MockDrainable(state=AgentState.RUNNING)
    manager, _ = make_manager(registry, drainable, config)

    await register_parent(registry)

    agent = make_mock_agent(
        label="Sub-multi",
        state=AgentState.COMPLETED,
        messages=[
            Message(role="user", content="Go"),
            Message(role="assistant", content="First response"),
            Message(role="tool", content="tool result"),
            Message(role="assistant", content="<think\nreasoning\n</think"),
            Message(role="assistant", content="Final visible response"),
        ],
    )
    await register_child(
        registry,
        "child_multi",
        result="task result",
        description="Multi-message task",
        agent=agent,
    )

    child_task = registry.get_task("child_multi")
    notif = manager._build_child_notification("child_multi", child_task)

    # Should pick the last non-thinking assistant message
    assert notif.summary == "Final visible response"


@pytest.mark.asyncio
async def test_assistant_message_with_empty_content_skipped(registry, config):
    """Assistant messages with empty/falsy content are skipped."""
    drainable = MockDrainable(state=AgentState.RUNNING)
    manager, _ = make_manager(registry, drainable, config)

    await register_parent(registry)

    agent = make_mock_agent(
        label="Sub-empty",
        state=AgentState.COMPLETED,
        messages=[
            Message(role="user", content="Go"),
            Message(role="assistant", content=""),
            Message(role="assistant", content="   "),
            Message(role="assistant", content="Actual content"),
        ],
    )
    await register_child(
        registry,
        "child_empty_msg",
        result="fallback result",
        description="Empty messages task",
        agent=agent,
    )

    child_task = registry.get_task("child_empty_msg")
    notif = manager._build_child_notification("child_empty_msg", child_task)

    # Empty/whitespace messages are skipped, picks actual content
    assert notif.summary == "Actual content"


@pytest.mark.asyncio
async def test_error_child_no_session_uses_final_result(registry, config):
    """Error child with no session: summary from _final_result."""
    drainable = MockDrainable(state=AgentState.RUNNING)
    manager, _ = make_manager(registry, drainable, config)

    await register_parent(registry)

    agent = make_mock_agent(
        label="Sub-no-session-err",
        state=AgentState.ERROR,
        final_result="RuntimeError: out of memory",
        messages=None,  # No session
    )
    await register_child(
        registry,
        "child_err_no_sess",
        result=None,
        description="Error without session",
        agent=agent,
    )

    child_task = registry.get_task("child_err_no_sess")
    notif = manager._build_child_notification("child_err_no_sess", child_task)

    assert notif.status == "error"
    assert notif.summary == "RuntimeError: out of memory"


@pytest.mark.asyncio
async def test_label_falls_back_to_task_id_when_agent_has_no_label(registry, config):
    """Label falls back to task_id when agent.label is None."""
    drainable = MockDrainable(state=AgentState.RUNNING)
    manager, _ = make_manager(registry, drainable, config)

    await register_parent(registry)

    agent = make_mock_agent(label=None, state=AgentState.COMPLETED)
    await register_child(
        registry,
        "child_no_label",
        result="result",
        description="No label task",
        agent=agent,
    )

    child_task = registry.get_task("child_no_label")
    notif = manager._build_child_notification("child_no_label", child_task)

    # getattr returns None, then `or task_id` kicks in
    assert notif.label == "child_no_label"


@pytest.mark.asyncio
async def test_notification_directly_on_child_completion_branch_b(registry, config):
    """End-to-end: _on_child_complete builds correct notification via Branch B."""
    drainable = MockDrainable(state=AgentState.RUNNING)
    manager, event_queue = make_manager(registry, drainable, config)

    await register_parent(registry)

    agent = make_mock_agent(
        label="Sub-e2e",
        state=AgentState.COMPLETED,
        messages=[
            Message(role="assistant", content="<think\nplanning...\n</think"),
            Message(role="assistant", content="Here is my final answer: 42"),
        ],
    )
    await register_child(
        registry,
        "child_e2e",
        result="42",
        description="End-to-end test task",
        agent=agent,
    )

    info = CompleteInfo(parent_task_id="parent_task", pending_children=0, pending_siblings=0)
    await manager._on_child_complete("child_e2e", info)

    event = event_queue[0]
    notif = event.notification

    # Verify all fields end-to-end
    assert notif.task_id == "child_e2e"
    assert notif.label == "Sub-e2e"
    assert notif.task == "End-to-end test task"
    assert notif.status == "completed"
    assert notif.summary == "Here is my final answer: 42"
    assert notif.session_file == "child_e2e.jsonl"

    # Verify to_prompt() works on the notification that flowed through
    prompt = notif.to_prompt()
    assert "[Child Agent Report]" in prompt
    assert "Sub-e2e" in prompt
    assert "Status: completed" in prompt
    assert "End-to-end test task" in prompt
    assert "Here is my final answer: 42" in prompt
