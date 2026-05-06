"""Integration tests for Branch C re-awaken mechanism in SubAgentManager.

Verifies that the _on_child_complete Branch C path (COMPLETED parent) works
correctly: the completed parent agent is re-awakened via drainable.run()
with trigger="reawaken", depth guards are enforced, and Branches A/B are
unaffected.

All tests use mock drainable objects — no real LLM API calls.
"""

import asyncio
import json
from typing import Optional

import pytest

from engine.config import Config
from engine.runtime.agent_models import AgentState
from engine.runtime.task_registry import AgentTaskRegistry
from engine.subagent.events import ChildCompletionEvent
from engine.subagent.manager import SubAgentManager


# ---------------------------------------------------------------------------
# Mock Drainable
# ---------------------------------------------------------------------------


class MockDrainable:
    """Mock Drainable that tracks run() calls and exposes configurable state."""

    def __init__(self, state: AgentState = AgentState.IDLE):
        self._state = state
        self._run_calls: list[dict] = []
        self._abort_calls: list[Exception] = []

    @property
    def state(self) -> AgentState:
        return self._state

    @state.setter
    def state(self, value: AgentState) -> None:
        self._state = value

    @property
    def result(self) -> Optional[str]:
        return "mock_parent_result"

    async def run(self, message: Optional[str] = None, *, trigger: str = "start") -> str:
        self._run_calls.append({"message": message, "trigger": trigger})
        return "mock_run_result"

    async def abort(self, error: Exception) -> None:
        self._abort_calls.append(error)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def registry() -> AgentTaskRegistry:
    """Fresh registry for each test."""
    return AgentTaskRegistry()


@pytest.fixture
def config() -> Config:
    """Config with generous re-awaken depth for most tests."""
    return Config(max_reawaken_depth=10, max_result_length=4000)


def make_manager(
    registry: AgentTaskRegistry,
    drainable: MockDrainable,
    config: Config,
    agent_task_id: str = "parent_task",
) -> tuple[SubAgentManager, list]:
    """Create a SubAgentManager wired to the mock drainable.

    Returns (manager, event_queue) tuple.
    """
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
    depth: int = 1,
) -> None:
    """Register a parent task in the registry (no parent_task_id)."""
    await registry.register(
        task_id=task_id,
        session_id=f"sess_{task_id}",
        description="Parent task",
        parent_agent=None,
        parent_task_id=None,
        depth=depth,
    )


async def register_child(
    registry: AgentTaskRegistry,
    child_task_id: str,
    parent_task_id: str = "parent_task",
    depth: int = 2,
) -> None:
    """Register a child task in the registry."""
    await registry.register(
        task_id=child_task_id,
        session_id=f"sess_{child_task_id}",
        description="Child task",
        parent_agent=None,
        parent_task_id=parent_task_id,
        depth=depth,
    )


# ---------------------------------------------------------------------------
# Test scenarios
# ---------------------------------------------------------------------------


class TestBranchCReawaken:
    """Tests for Branch C: parent is COMPLETED when child finishes."""

    @pytest.mark.asyncio
    async def test_branch_c_reawakens_completed_parent(self) -> None:
        """When a completed parent's last child finishes, the parent is re-awakened.

        Verify: drainable.run() called with trigger="reawaken".
        """
        registry = AgentTaskRegistry()
        config = Config(max_reawaken_depth=10)
        drainable = MockDrainable(state=AgentState.COMPLETED)
        manager, event_queue = make_manager(registry, drainable, config)

        await register_parent(registry, "parent_task", depth=1)
        registry.get_task("parent_task").result = "parent completed result"

        await register_child(registry, "child_1", "parent_task", depth=2)

        await registry.complete("child_1", "child result data")
        await asyncio.sleep(0.01)

        assert len(drainable._run_calls) == 1
        assert drainable._run_calls[0]["trigger"] == "reawaken"
        assert "child result data" in drainable._run_calls[0]["message"]

    @pytest.mark.asyncio
    async def test_branch_c_reawakens_root_agent(self) -> None:
        """Root agent (depth=0) in COMPLETED state is still re-awakened.

        Depth=0 is a valid target for re-awaken — no exception should occur.
        """
        registry = AgentTaskRegistry()
        config = Config(max_reawaken_depth=10)
        drainable = MockDrainable(state=AgentState.COMPLETED)
        manager, event_queue = make_manager(
            registry, drainable, config, agent_task_id="root_task"
        )

        await register_parent(registry, "root_task", depth=0)
        registry.get_task("root_task").result = "root completed"

        await register_child(registry, "child_root", "root_task", depth=1)

        await registry.complete("child_root", "child of root result")
        await asyncio.sleep(0.01)

        assert len(drainable._run_calls) == 1
        assert drainable._run_calls[0]["trigger"] == "reawaken"

    @pytest.mark.asyncio
    async def test_branch_c_depth_limit_exceeded(self) -> None:
        """When parent_depth > max_reawaken_depth, drainable.run() is NOT called.

        Verify that the depth guard blocks the re-awaken and logs a warning.
        """
        registry = AgentTaskRegistry()
        config = Config(max_reawaken_depth=1)
        drainable = MockDrainable(state=AgentState.COMPLETED)
        manager, event_queue = make_manager(registry, drainable, config)

        await register_parent(registry, "deep_parent", depth=2)
        registry.get_task("deep_parent").result = "deep result"

        await register_child(registry, "deep_child", "deep_parent", depth=3)

        await registry.complete("deep_child", "child result")
        await asyncio.sleep(0.01)

        assert len(drainable._run_calls) == 0

    @pytest.mark.asyncio
    async def test_branch_c_no_parent_result_returns_early(self) -> None:
        """When parent_task.result is None, Branch C returns early without re-awaken.

        The guard at line 525 checks parent_task.result is truthy.
        """
        registry = AgentTaskRegistry()
        config = Config(max_reawaken_depth=10)
        drainable = MockDrainable(state=AgentState.COMPLETED)
        manager, event_queue = make_manager(registry, drainable, config)

        await register_parent(registry, "no_result_parent", depth=1)

        await register_child(registry, "child_no_result", "no_result_parent", depth=2)

        await registry.complete("child_no_result", "child output")
        await asyncio.sleep(0.01)

        assert len(drainable._run_calls) == 0

    @pytest.mark.asyncio
    async def test_branch_a_unaffected(self) -> None:
        """Parent in WAITING_FOR_CHILDREN state triggers Branch A, not Branch C.

        Verify: drainable.run() called with trigger="children_settled" (NOT "reawaken").
        """
        registry = AgentTaskRegistry()
        config = Config(max_reawaken_depth=10)
        drainable = MockDrainable(state=AgentState.WAITING_FOR_CHILDREN)
        manager, event_queue = make_manager(
            registry, drainable, config, agent_task_id="waiting_parent"
        )

        await register_parent(registry, "waiting_parent", depth=1)
        registry.get_task("waiting_parent").result = "some result"

        await register_child(registry, "child_a", "waiting_parent", depth=2)

        await registry.complete("child_a", "child A result")
        await asyncio.sleep(0.01)

        assert len(drainable._run_calls) == 1
        assert drainable._run_calls[0]["trigger"] == "children_settled"
        assert drainable._run_calls[0]["trigger"] != "reawaken"

    @pytest.mark.asyncio
    async def test_branch_b_unaffected(self) -> None:
        """Parent in RUNNING state triggers Branch B (enqueue), not Branch C.

        Verify: a ChildCompletionEvent is appended to event_queue,
        drainable.run() is NOT called.
        """
        registry = AgentTaskRegistry()
        config = Config(max_reawaken_depth=10)
        drainable = MockDrainable(state=AgentState.RUNNING)
        manager, event_queue = make_manager(
            registry, drainable, config, agent_task_id="running_parent"
        )

        await register_parent(registry, "running_parent", depth=1)
        registry.get_task("running_parent").result = "running result"

        await register_child(registry, "child_b", "running_parent", depth=2)

        await registry.complete("child_b", "child B result")
        await asyncio.sleep(0.01)

        assert len(drainable._run_calls) == 0
        assert len(event_queue) == 1
        assert isinstance(event_queue[0], ChildCompletionEvent)
        assert "child B result" in event_queue[0].formatted_prompt

    @pytest.mark.asyncio
    async def test_branch_c_child_results_passed_to_reawaken(self) -> None:
        """The message passed to drainable.run() contains the child's result content.

        Verify that _format_child_results() correctly formats and the formatted
        prompt includes child task_id and result text.
        """
        registry = AgentTaskRegistry()
        config = Config(max_reawaken_depth=10, max_result_length=4000)
        drainable = MockDrainable(state=AgentState.COMPLETED)
        manager, event_queue = make_manager(
            registry, drainable, config, agent_task_id="result_parent"
        )

        await register_parent(registry, "result_parent", depth=1)
        registry.get_task("result_parent").result = "parent initial result"

        await register_child(
            registry, "child_result_test", "result_parent", depth=2
        )

        child_result = "Important analysis: the data shows a clear trend"
        await registry.complete("child_result_test", child_result)
        await asyncio.sleep(0.01)

        assert len(drainable._run_calls) == 1
        msg = drainable._run_calls[0]["message"]
        assert msg is not None
        assert "Important analysis: the data shows a clear trend" in msg
        assert "child_result_test" in msg

    @pytest.mark.asyncio
    async def test_branch_c_multiple_children_all_collected(self) -> None:
        """Multiple children complete: only the last one triggers re-awaken (pending_siblings=0).

        When there are multiple children and the first completes, pending_siblings > 0
        blocks the handler. Only when the last child completes (pending_siblings=0)
        does Branch C fire.
        """
        registry = AgentTaskRegistry()
        config = Config(max_reawaken_depth=10)
        drainable = MockDrainable(state=AgentState.COMPLETED)
        manager, event_queue = make_manager(
            registry, drainable, config, agent_task_id="multi_parent"
        )

        await register_parent(registry, "multi_parent", depth=1)
        registry.get_task("multi_parent").result = "multi parent result"

        await register_child(registry, "multi_child_1", "multi_parent", depth=2)
        await register_child(registry, "multi_child_2", "multi_parent", depth=2)

        # First child: pending_siblings=1, blocked by Gate 3
        await registry.complete("multi_child_1", "first result")
        await asyncio.sleep(0.01)
        assert len(drainable._run_calls) == 0

        # Last child: pending_siblings=0, Branch C fires
        await registry.complete("multi_child_2", "second result")
        await asyncio.sleep(0.01)
        assert len(drainable._run_calls) == 1
        assert drainable._run_calls[0]["trigger"] == "reawaken"

    @pytest.mark.asyncio
    async def test_branch_c_exact_depth_boundary(self) -> None:
        """Parent at depth == max_reawaken_depth is allowed (not blocked).

        The guard is `parent_depth > max_reawaken_depth`, so equal depth passes.
        """
        registry = AgentTaskRegistry()
        config = Config(max_reawaken_depth=3)
        drainable = MockDrainable(state=AgentState.COMPLETED)
        manager, event_queue = make_manager(
            registry, drainable, config, agent_task_id="boundary_parent"
        )

        await register_parent(registry, "boundary_parent", depth=3)
        registry.get_task("boundary_parent").result = "boundary result"

        await register_child(registry, "boundary_child", "boundary_parent", depth=4)

        await registry.complete("boundary_child", "boundary child result")
        await asyncio.sleep(0.01)

        # depth=3 == max_reawaken_depth=3: passes the guard
        assert len(drainable._run_calls) == 1
        assert drainable._run_calls[0]["trigger"] == "reawaken"

    @pytest.mark.asyncio
    async def test_branch_c_empty_child_result_still_reawakens(self) -> None:
        """Even with an empty string child result, Branch C still re-awakens.

        The result exists (is not None), so the flow should proceed.
        """
        registry = AgentTaskRegistry()
        config = Config(max_reawaken_depth=10)
        drainable = MockDrainable(state=AgentState.COMPLETED)
        manager, event_queue = make_manager(
            registry, drainable, config, agent_task_id="empty_parent"
        )

        await register_parent(registry, "empty_parent", depth=1)
        registry.get_task("empty_parent").result = "parent result"

        await register_child(registry, "empty_child", "empty_parent", depth=2)

        await registry.complete("empty_child", "")
        await asyncio.sleep(0.01)

        assert len(drainable._run_calls) == 1
        assert drainable._run_calls[0]["trigger"] == "reawaken"

    @pytest.mark.asyncio
    async def test_branch_c_no_config_uses_default_depth(self) -> None:
        """When config is None, default max_reawaken_depth=3 is used.

        Verify that depth=4 parent is blocked (4 > 3 default).
        """
        registry = AgentTaskRegistry()
        drainable = MockDrainable(state=AgentState.COMPLETED)
        event_queue: list = []
        manager = SubAgentManager(
            task_registry=registry,
            event_queue=event_queue,
            drainable=drainable,
            agent_task_id="no_config_parent",
            parent_label="NoConfigParent",
            config=None,
        )

        await register_parent(registry, "no_config_parent", depth=4)
        registry.get_task("no_config_parent").result = "no config result"

        await register_child(registry, "no_config_child", "no_config_parent", depth=5)

        await registry.complete("no_config_child", "child output")
        await asyncio.sleep(0.01)

        # depth=4 > default 3: blocked
        assert len(drainable._run_calls) == 0
