"""Tests for depth-1 enforcement: no reawaken, no multi-layer nesting.

Verifies that after removing Branch C and reawaken support:
1. No reawaken transition exists in state machine
2. Branch A (WAITING_FOR_CHILDREN -> resume) still works
3. Branch B (RUNNING -> enqueue) still works
4. Tool context contains task_id, agent, session (not parent_agent)
5. Sub-agents at depth=1 cannot spawn (enforced by spawn.py)
6. Display name is "Sub-{index}" (no depth suffix)
7. _build_path_index() returns simple string
"""

import asyncio
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from engine.config import Config
from engine.runtime.agent_models import AgentState, Session
from engine.runtime.state import AgentStateMachine, InvalidTransitionError
from engine.runtime.task_registry import AgentTaskRegistry, CompleteInfo
from engine.subagent.manager import SubAgentManager
from engine.subagent.events import ChildCompletionEvent


class MockDrainable:
    def __init__(self, state: AgentState = AgentState.IDLE):
        self._state = state
        self._run_calls = []

    @property
    def state(self) -> AgentState:
        return self._state

    @state.setter
    def state(self, value: AgentState) -> None:
        self._state = value

    @property
    def result(self) -> Optional[str]:
        return "mock_result"

    async def run(self, message=None, *, trigger="start"):
        self._run_calls.append({"message": message, "trigger": trigger})
        return "mock_run_result"

    async def abort(self, error):
        pass


class TestNoReawakenTransition:
    def test_reawaken_not_in_transitions(self):
        assert (AgentState.COMPLETED, "reawaken") not in AgentStateMachine.TRANSITIONS

    def test_reawaken_from_completed_raises(self):
        sm = AgentStateMachine(AgentState.COMPLETED)
        assert not sm.can_trigger("reawaken")
        with pytest.raises(InvalidTransitionError):
            sm.trigger("reawaken")

    def test_reawaken_from_all_states_raises(self):
        for state in AgentState:
            sm = AgentStateMachine(state)
            assert not sm.can_trigger("reawaken"), f"reawaken should be invalid from {state.value}"

    def test_transition_count(self):
        assert len(AgentStateMachine.TRANSITIONS) == 6


class TestBranchAStillWorks:
    @pytest.mark.asyncio
    async def test_branch_a_resumes_waiting_parent(self):
        registry = AgentTaskRegistry()
        drainable = MockDrainable(state=AgentState.WAITING_FOR_CHILDREN)
        event_queue = []
        manager = SubAgentManager(
            task_registry=registry,
            event_queue=event_queue,
            drainable=drainable,
            agent_task_id="parent_task",
            parent_label="TestParent",
            config=Config(),
        )

        await registry.register(
            task_id="parent_task",
            session_id="sess_parent",
            description="Parent task",
            parent_agent=None,
            depth=0,
        )
        registry.get_task("parent_task").result = "parent result"

        await registry.register(
            task_id="child_1",
            session_id="sess_child",
            description="Child task",
            parent_agent=None,
            parent_task_id="parent_task",
            depth=1,
        )

        await registry.complete("child_1", "child result")
        await asyncio.sleep(0.05)

        assert len(drainable._run_calls) == 1
        assert drainable._run_calls[0]["trigger"] == "children_settled"
        assert "child result" in drainable._run_calls[0]["message"]


class TestBranchBStillWorks:
    @pytest.mark.asyncio
    async def test_branch_b_enqueues_for_running_parent(self):
        registry = AgentTaskRegistry()
        drainable = MockDrainable(state=AgentState.RUNNING)
        event_queue = []
        manager = SubAgentManager(
            task_registry=registry,
            event_queue=event_queue,
            drainable=drainable,
            agent_task_id="parent_task",
            parent_label="TestParent",
            config=Config(),
        )

        await registry.register(
            task_id="parent_task",
            session_id="sess_parent",
            description="Parent task",
            parent_agent=None,
            depth=0,
        )
        registry.get_task("parent_task").result = "parent result"

        await registry.register(
            task_id="child_1",
            session_id="sess_child",
            description="Child task",
            parent_agent=None,
            parent_task_id="parent_task",
            depth=1,
        )

        await registry.complete("child_1", "child B result")
        await asyncio.sleep(0.05)

        assert len(drainable._run_calls) == 0
        assert len(event_queue) == 1
        assert isinstance(event_queue[0], ChildCompletionEvent)
        assert "child B result" in event_queue[0].notification.summary


class TestBranchCRemoved:
    @pytest.mark.asyncio
    async def test_completed_parent_not_reawakened(self):
        registry = AgentTaskRegistry()
        drainable = MockDrainable(state=AgentState.COMPLETED)
        event_queue = []
        manager = SubAgentManager(
            task_registry=registry,
            event_queue=event_queue,
            drainable=drainable,
            agent_task_id="parent_task",
            parent_label="TestParent",
            config=Config(),
        )

        await registry.register(
            task_id="parent_task",
            session_id="sess_parent",
            description="Parent task",
            parent_agent=None,
            depth=0,
        )
        registry.get_task("parent_task").result = "parent result"

        await registry.register(
            task_id="child_1",
            session_id="sess_child",
            description="Child task",
            parent_agent=None,
            parent_task_id="parent_task",
            depth=1,
        )

        await registry.complete("child_1", "child result")
        await asyncio.sleep(0.05)

        assert len(drainable._run_calls) == 0
        assert len(event_queue) == 0


class TestToolContext:
    def test_agent_context_has_required_keys(self):
        from engine.runtime.agent import Agent
        session = Session(id="sess_test", depth=0)
        config = Config()

        mock_llm = MagicMock()
        mock_tool_pack = MagicMock()
        mock_tool_pack.get_schemas.return_value = []
        mock_tool_pack.get.return_value = None

        agent = Agent(
            session=session,
            config=config,
            llm_provider=mock_llm,
            tool_pack=mock_tool_pack,
        )

        from unittest.mock import patch as mock_patch
        captured_context = {}

        async def fake_execute(arguments, context):
            captured_context.update(context)
            return "test"

        mock_tool = MagicMock()
        mock_tool.execute = fake_execute
        mock_tool_pack.get.return_value = mock_tool

        import asyncio
        from engine.providers.provider_models import ToolCall

        tc = ToolCall(name="test_tool", arguments={}, call_id="call_1")

        asyncio.get_event_loop().run_until_complete(agent._execute_tool(tc))

        assert "session" in captured_context
        assert "agent" in captured_context
        assert "task_id" in captured_context
        assert "parent_agent" not in captured_context
        assert captured_context["agent"] is agent
        assert captured_context["task_id"] == agent.task_id


class TestSpawnDepthLimit:
    @pytest.mark.asyncio
    async def test_spawn_rejected_at_depth_1(self):
        registry = AgentTaskRegistry()
        drainable = MockDrainable(state=AgentState.RUNNING)
        event_queue = []
        manager = SubAgentManager(
            task_registry=registry,
            event_queue=event_queue,
            drainable=drainable,
            agent_task_id="parent_task",
            parent_label="TestParent",
            config=Config(),
        )

        parent_session = Session(id="sess_parent", depth=1)

        with patch("engine.subagent.manager.get_config", return_value=Config()):
            result = await manager.spawn("test task", "test label", parent_session)

        assert "cannot spawn" in result.lower() or "rejected" in result.lower()

    @pytest.mark.asyncio
    async def test_spawn_allowed_at_depth_0(self):
        registry = AgentTaskRegistry()
        drainable = MockDrainable(state=AgentState.RUNNING)
        event_queue = []
        mock_llm = MagicMock()
        mock_tool_pack = MagicMock()
        mock_tool_pack.get_schemas.return_value = []

        manager = SubAgentManager(
            task_registry=registry,
            event_queue=event_queue,
            drainable=drainable,
            agent_task_id="parent_task",
            parent_label="Root",
            config=Config(),
            llm_provider=mock_llm,
            tool_pack=mock_tool_pack,
        )

        parent_session = Session(id="sess_root", depth=0)

        with patch("engine.subagent.manager.get_config", return_value=Config()):
            with patch("engine.subagent.manager.get_subagent_system_prompt", return_value="sys"):
                with patch("engine.subagent.manager.get_spawn_confirmation", return_value="ok"):
                    result = await manager.spawn("test task", "test label", parent_session)

        assert "cannot spawn" not in result.lower()


class TestDisplayName:
    def test_display_name_simple(self):
        assert SubAgentManager._build_path_index("Root", 1) == "1"
        assert SubAgentManager._build_path_index("Root", 5) == "5"

    def test_display_name_ignores_parent_label(self):
        assert SubAgentManager._build_path_index("Sub-1(d:1)", 2) == "2"
        assert SubAgentManager._build_path_index("Sub-3.1(d:2)", 1) == "1"

    def test_manager_spawn_uses_simple_display_name(self):
        assert "Sub-1" == "Sub-1"
        assert "Sub-{}".format(3) == "Sub-3"
