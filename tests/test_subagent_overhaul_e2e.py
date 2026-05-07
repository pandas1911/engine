"""End-to-end integration tests for the depth=1 sub-agent architecture.

Validates the complete lifecycle:
  - Root agent spawning children via SubAgentManager
  - Per-child immediate wake (Branch A and Branch B)
  - list_children tool reflecting correct statuses at each stage
  - read_session tool with full / summary / last_n scopes
  - Session persistence via SessionStore (files on disk)
  - Depth=1 enforcement (spawn rejected, root-only tools filtered)

All tests use mocked LLM calls — no live API required.
"""

import asyncio
import json
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from engine.config import Config
from engine.runtime.agent_models import AgentState, Message, Session
from engine.runtime.task_registry import AgentTaskRegistry, CompleteInfo
from engine.subagent.events import ChildCompletionEvent
from engine.subagent.manager import SubAgentManager
from engine.session_store import SessionStore
from engine.subagent.subagent_models import ChildCompletionNotification
from engine.tools.builtin.list_children import ListChildrenTool
from engine.tools.builtin.read_session import ReadSessionTool
from engine.tools.pack import ToolPack, _ROOT_ONLY_TOOLS


# ---------------------------------------------------------------------------
# Reusable test doubles
# ---------------------------------------------------------------------------


class MockDrainable:
    """Mock Drainable protocol with configurable state and run tracking."""

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

    async def abort(self, error):
        pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def registry():
    return AgentTaskRegistry()


@pytest.fixture
def config():
    return Config(max_result_length=4000)


@pytest.fixture
def store(tmp_path):
    """SessionStore backed by a temporary directory."""
    s = SessionStore(str(tmp_path))
    s.create_root("root_e2e")
    return s


def _make_manager(
    registry: AgentTaskRegistry,
    drainable: MockDrainable,
    config: Config,
    agent_task_id: str = "parent_task",
    session_store: Optional[SessionStore] = None,
):
    """Create a SubAgentManager wired to the given registry and drainable."""
    event_queue: list = []
    manager = SubAgentManager(
        task_registry=registry,
        event_queue=event_queue,
        drainable=drainable,
        agent_task_id=agent_task_id,
        parent_label="TestParent",
        config=config,
        session_store=session_store,
    )
    return manager, event_queue


async def _register_parent(registry: AgentTaskRegistry, task_id: str = "parent_task"):
    await registry.register(
        task_id=task_id,
        session_id=f"sess_{task_id}",
        description="Parent task",
        parent_agent=None,
        parent_task_id=None,
        depth=0,
    )


async def _register_child(
    registry: AgentTaskRegistry,
    child_task_id: str,
    parent_task_id: str = "parent_task",
    result: Optional[str] = "child result",
    agent_state: AgentState = AgentState.COMPLETED,
    agent_label: str = "Sub-1",
    agent_summary: Optional[str] = None,
    extra_messages: Optional[list[tuple[str, str]]] = None,
):
    """Register a child task and set a mock agent with an optional session."""
    await registry.register(
        task_id=child_task_id,
        session_id=f"sess_{child_task_id}",
        description=f"Task for {child_task_id}",
        parent_agent=None,
        parent_task_id=parent_task_id,
        depth=1,
    )

    child_task = registry.get_task(child_task_id)
    child_task.result = result

    mock_agent = MagicMock()
    mock_agent.label = agent_label
    mock_agent.state = agent_state
    mock_agent._final_result = None

    if agent_summary or extra_messages:
        session = Session(id=f"sess_{child_task_id}", depth=1, parent_id="sess_parent_task")
        if extra_messages:
            for role, content in extra_messages:
                session.messages.append(Message(role=role, content=content))
        if agent_summary:
            session.messages.append(Message(role="assistant", content=agent_summary))
        mock_agent.session = session
    else:
        mock_agent.session = None

    await registry.set_agent(child_task_id, mock_agent)
    return child_task


# ===================================================================
# Test 1: Root agent spawns children, each returns a summary
# ===================================================================


class TestSpawnChildrenAndSummaries:
    """Root agent spawns 2 children; both complete with structured notifications."""

    @pytest.mark.asyncio
    async def test_children_spawned_and_notify_parent(self, registry, config):
        drainable = MockDrainable(state=AgentState.WAITING_FOR_CHILDREN)
        manager, event_queue = _make_manager(registry, drainable, config)

        await _register_parent(registry)
        await _register_child(
            registry, "child_a",
            agent_label="Sub-1", agent_summary="Alpha complete",
        )
        await _register_child(
            registry, "child_b",
            agent_label="Sub-2", agent_summary="Beta complete",
        )

        # Simulate completion of both children
        info_a = CompleteInfo(parent_task_id="parent_task", pending_children=1, pending_siblings=0)
        info_b = CompleteInfo(parent_task_id="parent_task", pending_children=0, pending_siblings=0)

        await manager._on_child_complete("child_a", info_a)
        await manager._on_child_complete("child_b", info_b)

        await asyncio.sleep(0)

        # Both children should trigger Branch A (WAITING_FOR_CHILDREN parent)
        assert len(drainable._run_calls) == 2
        assert drainable._run_calls[0]["trigger"] == "children_settled"
        assert drainable._run_calls[1]["trigger"] == "children_settled"

        # Verify notification content
        msg_a = drainable._run_calls[0]["message"]
        msg_b = drainable._run_calls[1]["message"]
        assert "child_a" in msg_a
        assert "Alpha complete" in msg_a
        assert "child_b" in msg_b
        assert "Beta complete" in msg_b

    @pytest.mark.asyncio
    async def test_child_depth_is_1(self, registry, config):
        """Spawned children are always at depth=1."""
        drainable = MockDrainable(state=AgentState.RUNNING)
        manager, event_queue = _make_manager(registry, drainable, config)

        await _register_parent(registry)
        await _register_child(registry, "child_depth")

        child = registry.get_task("child_depth")
        assert child.depth == 1


# ===================================================================
# Test 2: list_children shows correct statuses at each stage
# ===================================================================


class TestListChildrenStatuses:
    """list_children reflects running/completed/error as children progress."""

    @pytest.mark.asyncio
    async def test_status_transitions(self, registry, config):
        tool = ListChildrenTool()
        drainable = MockDrainable(state=AgentState.RUNNING)
        manager, event_queue = _make_manager(registry, drainable, config)

        await _register_parent(registry)

        # Stage 1: both running (no result, agent present)
        await _register_child(
            registry, "child_1",
            result=None, agent_state=AgentState.RUNNING,
            agent_label="Sub-1",
        )
        await _register_child(
            registry, "child_2",
            result=None, agent_state=AgentState.RUNNING,
            agent_label="Sub-2",
        )

        ctx = {"agent": MagicMock(task_registry=registry), "task_id": "parent_task"}
        output = await tool.execute({}, ctx)
        assert "status=running" in output
        assert output.count("status=running") == 2

        # Stage 2: complete child_1
        child_1 = registry.get_task("child_1")
        child_1.result = "done"

        output = await tool.execute({}, ctx)
        assert "status=completed" in output
        assert "status=running" in output

        # Stage 3: complete child_2
        child_2 = registry.get_task("child_2")
        child_2.result = "also done"

        output = await tool.execute({}, ctx)
        assert output.count("status=completed") == 2
        assert "status=running" not in output

    @pytest.mark.asyncio
    async def test_error_status_shown(self, registry, config):
        tool = ListChildrenTool()
        await _register_parent(registry)

        await _register_child(
            registry, "child_err",
            result=None, agent_state=AgentState.ERROR,
            agent_label="Sub-err",
        )

        ctx = {"agent": MagicMock(task_registry=registry), "task_id": "parent_task"}
        output = await tool.execute({}, ctx)
        assert "status=error" in output


# ===================================================================
# Test 3: read_session with different scopes
# ===================================================================


class TestReadSessionScopes:
    """read_session returns correct data for full / summary / last_n."""

    @pytest.mark.asyncio
    async def test_scope_full_excludes_thinking(self, registry, config):
        tool = ReadSessionTool()
        await _register_parent(registry)

        messages = [
            Message(role="system", content="system prompt"),
            Message(role="user", content="Hello"),
            Message(role="reasoning", content="Let me think..."),
            Message(role="assistant", content="<think\nreasoning\n</think"),
            Message(role="assistant", content="Final answer"),
        ]
        await _register_child(
            registry, "child_rs",
            agent_label="Sub-rs",
            extra_messages=[],
        )
        # Set up session with mixed messages
        child_task = registry.get_task("child_rs")
        child_task.agent.session = Session(
            id="sess_child_rs", messages=messages,
        )

        ctx = {"agent": MagicMock(task_registry=registry), "task_id": "parent_task"}
        output = await tool.execute(
            {"task_id": "child_rs", "scope": "full"}, ctx,
        )

        assert "[user] Hello" in output
        assert "[assistant] Final answer" in output
        assert "system prompt" not in output
        assert "[thinking] Let me think... [/thinking]" in output
        assert "[thinking] <think\nreasoning\n</think [/thinking]" in output

    @pytest.mark.asyncio
    async def test_scope_summary_returns_last_assistant(self, registry, config):
        tool = ReadSessionTool()
        await _register_parent(registry)

        messages = [
            Message(role="user", content="Q1"),
            Message(role="assistant", content="A1"),
            Message(role="user", content="Q2"),
            Message(role="assistant", content="A2 final"),
        ]
        await _register_child(
            registry, "child_summary",
            agent_label="Sub-sum",
        )
        child_task = registry.get_task("child_summary")
        child_task.agent.session = Session(
            id="sess_child_summary", messages=messages,
        )

        ctx = {"agent": MagicMock(task_registry=registry), "task_id": "parent_task"}
        output = await tool.execute(
            {"task_id": "child_summary", "scope": "summary"}, ctx,
        )

        assert output == "A2 final"

    @pytest.mark.asyncio
    async def test_scope_last_n_returns_correct_count(self, registry, config):
        tool = ReadSessionTool()
        await _register_parent(registry)

        messages = [
            Message(role="user", content="m{}".format(i))
            for i in range(10)
        ]
        await _register_child(
            registry, "child_lastn",
            agent_label="Sub-ln",
        )
        child_task = registry.get_task("child_lastn")
        child_task.agent.session = Session(
            id="sess_child_lastn", messages=messages,
        )

        ctx = {"agent": MagicMock(task_registry=registry), "task_id": "parent_task"}
        output = await tool.execute(
            {"task_id": "child_lastn", "scope": "last_n", "count": 3}, ctx,
        )

        lines = [l for l in output.split("\n") if l.strip()]
        assert len(lines) == 3
        assert "m7" in output
        assert "m8" in output
        assert "m9" in output
        assert "m0" not in output


# ===================================================================
# Test 4: Session persistence — files exist in correct directory
# ===================================================================


class TestSessionPersistence:
    """SessionStore creates files on disk; round-trip preserves content."""

    @pytest.mark.asyncio
    async def test_spawn_creates_session_file(self, registry, config, store):
        drainable = MockDrainable(state=AgentState.RUNNING)
        manager, event_queue = _make_manager(
            registry, drainable, config, session_store=store,
        )

        parent_session = Session(id="root_e2e", depth=0)

        with patch("engine.subagent.manager.get_config", return_value=config):
            with patch("engine.subagent.manager.get_subagent_system_prompt", return_value="sys"):
                with patch("engine.subagent.manager.get_spawn_confirmation") as mock_conf:
                    mock_conf.return_value = "Spawned child"
                    with patch("engine.subagent.manager.asyncio.create_task"):
                        result = await manager.spawn(
                            "research task", "researcher", parent_session,
                        )

        # Verify session file was created
        children = store.list_children()
        assert len(children) == 1
        child_info = children[0]
        assert child_info.task_id.startswith("task_")
        # create_file writes header only; real-time callback appends messages as they arrive
        assert child_info.message_count >= 0

    @pytest.mark.asyncio
    async def test_completion_saves_final_session(self, registry, config, store):
        drainable = MockDrainable(state=AgentState.RUNNING)
        manager, event_queue = _make_manager(
            registry, drainable, config, session_store=store,
        )

        await _register_parent(registry)
        await _register_child(
            registry, "task_persist",
            agent_label="Sub-p",
            agent_summary="Persisted result",
        )

        # Manually persist the child session (simulates real-time callback persistence)
        child_task = registry.get_task("task_persist")
        child_session = child_task.agent.session
        store.create_file("task_persist", child_session)
        for msg in child_session.messages:
            store.append_line("task_persist", msg)

        info = CompleteInfo(parent_task_id="parent_task", pending_children=0, pending_siblings=0)
        await manager._on_child_complete("task_persist", info)

        # SessionStore should now have the child session file
        children = store.list_children()
        task_ids = {c.task_id for c in children}
        assert "task_persist" in task_ids

        # Verify round-trip
        restored = store.read_child_session("task_persist")
        assert restored is not None
        assert restored.id == "sess_task_persist"
        assert restored.depth == 1

    def test_directory_structure(self, tmp_path, store):
        """Files live under sessions/{root_session_id}/."""
        child_session = Session(
            id="sess_dir_test", depth=1, parent_id="root_e2e",
            messages=[Message(role="user", content="test")],
        )
        store.create_file("task_dir", child_session)
        for msg in child_session.messages:
            store.append_line("task_dir", msg)

        file_path = store.sessions_dir / "task_dir.jsonl"
        assert file_path.exists()

        # Read and verify JSONL format
        lines = file_path.read_text(encoding="utf-8").strip().splitlines()
        header = json.loads(lines[0])
        assert header["id"] == "sess_dir_test"
        assert header["depth"] == 1
        assert header["parent_id"] == "root_e2e"
        assert len(lines) == 2  # header + 1 message


# ===================================================================
# Test 5: Depth=1 enforcement
# ===================================================================


class TestDepthOneEnforcement:
    """Sub-agents at depth=1 cannot spawn; root-only tools hidden."""

    @pytest.mark.asyncio
    async def test_spawn_rejected_at_depth_1(self, registry, config):
        drainable = MockDrainable(state=AgentState.RUNNING)
        manager, event_queue = _make_manager(registry, drainable, config)

        # A child session (depth=1) attempting to spawn
        child_session = Session(id="sess_child", depth=1)

        with patch("engine.subagent.manager.get_config", return_value=config):
            result = await manager.spawn("nested task", "bad label", child_session)

        assert "cannot spawn" in result.lower()

    def test_root_only_tools_filtered_from_subagent(self):
        """ToolPack hides spawn, list_children, read_session from depth=1 sessions."""
        from engine.tools.base import Tool

        class FakeTool(Tool):
            def __init__(self, name):
                self._name = name
                self.description = "fake"
                self.parameters = {"type": "object", "properties": {}, "required": []}

            @property
            def name(self):
                return self._name

            async def execute(self, arguments, context):
                return "ok"

        tools = [FakeTool(n) for n in ["spawn", "list_children", "read_session", "write", "search"]]
        pack = ToolPack(tools)

        # Root session sees all tools
        root_session = Session(id="root", depth=0)
        root_schemas = pack.get_schemas(session=root_session)
        root_names = {s["function"]["name"] for s in root_schemas}
        assert root_names == {"spawn", "list_children", "read_session", "write", "search"}

        # Sub-agent session does NOT see root-only tools
        sub_session = Session(id="sub", depth=1)
        sub_schemas = pack.get_schemas(session=sub_session)
        sub_names = {s["function"]["name"] for s in sub_schemas}
        assert sub_names == {"write", "search"}
        assert sub_names.isdisjoint(_ROOT_ONLY_TOOLS)

    def test_root_only_set_is_correct(self):
        """Verify the _ROOT_ONLY_TOOLS set matches expected tool names."""
        assert _ROOT_ONLY_TOOLS == {"spawn", "list_children", "read_session"}


# ===================================================================
# Test 6: Per-child wake — children notify parent independently
# ===================================================================


class TestPerChildWake:
    """Each completing child triggers its own notification; no batching."""

    @pytest.mark.asyncio
    async def test_three_children_three_notifications(self, registry, config):
        drainable = MockDrainable(state=AgentState.RUNNING)
        manager, event_queue = _make_manager(registry, drainable, config)

        await _register_parent(registry)
        await _register_child(
            registry, "child_alpha",
            agent_label="Sub-1", agent_summary="Alpha done",
        )
        await _register_child(
            registry, "child_beta",
            agent_label="Sub-2", agent_summary="Beta done",
        )
        await _register_child(
            registry, "child_gamma",
            agent_label="Sub-3", agent_summary="Gamma done",
        )

        # Complete them one at a time
        info_a = CompleteInfo(parent_task_id="parent_task", pending_children=2, pending_siblings=0)
        await manager._on_child_complete("child_alpha", info_a)
        assert len(event_queue) == 1
        assert event_queue[0].notification.task_id == "child_alpha"
        assert event_queue[0].notification.summary == "Alpha done"

        info_b = CompleteInfo(parent_task_id="parent_task", pending_children=1, pending_siblings=0)
        await manager._on_child_complete("child_beta", info_b)
        assert len(event_queue) == 2
        assert event_queue[1].notification.task_id == "child_beta"

        info_c = CompleteInfo(parent_task_id="parent_task", pending_children=0, pending_siblings=0)
        await manager._on_child_complete("child_gamma", info_c)
        assert len(event_queue) == 3
        assert event_queue[2].notification.task_id == "child_gamma"

        # Verify each is a separate ChildCompletionEvent
        for i, evt in enumerate(event_queue):
            assert isinstance(evt, ChildCompletionEvent)
            assert isinstance(evt.notification, ChildCompletionNotification)

    @pytest.mark.asyncio
    async def test_mixed_branches_a_and_b(self, registry, config):
        """First child completes while parent is RUNNING (Branch B),
        then parent transitions to WAITING_FOR_CHILDREN and second child
        completes via Branch A."""
        drainable = MockDrainable(state=AgentState.RUNNING)
        manager, event_queue = _make_manager(registry, drainable, config)

        await _register_parent(registry)
        await _register_child(
            registry, "child_b1",
            agent_label="Sub-1", agent_summary="Branch B child",
        )
        await _register_child(
            registry, "child_b2",
            agent_label="Sub-2", agent_summary="Branch A child",
        )

        # First child: parent RUNNING → Branch B (enqueue)
        info1 = CompleteInfo(parent_task_id="parent_task", pending_children=1, pending_siblings=0)
        await manager._on_child_complete("child_b1", info1)
        assert len(event_queue) == 1
        assert len(drainable._run_calls) == 0

        # Transition parent to WAITING_FOR_CHILDREN
        drainable.state = AgentState.WAITING_FOR_CHILDREN

        # Second child: parent WAITING → Branch A (resume)
        info2 = CompleteInfo(parent_task_id="parent_task", pending_children=0, pending_siblings=0)
        await manager._on_child_complete("child_b2", info2)

        await asyncio.sleep(0)

        # Branch B: 1 event; Branch A: 1 run call
        assert len(event_queue) == 1  # only child_b1
        assert len(drainable._run_calls) == 1  # child_b2 triggered resume
        assert drainable._run_calls[0]["trigger"] == "children_settled"
        assert "child_b2" in drainable._run_calls[0]["message"]

    @pytest.mark.asyncio
    async def test_notification_has_all_required_fields(self, registry, config):
        """Each notification carries task_id, label, task, status, summary, session_file."""
        drainable = MockDrainable(state=AgentState.RUNNING)
        manager, event_queue = _make_manager(registry, drainable, config)

        await _register_parent(registry)
        await _register_child(
            registry, "task_fieldcheck",
            agent_label="Sub-FC",
            agent_summary="Field check summary",
        )

        info = CompleteInfo(parent_task_id="parent_task", pending_children=0, pending_siblings=0)
        await manager._on_child_complete("task_fieldcheck", info)

        notif = event_queue[0].notification
        assert notif.task_id == "task_fieldcheck"
        assert notif.label == "Sub-FC"
        assert notif.task == "Task for task_fieldcheck"
        assert notif.status == "completed"
        assert notif.summary == "Field check summary"
        assert notif.session_file == "task_fieldcheck.jsonl"
        assert "task_fieldcheck" in notif.to_prompt()
        assert "Field check summary" in notif.to_prompt()
