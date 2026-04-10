"""Comprehensive tests for the queue-based SubagentRegistry system.

Tests cover:
- Register: agent parameter handling, backward compatibility
- Complete: 3-gate + 2-branch logic (wake vs queue push)
- Multi-level: Chains, grandchildren, multiple children scenarios
"""

import pytest
import asyncio

from src.registry import SubagentRegistry
from src.models import SubagentTask, QueueEvent


class MockAgent:
    """Mock Agent for testing - simulates _resume_from_children and _event_queue."""
    
    def __init__(self, task_id: str, registry: SubagentRegistry):
        self.task_id = task_id
        self.registry = registry
        self._event_queue: list = []
        self.resumed = False
    
    async def _resume_from_children(self):
        """Mock resume method - tracks if it was called."""
        self.resumed = True


def make_test_registry() -> SubagentRegistry:
    """Helper to create a fresh registry for testing."""
    return SubagentRegistry()


class TestRegister:
    """Tests for registry.register() with agent parameter."""
    
    def test_register_with_agent(self):
        """Pass agent param → SubagentTask.agent is set."""
        registry = make_test_registry()
        mock_agent = MockAgent("parent-1", registry)
        
        task = asyncio.run(registry.register(
            task_id="task-1",
            session_id="session-1",
            description="Test task",
            parent_agent=mock_agent,
            agent=mock_agent,
            parent_task_id=None,
            depth=0
        ))
        
        assert isinstance(task, SubagentTask)
        assert task.task_id == "task-1"
        assert task.agent is mock_agent
        assert task.parent_agent is mock_agent
    
    def test_register_without_agent(self):
        """No agent param → SubagentTask.agent is None (backward compat)."""
        registry = make_test_registry()
        mock_parent = MockAgent("parent-1", registry)
        
        task = asyncio.run(registry.register(
            task_id="task-1",
            session_id="session-1",
            description="Test task",
            parent_agent=mock_parent,
            agent=None,  # Explicit None
            parent_task_id=None,
            depth=0
        ))
        
        assert isinstance(task, SubagentTask)
        assert task.agent is None
        assert task.parent_agent is mock_parent
    
    def test_register_root_agent(self):
        """Simulate main.py root registration flow."""
        registry = make_test_registry()
        root_agent = MockAgent("root-1", registry)
        
        # Root agent registers itself
        task = asyncio.run(registry.register(
            task_id=root_agent.task_id,
            session_id="root-session",
            description="Root agent",
            parent_agent=root_agent,
            agent=root_agent,  # Self-reference
            parent_task_id=None,
            depth=0
        ))
        
        assert task.task_id == "root-1"
        assert task.agent is root_agent
        assert task.depth == 0
        assert task.parent_task_id is None


class TestComplete:
    """Tests for registry.complete() with 3-gate + 2-branch logic."""
    
    def test_complete_push_to_queue_when_parent_running(self):
        """Parent running + all siblings done → QueueEvent pushed."""
        registry = make_test_registry()
        
        # Create parent agent
        parent_agent = MockAgent("parent-1", registry)
        
        # Register parent (running status is default)
        asyncio.run(registry.register(
            task_id="parent-1",
            session_id="session-1",
            description="Parent task",
            parent_agent=parent_agent,
            agent=parent_agent,
            parent_task_id=None,
            depth=0
        ))
        
        # Register single child
        child_agent = MockAgent("child-1", registry)
        asyncio.run(registry.register(
            task_id="child-1",
            session_id="session-1",
            description="Child task",
            parent_agent=parent_agent,
            agent=child_agent,
            parent_task_id="parent-1",
            depth=1
        ))
        
        # Complete the child - should push to parent's queue
        asyncio.run(registry.complete("child-1", "result-1", error=False))
        
        # Verify queue has the event
        assert len(parent_agent._event_queue) == 1
        event = parent_agent._event_queue[0]
        assert isinstance(event, QueueEvent)
        assert event.child_task_id == "child-1"
        assert event.result == "result-1"
        assert event.error is False
        assert parent_agent.resumed is False  # Should NOT wake
    
    def test_complete_wake_when_parent_ended(self):
        """Parent ended_with_pending_descendants → wake path (status=running, _resume_from_children called)."""
        registry = make_test_registry()
        
        # Create parent agent
        parent_agent = MockAgent("parent-1", registry)
        
        # Register parent
        asyncio.run(registry.register(
            task_id="parent-1",
            session_id="session-1",
            description="Parent task",
            parent_agent=parent_agent,
            agent=parent_agent,
            parent_task_id=None,
            depth=0
        ))
        
        # Register child
        child_agent = MockAgent("child-1", registry)
        asyncio.run(registry.register(
            task_id="child-1",
            session_id="session-1",
            description="Child task",
            parent_agent=parent_agent,
            agent=child_agent,
            parent_task_id="parent-1",
            depth=1
        ))
        
        # Mark parent as ended waiting
        asyncio.run(registry.mark_ended_with_pending_descendants("parent-1"))
        
        # Complete the child - should wake parent
        asyncio.run(registry.complete("child-1", "result-1", error=False))
        
        # Verify parent was woken
        assert parent_agent.resumed is True
        assert len(parent_agent._event_queue) == 0  # Should NOT push to queue
        
        # Verify parent status was reset to running
        parent_task = registry.get_task("parent-1")
        assert parent_task.status == "running"
    
    def test_complete_no_action_when_pending_descendants(self):
        """Gate 1 — pending descendants → no push, no wake."""
        registry = make_test_registry()
        
        # Create agents
        parent_agent = MockAgent("parent-1", registry)
        child1_agent = MockAgent("child-1", registry)
        grandchild_agent = MockAgent("grandchild-1", registry)
        
        # Register hierarchy: parent -> child1 -> grandchild
        asyncio.run(registry.register(
            task_id="parent-1",
            session_id="session-1",
            description="Parent task",
            parent_agent=parent_agent,
            agent=parent_agent,
            parent_task_id=None,
            depth=0
        ))
        
        asyncio.run(registry.register(
            task_id="child-1",
            session_id="session-1",
            description="Child task",
            parent_agent=parent_agent,
            agent=child1_agent,
            parent_task_id="parent-1",
            depth=1
        ))
        
        asyncio.run(registry.register(
            task_id="grandchild-1",
            session_id="session-1",
            description="Grandchild task",
            parent_agent=child1_agent,
            agent=grandchild_agent,
            parent_task_id="child-1",
            depth=2
        ))
        
        # Complete child1 while grandchild is still pending
        asyncio.run(registry.complete("child-1", "result-1", error=False))
        
        # Verify no action taken
        assert len(parent_agent._event_queue) == 0
        assert parent_agent.resumed is False
    
    def test_complete_no_action_when_pending_siblings(self):
        """Gate 3 — pending siblings → no push, no wake."""
        registry = make_test_registry()
        
        # Create parent and two children
        parent_agent = MockAgent("parent-1", registry)
        child1_agent = MockAgent("child-1", registry)
        child2_agent = MockAgent("child-2", registry)
        
        # Register parent
        asyncio.run(registry.register(
            task_id="parent-1",
            session_id="session-1",
            description="Parent task",
            parent_agent=parent_agent,
            agent=parent_agent,
            parent_task_id=None,
            depth=0
        ))
        
        # Register both children
        asyncio.run(registry.register(
            task_id="child-1",
            session_id="session-1",
            description="Child 1 task",
            parent_agent=parent_agent,
            agent=child1_agent,
            parent_task_id="parent-1",
            depth=1
        ))
        
        asyncio.run(registry.register(
            task_id="child-2",
            session_id="session-1",
            description="Child 2 task",
            parent_agent=parent_agent,
            agent=child2_agent,
            parent_task_id="parent-1",
            depth=1
        ))
        
        # Complete only child-1 (child-2 still pending)
        asyncio.run(registry.complete("child-1", "result-1", error=False))
        
        # Verify no action taken
        assert len(parent_agent._event_queue) == 0
        assert parent_agent.resumed is False
    
    def test_complete_second_sibling_triggers_notification(self):
        """First sibling → blocked by Gate 3; second → pushes to queue."""
        registry = make_test_registry()
        
        # Create parent and two children
        parent_agent = MockAgent("parent-1", registry)
        child1_agent = MockAgent("child-1", registry)
        child2_agent = MockAgent("child-2", registry)
        
        # Register parent
        asyncio.run(registry.register(
            task_id="parent-1",
            session_id="session-1",
            description="Parent task",
            parent_agent=parent_agent,
            agent=parent_agent,
            parent_task_id=None,
            depth=0
        ))
        
        # Register both children
        asyncio.run(registry.register(
            task_id="child-1",
            session_id="session-1",
            description="Child 1 task",
            parent_agent=parent_agent,
            agent=child1_agent,
            parent_task_id="parent-1",
            depth=1
        ))
        
        asyncio.run(registry.register(
            task_id="child-2",
            session_id="session-1",
            description="Child 2 task",
            parent_agent=parent_agent,
            agent=child2_agent,
            parent_task_id="parent-1",
            depth=1
        ))
        
        # Complete child-1 (blocked by Gate 3 - child-2 still pending)
        asyncio.run(registry.complete("child-1", "result-1", error=False))
        assert len(parent_agent._event_queue) == 0
        
        # Complete child-2 (all siblings done → push to queue)
        asyncio.run(registry.complete("child-2", "result-2", error=False))
        
        # Verify both events in queue (child-2 triggers the push, but only its own event)
        assert len(parent_agent._event_queue) == 1
        event = parent_agent._event_queue[0]
        assert event.child_task_id == "child-2"
        assert event.result == "result-2"
    
    def test_complete_error_result(self):
        """error=True → QueueEvent.error is True."""
        registry = make_test_registry()
        
        parent_agent = MockAgent("parent-1", registry)
        child_agent = MockAgent("child-1", registry)
        
        asyncio.run(registry.register(
            task_id="parent-1",
            session_id="session-1",
            description="Parent task",
            parent_agent=parent_agent,
            agent=parent_agent,
            parent_task_id=None,
            depth=0
        ))
        
        asyncio.run(registry.register(
            task_id="child-1",
            session_id="session-1",
            description="Child task",
            parent_agent=parent_agent,
            agent=child_agent,
            parent_task_id="parent-1",
            depth=1
        ))
        
        # Complete with error
        asyncio.run(registry.complete("child-1", "error-result", error=True))
        
        assert len(parent_agent._event_queue) == 1
        event = parent_agent._event_queue[0]
        assert event.error is True
        assert event.result == "error-result"
    
    def test_complete_queue_item_format(self):
        """Verify QueueEvent fields (child_task_id, result, error)."""
        registry = make_test_registry()
        
        parent_agent = MockAgent("parent-1", registry)
        child_agent = MockAgent("child-1", registry)
        
        asyncio.run(registry.register(
            task_id="parent-1",
            session_id="session-1",
            description="Parent task",
            parent_agent=parent_agent,
            agent=parent_agent,
            parent_task_id=None,
            depth=0
        ))
        
        asyncio.run(registry.register(
            task_id="child-1",
            session_id="session-1",
            description="Child task",
            parent_agent=parent_agent,
            agent=child_agent,
            parent_task_id="parent-1",
            depth=1
        ))
        
        asyncio.run(registry.complete("child-1", "test-result", error=False))
        
        assert len(parent_agent._event_queue) == 1
        event = parent_agent._event_queue[0]
        
        # Verify all fields
        assert isinstance(event, QueueEvent)
        assert hasattr(event, 'child_task_id')
        assert hasattr(event, 'result')
        assert hasattr(event, 'error')
        assert event.child_task_id == "child-1"
        assert event.result == "test-result"
        assert event.error is False
    
    def test_complete_no_direct_callbacks(self):
        """Verify _on_subagent_complete/_on_subagent_error not called."""
        registry = make_test_registry()
        
        # Track if any callback methods are called
        callback_called = []
        
        class MockAgentWithCallbacks(MockAgent):
            def _on_subagent_complete(self, task_id, result):
                callback_called.append(('complete', task_id, result))
            
            def _on_subagent_error(self, task_id, result):
                callback_called.append(('error', task_id, result))
        
        parent_agent = MockAgentWithCallbacks("parent-1", registry)
        child_agent = MockAgentWithCallbacks("child-1", registry)
        
        asyncio.run(registry.register(
            task_id="parent-1",
            session_id="session-1",
            description="Parent task",
            parent_agent=parent_agent,
            agent=parent_agent,
            parent_task_id=None,
            depth=0
        ))
        
        asyncio.run(registry.register(
            task_id="child-1",
            session_id="session-1",
            description="Child task",
            parent_agent=parent_agent,
            agent=child_agent,
            parent_task_id="parent-1",
            depth=1
        ))
        
        # Complete child
        asyncio.run(registry.complete("child-1", "result", error=False))
        
        # Verify callbacks were NOT called
        assert len(callback_called) == 0
        # Verify event was queued instead
        assert len(parent_agent._event_queue) == 1


class TestMultiLevel:
    """Tests for multi-level subagent chains."""
    
    def test_three_level_chain(self):
        """A(root)→B→C. C completes → B queue push. Then complete B → A queue push."""
        registry = make_test_registry()
        
        # Create agents
        agent_a = MockAgent("A", registry)
        agent_b = MockAgent("B", registry)
        agent_c = MockAgent("C", registry)
        
        # Register hierarchy: A (root) -> B -> C
        asyncio.run(registry.register(
            task_id="A",
            session_id="session-1",
            description="Agent A",
            parent_agent=agent_a,
            agent=agent_a,
            parent_task_id=None,
            depth=0
        ))
        
        asyncio.run(registry.register(
            task_id="B",
            session_id="session-1",
            description="Agent B",
            parent_agent=agent_a,
            agent=agent_b,
            parent_task_id="A",
            depth=1
        ))
        
        asyncio.run(registry.register(
            task_id="C",
            session_id="session-1",
            description="Agent C",
            parent_agent=agent_b,
            agent=agent_c,
            parent_task_id="B",
            depth=2
        ))
        
        # Complete C (B is running, so queue to B)
        asyncio.run(registry.complete("C", "result-C", error=False))
        assert len(agent_b._event_queue) == 1
        assert agent_b._event_queue[0].child_task_id == "C"
        assert len(agent_a._event_queue) == 0  # A shouldn't receive anything yet
        
        # Complete B (all B's descendants done, A is running)
        asyncio.run(registry.complete("B", "result-B", error=False))
        assert len(agent_a._event_queue) == 1
        assert agent_a._event_queue[0].child_task_id == "B"
    
    def test_grandchildren_wake_grandparent(self):
        """Grandparent ends waiting, grandchild completes → wake grandparent."""
        registry = make_test_registry()
        
        # Create agents
        grandparent_agent = MockAgent("grandparent", registry)
        parent_agent = MockAgent("parent", registry)
        grandchild_agent = MockAgent("grandchild", registry)
        
        # Register hierarchy: grandparent -> parent -> grandchild
        asyncio.run(registry.register(
            task_id="grandparent",
            session_id="session-1",
            description="Grandparent",
            parent_agent=grandparent_agent,
            agent=grandparent_agent,
            parent_task_id=None,
            depth=0
        ))
        
        asyncio.run(registry.register(
            task_id="parent",
            session_id="session-1",
            description="Parent",
            parent_agent=grandparent_agent,
            agent=parent_agent,
            parent_task_id="grandparent",
            depth=1
        ))
        
        asyncio.run(registry.register(
            task_id="grandchild",
            session_id="session-1",
            description="Grandchild",
            parent_agent=parent_agent,
            agent=grandchild_agent,
            parent_task_id="parent",
            depth=2
        ))
        
        # Mark grandparent as ended waiting FIRST (waiting for descendants to settle)
        asyncio.run(registry.mark_ended_with_pending_descendants("grandparent"))
        
        # Now complete grandchild - parent is running, so grandchild queues to parent
        # But wait, parent has no agent set up to receive... 
        # Actually, parent is running so grandchild should queue to parent
        asyncio.run(registry.complete("grandchild", "grandchild-result", error=False))
        
        # Parent's queue should have the event
        assert len(parent_agent._event_queue) == 1
        assert parent_agent._event_queue[0].child_task_id == "grandchild"
        
        # Grandparent should NOT be woken yet - parent is still running
        assert grandparent_agent.resumed is False
        
        # Now complete parent - this should wake grandparent since it's waiting
        asyncio.run(registry.complete("parent", "parent-result", error=False))
        
        # Now grandparent should be woken
        assert grandparent_agent.resumed is True
    
    def test_multiple_children_all_complete(self):
        """3 children all complete → only last one triggers notification."""
        registry = make_test_registry()
        
        # Create parent and 3 children
        parent_agent = MockAgent("parent-1", registry)
        child_agents = [
            MockAgent(f"child-{i}", registry)
            for i in range(1, 4)
        ]
        
        # Register parent
        asyncio.run(registry.register(
            task_id="parent-1",
            session_id="session-1",
            description="Parent task",
            parent_agent=parent_agent,
            agent=parent_agent,
            parent_task_id=None,
            depth=0
        ))
        
        # Register all 3 children
        for i, child_agent in enumerate(child_agents, 1):
            asyncio.run(registry.register(
                task_id=f"child-{i}",
                session_id="session-1",
                description=f"Child {i} task",
                parent_agent=parent_agent,
                agent=child_agent,
                parent_task_id="parent-1",
                depth=1
            ))
        
        # Complete first 2 children - should NOT trigger notification
        asyncio.run(registry.complete("child-1", "result-1", error=False))
        assert len(parent_agent._event_queue) == 0
        
        asyncio.run(registry.complete("child-2", "result-2", error=False))
        assert len(parent_agent._event_queue) == 0
        
        # Complete 3rd child - SHOULD trigger notification
        asyncio.run(registry.complete("child-3", "result-3", error=False))
        assert len(parent_agent._event_queue) == 1
        
        # Verify only child-3's event was pushed (the one that triggered)
        event = parent_agent._event_queue[0]
        assert event.child_task_id == "child-3"
        assert event.result == "result-3"


class TestEdgeCases:
    """Additional edge case tests."""
    
    def test_complete_nonexistent_task(self):
        """Completing a non-existent task should not raise error."""
        registry = make_test_registry()
        
        # Should not raise
        asyncio.run(registry.complete("nonexistent", "result", error=False))
    
    def test_complete_no_parent(self):
        """Complete root task with no parent."""
        registry = make_test_registry()
        
        root_agent = MockAgent("root", registry)
        
        asyncio.run(registry.register(
            task_id="root",
            session_id="session-1",
            description="Root task",
            parent_agent=root_agent,
            agent=root_agent,
            parent_task_id=None,
            depth=0
        ))
        
        # Complete root - no parent to notify
        asyncio.run(registry.complete("root", "root-result", error=False))
        
        # Should complete without error
        task = registry.get_task("root")
        assert task.status == "completed"
        assert task.result == "root-result"
    
    def test_cycle_detection(self):
        """Attempting to create cycle should raise ValueError."""
        registry = make_test_registry()
        
        agent_a = MockAgent("A", registry)
        agent_b = MockAgent("B", registry)
        
        # Register A
        asyncio.run(registry.register(
            task_id="A",
            session_id="session-1",
            description="Agent A",
            parent_agent=agent_a,
            agent=agent_a,
            parent_task_id=None,
            depth=0
        ))
        
        # Register B as child of A
        asyncio.run(registry.register(
            task_id="B",
            session_id="session-1",
            description="Agent B",
            parent_agent=agent_a,
            agent=agent_b,
            parent_task_id="A",
            depth=1
        ))
        
        # Try to register A as child of B (creates cycle A->B->A)
        with pytest.raises(ValueError) as exc_info:
            asyncio.run(registry.register(
                task_id="A",  # A already exists as ancestor
                session_id="session-1",
                description="Agent A again",
                parent_agent=agent_b,
                agent=agent_a,
                parent_task_id="B",
                depth=2
            ))
        
        assert "cycle" in str(exc_info.value).lower()
