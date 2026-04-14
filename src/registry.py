"""Subagent Registry - Support for multi-level nesting.

This module provides the SubagentRegistry for tracking subagent tasks.
Corresponds to OpenClaw's subagent-registry.ts
"""

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING, Dict, Optional, Set

from src.models import AgentState, QueueEvent, SubagentTask

if TYPE_CHECKING:
    from src.agent_core import Agent


class SubagentRegistry:
    """Subagent Registry - Support for multi-level nesting.

    Corresponds to OpenClaw's subagent-registry.ts
    """

    def __init__(self):
        """Initialize the registry."""
        self._tasks: Dict[str, SubagentTask] = {}
        self._pending: Set[str] = set()
        self._lock = asyncio.Lock()

    async def register(
        self,
        task_id: str,
        session_id: str,
        description: str,
        parent_agent: "Agent",  # deprecated
        agent: Optional["Agent"] = None,
        parent_task_id: Optional[str] = None,
        depth: int = 0,
    ) -> SubagentTask:
        """Register a subagent - Similar to OpenClaw's registerSubagentRun.

        Args:
            task_id: Unique identifier for this task
            session_id: Session identifier
            description: Task description
            parent_agent: Which agent spawned this task (deprecated, kept for backward compat)
            agent: Self-reference of the agent running this task (used by complete() to push events)
            parent_task_id: Optional parent task ID for nested subagents
            depth: Nesting depth level

        Returns:
            The created SubagentTask

        Raises:
            ValueError: If registering would create a cycle in the task hierarchy
        """
        print(f"[{task_id} 完成注册]")

        # Check for cycles before registering
        if parent_task_id and self._would_create_cycle(task_id, parent_task_id):
            raise ValueError(
                f"Cycle detected: agent {task_id} already exists in ancestor chain"
            )

        task = SubagentTask(
            task_id=task_id,
            session_id=session_id,
            task_description=description,
            parent_agent=parent_agent,
            agent=agent,
            parent_task_id=parent_task_id,
            depth=depth,
        )

        async with self._lock:
            self._tasks[task_id] = task
            self._pending.add(task_id)

            if parent_task_id and parent_task_id in self._tasks:
                self._tasks[parent_task_id].child_task_ids.add(task_id)

        return task

    def get_all_ancestors(self, task_id: str) -> Set[str]:
        """Get all ancestor task IDs using BFS traversal.

        Traverses from task_id up through parent_task_id chain to collect
        all ancestors in the task hierarchy.

        Args:
            task_id: The task ID to find ancestors for

        Returns:
            Set of all ancestor task IDs
        """
        ancestors: Set[str] = set()
        visited: Set[str] = set()
        queue = [task_id]

        while queue:
            current_id = queue.pop(0)
            if current_id in visited:
                continue
            visited.add(current_id)

            if current_id in self._tasks:
                task = self._tasks[current_id]
                if task.parent_task_id:
                    ancestors.add(task.parent_task_id)
                    queue.append(task.parent_task_id)

        return ancestors

    async def set_agent(self, task_id: str, agent: "Agent"):
        """Set the agent reference for a task.

        Args:
            task_id: The task ID
            agent: The corresponding Agent instance
        """
        async with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id].agent = agent

    def _would_create_cycle(self, new_agent_id: str, parent_task_id: str) -> bool:
        """Check if registering would create a cycle in the task hierarchy.

        A cycle would be created if new_agent_id already exists in the
        ancestor chain of parent_task_id.

        Args:
            new_agent_id: The agent/task ID being registered
            parent_task_id: The parent task ID this would be registered under

        Returns:
            True if a cycle would be created, False otherwise
        """
        ancestors = self.get_all_ancestors(parent_task_id)
        return new_agent_id in ancestors

    async def complete(self, task_id: str, result: str, error: bool = False):
        """Called when a subagent completes."""
        async with self._lock:
            if task_id not in self._tasks:
                return

            task = self._tasks[task_id]
            task.result = result
            task.ended_at = datetime.now().timestamp()
            self._pending.discard(task_id)

            parent_task_id = task.parent_task_id
            pending_children = self._count_pending_for_parent(task_id)
            pending_siblings = (
                self._count_pending_for_parent(parent_task_id) if parent_task_id else 0
            )

        # [Gate 1] Still have pending children → return
        if pending_children > 0:
            print(
                f"[Registry] {task_id} done, {pending_children} children pending"
            )
            return

        # [Gate 2] Parent doesn't exist or not registered → return
        if not (parent_task_id and parent_task_id in self._tasks):
            return

        # [Gate 3] Still have pending siblings → return
        if pending_siblings > 0:
            return

        # All gates passed → notify parent
        parent_task: SubagentTask = self._tasks[parent_task_id]

        # Collect aggregated child results
        child_results = self.collect_child_results(parent_task_id)

        parent_agent: "Agent" = parent_task.agent

        # [Branch A] Parent waiting for descendants → wake
        if (
            parent_agent is not None
            and parent_agent.state_machine.current_state
            == AgentState.WAITING_FOR_CHILDREN
        ):
            asyncio.create_task(parent_agent._resume_from_children(child_results))
            return

        # [Branch B] Parent is running → push to event queue
        if parent_agent is not None:
            event = QueueEvent(
                trigger_task_id=task_id, child_results=child_results, error=error
            )
            parent_agent._event_queue.append(event)

    # Replaced by _count_pending_for_parent. See docs/notebook.md for analysis.
    #
    # def _count_pending_descendants_locked(self, task_id: str) -> int:
    #     """Count pending descendants of a task - Corresponds to OpenClaw's countPendingDescendantRuns.
    #
    #     Must be called while holding self._lock.
    #
    #     Args:
    #         task_id: The task ID to count descendants for
    #
    #     Returns:
    #         Number of pending descendant tasks
    #     """
    #     if task_id not in self._tasks:
    #         return 0
    #
    #     task = self._tasks[task_id]
    #     count = 0
    #     visited = {task_id}
    #     queue = list(task.child_task_ids)
    #
    #     while queue:
    #         child_id = queue.pop(0)
    #         if child_id in visited:
    #             continue
    #         visited.add(child_id)
    #
    #         if child_id in self._tasks:
    #             child_task = self._tasks[child_id]
    #             if child_id in self._pending:
    #                 count += 1
    #             queue.extend(child_task.child_task_ids)
    #
    #     return count

    def has_pending(self) -> bool:
        """Check if there are any pending tasks.

        Returns:
            True if there are pending tasks, False otherwise
        """
        return len(self._pending) > 0

    def get_pending_count(self) -> int:
        """Get the count of pending tasks.

        Returns:
            Number of pending tasks
        """
        return len(self._pending)

    def _count_pending_for_parent(self, parent_task_id: str) -> int:
        if parent_task_id not in self._tasks:
            return 0
        return sum(1 for cid in self._tasks[parent_task_id].child_task_ids if cid in self._pending)

    def get_task(self, task_id: str) -> Optional[SubagentTask]:
        """Get a task by ID.

        Args:
            task_id: The task ID to retrieve

        Returns:
            The SubagentTask if found, None otherwise
        """
        return self._tasks.get(task_id)

    def collect_child_results(self, parent_task_id: str) -> Dict[str, str]:
        """Collect all direct child results - Corresponds to OpenClaw's childCompletionFindings.

        Args:
            parent_task_id: The parent task ID to collect results for

        Returns:
            Dictionary mapping child task IDs to their results
        """
        results = {}
        for task_id, task in self._tasks.items():
            if task.parent_task_id == parent_task_id and task.result is not None:
                results[task_id] = task.result
        return results


__all__ = ["SubagentRegistry"]
