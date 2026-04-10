"""Subagent Registry - Support for multi-level nesting.

This module provides the SubagentRegistry for tracking subagent tasks.
Corresponds to OpenClaw's subagent-registry.ts
"""

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING, Dict, Optional, Set

from src.models import SubagentTask

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
        parent_agent: "Agent",
        parent_task_id: Optional[str] = None,
        depth: int = 0,
    ) -> SubagentTask:
        """Register a subagent - Similar to OpenClaw's registerSubagentRun.

        Args:
            task_id: Unique identifier for this task
            session_id: Session identifier
            description: Task description
            parent_agent: The parent agent that spawned this subagent
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
        """Called when a subagent completes - Corresponds to OpenClaw's complete flow.

        Key logic:
        1. Mark self as completed
        2. Check if there are still pending descendants
        3. If not, notify parent agent

        Args:
            task_id: The task ID that completed
            result: The result string from the task
            error: Whether the task completed with an error
        """
        async with self._lock:
            if task_id not in self._tasks:
                return

            task = self._tasks[task_id]
            task.status = "error" if error else "completed"
            task.result = result
            task.completed_event.set()
            self._pending.discard(task_id)

            parent: Agent = task.parent_agent
            parent_task_id = task.parent_task_id
            pending_descendants = self._count_pending_descendants_locked(task_id)

        if pending_descendants > 0:
            print(
                f"[Registry] {task_id} done, {pending_descendants} descendants pending"
            )
            return

        if parent_task_id and parent_task_id in self._tasks:
            parent_task = self._tasks[parent_task_id]
            if (
                parent_task.status in ["completed", "ended_with_pending_descendants"]
                and parent_task.wake_on_descendants_settle
            ):
                print(f"[Registry] {parent_task_id} needs wake")
                asyncio.create_task(
                    self._wake_parent_agent(parent_task_id, task_id, result)
                )
                return

        if parent and parent_task_id:
            if error:
                print(f"[Registry] {task_id} error → {parent_task_id}")
                asyncio.create_task(parent._on_subagent_error(task_id, result))
            else:
                result_preview = result[:80] + "..." if len(result) > 80 else result
                print(f"[Registry] {task_id} done → {parent_task_id}: {result_preview}")
                asyncio.create_task(parent._on_subagent_complete(task_id, result))

    def _count_pending_descendants_locked(self, task_id: str) -> int:
        """Count pending descendants of a task - Corresponds to OpenClaw's countPendingDescendantRuns.

        Must be called while holding self._lock.

        Args:
            task_id: The task ID to count descendants for

        Returns:
            Number of pending descendant tasks
        """
        if task_id not in self._tasks:
            return 0

        task = self._tasks[task_id]
        count = 0
        visited = {task_id}
        queue = list(task.child_task_ids)

        while queue:
            child_id = queue.pop(0)
            if child_id in visited:
                continue
            visited.add(child_id)

            if child_id in self._tasks:
                child_task = self._tasks[child_id]
                if child_id in self._pending:
                    count += 1
                queue.extend(child_task.child_task_ids)

        return count

    async def _wake_parent_agent(
        self, parent_task_id: str, child_task_id: str, child_result: str
    ):
        """Wake parent agent - Corresponds to OpenClaw's wakeSubagentRunAfterDescendants.

        Args:
            parent_task_id: The parent task ID to wake
            child_task_id: The child task ID that triggered the wake
            child_result: The result from the child task
        """
        async with self._lock:
            if parent_task_id not in self._tasks:
                return
            parent_task = self._tasks[parent_task_id]
            parent_task.status = "running"
            parent_task.wake_on_descendants_settle = False

        parent = parent_task.agent
        if parent:
            print(f"[Registry] Wake: {parent_task_id}")
            await parent._on_descendant_wake(child_task_id, child_result)

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

    def count_pending_for_parent(self, parent_task_id: str) -> int:
        """Count pending subagents for a specific parent.

        Args:
            parent_task_id: The parent task ID to count for

        Returns:
            Number of pending child tasks for this parent
        """
        count = 0
        for task_id, task in self._tasks.items():
            if task.parent_task_id == parent_task_id and task_id in self._pending:
                count += 1
        return count

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

    async def mark_ended_with_pending_descendants(self, task_id: str):
        """Mark an Agent as ended but with pending descendants.

        Args:
            task_id: The task ID to mark
        """
        async with self._lock:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                task.status = "ended_with_pending_descendants"
                task.wake_on_descendants_settle = True
                task.ended_at = datetime.now().timestamp()


__all__ = ["SubagentRegistry"]
