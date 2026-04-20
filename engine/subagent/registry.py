"""Subagent Registry - Pure data CRUD with handler-based notification.

This module provides the SubagentRegistry for tracking subagent tasks.
Extracted from engine/registry.py with gate-check and notification logic
removed — that responsibility moves to SubAgentManager.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Set

from engine.logging import get_logger
from engine.subagent.models import CollectedChildResult, SubagentTask

if TYPE_CHECKING:
    from engine.agent_core import Agent


@dataclass
class CompleteInfo:
    """Completion info returned by store_result for handler notification."""

    parent_task_id: Optional[str]
    pending_children: int
    pending_siblings: int


class SubagentRegistry:
    """Subagent Registry - Support for multi-level nesting.

    Pure data CRUD operations with handler-based notification.
    Gate-check and notification logic lives in SubAgentManager.
    """

    def __init__(self):
        """Initialize the registry."""
        self._tasks: Dict[str, SubagentTask] = {}
        self._pending: Set[str] = set()
        self._lock = asyncio.Lock()
        self._completion_handlers: Dict[str, Callable] = {}  # maps parent_task_id → callback

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
        logger = get_logger()
        logger.info(
            "Registry",
            "Task registered | task_id={}, session_id={}, parent_task_id={}, depth={}".format(
                task_id, session_id, parent_task_id or "None (root)", depth
            ),
            task_id=task_id, state="running", depth=depth,
            event_type="registry_register",
            data={"session_id": session_id, "parent_task_id": parent_task_id, "depth": depth, "description": description}
        )

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

    def register_handler(self, parent_task_id: str, handler: Callable):
        """Register a completion handler for a parent task."""
        self._completion_handlers[parent_task_id] = handler

    async def store_result(self, task_id: str, result: str) -> CompleteInfo:
        """Store result and return completion info. Counts computed inside lock."""
        async with self._lock:
            if task_id not in self._tasks:
                return CompleteInfo(parent_task_id=None, pending_children=0, pending_siblings=0)
            task = self._tasks[task_id]
            task.result = result
            task.ended_at = datetime.now().timestamp()
            self._pending.discard(task_id)
            parent_task_id = task.parent_task_id
            pending_children = self._count_pending_for_parent(task_id)
            pending_siblings = (
                self._count_pending_for_parent(parent_task_id) if parent_task_id else 0
            )
        return CompleteInfo(
            parent_task_id=parent_task_id,
            pending_children=pending_children,
            pending_siblings=pending_siblings,
        )

    async def complete(self, task_id: str, result: str, error: bool = False):
        """Store result and notify registered handler. Called by child agents."""
        info = await self.store_result(task_id, result)
        handler = self._completion_handlers.get(info.parent_task_id)
        if handler:
            await handler(task_id, info)

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

    def count_pending_children(self, task_id: str) -> int:
        """Count direct children of `task_id` still pending."""
        return self._count_pending_for_parent(task_id)

    def get_task(self, task_id: str) -> Optional[SubagentTask]:
        """Get a task by ID.

        Args:
            task_id: The task ID to retrieve

        Returns:
            The SubagentTask if found, None otherwise
        """
        return self._tasks.get(task_id)

    def collect_child_results(self, parent_task_id: str) -> Dict[str, CollectedChildResult]:
        """Collect all direct child results with task_description.

        Args:
            parent_task_id: The parent task ID to collect results for

        Returns:
            Dictionary mapping child task IDs to their CollectedChildResult
        """
        parent = self._tasks.get(parent_task_id)
        if not parent:
            return {}
        results = {}
        for child_id in parent.child_task_ids:
            child_task = self._tasks.get(child_id)
            if child_task and child_task.result is not None:
                results[child_id] = CollectedChildResult(
                    task_description=child_task.task_description,
                    result=child_task.result,
                )
        return results

    def get_task_depth(self, task_id: str) -> int:
        """Get the depth of a task.

        Args:
            task_id: The task ID to look up.

        Returns:
            The task's depth, or 0 if the task is not found.
        """
        task = self._tasks.get(task_id)
        return task.depth if task else 0

    async def collect_and_cleanup(
        self, parent_task_id: str
    ) -> Dict[str, CollectedChildResult]:
        """Atomically collect child results, clear parent's child list, and remove child tasks.

        All operations are performed inside the registry lock for consistency.

        Args:
            parent_task_id: The parent task whose children should be collected and cleaned.

        Returns:
            Mapping from child task ID to CollectedChildResult.
        """
        async with self._lock:
            results = self.collect_child_results(parent_task_id)
            parent = self._tasks.get(parent_task_id)
            if parent:
                parent.child_task_ids.clear()
                for child_id in results:
                    self._tasks.pop(child_id, None)
            return results


__all__ = ["CompleteInfo", "SubagentRegistry"]
