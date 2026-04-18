"""Safety guards and resource limits for the Agent system.

This module provides resource limiting utilities to prevent:
- Unbounded concurrency (using semaphore)
- Infinite agent runs (using timeout)
- Unbounded result sizes (using truncation)
- Unbounded registry growth (using size limit with auto-purge)
"""

import asyncio
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from engine.subagent.registry import SubagentRegistry


class ConcurrencyLimiter:
    """Limits concurrent agent executions using asyncio.Semaphore."""

    def __init__(self, max_concurrent: int):
        """Initialize the concurrency limiter.

        Args:
            max_concurrent: Maximum number of concurrent agents allowed
        """
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def acquire(self):
        """Acquire the semaphore (blocks if at limit)."""
        await self._semaphore.acquire()

    async def release(self):
        """Release the semaphore."""
        self._semaphore.release()

    @property
    def max_concurrent(self) -> int:
        """Get the maximum concurrent limit."""
        return self._semaphore._value


class ResultTruncator:
    """Truncates results to prevent unbounded result sizes."""

    TRUNCATION_SUFFIX = "[truncated]"

    @staticmethod
    def truncate(result: str, max_length: int) -> str:
        """Truncate a result string if it exceeds max_length.

        Args:
            result: The result string to truncate
            max_length: Maximum allowed length

        Returns:
            The original string if within limits, or truncated string with suffix
        """
        if len(result) <= max_length:
            return result

        available_length = max_length - len(ResultTruncator.TRUNCATION_SUFFIX)
        if available_length <= 0:
            return result[:max_length]

        return result[:available_length] + ResultTruncator.TRUNCATION_SUFFIX


class RegistrySizeMonitor:
    """Monitors registry size and identifies tasks to purge."""

    @staticmethod
    def check_and_purge(registry: "SubagentRegistry", max_size: int) -> List[str]:
        """Check if registry exceeds max_size and return task_ids to purge.

        Purges oldest completed tasks first (based on ended_at timestamp).

        Args:
            registry: The SubagentRegistry to monitor
            max_size: Maximum allowed registry size

        Returns:
            List of task_ids that should be purged (empty if within limits)
        """
        current_size = len(registry._tasks)

        if current_size <= max_size:
            return []

        to_purge_count = current_size - max_size

        completed_tasks = [
            (task_id, task)
            for task_id, task in registry._tasks.items()
            if task.result is not None
        ]

        completed_tasks.sort(
            key=lambda x: x[1].ended_at if x[1].ended_at is not None else float("inf")
        )

        return [task_id for task_id, _ in completed_tasks[:to_purge_count]]


__all__ = ["ConcurrencyLimiter", "ResultTruncator", "RegistrySizeMonitor"]
