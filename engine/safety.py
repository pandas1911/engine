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
    from engine.runtime.task_registry import AgentTaskRegistry


class ConcurrencyLimiter:
    """Limits concurrent agent executions using asyncio.Semaphore.

    Wraps asyncio.Semaphore with a clean, observable interface.
    Uses a manual _active counter for active_count (no dependency
    on _semaphore._value private attribute).
    """

    def __init__(self, max_concurrent: int):
        if max_concurrent < 1:
            raise ValueError(
                "max_concurrent must be >= 1, got {}".format(max_concurrent)
            )
        self._max = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._active = 0

    async def acquire(self):
        """Acquire a slot (blocks if at limit)."""
        await self._semaphore.acquire()
        self._active += 1

    def release(self):
        """Release a slot (non-async). Raises on over-release."""
        if self._active <= 0:
            raise RuntimeError("release called too many times")
        self._active -= 1
        self._semaphore.release()

    @property
    def active_count(self) -> int:
        """Currently acquired count."""
        return self._active

    @property
    def max_concurrent(self) -> int:
        """Configured maximum (NOT remaining count)."""
        return self._max


class ResultTruncator:
    """Truncates results to prevent unbounded result sizes."""

    @staticmethod
    def truncate(result: str, max_length: int) -> str:
        """Truncate a result string if it exceeds max_length.

        Args:
            result: The result string to truncate
            max_length: Maximum allowed length

        Returns:
            The original string if within limits, or truncated string with suffix
            that includes the original character count.
        """
        original_length = len(result)
        if original_length <= max_length:
            return result

        suffix = "[truncated, original: {} chars]".format(original_length)
        available_length = max_length - len(suffix)
        if available_length <= 0:
            return result[:max_length]

        return result[:available_length] + suffix


class RegistrySizeMonitor:
    """Monitors registry size and identifies tasks to purge."""

    @staticmethod
    def check_and_purge(task_registry: "AgentTaskRegistry", max_size: int) -> List[str]:
        """Check if registry exceeds max_size and return task_ids to purge.

        Purges oldest completed tasks first (based on ended_at timestamp).

        Args:
            task_registry: The AgentTaskRegistry to monitor
            max_size: Maximum allowed registry size

        Returns:
            List of task_ids that should be purged (empty if within limits)
        """
        current_size = len(task_registry._tasks)

        if current_size <= max_size:
            return []

        to_purge_count = current_size - max_size

        completed_tasks = [
            (task_id, task)
            for task_id, task in task_registry._tasks.items()
            if task.result is not None
        ]

        completed_tasks.sort(
            key=lambda x: x[1].ended_at if x[1].ended_at is not None else float("inf")
        )

        return [task_id for task_id, _ in completed_tasks[:to_purge_count]]


__all__ = ["ConcurrencyLimiter", "ResultTruncator", "RegistrySizeMonitor"]
