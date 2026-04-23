"""Safety guards and resource limits for the Agent system.

This module provides resource limiting utilities to prevent:
- Unbounded concurrency (using semaphore)
- Infinite agent runs (using timeout)
- Unbounded result sizes (using truncation)
- Unbounded registry growth (using size limit with auto-purge)
"""

import asyncio
import time
from collections import deque
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


class TokenBucketRateLimiter:
    """Token bucket rate limiter (event-driven, no busy waiting).

    - FIFO fairness via explicit waiter queue (deque of Futures)
    - No thundering herd (scheduler wakes one waiter at a time)
    - Precise wake-up scheduling (background _scheduler task)
    - Fast path: token available + no waiters -> immediate return
    """

    def __init__(self, max_tokens: int, refill_rate: float):
        """Initialize token bucket rate limiter.

        Args:
            max_tokens: Maximum tokens in bucket (burst capacity).
                        Must be >= 1.
            refill_rate: Tokens added per second. Must be > 0.
                         Supports fractional rates (e.g., 0.5 = 1 token per 2 seconds).
        """
        if max_tokens < 1:
            raise ValueError(
                "max_tokens must be >= 1, got {}".format(max_tokens)
            )
        if refill_rate <= 0:
            raise ValueError(
                "refill_rate must be > 0, got {}".format(refill_rate)
            )
        self._max_tokens = max_tokens
        self._refill_rate = refill_rate

        self._tokens = float(max_tokens)
        self._last_refill = time.monotonic()

        self._waiters: deque = deque()  # FIFO queue of Futures
        self._lock = asyncio.Lock()
        self._wakeup_task = None  # background scheduler

    async def acquire(self) -> None:
        """Acquire one token (FIFO, event-driven).

        Fast path: if tokens available and no waiters, return immediately.
        Slow path: enqueue a Future and wait outside the lock.
        """
        loop = asyncio.get_running_loop()

        async with self._lock:
            self._refill()

            # Fast path: token available, no one waiting
            if self._tokens >= 1.0 and not self._waiters:
                self._tokens -= 1.0
                return

            # Slow path: enqueue and wait
            fut = loop.create_future()
            self._waiters.append(fut)
            self._ensure_scheduler_locked()

        # Wait OUTSIDE the lock — other acquire() calls can proceed
        try:
            await fut
        except BaseException:
            # Clean up on cancellation or error
            async with self._lock:
                if not fut.done():
                    fut.cancel()
                try:
                    self._waiters.remove(fut)
                except ValueError:
                    pass
            raise

    def _refill(self) -> None:
        """Add tokens based on elapsed time since last refill."""
        now = time.monotonic()
        elapsed = max(0.0, now - self._last_refill)
        if elapsed > 0:
            self._tokens = min(
                self._max_tokens,
                self._tokens + elapsed * self._refill_rate
            )
            self._last_refill = now

    def _ensure_scheduler_locked(self) -> None:
        """Start scheduler if not already running (lock must be held)."""
        if self._wakeup_task is None or self._wakeup_task.done():
            self._wakeup_task = asyncio.create_task(self._scheduler())

    async def _scheduler(self) -> None:
        """Background task: wake up waiters when tokens become available."""
        while True:
            async with self._lock:
                self._refill()

                if not self._waiters:
                    self._wakeup_task = None
                    return

                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    fut = self._waiters.popleft()
                    if not fut.done():
                        fut.set_result(None)
                    continue  # May be able to wake more waiters

                # Calculate precise wait time for next token
                deficit = 1.0 - self._tokens
                wait_time = deficit / self._refill_rate

            # Sleep OUTSIDE the lock
            await asyncio.sleep(wait_time)

    @property
    def available_tokens(self) -> float:
        """Current token count (may be fractional after refill)."""
        self._refill()
        return self._tokens

    @property
    def max_tokens(self) -> int:
        """Configured maximum burst capacity."""
        return self._max_tokens

    @property
    def refill_rate(self) -> float:
        """Configured refill rate (tokens per second)."""
        return self._refill_rate


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


__all__ = ["ConcurrencyLimiter", "ResultTruncator", "RegistrySizeMonitor", "TokenBucketRateLimiter"]
