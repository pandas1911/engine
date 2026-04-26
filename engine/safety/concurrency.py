"""Concurrency limiting utilities extracted from engine.safety.

Provides semaphore-based and per-lane concurrency controls.
"""

import asyncio
from dataclasses import dataclass
from typing import Dict

from engine.providers.provider_models import Lane
from engine.logging import get_logger


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


@dataclass
class LaneStatus:
    active: int
    waiting: int
    max_concurrent: int


class LaneSlot:
    """Async context manager representing an acquired lane slot."""

    def __init__(self, lane: Lane, queue: "LaneConcurrencyQueue"):
        self._lane = lane
        self._queue = queue
        self._released = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if not self._released:
            self._released = True
            self._queue._release_slot(self._lane)
        return False


class _LaneState:
    """Internal state for a single lane."""

    def __init__(self, lane: Lane, max_concurrent: int):
        self.lane = lane
        self.max_concurrent = max_concurrent
        self.active = 0
        self.waiting = 0
        self.condition = asyncio.Condition()


class LaneConcurrencyQueue:
    """Per-lane concurrency queue with independent limits.

    Uses asyncio.Condition for per-lane FIFO queuing.
    Each lane (MAIN, SUBAGENT) has its own max_concurrent limit.
    """

    def __init__(self):
        self._lanes: Dict[Lane, _LaneState] = {}

    def configure_lane(self, lane: Lane, max_concurrent: int) -> None:
        """Configure or update a lane's concurrency limit.

        Args:
            lane: The lane to configure.
            max_concurrent: Maximum concurrent slots for this lane. Must be >= 1.
        """
        if max_concurrent < 1:
            raise ValueError(
                "max_concurrent must be >= 1, got {}".format(max_concurrent)
            )
        if lane in self._lanes:
            self._lanes[lane].max_concurrent = max_concurrent
        else:
            self._lanes[lane] = _LaneState(lane, max_concurrent)

    async def acquire(self, lane: Lane, timeout: float = 120.0) -> LaneSlot:
        """Acquire a slot in the given lane.

        Fast path: if active < max_concurrent, acquire immediately.
        Slow path: wait in FIFO queue until a slot is available.

        Args:
            lane: The lane to acquire a slot in.
            timeout: Maximum seconds to wait for a slot.

        Returns:
            LaneSlot context manager.

        Raises:
            ValueError: If the lane is not configured.
            TimeoutError: If the timeout expires before a slot is available.
        """
        if lane not in self._lanes:
            raise ValueError("Lane {} is not configured".format(lane.value))

        lane_state = self._lanes[lane]
        async with lane_state.condition:
            if lane_state.active < lane_state.max_concurrent:
                lane_state.active += 1
                return LaneSlot(lane, self)

            lane_state.waiting += 1
            get_logger().info(
                "RateControl",
                "lane_queued | lane={} waiting={}".format(lane.value, lane_state.waiting),
                event_type="lane_queued",
                data={"lane": lane.value, "waiting": lane_state.waiting},
            )
            try:
                await asyncio.wait_for(
                    lane_state.condition.wait_for(
                        lambda: lane_state.active < lane_state.max_concurrent
                    ),
                    timeout=timeout,
                )
                lane_state.active += 1
                return LaneSlot(lane, self)
            except asyncio.TimeoutError:
                get_logger().warning(
                    "RateControl",
                    "lane_timeout | lane={} timeout={}".format(lane.value, timeout),
                    event_type="lane_timeout",
                    data={"lane": lane.value, "timeout": timeout},
                )
                raise TimeoutError(
                    "Timeout waiting for lane {} after {}s".format(lane.value, timeout)
                )
            finally:
                lane_state.waiting -= 1

    def _release_slot(self, lane: Lane) -> None:
        """Release a slot in the given lane.

        Decrements the active count and notifies one waiter.
        """
        lane_state = self._lanes[lane]
        lane_state.active -= 1
        get_logger().info(
            "RateControl",
            "lane_release | lane={} active={}".format(lane.value, lane_state.active),
            event_type="lane_release",
            data={"lane": lane.value, "active": lane_state.active},
        )
        asyncio.create_task(self._notify(lane_state))

    async def _notify(self, lane_state: _LaneState) -> None:
        """Notify one waiter on the lane's condition."""
        async with lane_state.condition:
            lane_state.condition.notify()

    def get_status(self) -> Dict[Lane, LaneStatus]:
        """Return a snapshot of all lanes' status."""
        return {
            lane: LaneStatus(
                active=ls.active,
                waiting=ls.waiting,
                max_concurrent=ls.max_concurrent,
            )
            for lane, ls in self._lanes.items()
        }

    async def wait_for_drain(self, timeout: float = 30.0) -> bool:
        """Wait for all lanes to have zero active slots.

        Args:
            timeout: Maximum seconds to wait.

        Returns:
            True if all lanes drained, False on timeout.
        """
        if not self._lanes:
            return True

        async def _drain_lane(lane_state: _LaneState) -> None:
            async with lane_state.condition:
                await lane_state.condition.wait_for(
                    lambda: lane_state.active == 0
                )

        try:
            await asyncio.wait_for(
                asyncio.gather(
                    *(_drain_lane(ls) for ls in self._lanes.values())
                ),
                timeout=timeout,
            )
            return True
        except asyncio.TimeoutError:
            return False
