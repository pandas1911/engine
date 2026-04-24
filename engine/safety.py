"""Safety guards and resource limits for the Agent system.

This module provides resource limiting utilities to prevent:
- Unbounded concurrency (using semaphore)
- Infinite agent runs (using timeout)
- Unbounded result sizes (using truncation)
- Unbounded registry growth (using size limit with auto-purge)
"""

import asyncio
import random
import re
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, TypeVar

from engine.providers.provider_models import Lane, ProviderProfile, ProviderHealth, PaceLevel, RateLimitSnapshot
from engine.logging import get_logger

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


class SlidingWindowRateLimiter:
    """Sliding window rate limiter with dual RPM/TPM tracking (event-driven, no busy waiting).

    - FIFO fairness via explicit waiter queue (deque of Futures)
    - No thundering herd (scheduler wakes one waiter at a time)
    - Precise wake-up scheduling (background _scheduler task)
    - Fast path: capacity available + no waiters -> immediate return
    - Tracks both requests-per-minute (RPM) and tokens-per-minute (TPM)
    """

    def __init__(
        self,
        rpm_limit: float,
        tpm_limit: float,
        window_seconds: float = 60.0,
        profile_name: str = "default",
    ):
        """Initialize sliding window rate limiter.

        Args:
            rpm_limit: Maximum requests per window. Zero disables RPM tracking.
            tpm_limit: Maximum tokens per window. Zero disables TPM tracking.
            window_seconds: Sliding window duration in seconds.
            profile_name: Identifier for logging.

        Raises:
            ValueError: If both rpm_limit and tpm_limit are <= 0.
        """
        if rpm_limit <= 0 and tpm_limit <= 0:
            raise ValueError(
                "at least one of rpm_limit or tpm_limit must be > 0, got rpm_limit={} tpm_limit={}".format(
                    rpm_limit, tpm_limit
                )
            )
        self._rpm_limit = rpm_limit
        self._tpm_limit = tpm_limit
        self._window_seconds = window_seconds
        self._profile_name = profile_name

        self._rpm_entries: deque = deque()  # (timestamp, 0)
        self._tpm_entries: deque = deque()  # (timestamp, token_count)

        self._waiters: deque = deque()  # FIFO queue of (Future, estimated_tokens)
        self._lock = asyncio.Lock()
        self._scheduler_task = None  # background scheduler

    def _prune_stale(self) -> None:
        """Remove entries older than window_seconds from both deques."""
        now = time.monotonic()
        cutoff = now - self._window_seconds
        while self._rpm_entries and self._rpm_entries[0][0] < cutoff:
            self._rpm_entries.popleft()
        while self._tpm_entries and self._tpm_entries[0][0] < cutoff:
            self._tpm_entries.popleft()

    def _current_tpm(self) -> float:
        """Sum of token counts in the TPM window."""
        return sum(tokens for _, tokens in self._tpm_entries)

    def _can_acquire(self, estimated_tokens: int = 0) -> bool:
        """Check if a request with estimated_tokens can proceed."""
        rpm_ok = self._rpm_limit <= 0 or len(self._rpm_entries) < self._rpm_limit
        if not rpm_ok:
            return False
        tpm_ok = (
            self._tpm_limit <= 0
            or (self._current_tpm() + estimated_tokens) <= self._tpm_limit
        )
        return tpm_ok

    async def acquire(self, estimated_tokens: int = 0) -> None:
        """Acquire rate limit slot (FIFO, event-driven).

        Fast path: if capacity available and no waiters, return immediately.
        Slow path: enqueue a Future and wait outside the lock.
        """
        loop = asyncio.get_running_loop()

        async with self._lock:
            self._prune_stale()

            # Fast path: capacity available, no one waiting
            if self._can_acquire(estimated_tokens) and not self._waiters:
                self._rpm_entries.append((time.monotonic(), 0))
                return

            # Slow path: enqueue and wait
            fut = loop.create_future()
            self._waiters.append((fut, estimated_tokens))
            self._ensure_scheduler_locked()

            get_logger().warning(
                "RateControl",
                "Rate limit blocked | profile={}, rpm={}/{}, tpm={}/{}, waiters={}".format(
                    self._profile_name,
                    len(self._rpm_entries),
                    self._rpm_limit,
                    self._current_tpm(),
                    self._tpm_limit,
                    len(self._waiters),
                ),
                event_type="rate_limit_blocked",
                data={
                    "profile": self._profile_name,
                    "rpm_entries": len(self._rpm_entries),
                    "rpm_limit": self._rpm_limit,
                    "tpm_entries": self._current_tpm(),
                    "tpm_limit": self._tpm_limit,
                    "waiters": len(self._waiters),
                },
            )

        # Wait OUTSIDE the lock -- other acquire() calls can proceed
        try:
            await fut
        except BaseException:
            # Clean up on cancellation or error
            async with self._lock:
                if not fut.done():
                    fut.cancel()
                try:
                    self._waiters.remove((fut, estimated_tokens))
                except ValueError:
                    pass
            raise

    def record_usage(self, tokens: int) -> None:
        """Record actual token usage after a request completes.

        Appends (timestamp, tokens) to the TPM window.
        """
        now = time.monotonic()
        self._tpm_entries.append((now, tokens))
        get_logger().info(
            "RateControl",
            "Rate limit usage recorded | profile={}, tokens={}, tpm={}/{}".format(
                self._profile_name,
                tokens,
                self._current_tpm(),
                self._tpm_limit,
            ),
            event_type="rate_limit_usage",
            data={
                "profile": self._profile_name,
                "tokens": tokens,
                "tpm": self._current_tpm(),
                "tpm_limit": self._tpm_limit,
            },
        )

    def _ensure_scheduler_locked(self) -> None:
        """Start scheduler if not already running (lock must be held)."""
        if self._scheduler_task is None or self._scheduler_task.done():
            self._scheduler_task = asyncio.create_task(self._scheduler())

    async def _scheduler(self) -> None:
        """Background task: wake up waiters when capacity becomes available."""
        while True:
            async with self._lock:
                self._prune_stale()

                if not self._waiters:
                    self._scheduler_task = None
                    return

                fut, estimated_tokens = self._waiters[0]
                if self._can_acquire(estimated_tokens):
                    self._waiters.popleft()
                    self._rpm_entries.append((time.monotonic(), 0))
                    if not fut.done():
                        fut.set_result(None)
                        get_logger().info(
                            "RateControl",
                            "Rate limit unblocked | profile={}, waiters_remaining={}".format(
                                self._profile_name,
                                len(self._waiters),
                            ),
                            event_type="rate_limit_unblock",
                            data={
                                "profile": self._profile_name,
                                "waiters_remaining": len(self._waiters),
                            },
                        )
                    continue  # May be able to wake more waiters

                # Calculate precise wait time until oldest entry expires
                oldest_ts = float("inf")
                if self._rpm_entries:
                    oldest_ts = min(oldest_ts, self._rpm_entries[0][0])
                if self._tpm_entries:
                    oldest_ts = min(oldest_ts, self._tpm_entries[0][0])

                now = time.monotonic()
                if oldest_ts != float("inf"):
                    sleep_time = max(0.0, oldest_ts + self._window_seconds - now)
                else:
                    sleep_time = 0.05  # Fallback, shouldn't happen in practice

            # Sleep OUTSIDE the lock
            await asyncio.sleep(sleep_time)

    def get_snapshot(self):
        """Return current RPM/TPM remaining counts.

        Uses lazy import to avoid circular dependencies.
        """
        from engine.providers.provider_models import RateLimitSnapshot

        remaining_rpm = (
            None
            if self._rpm_limit <= 0
            else max(0, int(self._rpm_limit - len(self._rpm_entries)))
        )
        remaining_tpm = (
            None
            if self._tpm_limit <= 0
            else max(0, int(self._tpm_limit - self._current_tpm()))
        )
        limit_rpm = None if self._rpm_limit <= 0 else int(self._rpm_limit)
        limit_tpm = None if self._tpm_limit <= 0 else int(self._tpm_limit)
        return RateLimitSnapshot(
            remaining_rpm=remaining_rpm,
            remaining_tpm=remaining_tpm,
            limit_rpm=limit_rpm,
            limit_tpm=limit_tpm,
        )

    def get_remaining_fraction(self) -> float:
        """Return the minimum of RPM and TPM remaining fractions."""
        rpm_frac = (
            1.0
            if self._rpm_limit <= 0
            else max(0.0, (self._rpm_limit - len(self._rpm_entries)) / self._rpm_limit)
        )
        tpm_frac = (
            1.0
            if self._tpm_limit <= 0
            else max(0.0, (self._tpm_limit - self._current_tpm()) / self._tpm_limit)
        )
        return min(rpm_frac, tpm_frac)

    @property
    def rpm_limit(self) -> float:
        """Configured RPM limit."""
        return self._rpm_limit

    @property
    def tpm_limit(self) -> float:
        """Configured TPM limit."""
        return self._tpm_limit

    @property
    def window_seconds(self) -> float:
        """Configured window duration in seconds."""
        return self._window_seconds


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


class APIKeyPool:
    """Multi-key management with staircase cooldown and automatic rotation.

    Manages a pool of provider API keys, tracking health state per key.
    On rate limit, keys enter a staircase cooldown (30s -> 60s -> 300s).
    Successful requests reset cooldown and error counts.
    """

    def __init__(
        self,
        profiles: List[ProviderProfile],
        cooldown_initial_ms: float = 30000.0,
        cooldown_max_ms: float = 300000.0,
    ):
        if not profiles:
            raise ValueError("at least one profile is required")
        self._profiles: Dict[str, ProviderProfile] = {p.name: p for p in profiles}
        self._health: Dict[str, ProviderHealth] = {
            p.name: ProviderHealth(profile_name=p.name) for p in profiles
        }
        self._cooldown_initial_ms = cooldown_initial_ms
        self._cooldown_max_ms = cooldown_max_ms

    def acquire_key(self) -> ProviderProfile:
        """Select the best available key.

        Filters out keys in cooldown. If none available, returns the
        least-recently-cooled key (sorted by cooldown_until ascending).
        Among available keys, prefers higher weight, then fewer errors.
        """
        now = time.monotonic()

        candidates = [
            (name, health)
            for name, health in self._health.items()
            if health.cooldown_until is None or health.cooldown_until <= now
        ]

        if not candidates:
            all_profiles = list(self._health.items())
            all_profiles.sort(key=lambda x: x[1].cooldown_until or 0.0)
            return self._profiles[all_profiles[0][0]]

        candidates.sort(
            key=lambda x: (-self._profiles[x[0]].weight, x[1].consecutive_errors)
        )
        return self._profiles[candidates[0][0]]

    def report_rate_limited(
        self, profile_name: str, retry_after_ms: Optional[float] = None
    ) -> None:
        """Report a rate limit for the given profile.

        Increments consecutive errors and applies staircase cooldown.
        Logs warning. If all keys are in cooldown, logs error.
        """
        health = self._health[profile_name]
        health.consecutive_errors += 1
        health.last_error_time = time.monotonic()

        steps = [
            self._cooldown_initial_ms,
            self._cooldown_initial_ms * 2.0,
            self._cooldown_max_ms,
        ]
        idx = min(health.consecutive_errors - 1, len(steps) - 1)
        cooldown_ms = max(steps[idx], retry_after_ms or 0.0)
        health.cooldown_until = time.monotonic() + cooldown_ms / 1000.0

        get_logger().warning(
            "RateControl",
            "Key cooldown | profile={} consecutive_errors={} cooldown_ms={}".format(
                profile_name, health.consecutive_errors, cooldown_ms
            ),
            event_type="key_cooldown",
            data={
                "profile": profile_name,
                "consecutive_errors": health.consecutive_errors,
                "cooldown_ms": cooldown_ms,
            },
        )

        if self.is_all_in_cooldown():
            get_logger().error(
                "RateControl",
                "Key pool exhausted | all_profiles_in_cooldown",
                event_type="key_pool_exhausted",
                data={"pool_size": len(self._profiles)},
            )

    def report_success(self, profile_name: str) -> None:
        """Report a successful request for the given profile.

        Resets consecutive_errors and cooldown. Logs recovery if
        the profile was previously in an error state.
        """
        health = self._health[profile_name]
        was_in_error = health.consecutive_errors > 0 or health.cooldown_until is not None

        health.consecutive_errors = 0
        health.cooldown_until = None

        if was_in_error:
            get_logger().info(
                "RateControl",
                "Key recovered | profile={}".format(profile_name),
                event_type="key_recovered",
                data={"profile": profile_name},
            )

    def is_all_in_cooldown(self) -> bool:
        """Return True if all profiles are currently in cooldown."""
        now = time.monotonic()
        return all(
            health.cooldown_until is not None and health.cooldown_until > now
            for health in self._health.values()
        )

    def get_cooldown_status(self) -> Dict[str, ProviderHealth]:
        """Return a copy of the health dict for all profiles."""
        return {
            name: ProviderHealth(
                profile_name=h.profile_name,
                consecutive_errors=h.consecutive_errors,
                last_error_time=h.last_error_time,
                cooldown_until=h.cooldown_until,
                pace_level=h.pace_level,
            )
            for name, h in self._health.items()
        }

    def get_active_profiles(self) -> List[ProviderProfile]:
        """Return profiles not currently in cooldown."""
        now = time.monotonic()
        return [
            self._profiles[name]
            for name, health in self._health.items()
            if health.cooldown_until is None or health.cooldown_until <= now
        ]


T = TypeVar("T")


class RetryEngine:
    """Error classification and retry logic with exponential backoff and jitter.

    Handles classification of errors into RATE_LIMITED, RETRYABLE, and
    NON_RETRYABLE categories, extracts retry-after hints from error messages,
    computes delays with exponential backoff and configurable jitter, and
    executes async functions with automatic retry.
    """

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter_mode: str = "symmetric",
    ):
        if max_attempts < 1:
            raise ValueError(
                "max_attempts must be >= 1, got {}".format(max_attempts)
            )
        if base_delay <= 0:
            raise ValueError(
                "base_delay must be > 0, got {}".format(base_delay)
            )
        if max_delay <= 0:
            raise ValueError(
                "max_delay must be > 0, got {}".format(max_delay)
            )
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter_mode = jitter_mode

    def classify_error(self, error: Exception) -> Any:
        """Classify an exception into an ErrorClass.

        Checks error string for common HTTP status codes and rate-limit
        patterns. Order matters: RATE_LIMITED first, then NON_RETRYABLE,
        then default RETRYABLE.
        """
        from engine.providers.provider_models import ErrorClass

        msg = str(error).lower()

        rate_limited_patterns = [
            "429",
            "rate limit",
            "too many requests",
            "quota exceeded",
            "resource_exhausted",
        ]
        if any(p in msg for p in rate_limited_patterns):
            return ErrorClass.RATE_LIMITED

        non_retryable_patterns = [
            "401",
            "403",
            "authentication",
            "invalid api key",
            "unauthorized",
        ]
        if any(p in msg for p in non_retryable_patterns):
            return ErrorClass.NON_RETRYABLE

        return ErrorClass.RETRYABLE

    def extract_retry_after(self, error: Exception) -> Optional[float]:
        """Extract retry-after value from error message in milliseconds.

        Checks for common patterns like 'retry-after: 30', 'retry after 5',
        and 'try again in X seconds'. Returns value in milliseconds, or
        None if no pattern is found.
        """
        msg = str(error)

        # Pattern: retry-after: 30 (with optional unit)
        match = re.search(
            r"retry-after[:\s]+(\d+(?:\.\d+)?)\s*(ms|milliseconds?)?",
            msg,
            re.IGNORECASE,
        )
        if match:
            value = float(match.group(1))
            if match.group(2):
                return value
            return value * 1000.0

        # Pattern: retry after 5 (without colon)
        match = re.search(
            r"retry after[:\s]+(\d+(?:\.\d+)?)\s*(ms|milliseconds?)?",
            msg,
            re.IGNORECASE,
        )
        if match:
            value = float(match.group(1))
            if match.group(2):
                return value
            return value * 1000.0

        # Pattern: try again in X seconds
        match = re.search(
            r"try again in[:\s]+(\d+(?:\.\d+)?)\s*(ms|milliseconds?|s|seconds?)?",
            msg,
            re.IGNORECASE,
        )
        if match:
            value = float(match.group(1))
            unit = match.group(2) or ""
            if unit.lower().startswith("ms"):
                return value
            return value * 1000.0

        return None

    def compute_delay(
        self, attempt: int, retry_after_ms: Optional[float] = None
    ) -> float:
        """Compute delay for a retry attempt with exponential backoff and jitter.

        Args:
            attempt: Current attempt number (1-based).
            retry_after_ms: Optional retry-after hint from provider in ms.

        Returns:
            Delay in seconds to wait before next attempt.
        """
        base = min(self.base_delay * (2 ** (attempt - 1)), self.max_delay)

        if retry_after_ms is not None:
            base = max(base, retry_after_ms / 1000.0)

        if retry_after_ms is not None:
            # Positive jitter only when retry_after is present
            jitter = random.uniform(1.0, 1.5)
        else:
            jitter = random.uniform(0.5, 1.5)

        return base * jitter

    async def execute_with_retry(
        self,
        fn: Callable[..., Any],
        on_retry: Optional[Callable[[int, Exception, float], None]] = None,
    ) -> Any:
        """Execute an async function with retry logic.

        Args:
            fn: Async callable to execute.
            on_retry: Optional callback invoked on each retry with
                (attempt, error, delay_seconds).

        Returns:
            Result from fn().

        Raises:
            LLMProviderError: When all retry attempts are exhausted.
            Exception: When a NON_RETRYABLE error is encountered.
        """
        from engine.logging import get_logger
        from engine.providers.llm_provider import LLMProviderError
        from engine.providers.provider_models import ErrorClass

        logger = get_logger()
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                return await fn()
            except Exception as e:
                last_error = e
                error_class = self.classify_error(e)

                if error_class == ErrorClass.NON_RETRYABLE:
                    raise

                if attempt >= self.max_attempts:
                    break

                retry_after_ms = self.extract_retry_after(e)
                delay = self.compute_delay(attempt, retry_after_ms)

                logger.warning(
                    "RateControl",
                    "Retry attempt | attempt={}/{}, delay={:.2f}s, error_class={}, error='{}'".format(
                        attempt, self.max_attempts, delay,
                        error_class.value, str(e)[:200]
                    ),
                    event_type="retry_attempt",
                    data={
                        "attempt": attempt,
                        "max_attempts": self.max_attempts,
                        "delay_seconds": delay,
                        "error_class": error_class.value,
                        "error_message": str(e)[:500],
                    },
                )

                if on_retry is not None:
                    on_retry(attempt, e, delay)

                await asyncio.sleep(delay)

        raise LLMProviderError(last_error)


class AdaptivePacer:
    """Provider-agnostic dynamic request throttling based on rate limit snapshots.

    Transitions between HEALTHY, PRESSING, and CRITICAL pace levels based on
    the remaining fraction of rate limit capacity. Enforces a minimum interval
    between calls and adds recommended delays at higher pace levels.
    """

    def __init__(self, min_interval_ms: float = 500, enabled: bool = True):
        self._min_interval_ms = min_interval_ms
        self._enabled = enabled
        self._remaining_fraction = 1.0
        self._last_call_timestamp: Optional[float] = None
        self._pace_level = PaceLevel.HEALTHY

    def update_from_snapshot(self, snapshot: RateLimitSnapshot) -> None:
        """Update pace level from a RateLimitSnapshot.

        Calculates the remaining fraction from RPM and TPM data, then
        transitions the pace level accordingly.
        """
        fractions = []
        if (
            snapshot.remaining_rpm is not None
            and snapshot.limit_rpm is not None
            and snapshot.limit_rpm > 0
        ):
            fractions.append(snapshot.remaining_rpm / snapshot.limit_rpm)
        if (
            snapshot.remaining_tpm is not None
            and snapshot.limit_tpm is not None
            and snapshot.limit_tpm > 0
        ):
            fractions.append(snapshot.remaining_tpm / snapshot.limit_tpm)

        if fractions:
            self._remaining_fraction = min(fractions)
        else:
            self._remaining_fraction = 1.0

        old_level = self._pace_level
        if self._remaining_fraction > 0.5:
            self._pace_level = PaceLevel.HEALTHY
        elif self._remaining_fraction >= 0.2:
            self._pace_level = PaceLevel.PRESSING
        else:
            self._pace_level = PaceLevel.CRITICAL

        if self._pace_level != old_level:
            get_logger().info(
                "RateControl",
                "pace_change | old={} new={} fraction={:.2f}".format(
                    old_level.value,
                    self._pace_level.value,
                    self._remaining_fraction,
                ),
                event_type="pace_change",
                data={
                    "old_level": old_level.value,
                    "new_level": self._pace_level.value,
                    "fraction": self._remaining_fraction,
                },
            )

    def update_from_tokens(self, used_tokens: int, limit_tokens: int) -> None:
        """Update pace level from manual token tracking.

        Fallback path when response headers are not available.
        Calculates fraction as 1.0 - (used_tokens / limit_tokens).
        """
        if limit_tokens > 0:
            self._remaining_fraction = 1.0 - (used_tokens / limit_tokens)
        else:
            self._remaining_fraction = 1.0

        old_level = self._pace_level
        if self._remaining_fraction > 0.5:
            self._pace_level = PaceLevel.HEALTHY
        elif self._remaining_fraction >= 0.2:
            self._pace_level = PaceLevel.PRESSING
        else:
            self._pace_level = PaceLevel.CRITICAL

        if self._pace_level != old_level:
            get_logger().info(
                "RateControl",
                "pace_change | old={} new={} fraction={:.2f}".format(
                    old_level.value,
                    self._pace_level.value,
                    self._remaining_fraction,
                ),
                event_type="pace_change",
                data={
                    "old_level": old_level.value,
                    "new_level": self._pace_level.value,
                    "fraction": self._remaining_fraction,
                },
            )

    def get_pace_level(self) -> PaceLevel:
        return self._pace_level

    def get_recommended_delay(self) -> float:
        """Return the recommended extra delay in milliseconds.

        - HEALTHY: 0 ms
        - PRESSING: 200 ms
        - CRITICAL: 1000 ms
        """
        if self._pace_level == PaceLevel.HEALTHY:
            return 0.0
        elif self._pace_level == PaceLevel.PRESSING:
            return 200.0
        else:
            return 1000.0

    async def wait_if_needed(self) -> None:
        """Enforce min interval and recommended delay before the next call.

        If the pacer is disabled, returns immediately.
        Updates _last_call_timestamp after waiting.
        """
        if not self._enabled:
            return

        now = time.monotonic()
        actual_delay_ms = 0.0

        if self._last_call_timestamp is not None:
            elapsed_ms = (now - self._last_call_timestamp) * 1000.0
            remaining_ms = max(0.0, self._min_interval_ms - elapsed_ms)
            actual_delay_ms += remaining_ms
        actual_delay_ms += self.get_recommended_delay()

        if actual_delay_ms > 0:
            await asyncio.sleep(actual_delay_ms / 1000.0)
            get_logger().info(
                "RateControl",
                "pace_wait | delay_ms={:.1f} level={}".format(
                    actual_delay_ms,
                    self._pace_level.value,
                ),
                event_type="pace_wait",
                data={
                    "delay_ms": actual_delay_ms,
                    "pace_level": self._pace_level.value,
                },
            )

        self._last_call_timestamp = time.monotonic()


__all__ = [
    "AdaptivePacer",
    "APIKeyPool",
    "ConcurrencyLimiter",
    "LaneConcurrencyQueue",
    "LaneSlot",
    "LaneStatus",
    "ResultTruncator",
    "RegistrySizeMonitor",
    "TokenBucketRateLimiter",
    "SlidingWindowRateLimiter",
    "RetryEngine",
]
