"""Sliding window rate limiter extracted from engine.safety.

Provides RPM/TPM rate limiting with event-driven scheduling.
"""

import asyncio
import sys
import time
from collections import deque
from typing import Optional

from engine.logging import get_logger


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
        self._tpm_entries: deque = deque()  # (timestamp, token_count, reservation_id_or_None)

        self._waiters: deque = deque()  # FIFO queue of (Future, estimated_tokens)
        self._lock = asyncio.Lock()
        self._scheduler_task = None  # background scheduler
        self._next_reservation_id: int = 1  # 0 is sentinel for "no reservation"

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
        return sum(tokens for _, tokens, _ in self._tpm_entries)

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

    def _remove_tpm_entry_by_rid(self, reservation_id: int) -> bool:
        """Remove TPM entry by reservation_id. Caller MUST hold self._lock.
        Returns True if found and removed, False otherwise."""
        if reservation_id <= 0:
            return False
        for i in range(len(self._tpm_entries) - 1, -1, -1):
            if self._tpm_entries[i][2] == reservation_id:
                del self._tpm_entries[i]
                return True
        return False

    async def acquire(self, estimated_tokens: int = 0) -> int:
        """Acquire rate limit slot (FIFO, event-driven).

        Fast path: if capacity available and no waiters, return immediately.
        Slow path: enqueue a Future and wait outside the lock.
        """
        loop = asyncio.get_running_loop()

        async with self._lock:
            self._prune_stale()

            # Prevent deadlock: cap estimated_tokens so it never exceeds tpm_limit
            # When window is empty, this ensures _can_acquire() can always return True
            if self._tpm_limit > 0:
                estimated_tokens = min(estimated_tokens, max(1, int(self._tpm_limit) - 1))

            # Fast path: capacity available, no one waiting
            if self._can_acquire(estimated_tokens) and not self._waiters:
                self._rpm_entries.append((time.monotonic(), 0))
                if estimated_tokens > 0:
                    rid = self._next_reservation_id
                    self._next_reservation_id += 1
                    self._tpm_entries.append((time.monotonic(), estimated_tokens, rid))
                    return rid
                return 0  # No reservation for estimated_tokens=0

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
                    self._current_tpm() + estimated_tokens,
                    self._tpm_limit,
                    len(self._waiters),
                ),
                event_type="rate_limit_blocked",
                data={
                    "profile": self._profile_name,
                    "rpm_entries": len(self._rpm_entries),
                    "rpm_limit": self._rpm_limit,
                    "tpm_entries": self._current_tpm() + estimated_tokens,
                    "tpm_limit": self._tpm_limit,
                    "waiters": len(self._waiters),
                },
            )

        # Wait OUTSIDE the lock -- other acquire() calls can proceed
        timeout_seconds = self._window_seconds * 2  # 2 full window rotations
        try:
            return await asyncio.wait_for(fut, timeout=timeout_seconds)
        except BaseException:
            async with self._lock:
                if not fut.done():
                    fut.cancel()
                else:
                    try:
                        rid = fut.result()
                    except BaseException:
                        rid = None
                    if rid and rid > 0:
                        self._remove_tpm_entry_by_rid(rid)
                try:
                    self._waiters.remove((fut, estimated_tokens))
                except ValueError:
                    pass
            if isinstance(sys.exc_info()[1], asyncio.TimeoutError):
                get_logger().error(
                    "RateControl",
                    "Rate limiter acquire timed out | profile={}, timeout={}s".format(
                        self._profile_name, timeout_seconds
                    ),
                    event_type="rate_limit_acquire_timeout",
                    data={"profile": self._profile_name, "timeout_seconds": timeout_seconds},
                )
            raise

    async def record_usage(self, tokens: int, reservation_id: Optional[int] = None) -> None:
        """Record actual token usage after a request completes.

        If reservation_id is provided, replaces the tentative entry with actual usage.
        If reservation_id is None or not found, appends as backward-compatible entry.
        """
        async with self._lock:
            if reservation_id is not None and reservation_id > 0:
                for i in range(len(self._tpm_entries) - 1, -1, -1):
                    entry = self._tpm_entries[i]
                    if entry[2] == reservation_id:
                        self._tpm_entries[i] = (entry[0], tokens, None)
                        get_logger().info(
                            "RateControl",
                            "Rate limit usage recorded (replaced reservation) | profile={}, tokens={}, reservation_id={}, tpm={}/{}".format(
                                self._profile_name,
                                tokens,
                                reservation_id,
                                self._current_tpm(),
                                self._tpm_limit,
                            ),
                            event_type="rate_limit_usage",
                            data={
                                "profile": self._profile_name,
                                "tokens": tokens,
                                "reservation_id": reservation_id,
                                "tpm": self._current_tpm(),
                                "tpm_limit": self._tpm_limit,
                            },
                        )
                        return
            self._tpm_entries.append((time.monotonic(), tokens, None))
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

    async def release_reserved(self, reservation_id: int) -> None:
        """Release a tentative TPM reservation.

        Idempotent: no-op if reservation_id not found or already released.
        """
        if reservation_id <= 0:
            return
        async with self._lock:
            if self._remove_tpm_entry_by_rid(reservation_id):
                get_logger().info(
                    "RateControl",
                    "Rate limit reservation released | profile={}, reservation_id={}".format(
                        self._profile_name,
                        reservation_id,
                    ),
                    event_type="rate_limit_reservation_released",
                    data={
                        "profile": self._profile_name,
                        "reservation_id": reservation_id,
                    },
                )
                return

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
                        rid = 0
                        if estimated_tokens > 0:
                            rid = self._next_reservation_id
                            self._next_reservation_id += 1
                            self._tpm_entries.append((time.monotonic(), estimated_tokens, rid))
                        fut.set_result(rid)
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

                # Deadlock detection: window is empty but waiter can't be released
                # This means estimated_tokens > tpm_limit (after Layer 1 cap, should never happen)
                # Force-release as safety net
                if not self._rpm_entries and not self._tpm_entries:
                    get_logger().error(
                        "RateControl",
                        "Rate limiter deadlock detected | profile={}, estimated_tokens={}, tpm_limit={}, waiters={}".format(
                            self._profile_name, estimated_tokens, self._tpm_limit, len(self._waiters)
                        ),
                        event_type="rate_limit_deadlock",
                        data={
                            "profile": self._profile_name,
                            "estimated_tokens": estimated_tokens,
                            "tpm_limit": self._tpm_limit,
                            "waiters": len(self._waiters),
                        },
                    )
                    self._waiters.popleft()
                    self._rpm_entries.append((time.monotonic(), 0))
                    if not fut.done():
                        fut.set_result(0)  # No TPM reservation for force-released
                    continue

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
