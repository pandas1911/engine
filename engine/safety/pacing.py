"""Adaptive pacing and resource monitoring extracted from engine.safety.

Provides result truncation, registry size monitoring, and adaptive request throttling.
"""

import asyncio
import time
from typing import TYPE_CHECKING, List, Optional

from engine.providers.provider_models import PaceLevel, RateLimitSnapshot
from engine.logging import get_logger

if TYPE_CHECKING:
    from engine.runtime.task_registry import AgentTaskRegistry


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


class AdaptivePacer:
    """Provider-agnostic dynamic request throttling based on rate limit snapshots.

    Transitions between HEALTHY, PRESSING, and CRITICAL pace levels based on
    the remaining fraction of rate limit capacity. Enforces a minimum interval
    between calls and adds recommended delays at higher pace levels.
    """

    def __init__(self, min_interval_ms: float = 500, enabled: bool = True, rpm_limit: float = 0):
        self._min_interval_ms = min_interval_ms
        self._enabled = enabled
        self._remaining_fraction = 1.0
        self._last_call_timestamp: Optional[float] = None
        self._pace_level = PaceLevel.HEALTHY
        self._lock = asyncio.Lock()

        # Effective min interval: never exceed RPM limit rate.
        if rpm_limit > 0:
            rpm_derived_ms = 60000.0 / rpm_limit
            self._effective_min_interval_ms = max(min_interval_ms, rpm_derived_ms)
        else:
            self._effective_min_interval_ms = min_interval_ms

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

        async with self._lock:
            now = time.monotonic()
            actual_delay_ms = 0.0

            if self._last_call_timestamp is not None:
                elapsed_ms = (now - self._last_call_timestamp) * 1000.0
                remaining_ms = max(0.0, self._effective_min_interval_ms - elapsed_ms)
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
