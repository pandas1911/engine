"""API key pool management extracted from engine.safety.

Provides multi-key rotation with staircase cooldown.
"""

import time
from typing import Dict, List, Optional

from engine.providers.provider_models import ProviderHealth
from engine.logging import get_logger


class APIKeyPool:
    """Multi-key management with staircase cooldown and automatic rotation.

    Manages a pool of provider key names, tracking health state per key.
    On rate limit, keys enter a staircase cooldown (30s -> 60s -> 300s).
    Successful requests reset cooldown and error counts.

    Keys are identified by composite strings (e.g., "aliyun/deepseek-v4-pro").
    Selection prefers keys with the lowest consecutive_errors among those
    not in cooldown, providing sequential primary -> fallback ordering.
    """

    def __init__(
        self,
        names: List[str],
        cooldown_initial_ms: float = 30000.0,
        cooldown_max_ms: float = 300000.0,
    ):
        if not names:
            raise ValueError("at least one name is required")
        self._names: List[str] = list(names)
        self._health: Dict[str, ProviderHealth] = {
            name: ProviderHealth(profile_name=name) for name in names
        }
        self._cooldown_initial_ms = cooldown_initial_ms
        self._cooldown_max_ms = cooldown_max_ms

    def acquire_key(self) -> str:
        """Select the best available key.

        Filters out keys in cooldown. If none available, returns the
        least-recently-cooled key (sorted by cooldown_until ascending).
        Among available keys, prefers fewer consecutive_errors, preserving
        the original insertion order (primary first, then fallbacks).
        """
        now = time.monotonic()

        candidates = [
            (name, health)
            for name, health in self._health.items()
            if health.cooldown_until is None or health.cooldown_until <= now
        ]

        if not candidates:
            all_entries = list(self._health.items())
            all_entries.sort(key=lambda x: x[1].cooldown_until or 0.0)
            return all_entries[0][0]

        candidates.sort(key=lambda x: x[1].consecutive_errors)
        return candidates[0][0]

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
                data={"pool_size": len(self._names)},
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

    def get_active_names(self) -> List[str]:
        """Return key names not currently in cooldown."""
        now = time.monotonic()
        return [
            name
            for name, health in self._health.items()
            if health.cooldown_until is None or health.cooldown_until <= now
        ]
