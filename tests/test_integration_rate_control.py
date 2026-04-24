"""Integration tests for the rate control system.

These tests verify end-to-end behavior of all rate control components
working together. All external API calls are mocked.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from engine.safety import (
    LaneConcurrencyQueue,
    SlidingWindowRateLimiter,
    AdaptivePacer,
    APIKeyPool,
    RetryEngine,
)
from engine.providers.fallback_provider import FallbackLLMProvider
from engine.providers.provider_models import (
    ProviderProfile,
    Lane,
    PaceLevel,
    RateLimitSnapshot,
)
from engine.providers.llm_provider import LLMProviderError


def _make_profiles(names=None):
    """Create provider profiles for testing."""
    if names is None:
        names = ["profile-a", "profile-b"]
    return [
        ProviderProfile(
            name=name,
            api_key="key-{}".format(name),
            base_url="https://api.test.com",
            model="gpt-4",
            rpm_limit=60.0,
            tpm_limit=100000.0,
            weight=1,
        )
        for name in names
    ]


def _make_mock_provider(response_content="OK", raise_error=None, usage=None, snapshot=None):
    """Create a mock LLMProvider for testing."""
    provider = MagicMock()
    if raise_error:
        provider.chat = AsyncMock(side_effect=raise_error)
    else:
        mock_resp = MagicMock()
        mock_resp.content = response_content
        mock_resp.tool_calls = []
        mock_resp.has_tool_calls.return_value = False
        provider.chat = AsyncMock(return_value=mock_resp)
    provider.get_last_usage = MagicMock(return_value=usage)
    provider.get_rate_limit_snapshot = MagicMock(return_value=snapshot)
    provider.stream_chat = AsyncMock(return_value=None)
    return provider


class TestIntegrationRateControl:
    """Integration tests for the full rate control pipeline."""

    @pytest.mark.asyncio
    async def test_full_pipeline_single_provider_backward_compat(self):
        """Full pipeline with single provider: verify component wiring,
        rate limiting blocks when limits exceeded, and lane queue queues
        subagent spawns.
        """
        profile = ProviderProfile(
            name="default",
            api_key="test-key",
            base_url="https://api.test.com",
            model="gpt-4",
            rpm_limit=3.0,
            tpm_limit=100.0,
            weight=1,
        )

        mock_provider = _make_mock_provider(response_content="Hello", usage=(10, 20))

        rate_limiter = SlidingWindowRateLimiter(
            rpm_limit=profile.rpm_limit,
            tpm_limit=profile.tpm_limit,
            window_seconds=60.0,
            profile_name=profile.name,
        )
        pacer = AdaptivePacer(min_interval_ms=10, enabled=True)
        key_pool = APIKeyPool([profile])
        retry_engine = RetryEngine(max_attempts=2, base_delay=0.01)

        fallback = FallbackLLMProvider(
            providers={"default": mock_provider},
            key_pool=key_pool,
            rate_limiters={"default": rate_limiter},
            pacers={"default": pacer},
            retry_engine=retry_engine,
        )

        lane_queue = LaneConcurrencyQueue()
        lane_queue.configure_lane(Lane.MAIN, max_concurrent=2)
        lane_queue.configure_lane(Lane.SUBAGENT, max_concurrent=1)

        assert "default" in fallback._providers
        assert fallback._rate_limiters["default"] is rate_limiter
        assert fallback._pacers["default"] is pacer
        assert fallback._retry_engine is retry_engine
        assert fallback._key_pool is key_pool

        status = lane_queue.get_status()
        assert status[Lane.MAIN].max_concurrent == 2
        assert status[Lane.SUBAGENT].max_concurrent == 1

        result = await fallback.chat(
            messages=[{"role": "user", "content": "Hi"}],
            tools=[],
            agent_label="Test",
            task_id="t1",
        )
        assert result.content == "Hello"

        tiny_limiter = SlidingWindowRateLimiter(
            rpm_limit=2.0, tpm_limit=0.0, window_seconds=0.05, profile_name="tiny"
        )
        await tiny_limiter.acquire()
        await tiny_limiter.acquire()

        start = time.monotonic()
        await tiny_limiter.acquire()
        elapsed = time.monotonic() - start
        assert elapsed >= 0.01, (
            "Expected blocking when RPM limit exceeded, got {:.4f}s".format(elapsed)
        )

        slot = await lane_queue.acquire(Lane.SUBAGENT)
        assert lane_queue.get_status()[Lane.SUBAGENT].active == 1

        with pytest.raises(TimeoutError):
            await lane_queue.acquire(Lane.SUBAGENT, timeout=0.05)

        slot._queue._release_slot(Lane.SUBAGENT)
        assert lane_queue.get_status()[Lane.SUBAGENT].active == 0

        assert len(rate_limiter._tpm_entries) == 1

    @pytest.mark.asyncio
    async def test_multi_profile_key_rotation(self):
        """Multi-profile with key rotation: simulate 429 on first key,
        verify rotation to second key and cooldown escalation.
        """
        profiles = _make_profiles(["p1", "p2"])

        mock_p1 = _make_mock_provider(
            raise_error=Exception("429 rate limit exceeded")
        )
        mock_p2 = _make_mock_provider(response_content="Fallback OK")

        key_pool = APIKeyPool(profiles)
        rate_limiters = {
            "p1": SlidingWindowRateLimiter(
                rpm_limit=60.0, tpm_limit=100000.0, profile_name="p1"
            ),
            "p2": SlidingWindowRateLimiter(
                rpm_limit=60.0, tpm_limit=100000.0, profile_name="p2"
            ),
        }
        pacers = {
            "p1": AdaptivePacer(min_interval_ms=10, enabled=True),
            "p2": AdaptivePacer(min_interval_ms=10, enabled=True),
        }
        retry_engine = RetryEngine(max_attempts=2, base_delay=0.01)

        fallback = FallbackLLMProvider(
            providers={"p1": mock_p1, "p2": mock_p2},
            key_pool=key_pool,
            rate_limiters=rate_limiters,
            pacers=pacers,
            retry_engine=retry_engine,
            max_profile_rotations=1,
        )

        with patch("engine.safety.time.monotonic", return_value=1000.0):
            result = await fallback.chat(
                messages=[{"role": "user", "content": "Hi"}],
                tools=[],
                agent_label="Test",
                task_id="t1",
            )

        assert result.content == "Fallback OK"
        mock_p1.chat.assert_awaited_once()
        mock_p2.chat.assert_awaited_once()

        status = key_pool.get_cooldown_status()["p1"]
        assert status.consecutive_errors == 1
        assert status.cooldown_until == 1000.0 + 30.0

        status_p2 = key_pool.get_cooldown_status()["p2"]
        assert status_p2.consecutive_errors == 0
        assert status_p2.cooldown_until is None

    @pytest.mark.asyncio
    async def test_provider_fallback_ping_pong(self):
        """Provider fallback (ping-pong): main provider returns 429 on all
        keys, verify fallback to second provider, then ping back to main
        after it recovers.
        """
        profiles = _make_profiles(["main", "fallback"])

        call_count_main = [0]

        async def main_side_effect(*args, **kwargs):
            call_count_main[0] += 1
            if call_count_main[0] == 1:
                raise Exception("429 rate limit exceeded")
            mock_resp = MagicMock()
            mock_resp.content = "Main recovered"
            mock_resp.tool_calls = []
            mock_resp.has_tool_calls.return_value = False
            return mock_resp

        mock_main = MagicMock()
        mock_main.chat = AsyncMock(side_effect=main_side_effect)
        mock_main.get_last_usage = MagicMock(return_value=None)
        mock_main.get_rate_limit_snapshot = MagicMock(return_value=None)
        mock_main.stream_chat = AsyncMock(return_value=None)

        mock_fallback = _make_mock_provider(response_content="Fallback OK")

        key_pool = APIKeyPool(profiles)
        rate_limiters = {
            "main": SlidingWindowRateLimiter(
                rpm_limit=60.0, tpm_limit=100000.0, profile_name="main"
            ),
            "fallback": SlidingWindowRateLimiter(
                rpm_limit=60.0, tpm_limit=100000.0, profile_name="fallback"
            ),
        }
        pacers = {
            "main": AdaptivePacer(min_interval_ms=10, enabled=True),
            "fallback": AdaptivePacer(min_interval_ms=10, enabled=True),
        }
        retry_engine = RetryEngine(max_attempts=2, base_delay=0.01)

        fallback_provider = FallbackLLMProvider(
            providers={"main": mock_main, "fallback": mock_fallback},
            key_pool=key_pool,
            rate_limiters=rate_limiters,
            pacers=pacers,
            retry_engine=retry_engine,
            max_profile_rotations=1,
        )

        with patch("engine.safety.time.monotonic", return_value=1000.0):
            result1 = await fallback_provider.chat(
                messages=[{"role": "user", "content": "Hi"}],
                tools=[],
                agent_label="Test",
                task_id="t1",
            )

        assert result1.content == "Fallback OK"
        assert mock_main.chat.await_count == 1
        assert mock_fallback.chat.await_count == 1

        key_pool._health["main"].cooldown_until = None
        key_pool._health["main"].consecutive_errors = 0

        with patch("engine.safety.time.monotonic", return_value=2000.0):
            result2 = await fallback_provider.chat(
                messages=[{"role": "user", "content": "Hi"}],
                tools=[],
                agent_label="Test",
                task_id="t2",
            )

        assert result2.content == "Main recovered"
        assert mock_main.chat.await_count == 2

    @pytest.mark.asyncio
    async def test_adaptive_pacing_integration(self):
        """Adaptive pacing integration: mock responses with varying rate
        limit headers, verify pace level transitions
        (healthy -> pressing -> critical), and verify call interval adjustment.
        """
        profiles = _make_profiles(["p1"])

        snapshots = [
            RateLimitSnapshot(
                remaining_rpm=55, limit_rpm=60,
                remaining_tpm=90000, limit_tpm=100000,
            ),
            RateLimitSnapshot(
                remaining_rpm=25, limit_rpm=60,
                remaining_tpm=50000, limit_tpm=100000,
            ),
            RateLimitSnapshot(
                remaining_rpm=5, limit_rpm=60,
                remaining_tpm=10000, limit_tpm=100000,
            ),
        ]
        snapshot_iter = iter(snapshots)

        def get_next_snapshot():
            return next(snapshot_iter)

        mock_p1 = _make_mock_provider(response_content="OK")
        mock_p1.get_rate_limit_snapshot = MagicMock(side_effect=get_next_snapshot)

        key_pool = APIKeyPool(profiles)
        rate_limiters = {
            "p1": SlidingWindowRateLimiter(
                rpm_limit=60.0, tpm_limit=100000.0, profile_name="p1"
            ),
        }
        pacer = AdaptivePacer(min_interval_ms=50, enabled=True)
        pacers = {"p1": pacer}
        retry_engine = RetryEngine(max_attempts=2, base_delay=0.01)

        fallback = FallbackLLMProvider(
            providers={"p1": mock_p1},
            key_pool=key_pool,
            rate_limiters=rate_limiters,
            pacers=pacers,
            retry_engine=retry_engine,
        )

        # First call -> HEALTHY
        await fallback.chat(
            messages=[{"role": "user", "content": "Hi"}],
            tools=[],
            agent_label="Test",
            task_id="t1",
        )
        assert pacer.get_pace_level() == PaceLevel.HEALTHY
        assert pacer.get_recommended_delay() == 0.0

        await fallback.chat(
            messages=[{"role": "user", "content": "Hi"}],
            tools=[],
            agent_label="Test",
            task_id="t2",
        )
        assert pacer.get_pace_level() == PaceLevel.PRESSING
        assert pacer.get_recommended_delay() == 200.0

        await fallback.chat(
            messages=[{"role": "user", "content": "Hi"}],
            tools=[],
            agent_label="Test",
            task_id="t3",
        )
        assert pacer.get_pace_level() == PaceLevel.CRITICAL
        assert pacer.get_recommended_delay() == 1000.0

        pacer._last_call_timestamp = None
        await pacer.wait_if_needed()

        start = time.monotonic()
        await pacer.wait_if_needed()
        elapsed_ms = (time.monotonic() - start) * 1000.0
        assert elapsed_ms >= 900.0, (
            "Expected >=900ms delay in CRITICAL, got {:.1f}ms".format(elapsed_ms)
        )

    @pytest.mark.asyncio
    async def test_concurrent_load_lane_queue(self):
        """Concurrent load test: spawn 20 concurrent tasks through the lane
        queue, verify Main lane caps at configured limit, Subagent lane caps
        independently, and verify all tasks complete eventually (no deadlock
        or starvation).
        """
        lane_queue = LaneConcurrencyQueue()
        lane_queue.configure_lane(Lane.MAIN, max_concurrent=3)
        lane_queue.configure_lane(Lane.SUBAGENT, max_concurrent=2)

        main_peak = [0]
        subagent_peak = [0]
        main_completed = []
        subagent_completed = []

        async def main_task(task_id: int):
            async with await lane_queue.acquire(Lane.MAIN):
                status = lane_queue.get_status()
                main_peak[0] = max(main_peak[0], status[Lane.MAIN].active)
                await asyncio.sleep(0.01)
                main_completed.append(task_id)

        async def subagent_task(task_id: int):
            async with await lane_queue.acquire(Lane.SUBAGENT):
                status = lane_queue.get_status()
                subagent_peak[0] = max(
                    subagent_peak[0], status[Lane.SUBAGENT].active
                )
                await asyncio.sleep(0.01)
                subagent_completed.append(task_id)

        tasks = [
            asyncio.create_task(main_task(i)) for i in range(10)
        ] + [
            asyncio.create_task(subagent_task(i)) for i in range(10)
        ]

        await asyncio.wait_for(asyncio.gather(*tasks), timeout=5.0)

        # Verify Main lane never exceeded its cap.
        assert main_peak[0] <= 3, (
            "Main lane peaked at {}, expected <= 3".format(main_peak[0])
        )

        assert subagent_peak[0] <= 2, (
            "Subagent lane peaked at {}, expected <= 2".format(subagent_peak[0])
        )

        assert len(main_completed) == 10
        assert len(subagent_completed) == 10

        final_status = lane_queue.get_status()
        assert final_status[Lane.MAIN].active == 0
        assert final_status[Lane.MAIN].waiting == 0
        assert final_status[Lane.SUBAGENT].active == 0
        assert final_status[Lane.SUBAGENT].waiting == 0

    @pytest.mark.asyncio
    async def test_rate_limited_no_internal_retry_rotates_immediately(self):
        """429 raises immediately from LLMProvider (no internal retry),
        so FallbackLLMProvider rotates to the next key on first 429."""
        profiles = _make_profiles(["p1", "p2"])

        mock_p1 = _make_mock_provider(
            raise_error=Exception("429 rate limit exceeded")
        )
        mock_p2 = _make_mock_provider(response_content="Fallback OK")

        key_pool = APIKeyPool(profiles)
        rate_limiters = {
            "p1": SlidingWindowRateLimiter(
                rpm_limit=60.0, tpm_limit=100000.0, profile_name="p1"
            ),
            "p2": SlidingWindowRateLimiter(
                rpm_limit=60.0, tpm_limit=100000.0, profile_name="p2"
            ),
        }
        pacers = {
            "p1": AdaptivePacer(min_interval_ms=10, enabled=True),
            "p2": AdaptivePacer(min_interval_ms=10, enabled=True),
        }
        retry_engine = RetryEngine(max_attempts=2, base_delay=0.01)

        fallback = FallbackLLMProvider(
            providers={"p1": mock_p1, "p2": mock_p2},
            key_pool=key_pool,
            rate_limiters=rate_limiters,
            pacers=pacers,
            retry_engine=retry_engine,
            max_profile_rotations=1,
        )

        with patch("engine.safety.time.monotonic", return_value=1000.0):
            result = await fallback.chat(
                messages=[{"role": "user", "content": "Hi"}],
                tools=[],
                agent_label="Test",
                task_id="t1",
            )

        assert result.content == "Fallback OK"
        assert mock_p1.chat.await_count == 1, (
            "p1 should be called exactly once (no internal retry), got {}".format(
                mock_p1.chat.await_count
            )
        )
        assert mock_p2.chat.await_count == 1
