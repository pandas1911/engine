"""TDD tests for FallbackLLMProvider.

Tests cover:
- Normal operation with single provider
- Key rotation on rate limit errors
- Provider fallback (ping-pong) when all keys exhausted
- Success resets provider state
- Non-retryable errors raise immediately
- Token estimation
- stream_chat delegation
- get_active_provider_info
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from engine.providers.fallback_provider import FallbackLLMProvider
from engine.providers.llm_provider import LLMProviderError
from engine.providers.provider_models import (
    ErrorClass,
    LLMResponse,
    ProviderProfile,
    RateLimitSnapshot,
)
from engine.safety import (
    APIKeyPool,
    AdaptivePacer,
    RetryEngine,
    SlidingWindowRateLimiter,
)


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


def _make_fallback_provider(
    providers,
    profiles,
    max_profile_rotations=3,
    with_limiters=True,
    with_pacers=True,
):
    """Create a FallbackLLMProvider with real supporting components."""
    key_pool = APIKeyPool(profiles)
    rate_limiters = {}
    pacers = {}
    for p in profiles:
        if with_limiters:
            rate_limiters[p.name] = SlidingWindowRateLimiter(
                rpm_limit=p.rpm_limit,
                tpm_limit=p.tpm_limit,
                window_seconds=60.0,
                profile_name=p.name,
            )
        if with_pacers:
            pacers[p.name] = AdaptivePacer(min_interval_ms=10, enabled=True)
    retry_engine = RetryEngine(max_attempts=3, base_delay=0.01)
    return FallbackLLMProvider(
        providers=providers,
        key_pool=key_pool,
        rate_limiters=rate_limiters,
        pacers=pacers,
        retry_engine=retry_engine,
        max_profile_rotations=max_profile_rotations,
    )


class TestFallbackProviderInit:
    """Constructor and basic property tests."""

    def test_init_stores_params(self):
        """Constructor stores all parameters correctly."""
        profiles = _make_profiles(["p1"])
        provider = _make_fallback_provider(
            {"p1": _make_mock_provider()}, profiles
        )
        assert provider._max_profile_rotations == 3
        assert provider._current_profile is None
        assert provider._rotation_count == 0

    def test_init_custom_rotations(self):
        """Custom max_profile_rotations is stored."""
        profiles = _make_profiles(["p1"])
        provider = _make_fallback_provider(
            {"p1": _make_mock_provider()}, profiles, max_profile_rotations=5
        )
        assert provider._max_profile_rotations == 5


class TestFallbackProviderChatSuccess:
    """Tests for successful chat operations."""

    @pytest.mark.asyncio
    async def test_single_provider_success(self):
        """Single provider returns successful response."""
        profiles = _make_profiles(["p1"])
        mock_provider = _make_mock_provider(response_content="Hello")
        fallback = _make_fallback_provider({"p1": mock_provider}, profiles)

        result = await fallback.chat(
            messages=[{"role": "user", "content": "Hi"}],
            tools=[],
            agent_label="Test",
            task_id="t1",
        )

        assert result.content == "Hello"
        assert fallback._current_profile == "p1"
        mock_provider.chat.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_success_records_usage(self):
        """Success path records token usage with limiter."""
        profiles = _make_profiles(["p1"])
        mock_provider = _make_mock_provider(
            response_content="OK", usage=(10, 20)
        )
        fallback = _make_fallback_provider({"p1": mock_provider}, profiles)

        await fallback.chat(
            messages=[{"role": "user", "content": "Hi"}],
            tools=[],
            agent_label="Test",
            task_id="t1",
        )

        limiter = fallback._rate_limiters["p1"]
        assert len(limiter._tpm_entries) == 1
        assert limiter._tpm_entries[0][1] == 30

    @pytest.mark.asyncio
    async def test_success_updates_pacer(self):
        """Success path updates pacer with rate limit snapshot."""
        profiles = _make_profiles(["p1"])
        snapshot = RateLimitSnapshot(
            remaining_rpm=30, remaining_tpm=50000, limit_rpm=60, limit_tpm=100000
        )
        mock_provider = _make_mock_provider(
            response_content="OK", snapshot=snapshot
        )
        fallback = _make_fallback_provider({"p1": mock_provider}, profiles)

        await fallback.chat(
            messages=[{"role": "user", "content": "Hi"}],
            tools=[],
            agent_label="Test",
            task_id="t1",
        )

        pacer = fallback._pacers["p1"]
        assert pacer._remaining_fraction == 0.5

    @pytest.mark.asyncio
    async def test_success_resets_rotation_count(self):
        """After a rate limit, success resets rotation count."""
        profiles = _make_profiles(["p1", "p2"])
        mock_p1 = _make_mock_provider(
            raise_error=Exception("429 rate limit exceeded")
        )
        mock_p2 = _make_mock_provider(response_content="OK")
        fallback = _make_fallback_provider(
            {"p1": mock_p1, "p2": mock_p2}, profiles
        )

        with patch("engine.safety.time.monotonic", return_value=1000.0):
            await fallback.chat(
                messages=[{"role": "user", "content": "Hi"}],
                tools=[],
                agent_label="Test",
                task_id="t1",
            )

        assert fallback._rotation_count == 0


class TestFallbackProviderKeyRotation:
    """Tests for key rotation on rate limit errors."""

    @pytest.mark.asyncio
    async def test_rate_limit_rotates_to_next_key(self):
        """429 error causes rotation to next available key."""
        profiles = _make_profiles(["p1", "p2"])
        mock_p1 = _make_mock_provider(
            raise_error=Exception("429 rate limit exceeded")
        )
        mock_p2 = _make_mock_provider(response_content="Fallback OK")
        fallback = _make_fallback_provider(
            {"p1": mock_p1, "p2": mock_p2}, profiles
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
        assert fallback._rotation_count == 0

    @pytest.mark.asyncio
    async def test_rate_limit_reports_to_key_pool(self):
        """Rate limit reports cooldown to key pool."""
        profiles = _make_profiles(["p1", "p2"])
        mock_p1 = _make_mock_provider(
            raise_error=Exception("429 rate limit exceeded")
        )
        mock_p2 = _make_mock_provider(response_content="OK")
        fallback = _make_fallback_provider(
            {"p1": mock_p1, "p2": mock_p2}, profiles
        )

        with patch("engine.safety.time.monotonic", return_value=1000.0):
            await fallback.chat(
                messages=[{"role": "user", "content": "Hi"}],
                tools=[],
                agent_label="Test",
                task_id="t1",
            )

        status = fallback._key_pool.get_cooldown_status()["p1"]
        assert status.consecutive_errors == 1
        assert status.cooldown_until is not None

    @pytest.mark.asyncio
    async def test_rate_limit_increments_rotation_count(self):
        """Each rate limit increments rotation counter."""
        profiles = _make_profiles(["p1", "p2"])
        mock_p1 = _make_mock_provider(
            raise_error=Exception("429 rate limit exceeded")
        )
        mock_p2 = _make_mock_provider(
            raise_error=Exception("429 rate limit exceeded")
        )
        fallback = _make_fallback_provider(
            {"p1": mock_p1, "p2": mock_p2}, profiles,
            with_limiters=False, with_pacers=False,
        )

        with patch("engine.safety.time.monotonic", return_value=1000.0):
            with pytest.raises(LLMProviderError) as exc_info:
                await fallback.chat(
                    messages=[{"role": "user", "content": "Hi"}],
                    tools=[],
                    agent_label="Test",
                    task_id="t1",
                )
            assert "exceeded maximum iterations" in str(exc_info.value)
        assert fallback._rotation_count > 0


class TestFallbackProviderPingPong:
    """Tests for provider fallback (ping-pong) behavior."""

    @pytest.mark.asyncio
    async def test_ping_pong_when_all_keys_exhausted(self):
        """When max rotations exceeded, fallback resets and tries again."""
        profiles = _make_profiles(["p1", "p2"])
        # p1 always rate limited, p2 succeeds
        mock_p1 = _make_mock_provider(
            raise_error=Exception("429 rate limit exceeded")
        )
        mock_p2 = _make_mock_provider(response_content="PingPong OK")
        fallback = _make_fallback_provider(
            {"p1": mock_p1, "p2": mock_p2}, profiles, max_profile_rotations=1
        )

        with patch("engine.safety.time.monotonic", return_value=1000.0):
            result = await fallback.chat(
                messages=[{"role": "user", "content": "Hi"}],
                tools=[],
                agent_label="Test",
                task_id="t1",
            )

        assert result.content == "PingPong OK"
        # Should have tried p1, hit rotation limit, then p2
        assert mock_p1.chat.await_count >= 1
        assert mock_p2.chat.await_count >= 1

    @pytest.mark.asyncio
    async def test_rotation_counter_reset_on_fallback(self):
        """Rotation counter resets when provider fallback occurs."""
        profiles = _make_profiles(["p1", "p2"])
        mock_p1 = _make_mock_provider(
            raise_error=Exception("429 rate limit exceeded")
        )
        mock_p2 = _make_mock_provider(response_content="OK")
        fallback = _make_fallback_provider(
            {"p1": mock_p1, "p2": mock_p2}, profiles, max_profile_rotations=1
        )

        with patch("engine.safety.time.monotonic", return_value=1000.0):
            await fallback.chat(
                messages=[{"role": "user", "content": "Hi"}],
                tools=[],
                agent_label="Test",
                task_id="t1",
            )

        assert fallback._rotation_count == 0


class TestFallbackProviderErrorHandling:
    """Tests for error classification and handling."""

    @pytest.mark.asyncio
    async def test_non_retryable_raises_immediately(self):
        """401/403 errors raise immediately without retry."""
        profiles = _make_profiles(["p1"])
        mock_p1 = _make_mock_provider(
            raise_error=Exception("401 unauthorized: invalid api key")
        )
        fallback = _make_fallback_provider({"p1": mock_p1}, profiles)

        with pytest.raises(LLMProviderError) as exc_info:
            await fallback.chat(
                messages=[{"role": "user", "content": "Hi"}],
                tools=[],
                agent_label="Test",
                task_id="t1",
            )

        assert "401" in str(exc_info.value)
        mock_p1.chat.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_retryable_raises_for_caller(self):
        """Retryable errors are raised for the caller to handle."""
        profiles = _make_profiles(["p1"])
        mock_p1 = _make_mock_provider(
            raise_error=Exception("500 internal server error")
        )
        fallback = _make_fallback_provider({"p1": mock_p1}, profiles)

        with pytest.raises(Exception) as exc_info:
            await fallback.chat(
                messages=[{"role": "user", "content": "Hi"}],
                tools=[],
                agent_label="Test",
                task_id="t1",
            )

        assert "500" in str(exc_info.value)
        mock_p1.chat.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_cancelled_error_propagates(self):
        """CancelledError propagates without being caught."""
        profiles = _make_profiles(["p1"])
        mock_p1 = _make_mock_provider(
            raise_error=asyncio.CancelledError()
        )
        fallback = _make_fallback_provider({"p1": mock_p1}, profiles)

        with pytest.raises(asyncio.CancelledError):
            await fallback.chat(
                messages=[{"role": "user", "content": "Hi"}],
                tools=[],
                agent_label="Test",
                task_id="t1",
            )


class TestFallbackProviderStreamChat:
    """Tests for stream_chat delegation."""

    @pytest.mark.asyncio
    async def test_stream_chat_delegates_to_provider(self):
        """stream_chat delegates to the active provider."""
        profiles = _make_profiles(["p1"])
        mock_provider = _make_mock_provider()
        fallback = _make_fallback_provider({"p1": mock_provider}, profiles)

        await fallback.stream_chat(
            messages=[{"role": "user", "content": "Hi"}],
            tools=[],
            agent_label="Test",
            task_id="t1",
        )

        mock_provider.stream_chat.assert_awaited_once()
        assert fallback._current_profile == "p1"


class TestFallbackProviderInfo:
    """Tests for get_active_provider_info."""

    def test_get_info_before_chat(self):
        """get_active_provider_info before any chat returns None profile."""
        profiles = _make_profiles(["p1"])
        fallback = _make_fallback_provider(
            {"p1": _make_mock_provider()}, profiles
        )
        info = fallback.get_active_provider_info()
        assert info["current_profile"] is None
        assert "pool_status" in info

    @pytest.mark.asyncio
    async def test_get_info_after_chat(self):
        """get_active_provider_info reflects last used profile."""
        profiles = _make_profiles(["p1"])
        fallback = _make_fallback_provider(
            {"p1": _make_mock_provider()}, profiles
        )
        await fallback.chat(
            messages=[{"role": "user", "content": "Hi"}],
            tools=[],
            agent_label="Test",
            task_id="t1",
        )
        info = fallback.get_active_provider_info()
        assert info["current_profile"] == "p1"


class TestEstimateTokens:
    """Tests for _estimate_tokens static method."""

    def test_empty_messages_returns_one(self):
        """Empty input returns at least 1 token."""
        result = FallbackLLMProvider._estimate_tokens([], None)
        assert result == 1

    def test_messages_only(self):
        """Token estimate from messages."""
        messages = [
            {"role": "user", "content": "Hello world"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result = FallbackLLMProvider._estimate_tokens(messages, None)
        total_chars = sum(len(str(m)) for m in messages)
        expected = max(1, total_chars // 3)
        assert result == expected

    def test_messages_and_tools(self):
        """Token estimate includes tools."""
        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "name": "test_tool"}]
        result = FallbackLLMProvider._estimate_tokens(messages, tools)
        total_chars = sum(len(str(m)) for m in messages) + sum(
            len(str(t)) for t in tools
        )
        expected = max(1, total_chars // 3)
        assert result == expected

    def test_tools_none(self):
        """None tools is treated as empty list."""
        messages = [{"role": "user", "content": "Hello"}]
        result_with_none = FallbackLLMProvider._estimate_tokens(messages, None)
        result_with_empty = FallbackLLMProvider._estimate_tokens(messages, [])
        assert result_with_none == result_with_empty

    def test_large_input(self):
        """Large input produces reasonable estimate."""
        messages = [{"role": "user", "content": "x" * 3000}]
        result = FallbackLLMProvider._estimate_tokens(messages, None)
        assert result >= 1000


class TestFallbackProviderNoLimiterPacer:
    """Tests for missing limiter/pacer configuration."""

    @pytest.mark.asyncio
    async def test_chat_without_limiter_or_pacer(self):
        """chat works when limiters and pacers are not configured."""
        profiles = _make_profiles(["p1"])
        mock_provider = _make_mock_provider(response_content="No limits")
        key_pool = APIKeyPool(profiles)
        retry_engine = RetryEngine(max_attempts=3, base_delay=0.01)
        fallback = FallbackLLMProvider(
            providers={"p1": mock_provider},
            key_pool=key_pool,
            rate_limiters={},
            pacers={},
            retry_engine=retry_engine,
        )

        result = await fallback.chat(
            messages=[{"role": "user", "content": "Hi"}],
            tools=[],
            agent_label="Test",
            task_id="t1",
        )

        assert result.content == "No limits"

    @pytest.mark.asyncio
    async def test_no_usage_recorded_without_limiter(self):
        """No usage recorded when limiter is missing."""
        profiles = _make_profiles(["p1"])
        mock_provider = _make_mock_provider(
            response_content="OK", usage=(10, 20)
        )
        key_pool = APIKeyPool(profiles)
        retry_engine = RetryEngine(max_attempts=3, base_delay=0.01)
        fallback = FallbackLLMProvider(
            providers={"p1": mock_provider},
            key_pool=key_pool,
            rate_limiters={},
            pacers={},
            retry_engine=retry_engine,
        )

        await fallback.chat(
            messages=[{"role": "user", "content": "Hi"}],
            tools=[],
            agent_label="Test",
            task_id="t1",
        )

        mock_provider.get_last_usage.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_pacer_update_without_pacer(self):
        """No pacer update when pacer is missing."""
        profiles = _make_profiles(["p1"])
        snapshot = RateLimitSnapshot(remaining_rpm=30, limit_rpm=60)
        mock_provider = _make_mock_provider(
            response_content="OK", snapshot=snapshot
        )
        key_pool = APIKeyPool(profiles)
        retry_engine = RetryEngine(max_attempts=3, base_delay=0.01)
        fallback = FallbackLLMProvider(
            providers={"p1": mock_provider},
            key_pool=key_pool,
            rate_limiters={},
            pacers={},
            retry_engine=retry_engine,
        )

        await fallback.chat(
            messages=[{"role": "user", "content": "Hi"}],
            tools=[],
            agent_label="Test",
            task_id="t1",
        )

        mock_provider.get_rate_limit_snapshot.assert_not_called()


class TestFallbackProviderSuccessResets:
    """Tests verifying success resets provider state after failures."""

    @pytest.mark.asyncio
    async def test_success_after_rate_limit_resets_key_pool(self):
        """Success after a rate-limited key resets its cooldown."""
        profiles = _make_profiles(["p1", "p2"])
        # p1 fails first, then succeeds
        mock_p1 = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = "Recovered"
        mock_resp.tool_calls = []
        mock_resp.has_tool_calls.return_value = False

        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("429 rate limit")
            return mock_resp

        mock_p1.chat = AsyncMock(side_effect=side_effect)
        mock_p1.get_last_usage = MagicMock(return_value=None)
        mock_p1.get_rate_limit_snapshot = MagicMock(return_value=None)
        mock_p1.stream_chat = AsyncMock(return_value=None)

        mock_p2 = _make_mock_provider(response_content="p2 OK")
        fallback = _make_fallback_provider(
            {"p1": mock_p1, "p2": mock_p2}, profiles, max_profile_rotations=1,
            with_limiters=False, with_pacers=False,
        )

        with patch("engine.safety.time.monotonic", return_value=1000.0):
            # First call: p1 fails, falls back to p2
            result1 = await fallback.chat(
                messages=[{"role": "user", "content": "Hi"}],
                tools=[],
                agent_label="Test",
                task_id="t1",
            )
            assert result1.content == "p2 OK"

            # Now manually reset p1 cooldown to simulate time passing
            fallback._key_pool._health["p1"].cooldown_until = None
            fallback._key_pool._health["p1"].consecutive_errors = 0

            # Second call: p1 succeeds
            result2 = await fallback.chat(
                messages=[{"role": "user", "content": "Hi"}],
                tools=[],
                agent_label="Test",
                task_id="t2",
            )
            assert result2.content == "Recovered"

        # After success, p1 should have no errors
        status = fallback._key_pool.get_cooldown_status()["p1"]
        assert status.consecutive_errors == 0
        assert status.cooldown_until is None
