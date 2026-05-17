"""Tests for FallbackLLMProvider max_iterations formula.

Verifies that the retry loop in chat() and stream_chat() stops at the
correct iteration count based on the new formula:

    max_iterations = len(providers) * (max_profile_rotations + 1)

Previously the formula was max(50, len(providers) * (max_profile_rotations + 1) * 2)
which could lead to excessively long retry loops (minimum 50 iterations).
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from engine.providers.fallback_provider import FallbackLLMProvider
from engine.providers.provider_models import LLMResponse
from engine.providers.llm_provider import LLMProviderError
from engine.providers.provider_models import ErrorClass
from engine.safety.key_pool import APIKeyPool
from engine.safety.retry import RetryEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_provider(name="test/model"):
    """Create a mock LLM provider with async chat."""
    provider = AsyncMock()
    provider.chat = AsyncMock(return_value=LLMResponse(content="ok"))
    provider.get_last_usage = MagicMock(return_value=None)
    return provider


def _make_key_pool(keys):
    """Create a real APIKeyPool."""
    return APIKeyPool(names=keys)


def _make_limiter():
    """Create a mock rate limiter."""
    limiter = AsyncMock()
    limiter.tpm_limit = 100000
    limiter.acquire = AsyncMock(return_value=1)
    limiter.record_usage = AsyncMock()
    limiter.release_reserved = AsyncMock()
    return limiter


def _build_fallback(providers, key_pool, rate_limiters, retry_engine=None, max_profile_rotations=3):
    """Construct FallbackLLMProvider with defaults."""
    if retry_engine is None:
        retry_engine = RetryEngine(max_attempts=1)
    return FallbackLLMProvider(
        providers=providers,
        key_pool=key_pool,
        rate_limiters=rate_limiters,
        retry_engine=retry_engine,
        max_profile_rotations=max_profile_rotations,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMaxIterationsChat:
    """Tests for chat() max_iterations behaviour."""

    @pytest.mark.asyncio
    async def test_max_iterations_all_rate_limited(self):
        """When all providers always hit rate-limit, the loop stops at
        len(providers) * (max_profile_rotations + 1) iterations, NOT 50.

        With 2 providers and max_profile_rotations=3:
            max_iterations = 2 * (3 + 1) = 8
        """
        provider_a = _make_provider("pA/m-a")
        provider_b = _make_provider("pB/m-b")
        # Both providers always raise an exception classified as RATE_LIMITED
        provider_a.chat = AsyncMock(side_effect=Exception("429 rate limited"))
        provider_b.chat = AsyncMock(side_effect=Exception("429 rate limited"))

        key_pool = _make_key_pool(["pA/m-a", "pB/m-b"])
        rate_limiters = {"pA": _make_limiter(), "pB": _make_limiter()}

        retry_engine = RetryEngine(max_attempts=1)
        retry_engine.classify_error = MagicMock(return_value=ErrorClass.RATE_LIMITED)
        retry_engine.extract_retry_after = MagicMock(return_value=0)

        fb = _build_fallback(
            providers={"pA/m-a": provider_a, "pB/m-b": provider_b},
            key_pool=key_pool,
            rate_limiters=rate_limiters,
            retry_engine=retry_engine,
            max_profile_rotations=3,
        )

        with pytest.raises(LLMProviderError):
            await fb.chat(
                messages=[{"role": "user", "content": "test"}],
                tools=[],
            )

        total_calls = provider_a.chat.call_count + provider_b.chat.call_count
        # max_iterations = 2 * (3 + 1) = 8
        assert total_calls == 8, (
            "Expected exactly 8 calls (2 providers x 4 attempts), got {}".format(
                total_calls
            )
        )

    @pytest.mark.asyncio
    async def test_succeeds_within_iterations(self):
        """Provider succeeds on the N-th attempt within max_iterations.

        With 1 provider and max_profile_rotations=3:
            max_iterations = 1 * (3 + 1) = 4

        The provider raises rate-limit 3 times, then succeeds.
        Total call count should be 4.
        """
        provider = _make_provider("p/m")
        provider.chat = AsyncMock(
            side_effect=[
                Exception("429"),
                Exception("429"),
                Exception("429"),
                LLMResponse(content="ok"),
            ]
        )

        key_pool = _make_key_pool(["p/m"])
        rate_limiters = {"p": _make_limiter()}

        retry_engine = RetryEngine(max_attempts=1)
        retry_engine.classify_error = MagicMock(return_value=ErrorClass.RATE_LIMITED)
        retry_engine.extract_retry_after = MagicMock(return_value=0)

        fb = _build_fallback(
            providers={"p/m": provider},
            key_pool=key_pool,
            rate_limiters=rate_limiters,
            retry_engine=retry_engine,
            max_profile_rotations=3,
        )

        result = await fb.chat(
            messages=[{"role": "user", "content": "test"}],
            tools=[],
        )

        assert isinstance(result, LLMResponse)
        assert result.content == "ok"
        assert provider.chat.call_count == 4, (
            "Expected exactly 4 calls (3 failures + 1 success), got {}".format(
                provider.chat.call_count
            )
        )

    @pytest.mark.asyncio
    async def test_max_iterations_value_with_many_providers(self):
        """With 3 providers and max_profile_rotations=2:
            max_iterations = 3 * (2 + 1) = 9

        All providers always rate-limited → exactly 9 total calls.
        """
        providers = {}
        rate_limiters = {}
        keys = []
        for i in range(3):
            name = "p{}/m-{}".format(i, i)
            p = _make_provider(name)
            p.chat = AsyncMock(side_effect=Exception("429"))
            providers[name] = p
            rate_limiters["p{}".format(i)] = _make_limiter()
            keys.append(name)

        key_pool = _make_key_pool(keys)

        retry_engine = RetryEngine(max_attempts=1)
        retry_engine.classify_error = MagicMock(return_value=ErrorClass.RATE_LIMITED)
        retry_engine.extract_retry_after = MagicMock(return_value=0)

        fb = _build_fallback(
            providers=providers,
            key_pool=key_pool,
            rate_limiters=rate_limiters,
            retry_engine=retry_engine,
            max_profile_rotations=2,
        )

        with pytest.raises(LLMProviderError):
            await fb.chat(
                messages=[{"role": "user", "content": "test"}],
                tools=[],
            )

        total_calls = sum(p.chat.call_count for p in providers.values())
        assert total_calls == 9, (
            "Expected exactly 9 calls (3 providers x 3 attempts), got {}".format(
                total_calls
            )
        )
