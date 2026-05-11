"""Tests for per-provider concurrency guard in FallbackLLMProvider.

Verifies that the concurrency_guards dict correctly limits in-flight
requests per provider using asyncio.Semaphore instances.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from engine.providers.fallback_provider import FallbackLLMProvider
from engine.providers.provider_models import LLMResponse, ProviderConfig
from engine.safety.key_pool import APIKeyPool
from engine.safety.retry import RetryEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class BlockingMockProvider:
    """Mock provider that blocks to allow concurrency measurement."""

    def __init__(self):
        self.active_count = 0
        self.max_active = 0
        self._block_event = asyncio.Event()
        self._lock = asyncio.Lock()
        self.get_last_usage = MagicMock(return_value=None)

    async def chat(self, **kwargs):
        async with self._lock:
            self.active_count += 1
            self.max_active = max(self.max_active, self.active_count)
        try:
            await self._block_event.wait()
        finally:
            async with self._lock:
                self.active_count -= 1
        return LLMResponse(content="ok")


def _make_limiter():
    """Create a mock rate limiter."""
    limiter = AsyncMock()
    limiter.tpm_limit = 100000
    limiter.acquire = AsyncMock(return_value=1)
    limiter.record_usage = AsyncMock()
    limiter.release_reserved = AsyncMock()
    return limiter


def _build_fallback(provider, key_pool, rate_limiters, concurrency_guards=None):
    """Construct a FallbackLLMProvider with sensible defaults."""
    return FallbackLLMProvider(
        providers=provider,
        key_pool=key_pool,
        rate_limiters=rate_limiters,
        retry_engine=RetryEngine(max_attempts=1),
        concurrency_guards=concurrency_guards,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestConcurrencyLimitEnforced:
    """THE KEY TEST: proves max in-flight count never exceeds configured limit."""

    @pytest.mark.asyncio
    async def test_concurrent_limit_enforced(self):
        """With semaphore(2), launching 5 concurrent calls never exceeds 2 active."""
        mock_provider = BlockingMockProvider()
        key_pool = APIKeyPool(names=["prov/model"])
        limiter = _make_limiter()
        rate_limiters = {"prov": limiter}
        concurrency_guards = {"prov": asyncio.Semaphore(2)}

        fb = _build_fallback(
            provider={"prov/model": mock_provider},
            key_pool=key_pool,
            rate_limiters=rate_limiters,
            concurrency_guards=concurrency_guards,
        )

        # Launch 5 concurrent chat() calls — they will block on the event
        tasks = [
            asyncio.create_task(fb.chat(messages=[{"role": "user", "content": "hi"}], tools=[]))
            for _ in range(5)
        ]

        # Give the event loop a moment to schedule all coroutines
        await asyncio.sleep(0.1)

        # Release the blocking event so all waiting calls can proceed
        mock_provider._block_event.set()

        results = await asyncio.gather(*tasks)

        # All 5 calls complete successfully
        assert len(results) == 5
        assert all(r.content == "ok" for r in results)

        # max_active must never exceed the semaphore limit of 2
        assert mock_provider.max_active <= 2, (
            "max_active={} should be <= 2 (semaphore limit)".format(
                mock_provider.max_active
            )
        )


class TestNoGuardUnlimited:
    """Without a concurrency guard, all calls proceed without blocking."""

    @pytest.mark.asyncio
    async def test_no_guard_unlimited(self):
        """Empty concurrency_guards dict allows all 5 calls to proceed at once."""
        mock_provider = BlockingMockProvider()
        key_pool = APIKeyPool(names=["prov/model"])
        limiter = _make_limiter()
        rate_limiters = {"prov": limiter}

        fb = _build_fallback(
            provider={"prov/model": mock_provider},
            key_pool=key_pool,
            rate_limiters=rate_limiters,
            concurrency_guards={},  # no guard
        )

        tasks = [
            asyncio.create_task(fb.chat(messages=[{"role": "user", "content": "hi"}], tools=[]))
            for _ in range(5)
        ]

        await asyncio.sleep(0.1)
        mock_provider._block_event.set()
        results = await asyncio.gather(*tasks)

        assert len(results) == 5
        assert all(r.content == "ok" for r in results)

        # All 5 should have been active concurrently (no semaphore)
        assert mock_provider.max_active == 5, (
            "max_active={} should be 5 without guard".format(
                mock_provider.max_active
            )
        )


class TestSemaphoreReleasedOnError:
    """Semaphore is released when provider raises an exception."""

    @pytest.mark.asyncio
    async def test_semaphore_released_on_error(self):
        """After an error, subsequent calls can acquire the semaphore."""
        call_count = 0
        error_event = asyncio.Event()

        class ErrorOnceProvider:
            def __init__(self):
                self.get_last_usage = MagicMock(return_value=None)

            async def chat(self, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise RuntimeError("provider error")
                return LLMResponse(content="ok")

        provider = ErrorOnceProvider()
        key_pool = APIKeyPool(names=["prov/model"])
        limiter = _make_limiter()
        rate_limiters = {"prov": limiter}
        concurrency_guards = {"prov": asyncio.Semaphore(1)}

        fb = _build_fallback(
            provider={"prov/model": provider},
            key_pool=key_pool,
            rate_limiters=rate_limiters,
            concurrency_guards=concurrency_guards,
        )

        # First call fails — semaphore(1) must be released
        with pytest.raises(Exception):
            await fb.chat(messages=[{"role": "user", "content": "fail"}], tools=[])

        # Second call must succeed — if semaphore leaked, this would hang
        result = await fb.chat(messages=[{"role": "user", "content": "ok"}], tools=[])
        assert result.content == "ok"


class TestSemaphoreReleasedOnSuccess:
    """Semaphore is released after successful call."""

    @pytest.mark.asyncio
    async def test_semaphore_released_on_success(self):
        """After a successful call, subsequent calls can acquire the semaphore."""
        provider = AsyncMock()
        provider.chat = AsyncMock(return_value=LLMResponse(content="ok"))
        provider.get_last_usage = MagicMock(return_value=None)

        key_pool = APIKeyPool(names=["prov/model"])
        limiter = _make_limiter()
        rate_limiters = {"prov": limiter}
        concurrency_guards = {"prov": asyncio.Semaphore(1)}

        fb = _build_fallback(
            provider={"prov/model": provider},
            key_pool=key_pool,
            rate_limiters=rate_limiters,
            concurrency_guards=concurrency_guards,
        )

        # First call succeeds — semaphore(1) must be released
        r1 = await fb.chat(messages=[{"role": "user", "content": "first"}], tools=[])
        assert r1.content == "ok"

        # Second call must also succeed — if semaphore leaked, this would hang
        r2 = await fb.chat(messages=[{"role": "user", "content": "second"}], tools=[])
        assert r2.content == "ok"


class TestConfigParsing:
    """Verify ProviderConfig.max_concurrent_requests field."""

    def test_config_default_zero(self):
        """Default max_concurrent_requests is 0 (unlimited / not configured)."""
        cfg = ProviderConfig(name="test", api_key="key", base_url="http://localhost")
        assert cfg.max_concurrent_requests == 0

    def test_config_custom_value(self):
        """Custom max_concurrent_requests is stored correctly."""
        cfg = ProviderConfig(
            name="test",
            api_key="key",
            base_url="http://localhost",
            max_concurrent_requests=3,
        )
        assert cfg.max_concurrent_requests == 3
