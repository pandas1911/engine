"""Unit tests for TokenBucketRateLimiter."""

import asyncio
import time

import pytest

from engine.safety import TokenBucketRateLimiter


class TestTokenBucketRateLimiterInit:
    """Constructor validation tests."""

    def test_valid_params(self):
        limiter = TokenBucketRateLimiter(max_tokens=5, refill_rate=10.0)
        assert limiter.max_tokens == 5
        assert limiter.refill_rate == 10.0
        assert limiter.available_tokens == 5.0

    def test_max_tokens_zero_raises(self):
        with pytest.raises(ValueError, match="max_tokens"):
            TokenBucketRateLimiter(max_tokens=0, refill_rate=1.0)

    def test_max_tokens_negative_raises(self):
        with pytest.raises(ValueError, match="max_tokens"):
            TokenBucketRateLimiter(max_tokens=-1, refill_rate=1.0)

    def test_refill_rate_zero_raises(self):
        with pytest.raises(ValueError, match="refill_rate"):
            TokenBucketRateLimiter(max_tokens=5, refill_rate=0)

    def test_refill_rate_negative_raises(self):
        with pytest.raises(ValueError, match="refill_rate"):
            TokenBucketRateLimiter(max_tokens=5, refill_rate=-1.0)

    def test_fractional_refill_rate(self):
        """0.5 refill_rate = 1 token per 2 seconds."""
        limiter = TokenBucketRateLimiter(max_tokens=3, refill_rate=0.5)
        assert limiter.refill_rate == 0.5


class TestTokenBucketRateLimiterAcquire:
    """Basic acquire behavior tests."""

    @pytest.mark.asyncio
    async def test_acquire_within_capacity(self):
        """Acquires within burst capacity return immediately (fast path)."""
        limiter = TokenBucketRateLimiter(max_tokens=5, refill_rate=10.0)
        for _ in range(5):
            await limiter.acquire()  # Should not raise or block

    @pytest.mark.asyncio
    async def test_acquire_blocks_when_empty(self):
        """Acquire blocks when bucket is drained, then unblocks after refill."""
        limiter = TokenBucketRateLimiter(max_tokens=1, refill_rate=100.0)
        await limiter.acquire()  # Drain the bucket

        start = time.monotonic()
        await limiter.acquire()  # Should block briefly
        elapsed = time.monotonic() - start

        # At 100 tokens/sec, should take ~10ms to get a new token
        assert elapsed >= 0.005, f"Should have blocked, but only {elapsed:.4f}s elapsed"
        assert elapsed < 0.5, f"Blocked too long: {elapsed:.4f}s"

    @pytest.mark.asyncio
    async def test_token_accumulation_caps_at_max(self):
        """Tokens don't exceed max_tokens after idle period."""
        limiter = TokenBucketRateLimiter(max_tokens=3, refill_rate=100.0)
        await asyncio.sleep(0.1)  # Let tokens accumulate
        assert limiter.available_tokens <= 3.0 + 0.01  # Small tolerance

    @pytest.mark.asyncio
    async def test_concurrent_acquire_all_complete(self):
        """Multiple concurrent acquirers all eventually succeed."""
        limiter = TokenBucketRateLimiter(max_tokens=1, refill_rate=20.0)
        await limiter.acquire()  # Drain

        results = []

        async def worker(worker_id: int):
            await limiter.acquire()
            results.append(worker_id)

        tasks = [asyncio.create_task(worker(i)) for i in range(5)]
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=2.0)

        assert len(results) == 5, f"Expected 5 completions, got {len(results)}"

    @pytest.mark.asyncio
    async def test_fast_path_no_scheduler_created(self):
        """Fast path: no scheduler task when tokens available and no waiters."""
        limiter = TokenBucketRateLimiter(max_tokens=5, refill_rate=10.0)
        await limiter.acquire()
        # _wakeup_task should remain None (fast path doesn't start scheduler)
        assert limiter._wakeup_task is None

    @pytest.mark.asyncio
    async def test_slow_path_starts_scheduler(self):
        """Slow path: scheduler starts when a waiter is enqueued."""
        limiter = TokenBucketRateLimiter(max_tokens=1, refill_rate=100.0)
        await limiter.acquire()  # Drain

        # Launch a blocked acquire
        task = asyncio.create_task(limiter.acquire())
        await asyncio.sleep(0.001)  # Let it enter the slow path

        # Scheduler should be running
        assert limiter._wakeup_task is not None
        await asyncio.wait_for(task, timeout=1.0)  # Clean up

    @pytest.mark.asyncio
    async def test_cancellation_cleans_up_waiter(self):
        """Cancelled acquire removes itself from waiter queue."""
        limiter = TokenBucketRateLimiter(max_tokens=1, refill_rate=2.0)
        await limiter.acquire()  # Drain

        # Launch a blocked acquire, then cancel it
        task = asyncio.create_task(limiter.acquire())
        await asyncio.sleep(0.05)  # Let it enqueue
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        # Queue should be clean — new acquire should succeed after refill
        await asyncio.sleep(0.6)  # Wait for refill
        await asyncio.wait_for(limiter.acquire(), timeout=1.0)


from unittest.mock import AsyncMock, MagicMock

from engine.providers.llm_provider import LLMProvider, LLMProviderError
from engine.safety import TokenBucketRateLimiter


class TestLLMProviderRetry:
    """Tests for LLMProvider retry logic."""

    def _make_config(self, **overrides):
        """Create a mock Config for testing."""
        config = MagicMock()
        config.api_key = "test-key"
        config.base_url = "https://api.test.com/v1"
        config.model = "test-model"
        config.strip_thinking = True
        config.llm_retry_max_attempts = overrides.get("max_attempts", 3)
        config.llm_retry_base_delay = overrides.get("base_delay", 0.01)  # Fast for tests
        return config

    @pytest.mark.asyncio
    async def test_success_first_attempt(self):
        """Successful call returns immediately, no retry."""
        config = self._make_config()
        provider = LLMProvider(config, rate_limiter=None)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello"
        mock_response.choices[0].message.tool_calls = None

        provider.client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await provider.chat(
            messages=[{"role": "user", "content": "Hi"}],
            tools=[],
            agent_label="Test",
            task_id="test-1",
        )

        assert result.content == "Hello"
        assert not result.has_tool_calls()
        assert provider.client.chat.completions.create.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_error_then_succeed(self):
        """Retry on transient error, succeed on 2nd attempt."""
        config = self._make_config(max_attempts=3, base_delay=0.01)
        provider = LLMProvider(config, rate_limiter=None)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Retried OK"
        mock_response.choices[0].message.tool_calls = None

        provider.client.chat.completions.create = AsyncMock(
            side_effect=[ConnectionError("timeout"), mock_response]
        )

        result = await provider.chat(
            messages=[{"role": "user", "content": "Hi"}],
            tools=[],
            agent_label="Test",
            task_id="test-2",
        )

        assert result.content == "Retried OK"
        assert provider.client.chat.completions.create.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_exhausted_raises_llm_provider_error(self):
        """All retries exhausted raises LLMProviderError."""
        config = self._make_config(max_attempts=3, base_delay=0.01)
        provider = LLMProvider(config, rate_limiter=None)

        provider.client.chat.completions.create = AsyncMock(
            side_effect=ConnectionError("persistent failure")
        )

        with pytest.raises(LLMProviderError) as exc_info:
            await provider.chat(
                messages=[{"role": "user", "content": "Hi"}],
                tools=[],
                agent_label="Test",
                task_id="test-3",
            )

        assert isinstance(exc_info.value.original_error, ConnectionError)
        assert provider.client.chat.completions.create.call_count == 3

    @pytest.mark.asyncio
    async def test_cancelled_error_propagates(self):
        """CancelledError is NOT caught by retry logic."""
        config = self._make_config(max_attempts=3, base_delay=0.01)
        provider = LLMProvider(config, rate_limiter=None)

        provider.client.chat.completions.create = AsyncMock(
            side_effect=asyncio.CancelledError()
        )

        with pytest.raises(asyncio.CancelledError):
            await provider.chat(
                messages=[{"role": "user", "content": "Hi"}],
                tools=[],
                agent_label="Test",
                task_id="test-4",
            )

        # Should only have been called once — no retry on CancelledError
        assert provider.client.chat.completions.create.call_count == 1

    @pytest.mark.asyncio
    async def test_rate_limiter_called_before_api(self):
        """Rate limiter acquire() is called before each API attempt."""
        config = self._make_config(base_delay=0.01)
        mock_limiter = AsyncMock()
        provider = LLMProvider(config, rate_limiter=mock_limiter)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "OK"
        mock_response.choices[0].message.tool_calls = None

        provider.client.chat.completions.create = AsyncMock(return_value=mock_response)

        await provider.chat(
            messages=[{"role": "user", "content": "Hi"}],
            tools=[],
            agent_label="Test",
            task_id="test-5",
        )

        mock_limiter.acquire.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_rate_limiter_overhead(self):
        """When rate_limiter is None, no rate-limit code runs."""
        config = self._make_config()
        provider = LLMProvider(config, rate_limiter=None)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Fast"
        mock_response.choices[0].message.tool_calls = None

        provider.client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await provider.chat(
            messages=[{"role": "user", "content": "Hi"}],
            tools=[],
            agent_label="Test",
            task_id="test-6",
        )

        assert result.content == "Fast"

    @pytest.mark.asyncio
    async def test_retry_with_rate_limiter(self):
        """Rate limiter re-acquired on each retry attempt."""
        config = self._make_config(max_attempts=3, base_delay=0.01)
        mock_limiter = AsyncMock()
        provider = LLMProvider(config, rate_limiter=mock_limiter)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "OK after retry"
        mock_response.choices[0].message.tool_calls = None

        provider.client.chat.completions.create = AsyncMock(
            side_effect=[ConnectionError("fail"), mock_response]
        )

        result = await provider.chat(
            messages=[{"role": "user", "content": "Hi"}],
            tools=[],
            agent_label="Test",
            task_id="test-7",
        )

        assert result.content == "OK after retry"
        # Rate limiter should be called once per attempt (2 attempts = 2 calls)
        assert mock_limiter.acquire.call_count == 2
