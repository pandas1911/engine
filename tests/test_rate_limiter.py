"""Unit tests for rate limiting and retry components."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from engine.providers.llm_provider import LLMProvider, LLMProviderError
from engine.safety import (
    SlidingWindowRateLimiter,
    RetryEngine,
    AdaptivePacer,
    APIKeyPool,
)

from engine.providers.provider_models import (
    ProviderProfile,
    RateLimitSnapshot,
    PaceLevel,
    Lane,
    ErrorClass,
    ProviderHealth,
)


class TestProviderProfile:
    """ProviderProfile dataclass validation."""

    def test_provider_profile_creation(self):
        """Create ProviderProfile with valid params."""
        profile = ProviderProfile(
            name="test",
            api_key="key",
            base_url="https://api.test.com",
            model="gpt-4",
            rpm_limit=60.0,
            tpm_limit=100000.0,
            weight=2,
        )
        assert profile.name == "test"
        assert profile.api_key == "key"
        assert profile.base_url == "https://api.test.com"
        assert profile.model == "gpt-4"
        assert profile.rpm_limit == 60.0
        assert profile.tpm_limit == 100000.0
        assert profile.weight == 2

    def test_provider_profile_default_weight(self):
        """Default weight is 1 when omitted."""
        profile = ProviderProfile(
            name="test",
            api_key="key",
            base_url="https://api.test.com",
            model="gpt-4",
            rpm_limit=60.0,
            tpm_limit=100000.0,
        )
        assert profile.weight == 1


class TestRateLimitSnapshot:
    """RateLimitSnapshot dataclass validation."""

    def test_rate_limit_snapshot_creation(self):
        """Create RateLimitSnapshot with all fields."""
        snapshot = RateLimitSnapshot(
            remaining_rpm=50,
            remaining_tpm=90000,
            limit_rpm=60,
            limit_tpm=100000,
            retry_after_ms=100.0,
        )
        assert snapshot.remaining_rpm == 50
        assert snapshot.remaining_tpm == 90000
        assert snapshot.limit_rpm == 60
        assert snapshot.limit_tpm == 100000
        assert snapshot.retry_after_ms == 100.0

    def test_rate_limit_snapshot_defaults(self):
        """All fields default to None."""
        snapshot = RateLimitSnapshot()
        assert snapshot.remaining_rpm is None
        assert snapshot.remaining_tpm is None
        assert snapshot.limit_rpm is None
        assert snapshot.limit_tpm is None
        assert snapshot.retry_after_ms is None


class TestPaceLevel:
    """PaceLevel enum validation."""

    def test_pace_level_values(self):
        """PaceLevel enum members have expected values."""
        assert PaceLevel.HEALTHY.value == "healthy"
        assert PaceLevel.PRESSING.value == "pressing"
        assert PaceLevel.CRITICAL.value == "critical"


class TestLane:
    """Lane enum validation."""

    def test_lane_values(self):
        """Lane enum members have expected values."""
        assert Lane.MAIN.value == "main"
        assert Lane.SUBAGENT.value == "subagent"


class TestErrorClass:
    """ErrorClass enum validation."""

    def test_error_class_values(self):
        """ErrorClass enum members have expected values."""
        assert ErrorClass.RETRYABLE.value == "retryable"
        assert ErrorClass.NON_RETRYABLE.value == "non_retryable"
        assert ErrorClass.RATE_LIMITED.value == "rate_limited"


class TestProviderHealth:
    """ProviderHealth dataclass validation."""

    def test_provider_health_defaults(self):
        """ProviderHealth defaults are correct."""
        health = ProviderHealth(profile_name="test")
        assert health.profile_name == "test"
        assert health.consecutive_errors == 0
        assert health.last_error_time is None
        assert health.cooldown_until is None
        assert health.pace_level == PaceLevel.HEALTHY


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
        provider = LLMProvider(config)

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
        provider = LLMProvider(config)

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
        provider = LLMProvider(config)

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
        provider = LLMProvider(config)

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

class TestLLMProviderHeaderExtraction:
    """Tests for rate limit header and usage extraction."""

    def _make_config(self, **overrides):
        """Create a mock Config for testing."""
        config = MagicMock()
        config.api_key = "test-key"
        config.base_url = "https://api.test.com/v1"
        config.model = "test-model"
        config.strip_thinking = True
        config.llm_retry_max_attempts = overrides.get("max_attempts", 3)
        config.llm_retry_base_delay = overrides.get("base_delay", 0.01)
        return config

    @pytest.mark.asyncio
    async def test_header_extraction_standard_headers(self):
        """Extract rate limit info from standard OpenAI headers."""
        config = self._make_config()
        provider = LLMProvider(config)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello"
        mock_response.choices[0].message.tool_calls = None
        mock_response.headers = {
            'x-ratelimit-remaining-requests': '45',
            'x-ratelimit-remaining-tokens': '95000',
            'x-ratelimit-limit-requests': '60',
            'x-ratelimit-limit-tokens': '100000',
        }

        provider.client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await provider.chat(
            messages=[{"role": "user", "content": "Hi"}],
            tools=[],
            agent_label="Test",
            task_id="test-header-1",
        )

        assert result.content == "Hello"
        snapshot = provider.get_rate_limit_snapshot()
        assert snapshot is not None
        assert snapshot.remaining_rpm == 45
        assert snapshot.remaining_tpm == 95000
        assert snapshot.limit_rpm == 60
        assert snapshot.limit_tpm == 100000

    @pytest.mark.asyncio
    async def test_header_extraction_no_headers(self):
        """Gracefully handle responses with no rate limit headers (e.g., DashScope)."""
        config = self._make_config()
        provider = LLMProvider(config)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello"
        mock_response.choices[0].message.tool_calls = None
        mock_response.headers = None

        provider.client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await provider.chat(
            messages=[{"role": "user", "content": "Hi"}],
            tools=[],
            agent_label="Test",
            task_id="test-header-2",
        )

        assert result.content == "Hello"
        assert provider.get_rate_limit_snapshot() is None

    @pytest.mark.asyncio
    async def test_header_extraction_alternate_headers(self):
        """Extract rate limit info from alternate header names."""
        config = self._make_config()
        provider = LLMProvider(config)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello"
        mock_response.choices[0].message.tool_calls = None
        mock_response.headers = {
            'ratelimit-remaining': '30',
            'ratelimit-limit': '60',
        }

        provider.client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await provider.chat(
            messages=[{"role": "user", "content": "Hi"}],
            tools=[],
            agent_label="Test",
            task_id="test-header-3",
        )

        assert result.content == "Hello"
        snapshot = provider.get_rate_limit_snapshot()
        assert snapshot is not None
        assert snapshot.remaining_rpm == 30
        assert snapshot.limit_rpm == 60

    @pytest.mark.asyncio
    async def test_usage_extraction(self):
        """Extract token usage from response."""
        config = self._make_config()
        provider = LLMProvider(config)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello"
        mock_response.choices[0].message.tool_calls = None
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20

        provider.client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await provider.chat(
            messages=[{"role": "user", "content": "Hi"}],
            tools=[],
            agent_label="Test",
            task_id="test-usage-1",
        )

        assert result.content == "Hello"
        usage = provider.get_last_usage()
        assert usage == (10, 20)


class TestRetryEngine:
    """Tests for RetryEngine error classification, delay computation, and execution."""

    # --- classify_error tests ---

    def test_classify_error_429_rate_limited(self):
        """HTTP 429 is classified as RATE_LIMITED."""
        engine = RetryEngine()
        error = Exception("HTTP 429: too many requests")
        assert engine.classify_error(error) == ErrorClass.RATE_LIMITED

    def test_classify_error_rate_limit_string(self):
        """'rate limit exceeded' is classified as RATE_LIMITED."""
        engine = RetryEngine()
        error = Exception("rate limit exceeded")
        assert engine.classify_error(error) == ErrorClass.RATE_LIMITED

    def test_classify_error_resource_exhausted(self):
        """'resource_exhausted' is classified as RATE_LIMITED."""
        engine = RetryEngine()
        error = Exception("resource_exhausted")
        assert engine.classify_error(error) == ErrorClass.RATE_LIMITED

    def test_classify_error_quota_exceeded(self):
        """'quota exceeded' is classified as RATE_LIMITED."""
        engine = RetryEngine()
        error = Exception("quota exceeded for today")
        assert engine.classify_error(error) == ErrorClass.RATE_LIMITED

    def test_classify_error_500_retryable(self):
        """HTTP 500 is classified as RETRYABLE (default)."""
        engine = RetryEngine()
        error = Exception("HTTP 500: internal server error")
        assert engine.classify_error(error) == ErrorClass.RETRYABLE

    def test_classify_error_502_retryable(self):
        """HTTP 502 is classified as RETRYABLE."""
        engine = RetryEngine()
        error = Exception("HTTP 502: bad gateway")
        assert engine.classify_error(error) == ErrorClass.RETRYABLE

    def test_classify_error_503_retryable(self):
        """HTTP 503 is classified as RETRYABLE."""
        engine = RetryEngine()
        error = Exception("HTTP 503: service unavailable")
        assert engine.classify_error(error) == ErrorClass.RETRYABLE

    def test_classify_error_timeout_retryable(self):
        """Timeout errors are classified as RETRYABLE."""
        engine = RetryEngine()
        error = Exception("connection timeout")
        assert engine.classify_error(error) == ErrorClass.RETRYABLE

    def test_classify_error_401_non_retryable(self):
        """HTTP 401 is classified as NON_RETRYABLE."""
        engine = RetryEngine()
        error = Exception("HTTP 401: unauthorized")
        assert engine.classify_error(error) == ErrorClass.NON_RETRYABLE

    def test_classify_error_403_non_retryable(self):
        """HTTP 403 is classified as NON_RETRYABLE."""
        engine = RetryEngine()
        error = Exception("HTTP 403: forbidden")
        assert engine.classify_error(error) == ErrorClass.NON_RETRYABLE

    def test_classify_error_invalid_api_key(self):
        """'invalid api key' is classified as NON_RETRYABLE."""
        engine = RetryEngine()
        error = Exception("invalid api key provided")
        assert engine.classify_error(error) == ErrorClass.NON_RETRYABLE

    def test_classify_error_authentication(self):
        """'authentication' failures are classified as NON_RETRYABLE."""
        engine = RetryEngine()
        error = Exception("authentication failed")
        assert engine.classify_error(error) == ErrorClass.NON_RETRYABLE

    def test_classify_error_order_rate_limited_before_non_retryable(self):
        """RATE_LIMITED checked before NON_RETRYABLE — 429 in auth context."""
        engine = RetryEngine()
        # Contains both "429" and "unauthorized" — 429 should win
        error = Exception("429 rate limit — unauthorized access")
        assert engine.classify_error(error) == ErrorClass.RATE_LIMITED

    # --- extract_retry_after tests ---

    def test_extract_retry_after_with_colon(self):
        """'retry-after: 30' extracts 30000 ms."""
        engine = RetryEngine()
        error = Exception("Rate limited. retry-after: 30")
        assert engine.extract_retry_after(error) == 30000.0

    def test_extract_retry_after_without_colon(self):
        """'retry after 5' extracts 5000 ms."""
        engine = RetryEngine()
        error = Exception("Please retry after 5")
        assert engine.extract_retry_after(error) == 5000.0

    def test_extract_retry_after_try_again_seconds(self):
        """'try again in 5 seconds' extracts 5000 ms."""
        engine = RetryEngine()
        error = Exception("Too busy, try again in 5 seconds")
        assert engine.extract_retry_after(error) == 5000.0

    def test_extract_retry_after_try_again_no_unit(self):
        """'try again in 10' with no unit defaults to seconds."""
        engine = RetryEngine()
        error = Exception("try again in 10")
        assert engine.extract_retry_after(error) == 10000.0

    def test_extract_retry_after_milliseconds(self):
        """Explicit millisecond unit extracts directly."""
        engine = RetryEngine()
        error = Exception("retry-after: 1500 ms")
        assert engine.extract_retry_after(error) == 1500.0

    def test_extract_retry_after_no_match(self):
        """No retry-after pattern returns None."""
        engine = RetryEngine()
        error = Exception("Something went wrong")
        assert engine.extract_retry_after(error) is None

    # --- compute_delay tests ---

    def test_compute_delay_exponential_base(self):
        """Base delay doubles each attempt without exceeding max_delay."""
        engine = RetryEngine(base_delay=1.0, max_delay=60.0)
        with patch("engine.safety.random.uniform", return_value=1.0):
            assert engine.compute_delay(1) == 1.0
            assert engine.compute_delay(2) == 2.0
            assert engine.compute_delay(3) == 4.0

    def test_compute_delay_respects_max_delay(self):
        """Computed base is capped at max_delay."""
        engine = RetryEngine(base_delay=1.0, max_delay=5.0)
        with patch("engine.safety.random.uniform", return_value=1.0):
            assert engine.compute_delay(1) == 1.0
            assert engine.compute_delay(2) == 2.0
            assert engine.compute_delay(3) == 4.0
            assert engine.compute_delay(4) == 5.0  # capped
            assert engine.compute_delay(5) == 5.0  # capped

    def test_compute_delay_with_retry_after_overrides(self):
        """retry_after_ms overrides base delay when larger."""
        engine = RetryEngine(base_delay=1.0, max_delay=60.0)
        with patch("engine.safety.random.uniform", return_value=1.0):
            # retry_after_ms=10000 (10s) > base=1s -> uses 10s
            assert engine.compute_delay(1, retry_after_ms=10000.0) == 10.0

    def test_compute_delay_with_retry_after_keeps_base_when_smaller(self):
        """Base delay wins when retry_after_ms is smaller."""
        engine = RetryEngine(base_delay=5.0, max_delay=60.0)
        with patch("engine.safety.random.uniform", return_value=1.0):
            # retry_after_ms=1000 (1s) < base=5s -> uses 5s
            assert engine.compute_delay(1, retry_after_ms=1000.0) == 5.0

    def test_compute_delay_symmetric_jitter_bounds(self):
        """Symmetric jitter multiplies by [0.5, 1.5]."""
        engine = RetryEngine(base_delay=2.0, max_delay=60.0)
        # No retry_after: symmetric jitter [0.5, 1.5]
        with patch("engine.safety.random.uniform", return_value=0.5):
            assert engine.compute_delay(1) == 1.0
        with patch("engine.safety.random.uniform", return_value=1.5):
            assert engine.compute_delay(1) == 3.0

    def test_compute_delay_positive_jitter_with_retry_after(self):
        """When retry_after present, jitter is positive only [1.0, 1.5]."""
        engine = RetryEngine(base_delay=1.0, max_delay=60.0)
        with patch("engine.safety.random.uniform", return_value=1.0):
            assert engine.compute_delay(1, retry_after_ms=5000.0) == 5.0
        with patch("engine.safety.random.uniform", return_value=1.5):
            assert engine.compute_delay(1, retry_after_ms=5000.0) == 7.5

    # --- execute_with_retry tests ---

    @pytest.mark.asyncio
    async def test_execute_success_first_try(self):
        """Successful call returns immediately with no retries."""
        engine = RetryEngine(max_attempts=3, base_delay=0.01)
        mock_fn = AsyncMock(return_value="success")

        result = await engine.execute_with_retry(mock_fn)
        assert result == "success"
        assert mock_fn.call_count == 1

    @pytest.mark.asyncio
    async def test_execute_retry_then_success(self):
        """Retry on transient error, succeed on 2nd attempt."""
        engine = RetryEngine(max_attempts=3, base_delay=0.01)
        mock_fn = AsyncMock(
            side_effect=[ConnectionError("timeout"), "success"]
        )

        result = await engine.execute_with_retry(mock_fn)
        assert result == "success"
        assert mock_fn.call_count == 2

    @pytest.mark.asyncio
    async def test_execute_all_retries_exhausted_raises(self):
        """All retries exhausted raises LLMProviderError."""
        engine = RetryEngine(max_attempts=3, base_delay=0.01)
        mock_fn = AsyncMock(side_effect=ConnectionError("persistent failure"))

        with pytest.raises(LLMProviderError) as exc_info:
            await engine.execute_with_retry(mock_fn)

        assert isinstance(exc_info.value.original_error, ConnectionError)
        assert mock_fn.call_count == 3

    @pytest.mark.asyncio
    async def test_execute_non_retryable_raises_immediately(self):
        """NON_RETRYABLE errors are raised immediately without retry."""
        engine = RetryEngine(max_attempts=3, base_delay=0.01)
        mock_fn = AsyncMock(side_effect=Exception("401 unauthorized"))

        with pytest.raises(Exception) as exc_info:
            await engine.execute_with_retry(mock_fn)

        assert "401" in str(exc_info.value)
        assert mock_fn.call_count == 1

    @pytest.mark.asyncio
    async def test_execute_on_retry_callback(self):
        """on_retry callback is invoked with (attempt, error, delay)."""
        engine = RetryEngine(max_attempts=3, base_delay=0.01)
        mock_fn = AsyncMock(
            side_effect=[ConnectionError("timeout"), "success"]
        )
        callback_calls = []

        def on_retry(attempt, error, delay):
            callback_calls.append((attempt, error, delay))

        await engine.execute_with_retry(mock_fn, on_retry=on_retry)
        assert len(callback_calls) == 1
        assert callback_calls[0][0] == 1
        assert isinstance(callback_calls[0][1], ConnectionError)
        assert callback_calls[0][2] > 0

    @pytest.mark.asyncio
    async def test_execute_rate_limited_retries(self):
        """RATE_LIMITED errors are retried like other retryable errors."""
        engine = RetryEngine(max_attempts=3, base_delay=0.01)
        mock_fn = AsyncMock(
            side_effect=[
                Exception("429 rate limit"),
                Exception("429 rate limit"),
                "success",
            ]
        )

        result = await engine.execute_with_retry(mock_fn)
        assert result == "success"
        assert mock_fn.call_count == 3

    # --- constructor validation ---

    def test_init_default_params(self):
        """Default constructor parameters."""
        engine = RetryEngine()
        assert engine.max_attempts == 3
        assert engine.base_delay == 1.0
        assert engine.max_delay == 60.0
        assert engine.jitter_mode == "symmetric"

    def test_init_custom_params(self):
        """Custom constructor parameters."""
        engine = RetryEngine(
            max_attempts=5, base_delay=2.0, max_delay=30.0, jitter_mode="full"
        )
        assert engine.max_attempts == 5
        assert engine.base_delay == 2.0
        assert engine.max_delay == 30.0
        assert engine.jitter_mode == "full"

    def test_init_invalid_max_attempts(self):
        """max_attempts < 1 raises ValueError."""
        with pytest.raises(ValueError, match="max_attempts"):
            RetryEngine(max_attempts=0)

    def test_init_invalid_base_delay(self):
        """base_delay <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="base_delay"):
            RetryEngine(base_delay=0)

    def test_init_invalid_max_delay(self):
        """max_delay <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="max_delay"):
            RetryEngine(max_delay=0)


class TestSlidingWindowRateLimiterInit:
    """Constructor validation tests for SlidingWindowRateLimiter."""

    def test_valid_params_rpm_only(self):
        """Valid RPM limit with TPM=0 (disabled)."""
        limiter = SlidingWindowRateLimiter(
            rpm_limit=5.0, tpm_limit=0.0, window_seconds=60.0, profile_name="test"
        )
        assert limiter.rpm_limit == 5.0
        assert limiter.tpm_limit == 0.0
        assert limiter.window_seconds == 60.0

    def test_valid_params_tpm_only(self):
        """Valid TPM limit with RPM=0 (disabled)."""
        limiter = SlidingWindowRateLimiter(
            rpm_limit=0.0, tpm_limit=1000.0, window_seconds=60.0, profile_name="test"
        )
        assert limiter.rpm_limit == 0.0
        assert limiter.tpm_limit == 1000.0

    def test_valid_params_both_limits(self):
        """Both RPM and TPM limits enabled."""
        limiter = SlidingWindowRateLimiter(
            rpm_limit=5.0, tpm_limit=1000.0, window_seconds=60.0, profile_name="test"
        )
        assert limiter.rpm_limit == 5.0
        assert limiter.tpm_limit == 1000.0

    def test_both_limits_zero_raises(self):
        """Both limits <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="at least one of rpm_limit or tpm_limit"):
            SlidingWindowRateLimiter(rpm_limit=0.0, tpm_limit=0.0)

    def test_both_limits_negative_raises(self):
        """Both limits negative raises ValueError."""
        with pytest.raises(ValueError, match="at least one of rpm_limit or tpm_limit"):
            SlidingWindowRateLimiter(rpm_limit=-1.0, tpm_limit=-5.0)

    def test_default_window_seconds(self):
        """Default window_seconds is 60.0."""
        limiter = SlidingWindowRateLimiter(rpm_limit=5.0, tpm_limit=0.0)
        assert limiter.window_seconds == 60.0

    def test_default_profile_name(self):
        """Default profile_name is 'default'."""
        limiter = SlidingWindowRateLimiter(rpm_limit=5.0, tpm_limit=0.0)
        assert limiter.get_snapshot() is not None


class TestSlidingWindowRateLimiterAcquire:
    """Basic acquire behavior tests for SlidingWindowRateLimiter."""

    @pytest.mark.asyncio
    async def test_acquire_within_capacity(self):
        """Acquires within burst capacity return immediately (fast path)."""
        limiter = SlidingWindowRateLimiter(rpm_limit=5.0, tpm_limit=0.0)
        for _ in range(5):
            await limiter.acquire()

    @pytest.mark.asyncio
    async def test_acquire_blocks_when_rpm_full(self):
        """Acquire blocks when RPM limit is reached."""
        limiter = SlidingWindowRateLimiter(rpm_limit=1.0, tpm_limit=0.0, window_seconds=0.1)
        await limiter.acquire()

        start = time.monotonic()
        await limiter.acquire()
        elapsed = time.monotonic() - start

        assert elapsed >= 0.05, f"Should have blocked, but only {elapsed:.4f}s elapsed"

    @pytest.mark.asyncio
    async def test_fast_path_no_scheduler_created(self):
        """Fast path: no scheduler task when capacity available and no waiters."""
        limiter = SlidingWindowRateLimiter(rpm_limit=5.0, tpm_limit=0.0)
        await limiter.acquire()
        assert limiter._scheduler_task is None

    @pytest.mark.asyncio
    async def test_slow_path_starts_scheduler(self):
        """Slow path: scheduler starts when a waiter is enqueued."""
        limiter = SlidingWindowRateLimiter(
            rpm_limit=1.0, tpm_limit=0.0, window_seconds=0.05
        )
        await limiter.acquire()

        task = asyncio.create_task(limiter.acquire())
        await asyncio.sleep(0.001)
        assert limiter._scheduler_task is not None
        await asyncio.wait_for(task, timeout=1.0)

    @pytest.mark.asyncio
    async def test_rpm_enforcement_five_requests(self):
        """RPM limit of 5: 5 requests succeed, 6th blocks."""
        limiter = SlidingWindowRateLimiter(
            rpm_limit=5.0, tpm_limit=0.0, window_seconds=0.1
        )

        for i in range(5):
            await limiter.acquire()

        start = time.monotonic()
        await limiter.acquire()
        elapsed = time.monotonic() - start

        assert elapsed > 0.001, f"6th request should have blocked, took {elapsed:.4f}s"

    @pytest.mark.asyncio
    async def test_concurrent_acquire_all_complete(self):
        """Multiple concurrent acquirers all eventually succeed."""
        limiter = SlidingWindowRateLimiter(
            rpm_limit=1.0, tpm_limit=0.0, window_seconds=0.02
        )
        await limiter.acquire()

        results = []

        async def worker(worker_id: int):
            await limiter.acquire()
            results.append(worker_id)

        tasks = [asyncio.create_task(worker(i)) for i in range(5)]
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=2.0)

        assert len(results) == 5, f"Expected 5 completions, got {len(results)}"

    @pytest.mark.asyncio
    async def test_fifo_fairness(self):
        """Earlier waiters are served before later ones (FIFO fairness)."""
        limiter = SlidingWindowRateLimiter(
            rpm_limit=1.0, tpm_limit=0.0, window_seconds=0.03
        )
        await limiter.acquire()

        completion_order = []

        async def worker(worker_id: int):
            await limiter.acquire()
            completion_order.append(worker_id)

        tasks = [asyncio.create_task(worker(i)) for i in range(3)]
        await asyncio.sleep(0.01)
        await asyncio.gather(*tasks)

        assert completion_order == [0, 1, 2], f"FIFO order violated: {completion_order}"

    @pytest.mark.asyncio
    async def test_cancellation_cleans_up_waiter(self):
        """Cancelled acquire removes itself from waiter queue."""
        limiter = SlidingWindowRateLimiter(
            rpm_limit=1.0, tpm_limit=0.0, window_seconds=1.0
        )
        await limiter.acquire()

        task = asyncio.create_task(limiter.acquire())
        await asyncio.sleep(0.01)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        await asyncio.sleep(1.1)
        await asyncio.wait_for(limiter.acquire(), timeout=1.0)


class TestSlidingWindowRateLimiterTPM:
    """TPM (tokens per minute) tracking tests."""

    @pytest.mark.asyncio
    async def test_tpm_tracking_record_usage(self):
        """record_usage affects remaining fraction."""
        limiter = SlidingWindowRateLimiter(
            rpm_limit=0.0, tpm_limit=100.0, window_seconds=60.0
        )

        assert limiter.get_remaining_fraction() == 1.0
        limiter.record_usage(50)
        assert limiter.get_remaining_fraction() == 0.5

    @pytest.mark.asyncio
    async def test_tpm_blocks_when_full(self):
        """Acquire with estimated_tokens blocks when TPM would be exceeded."""
        limiter = SlidingWindowRateLimiter(
            rpm_limit=10.0, tpm_limit=100.0, window_seconds=0.1
        )

        limiter.record_usage(100)

        start = time.monotonic()
        await limiter.acquire(estimated_tokens=0)
        elapsed = time.monotonic() - start

        assert elapsed < 0.05, f"Should be fast, took {elapsed:.4f}s"

    @pytest.mark.asyncio
    async def test_tpm_and_rpm_together(self):
        """Both TPM and RPM limits are enforced together."""
        limiter = SlidingWindowRateLimiter(
            rpm_limit=2.0, tpm_limit=100.0, window_seconds=0.1
        )

        await limiter.acquire()
        assert limiter.get_remaining_fraction() == 0.5

        await limiter.acquire()
        assert limiter.get_remaining_fraction() == 0.0


class TestSlidingWindowRateLimiterWindowExpiry:
    """Window expiration tests."""

    @pytest.mark.asyncio
    async def test_window_expiry_clears_entries(self):
        """Entries expire after window_seconds."""
        limiter = SlidingWindowRateLimiter(
            rpm_limit=1.0, tpm_limit=0.0, window_seconds=0.05
        )

        await limiter.acquire()  # Add entry
        assert len(limiter._rpm_entries) == 1

        # Wait for window to expire
        await asyncio.sleep(0.06)

        # Prune and check
        limiter._prune_stale()
        assert len(limiter._rpm_entries) == 0

    @pytest.mark.asyncio
    async def test_capacity_returns_after_expiry(self):
        """After window expires, capacity is restored."""
        limiter = SlidingWindowRateLimiter(
            rpm_limit=1.0, tpm_limit=0.0, window_seconds=0.05
        )

        await limiter.acquire()
        await asyncio.sleep(0.06)

        start = time.monotonic()
        await limiter.acquire()
        elapsed = time.monotonic() - start

        assert elapsed < 0.01, f"Should be immediate after expiry, took {elapsed:.4f}s"


class TestKey_poolInit:
    """Constructor validation tests for APIKeyPool."""

    def test_valid_params(self):
        """Valid profile list creates pool successfully."""
        profiles = [
            ProviderProfile(
                name="key-a",
                api_key="ak1",
                base_url="https://api.test.com",
                model="gpt-4",
                rpm_limit=60.0,
                tpm_limit=100000.0,
                weight=2,
            ),
        ]
        pool = APIKeyPool(profiles)
        assert pool.get_active_profiles() == profiles

    def test_empty_profiles_raises(self):
        """Empty profile list raises ValueError."""
        with pytest.raises(ValueError, match="at least one profile"):
            APIKeyPool([])

    def test_default_cooldown_params(self):
        """Default cooldown params are 30s initial and 300s max."""
        profiles = [
            ProviderProfile(
                name="key-a",
                api_key="ak1",
                base_url="https://api.test.com",
                model="gpt-4",
                rpm_limit=60.0,
                tpm_limit=100000.0,
            ),
        ]
        pool = APIKeyPool(profiles)
        assert pool._cooldown_initial_ms == 30000.0
        assert pool._cooldown_max_ms == 300000.0


class TestKey_poolAcquire:
    """Key selection and rotation tests."""

    def _make_profiles(self):
        return [
            ProviderProfile(
                name="key-a",
                api_key="ak1",
                base_url="https://api.test.com",
                model="gpt-4",
                rpm_limit=60.0,
                tpm_limit=100000.0,
                weight=2,
            ),
            ProviderProfile(
                name="key-b",
                api_key="ak2",
                base_url="https://api.test.com",
                model="gpt-4",
                rpm_limit=60.0,
                tpm_limit=100000.0,
                weight=1,
            ),
        ]

    def test_acquire_highest_weight_first(self):
        """Highest weight profile is selected first."""
        profiles = self._make_profiles()
        pool = APIKeyPool(profiles)
        key = pool.acquire_key()
        assert key.name == "key-a"

    def test_acquire_fewest_errors_when_weights_equal(self):
        """When weights are equal, profile with fewer errors is selected."""
        profiles = [
            ProviderProfile(
                name="key-a",
                api_key="ak1",
                base_url="https://api.test.com",
                model="gpt-4",
                rpm_limit=60.0,
                tpm_limit=100000.0,
                weight=1,
            ),
            ProviderProfile(
                name="key-b",
                api_key="ak2",
                base_url="https://api.test.com",
                model="gpt-4",
                rpm_limit=60.0,
                tpm_limit=100000.0,
                weight=1,
            ),
        ]
        pool = APIKeyPool(profiles)
        pool._health["key-a"].consecutive_errors = 2
        key = pool.acquire_key()
        assert key.name == "key-b"

    def test_key_rotation_on_rate_limit(self):
        """Rate limiting key-a causes rotation to key-b."""
        profiles = self._make_profiles()
        pool = APIKeyPool(profiles)

        with patch("engine.safety.time.monotonic", return_value=1000.0):
            pool.report_rate_limited("key-a")
            key = pool.acquire_key()
        assert key.name == "key-b"

    def test_all_in_cooldown_returns_least_recently_cooled(self):
        """When all keys are in cooldown, return the one with earliest cooldown_until."""
        profiles = self._make_profiles()
        pool = APIKeyPool(profiles)

        with patch("engine.safety.time.monotonic", return_value=1000.0):
            pool.report_rate_limited("key-a")

        with patch("engine.safety.time.monotonic", return_value=1001.0):
            pool.report_rate_limited("key-b")

        key = pool.acquire_key()
        assert key.name == "key-a"


class TestKey_poolCooldown:
    """Cooldown escalation and reset tests."""

    def _make_pool(self):
        profiles = [
            ProviderProfile(
                name="key-a",
                api_key="ak1",
                base_url="https://api.test.com",
                model="gpt-4",
                rpm_limit=60.0,
                tpm_limit=100000.0,
            ),
        ]
        return APIKeyPool(profiles)

    def test_cooldown_escalation_first(self):
        """1st rate limit triggers initial cooldown (30s)."""
        pool = self._make_pool()
        with patch("engine.safety.time.monotonic", return_value=1000.0):
            pool.report_rate_limited("key-a")
        health = pool.get_cooldown_status()["key-a"]
        assert health.cooldown_until == 1000.0 + 30.0
        assert health.consecutive_errors == 1

    def test_cooldown_escalation_second(self):
        """2nd consecutive rate limit triggers 60s cooldown."""
        pool = self._make_pool()
        with patch("engine.safety.time.monotonic", return_value=1000.0):
            pool.report_rate_limited("key-a")
        with patch("engine.safety.time.monotonic", return_value=1001.0):
            pool.report_rate_limited("key-a")
        health = pool.get_cooldown_status()["key-a"]
        assert health.cooldown_until == 1001.0 + 60.0
        assert health.consecutive_errors == 2

    def test_cooldown_escalation_third(self):
        """3rd+ consecutive rate limit triggers max cooldown (300s)."""
        pool = self._make_pool()
        with patch("engine.safety.time.monotonic", return_value=1000.0):
            pool.report_rate_limited("key-a")
        with patch("engine.safety.time.monotonic", return_value=1001.0):
            pool.report_rate_limited("key-a")
        with patch("engine.safety.time.monotonic", return_value=1002.0):
            pool.report_rate_limited("key-a")
        health = pool.get_cooldown_status()["key-a"]
        assert health.cooldown_until == 1002.0 + 300.0
        assert health.consecutive_errors == 3

    def test_cooldown_escalation_max_capped(self):
        """4th+ rate limit still triggers max cooldown (300s)."""
        pool = self._make_pool()
        base_time = 1000.0
        for i in range(5):
            with patch("engine.safety.time.monotonic", return_value=base_time + i):
                pool.report_rate_limited("key-a")
        health = pool.get_cooldown_status()["key-a"]
        assert health.cooldown_until == 1004.0 + 300.0
        assert health.consecutive_errors == 5

    def test_cooldown_uses_retry_after_when_larger(self):
        """retry_after_ms overrides staircase when larger."""
        pool = self._make_pool()
        with patch("engine.safety.time.monotonic", return_value=1000.0):
            pool.report_rate_limited("key-a", retry_after_ms=60000.0)
        health = pool.get_cooldown_status()["key-a"]
        assert health.cooldown_until == 1000.0 + 60.0

    def test_cooldown_uses_staircase_when_retry_after_smaller(self):
        """Staircase wins when retry_after_ms is smaller."""
        pool = self._make_pool()
        with patch("engine.safety.time.monotonic", return_value=1000.0):
            pool.report_rate_limited("key-a", retry_after_ms=10000.0)
        health = pool.get_cooldown_status()["key-a"]
        assert health.cooldown_until == 1000.0 + 30.0

    def test_success_resets_errors_and_cooldown(self):
        """Success resets consecutive_errors and cooldown_until."""
        pool = self._make_pool()
        with patch("engine.safety.time.monotonic", return_value=1000.0):
            pool.report_rate_limited("key-a")

        with patch("engine.safety.time.monotonic", return_value=1001.0):
            pool.report_success("key-a")

        health = pool.get_cooldown_status()["key-a"]
        assert health.consecutive_errors == 0
        assert health.cooldown_until is None

    def test_success_does_not_log_on_healthy_key(self):
        """Success on a healthy key does not log recovery."""
        pool = self._make_pool()
        with patch("engine.safety.get_logger") as mock_logger:
            pool.report_success("key-a")
            mock_logger.return_value.info.assert_not_called()


class TestKey_poolExhaustion:
    """Pool exhaustion detection tests."""

    def _make_pool(self):
        profiles = [
            ProviderProfile(
                name="key-a",
                api_key="ak1",
                base_url="https://api.test.com",
                model="gpt-4",
                rpm_limit=60.0,
                tpm_limit=100000.0,
            ),
            ProviderProfile(
                name="key-b",
                api_key="ak2",
                base_url="https://api.test.com",
                model="gpt-4",
                rpm_limit=60.0,
                tpm_limit=100000.0,
            ),
        ]
        return APIKeyPool(profiles)

    def test_is_all_in_cooldown_false(self):
        """is_all_in_cooldown returns False when some keys are active."""
        pool = self._make_pool()
        with patch("engine.safety.time.monotonic", return_value=1000.0):
            pool.report_rate_limited("key-a")
        assert not pool.is_all_in_cooldown()

    def test_is_all_in_cooldown_true(self):
        """is_all_in_cooldown returns True when all keys are in cooldown."""
        pool = self._make_pool()
        with patch("engine.safety.time.monotonic", return_value=1000.0):
            pool.report_rate_limited("key-a")
            pool.report_rate_limited("key-b")
            assert pool.is_all_in_cooldown()

    def test_is_all_in_cooldown_expired(self):
        """is_all_in_cooldown returns False when cooldown has expired."""
        pool = self._make_pool()
        with patch("engine.safety.time.monotonic", return_value=1000.0):
            pool.report_rate_limited("key-a")
            pool.report_rate_limited("key-b")

        with patch("engine.safety.time.monotonic", return_value=2000.0):
            assert not pool.is_all_in_cooldown()

    def test_all_in_cooldown_logs_error(self):
        """When all keys enter cooldown, an error is logged."""
        pool = self._make_pool()
        with patch("engine.safety.get_logger") as mock_logger:
            with patch("engine.safety.time.monotonic", return_value=1000.0):
                pool.report_rate_limited("key-a")
                pool.report_rate_limited("key-b")

            mock_logger.return_value.error.assert_called_once()
            call_args = mock_logger.return_value.error.call_args
            assert call_args[1]["event_type"] == "key_pool_exhausted"

    def test_get_active_profiles(self):
        """get_active_profiles returns only profiles not in cooldown."""
        pool = self._make_pool()
        with patch("engine.safety.time.monotonic", return_value=1000.0):
            pool.report_rate_limited("key-a")
            active = pool.get_active_profiles()
            assert len(active) == 1
            assert active[0].name == "key-b"

    def test_get_cooldown_status_returns_copy(self):
        """get_cooldown_status returns a copy that does not affect internal state."""
        pool = self._make_pool()
        status = pool.get_cooldown_status()
        status["key-a"].consecutive_errors = 99
        assert pool.get_cooldown_status()["key-a"].consecutive_errors == 0


class TestSlidingWindowRateLimiterSnapshot:
    """Snapshot and remaining fraction tests."""

    @pytest.mark.asyncio
    async def test_get_snapshot_rpm_only(self):
        """Snapshot shows RPM remaining, TPM is None when disabled."""
        limiter = SlidingWindowRateLimiter(
            rpm_limit=5.0, tpm_limit=0.0, window_seconds=60.0
        )

        snapshot = limiter.get_snapshot()
        assert snapshot.remaining_rpm == 5
        assert snapshot.limit_rpm == 5
        assert snapshot.remaining_tpm is None
        assert snapshot.limit_tpm is None

    @pytest.mark.asyncio
    async def test_get_snapshot_tpm_only(self):
        """Snapshot shows TPM remaining, RPM is None when disabled."""
        limiter = SlidingWindowRateLimiter(
            rpm_limit=0.0, tpm_limit=100.0, window_seconds=60.0
        )

        snapshot = limiter.get_snapshot()
        assert snapshot.remaining_rpm is None
        assert snapshot.limit_rpm is None
        assert snapshot.remaining_tpm == 100
        assert snapshot.limit_tpm == 100

    @pytest.mark.asyncio
    async def test_get_remaining_fraction_combined(self):
        """Remaining fraction is min of RPM and TPM fractions."""
        limiter = SlidingWindowRateLimiter(
            rpm_limit=10.0, tpm_limit=100.0, window_seconds=60.0
        )

        for _ in range(5):
            await limiter.acquire()
        limiter.record_usage(25)

        fraction = limiter.get_remaining_fraction()
        assert fraction == min(0.5, 0.75)

    @pytest.mark.asyncio
    async def test_get_remaining_fraction_zero_limits(self):
        """Disabled limits contribute 1.0 (100%) to fraction calculation."""
        limiter = SlidingWindowRateLimiter(
            rpm_limit=5.0, tpm_limit=0.0, window_seconds=60.0
        )

        await limiter.acquire()
        assert limiter.get_remaining_fraction() == 0.8


class TestAdaptivePacer:
    """Tests for AdaptivePacer pace levels, delays, and waiting."""

    def test_pace_level_healthy_at_80_percent(self):
        """HEALTHY when remaining fraction is above 50%."""
        pacer = AdaptivePacer()
        snapshot = RateLimitSnapshot(
            remaining_rpm=48, limit_rpm=60,
            remaining_tpm=80000, limit_tpm=100000,
        )
        pacer.update_from_snapshot(snapshot)
        assert pacer.get_pace_level() == PaceLevel.HEALTHY
        assert pacer._remaining_fraction == 0.8

    def test_pace_level_pressing_at_30_percent(self):
        """PRESSING when remaining fraction is between 20% and 50%."""
        pacer = AdaptivePacer()
        snapshot = RateLimitSnapshot(
            remaining_rpm=18, limit_rpm=60,
            remaining_tpm=30000, limit_tpm=100000,
        )
        pacer.update_from_snapshot(snapshot)
        assert pacer.get_pace_level() == PaceLevel.PRESSING
        assert pacer._remaining_fraction == 0.3

    def test_pace_level_critical_at_10_percent(self):
        """CRITICAL when remaining fraction is below 20%."""
        pacer = AdaptivePacer()
        snapshot = RateLimitSnapshot(
            remaining_rpm=6, limit_rpm=60,
            remaining_tpm=10000, limit_tpm=100000,
        )
        pacer.update_from_snapshot(snapshot)
        assert pacer.get_pace_level() == PaceLevel.CRITICAL
        assert pacer._remaining_fraction == 0.1

    def test_pace_level_uses_min_fraction(self):
        """Pace level uses the minimum of available fractions."""
        pacer = AdaptivePacer()
        snapshot = RateLimitSnapshot(
            remaining_rpm=50, limit_rpm=60,
            remaining_tpm=5000, limit_tpm=100000,
        )
        pacer.update_from_snapshot(snapshot)
        assert pacer.get_pace_level() == PaceLevel.CRITICAL
        assert pacer._remaining_fraction == 0.05

    def test_pace_level_defaults_to_healthy_with_empty_snapshot(self):
        """Default to HEALTHY when snapshot has no usable limits."""
        pacer = AdaptivePacer()
        snapshot = RateLimitSnapshot()
        pacer.update_from_snapshot(snapshot)
        assert pacer.get_pace_level() == PaceLevel.HEALTHY
        assert pacer._remaining_fraction == 1.0

    def test_update_from_tokens_critical_at_10_percent(self):
        """CRITICAL when 90% of tokens are used."""
        pacer = AdaptivePacer()
        pacer.update_from_tokens(used_tokens=90, limit_tokens=100)
        assert pacer.get_pace_level() == PaceLevel.CRITICAL
        assert pacer._remaining_fraction == pytest.approx(0.1)

    def test_update_from_tokens_healthy_at_80_percent(self):
        """HEALTHY when 20% of tokens are used."""
        pacer = AdaptivePacer()
        pacer.update_from_tokens(used_tokens=20, limit_tokens=100)
        assert pacer.get_pace_level() == PaceLevel.HEALTHY
        assert pacer._remaining_fraction == 0.8

    def test_update_from_tokens_zero_limit(self):
        """Zero limit_tokens defaults to HEALTHY with fraction 1.0."""
        pacer = AdaptivePacer()
        pacer.update_from_tokens(used_tokens=0, limit_tokens=0)
        assert pacer.get_pace_level() == PaceLevel.HEALTHY
        assert pacer._remaining_fraction == 1.0

    def test_recommended_delay_healthy(self):
        """HEALTHY level has no recommended delay."""
        pacer = AdaptivePacer()
        pacer.update_from_tokens(used_tokens=0, limit_tokens=100)
        assert pacer.get_recommended_delay() == 0.0

    def test_recommended_delay_pressing(self):
        """PRESSING level adds 200 ms delay."""
        pacer = AdaptivePacer()
        pacer.update_from_tokens(used_tokens=70, limit_tokens=100)
        assert pacer.get_recommended_delay() == 200.0

    def test_recommended_delay_critical(self):
        """CRITICAL level adds 1000 ms delay."""
        pacer = AdaptivePacer()
        pacer.update_from_tokens(used_tokens=95, limit_tokens=100)
        assert pacer.get_recommended_delay() == 1000.0

    @pytest.mark.asyncio
    async def test_min_interval_enforcement(self):
        """Two rapid calls: second call waits for min interval."""
        pacer = AdaptivePacer(min_interval_ms=100)
        await pacer.wait_if_needed()
        start = time.monotonic()
        await pacer.wait_if_needed()
        elapsed_ms = (time.monotonic() - start) * 1000.0
        assert elapsed_ms >= 90.0, f"Expected wait ~100ms, got {elapsed_ms:.1f}ms"

    @pytest.mark.asyncio
    async def test_disabled_pacer_no_waiting(self):
        """Disabled pacer returns immediately without waiting."""
        pacer = AdaptivePacer(min_interval_ms=200, enabled=False)
        await pacer.wait_if_needed()
        start = time.monotonic()
        await pacer.wait_if_needed()
        elapsed_ms = (time.monotonic() - start) * 1000.0
        assert elapsed_ms < 50.0, f"Disabled pacer should not wait, got {elapsed_ms:.1f}ms"

    @pytest.mark.asyncio
    async def test_wait_includes_recommended_delay(self):
        """Wait includes both min interval and recommended delay."""
        pacer = AdaptivePacer(min_interval_ms=50)
        pacer.update_from_tokens(used_tokens=95, limit_tokens=100)
        await pacer.wait_if_needed()
        start = time.monotonic()
        await pacer.wait_if_needed()
        elapsed_ms = (time.monotonic() - start) * 1000.0
        assert elapsed_ms >= 1040.0, f"Expected wait ~1050ms, got {elapsed_ms:.1f}ms"

    def test_init_defaults(self):
        """Default constructor parameters."""
        pacer = AdaptivePacer()
        assert pacer._min_interval_ms == 500
        assert pacer._enabled is True
        assert pacer._remaining_fraction == 1.0
        assert pacer._last_call_timestamp is None
        assert pacer.get_pace_level() == PaceLevel.HEALTHY

    def test_init_custom_params(self):
        """Custom constructor parameters."""
        pacer = AdaptivePacer(min_interval_ms=100, enabled=False)
        assert pacer._min_interval_ms == 100
        assert pacer._enabled is False
