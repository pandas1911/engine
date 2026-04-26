"""Integration tests for the provider chain: config → runner → LLM provider.

Verifies parameter merge, key rotation, fallback behaviour, and wrap-around
using mocks — no real API calls are made.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from engine.config import Config
from engine.providers.llm_provider import LLMProvider, LLMProviderError
from engine.providers.fallback_provider import FallbackLLMProvider
from engine.providers.provider_models import ProviderParams, LLMResponse, ErrorClass
from engine.safety import APIKeyPool, RetryEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides) -> Config:
    """Create a minimal Config instance for testing."""
    cfg = Config(
        strip_thinking=overrides.get("strip_thinking", True),
        llm_retry_max_attempts=overrides.get("llm_retry_max_attempts", 1),
        llm_retry_base_delay=overrides.get("llm_retry_base_delay", 0.01),
    )
    return cfg


def _make_mock_response(content: str = "test response") -> MagicMock:
    """Build a fake OpenAI chat-completion response."""
    mock_msg = MagicMock()
    mock_msg.content = content
    mock_msg.tool_calls = None

    mock_choice = MagicMock()
    mock_choice.message = mock_msg

    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    return mock_resp


def _make_provider_with_mock_client(
    config: Config,
    model: str = "test-model",
    model_params=None,
) -> tuple[LLMProvider, MagicMock]:
    """Create an LLMProvider whose OpenAI client is fully mocked.

    Returns (provider, mock_client) so the caller can inspect call args.
    """
    mock_client = MagicMock()
    mock_resp = _make_mock_response()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

    with patch(
        "engine.providers.llm_provider.AsyncOpenAI", return_value=mock_client
    ):
        provider = LLMProvider(
            provider_params=ProviderParams(
                api_key="test-key",
                base_url="https://api.test.com",
                model=model,
            ),
            runtime_config=config,
            model_params=model_params,
        )
    # Swap client directly to guarantee the mock is used
    provider.client = mock_client
    return provider, mock_client


# ===========================================================================
# 1. LLMProvider — model params merge
# ===========================================================================

@pytest.mark.asyncio
async def test_llm_provider_model_params_merge():
    """model_params dict should be merged into the API call kwargs."""
    config = _make_config()
    provider, mock_client = _make_provider_with_mock_client(
        config,
        model_params={"reasoning_effort": "high"},
    )

    await provider.chat(
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
    )

    call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert call_kwargs["reasoning_effort"] == "high"
    assert call_kwargs["model"] == "test-model"
    assert call_kwargs["messages"] == [{"role": "user", "content": "hi"}]


# ===========================================================================
# 2. LLMProvider — empty model_params
# ===========================================================================

@pytest.mark.asyncio
async def test_llm_provider_empty_model_params():
    """When model_params is omitted, only model and messages should be sent."""
    config = _make_config()
    provider, mock_client = _make_provider_with_mock_client(config)

    await provider.chat(
        messages=[{"role": "user", "content": "hello"}],
        tools=[],
    )

    call_kwargs = mock_client.chat.completions.create.call_args[1]
    # Only base keys present — no extra model_params
    assert set(call_kwargs.keys()) == {"model", "messages"}
    assert call_kwargs["model"] == "test-model"


# ===========================================================================
# 3. FallbackLLMProvider — rotation on rate limit
# ===========================================================================

@pytest.mark.asyncio
async def test_fallback_rotation_on_rate_limit():
    """On rate-limit error from primary, fallback provider should be called."""
    config = _make_config()

    provider_a, _ = _make_provider_with_mock_client(config, model="model-a")
    provider_b, _ = _make_provider_with_mock_client(config, model="model-b")

    # provider_a.chat raises a rate-limit-like error
    provider_a.chat = AsyncMock(
        side_effect=Exception("rate_limit_exceeded")
    )
    provider_b.chat = AsyncMock(
        return_value=LLMResponse(content="fallback ok")
    )

    retry_engine = RetryEngine(max_attempts=1, base_delay=0.01)
    retry_engine.classify_error = MagicMock(return_value=ErrorClass.RATE_LIMITED)
    retry_engine.extract_retry_after = MagicMock(return_value=None)

    key_pool = APIKeyPool(["prov-a/model-a", "prov-b/model-b"])
    fallback = FallbackLLMProvider(
        providers={
            "prov-a/model-a": provider_a,
            "prov-b/model-b": provider_b,
        },
        key_pool=key_pool,
        rate_limiters={},
        pacers={},
        retry_engine=retry_engine,
    )

    result = await fallback.chat(
        messages=[{"role": "user", "content": "test"}],
        tools=[],
    )

    assert provider_a.chat.called
    assert provider_b.chat.called
    assert result.content == "fallback ok"


# ===========================================================================
# 4. FallbackLLMProvider — non-retryable raises immediately
# ===========================================================================

@pytest.mark.asyncio
async def test_fallback_non_retryable_raises():
    """Non-retryable error should raise LLMProviderError immediately."""

    class BadAuthError(Exception):
        pass

    config = _make_config()
    provider_a, _ = _make_provider_with_mock_client(config)
    provider_a.chat = AsyncMock(side_effect=BadAuthError("invalid api key"))

    retry_engine = RetryEngine(max_attempts=1, base_delay=0.01)
    retry_engine.classify_error = MagicMock(return_value=ErrorClass.NON_RETRYABLE)

    key_pool = APIKeyPool(["prov-a/model-a"])
    fallback = FallbackLLMProvider(
        providers={"prov-a/model-a": provider_a},
        key_pool=key_pool,
        rate_limiters={},
        pacers={},
        retry_engine=retry_engine,
    )

    with pytest.raises(LLMProviderError) as exc_info:
        await fallback.chat(
            messages=[{"role": "user", "content": "test"}],
            tools=[],
        )

    assert "invalid api key" in str(exc_info.value)


# ===========================================================================
# 5. APIKeyPool — returns strings
# ===========================================================================

def test_api_key_pool_returns_strings():
    """acquire_key() should always return a string."""
    pool = APIKeyPool(["alpha", "beta"])
    key = pool.acquire_key()
    assert isinstance(key, str)
    assert key in ("alpha", "beta")


# ===========================================================================
# 6. APIKeyPool — cooldown rotation
# ===========================================================================

def test_api_key_pool_cooldown_rotation():
    """A rate-limited key should be skipped in favour of the next one."""
    pool = APIKeyPool(["key1", "key2"], cooldown_initial_ms=60000.0)

    # Send key1 into cooldown
    pool.report_rate_limited("key1")

    key = pool.acquire_key()
    assert key == "key2", f"Expected 'key2' but got '{key}'"


# ===========================================================================
# 7. APIKeyPool — success resets cooldown
# ===========================================================================

def test_api_key_pool_success_resets():
    """report_success() should clear cooldown so the key is usable again."""
    pool = APIKeyPool(["key1", "key2"], cooldown_initial_ms=60000.0)

    pool.report_rate_limited("key1")
    pool.report_success("key1")

    key = pool.acquire_key()
    assert key == "key1", "key1 should be available again after success reset"


# ===========================================================================
# 8. ProviderParams dataclass — fields accessible
# ===========================================================================

def test_provider_params_dataclass():
    """ProviderParams fields should be directly accessible."""
    params = ProviderParams(
        api_key="sk-abc",
        base_url="https://api.example.com",
        model="gpt-4o",
    )
    assert params.api_key == "sk-abc"
    assert params.base_url == "https://api.example.com"
    assert params.model == "gpt-4o"
