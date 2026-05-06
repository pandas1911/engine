"""Integration tests for FallbackLLMProvider's context truncation behavior.

Verifies that FallbackLLMProvider correctly integrates with
truncate_messages_for_tpm before forwarding messages to underlying
providers in both chat() and stream_chat() paths.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from engine.providers.fallback_provider import FallbackLLMProvider
from engine.providers.provider_models import LLMResponse
from engine.providers.chunk_types import StreamChunk
from engine.safety.key_pool import APIKeyPool
from engine.safety.retry import RetryEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_provider(name="test/model"):
    """Create a mock LLM provider with async chat and stream_chat."""
    provider = AsyncMock()
    provider.chat = AsyncMock(return_value=LLMResponse(content="ok"))
    provider.get_last_usage = MagicMock(return_value=None)
    # stream_chat returns an async generator
    async def _stream(*, messages, tools, agent_label="Root", task_id="unknown", depth=0):
        yield StreamChunk(delta_text="ok", finish_reason="stop")
    provider.stream_chat = MagicMock(side_effect=_stream)
    return provider


def _make_limiter(tpm_limit=100000):
    """Create a mock rate limiter with the given tpm_limit."""
    limiter = AsyncMock()
    limiter.tpm_limit = tpm_limit
    limiter.acquire = AsyncMock(return_value=1)
    limiter.record_usage = AsyncMock()
    limiter.release_reserved = AsyncMock()
    return limiter


def _make_key_pool(keys):
    """Create a real APIKeyPool with the given composite key names."""
    return APIKeyPool(names=keys)


def _build_large_messages(num_rounds=3, content_size=200):
    """Build a message list with system prompt + N full rounds.

    Each round = (user, assistant).  The last message is always a user message.
    With content_size=200 and coefficient=3.0 each content field contributes
    ~67 tokens, so the total easily exceeds a small tpm_limit.
    """
    messages = [{"role": "system", "content": "You are helpful."}]
    for i in range(num_rounds):
        messages.append({"role": "user", "content": "U" * content_size})
        messages.append({"role": "assistant", "content": "A" * content_size})
    # Final user turn (the "current" input that must be preserved)
    messages.append({"role": "user", "content": "Final query"})
    return messages


def _build_small_messages():
    """Build a message list that stays well under a low TPM limit."""
    return [
        {"role": "system", "content": "Hi"},
        {"role": "user", "content": "ping"},
    ]


def _build_fallback(
    providers,
    key_pool,
    rate_limiters,
    retry_engine=None,
):
    """Construct a FallbackLLMProvider with sensible defaults."""
    if retry_engine is None:
        retry_engine = RetryEngine(max_attempts=1)
    return FallbackLLMProvider(
        providers=providers,
        key_pool=key_pool,
        rate_limiters=rate_limiters,
        retry_engine=retry_engine,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestChatTruncation:
    """Tests for FallbackLLMProvider.chat() truncation path."""

    @pytest.mark.asyncio
    async def test_chat_truncation_applied(self):
        """Messages exceeding tpm_limit are truncated before reaching provider.chat()."""
        tpm_limit = 100
        provider = _make_provider()
        key_pool = _make_key_pool(["prov/model-a"])
        limiter = _make_limiter(tpm_limit=tpm_limit)
        rate_limiters = {"prov": limiter}

        fb = _build_fallback(
            providers={"prov/model-a": provider},
            key_pool=key_pool,
            rate_limiters=rate_limiters,
        )

        original_messages = _build_large_messages(num_rounds=3, content_size=200)
        # Original has 8 messages: system + 3*(user+assistant) + final user
        assert len(original_messages) == 8

        result = await fb.chat(messages=original_messages, tools=[])

        # The response should be successful
        assert result.content == "ok"

        # provider.chat should have been called once
        provider.chat.assert_called_once()

        # The messages passed to provider.chat should be fewer than original
        call_kwargs = provider.chat.call_args
        passed_messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
        assert len(passed_messages) < len(original_messages), (
            "Expected truncated messages ({} < {}), but got same or more".format(
                len(passed_messages), len(original_messages)
            )
        )

        # The system prompt must be preserved
        assert passed_messages[0]["role"] == "system"

        # The last message must be the final user query
        assert passed_messages[-1]["content"] == "Final query"

    @pytest.mark.asyncio
    async def test_chat_no_truncation_when_under_limit(self):
        """Messages under tpm_limit are passed through unchanged."""
        tpm_limit = 100000  # High limit
        provider = _make_provider()
        key_pool = _make_key_pool(["prov/model-a"])
        limiter = _make_limiter(tpm_limit=tpm_limit)
        rate_limiters = {"prov": limiter}

        fb = _build_fallback(
            providers={"prov/model-a": provider},
            key_pool=key_pool,
            rate_limiters=rate_limiters,
        )

        original_messages = _build_small_messages()
        # Small: system + user = 2 messages

        result = await fb.chat(messages=original_messages, tools=[])
        assert result.content == "ok"

        call_kwargs = provider.chat.call_args
        passed_messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")

        # Messages should be identical (same count, same content)
        assert len(passed_messages) == len(original_messages)
        for orig, passed in zip(original_messages, passed_messages):
            assert orig == passed


class TestStreamChatTruncation:
    """Tests for FallbackLLMProvider.stream_chat() truncation path."""

    @pytest.mark.asyncio
    async def test_stream_chat_truncation_applied(self):
        """Messages exceeding tpm_limit are truncated before reaching stream_chat()."""
        tpm_limit = 100
        provider = _make_provider()
        key_pool = _make_key_pool(["prov/model-a"])
        limiter = _make_limiter(tpm_limit=tpm_limit)
        rate_limiters = {"prov": limiter}

        fb = _build_fallback(
            providers={"prov/model-a": provider},
            key_pool=key_pool,
            rate_limiters=rate_limiters,
        )

        original_messages = _build_large_messages(num_rounds=3, content_size=200)

        # Consume the async generator
        chunks = []
        async for chunk in fb.stream_chat(messages=original_messages, tools=[]):
            chunks.append(chunk)

        # At least one chunk should be yielded
        assert len(chunks) >= 1
        assert chunks[0].delta_text == "ok"

        # stream_chat on the mock provider should have been called once
        provider.stream_chat.assert_called_once()

        call_kwargs = provider.stream_chat.call_args
        passed_messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")

        assert len(passed_messages) < len(original_messages), (
            "Expected truncated messages ({} < {})".format(
                len(passed_messages), len(original_messages)
            )
        )

        # System prompt preserved
        assert passed_messages[0]["role"] == "system"
        # Last message preserved
        assert passed_messages[-1]["content"] == "Final query"

    @pytest.mark.asyncio
    async def test_stream_chat_no_truncation_when_under_limit(self):
        """stream_chat passes messages through when under tpm_limit."""
        tpm_limit = 100000
        provider = _make_provider()
        key_pool = _make_key_pool(["prov/model-a"])
        limiter = _make_limiter(tpm_limit=tpm_limit)
        rate_limiters = {"prov": limiter}

        fb = _build_fallback(
            providers={"prov/model-a": provider},
            key_pool=key_pool,
            rate_limiters=rate_limiters,
        )

        original_messages = _build_small_messages()

        chunks = []
        async for chunk in fb.stream_chat(messages=original_messages, tools=[]):
            chunks.append(chunk)

        assert len(chunks) >= 1
        provider.stream_chat.assert_called_once()

        call_kwargs = provider.stream_chat.call_args
        passed_messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
        assert len(passed_messages) == len(original_messages)


class TestPerProviderTPMDifference:
    """Tests verifying per-provider TPM limits are respected."""

    @pytest.mark.asyncio
    async def test_per_provider_tpm_difference(self):
        """Truncation uses the specific provider's limiter, not a global value."""
        # Provider A has low TPM (truncation needed)
        # Provider B has high TPM (no truncation needed)
        provider_a = _make_provider()
        provider_b = _make_provider()

        key_pool = _make_key_pool(["provA/model-a", "provB/model-b"])
        limiter_a = _make_limiter(tpm_limit=100)   # Low → triggers truncation
        limiter_b = _make_limiter(tpm_limit=100000) # High → no truncation

        rate_limiters = {
            "provA": limiter_a,
            "provB": limiter_b,
        }

        fb = _build_fallback(
            providers={
                "provA/model-a": provider_a,
                "provB/model-b": provider_b,
            },
            key_pool=key_pool,
            rate_limiters=rate_limiters,
        )

        original_messages = _build_large_messages(num_rounds=3, content_size=200)

        # First call will use the first key from pool (provA/model-a, low TPM)
        result = await fb.chat(messages=original_messages, tools=[])
        assert result.content == "ok"

        # provider_a should have received truncated messages
        call_kwargs_a = provider_a.chat.call_args
        passed_a = call_kwargs_a.kwargs.get("messages") or call_kwargs_a[1].get("messages")
        assert len(passed_a) < len(original_messages), (
            "Provider A (low TPM) should have received truncated messages"
        )

        # Now force key pool to use provB by putting provA in cooldown
        key_pool.report_rate_limited("provA/model-a", retry_after_ms=60000)

        result2 = await fb.chat(messages=original_messages, tools=[])
        assert result2.content == "ok"

        # provider_b should have received the full messages (high TPM)
        call_kwargs_b = provider_b.chat.call_args
        passed_b = call_kwargs_b.kwargs.get("messages") or call_kwargs_b[1].get("messages")
        assert len(passed_b) == len(original_messages), (
            "Provider B (high TPM) should have received original messages unchanged"
        )


class TestAcquireTokenReflection:
    """Tests verifying limiter.acquire() receives truncated token count."""

    @pytest.mark.asyncio
    async def test_estimated_tokens_passed_to_acquire_reflects_truncation(self):
        """limiter.acquire() should receive the truncated token count, not original."""
        tpm_limit = 100
        provider = _make_provider()
        key_pool = _make_key_pool(["prov/model-a"])
        limiter = _make_limiter(tpm_limit=tpm_limit)
        rate_limiters = {"prov": limiter}

        fb = _build_fallback(
            providers={"prov/model-a": provider},
            key_pool=key_pool,
            rate_limiters=rate_limiters,
        )

        original_messages = _build_large_messages(num_rounds=3, content_size=200)

        await fb.chat(messages=original_messages, tools=[])

        # limiter.acquire should have been called once
        limiter.acquire.assert_called_once()

        # The estimated_tokens argument should be the truncated count
        call_kwargs = limiter.acquire.call_args
        token_arg = call_kwargs.kwargs.get("estimated_tokens")
        assert token_arg is not None, "limiter.acquire() was not called with estimated_tokens"

        # Truncated tokens should be <= tpm_limit
        assert token_arg <= tpm_limit, (
            "Truncated tokens ({}) should be <= tpm_limit ({})".format(
                token_arg, tpm_limit
            )
        )

        # Also verify it's strictly less than what the original messages would estimate
        # (since truncation removed rounds)
        original_tokens = fb._estimate_tokens(original_messages, None)
        assert token_arg < original_tokens, (
            "Truncated tokens ({}) should be less than original ({})".format(
                token_arg, original_tokens
            )
        )
