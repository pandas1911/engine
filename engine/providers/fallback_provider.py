"""Fallback LLM Provider with key rotation and provider fallback (ping-pong).

This module provides a FallbackLLMProvider that wraps multiple LLMProvider
instances, automatically rotating between API keys on rate limit errors and
falling back between providers when all keys are exhausted.
"""

import asyncio
from typing import AsyncGenerator, Dict, List, Optional

from engine.providers.llm_provider import BaseLLMProvider, LLMProvider, LLMProviderError
from engine.providers.provider_models import LLMResponse, ErrorClass
from engine.providers.chunk_types import StreamChunk
from engine.safety import APIKeyPool, SlidingWindowRateLimiter, RetryEngine
from engine.safety.token_estimator import EmaTokenEstimator
from engine.safety.context_truncation import truncate_messages_for_tpm
from engine.logging import get_logger


class FallbackLLMProvider(BaseLLMProvider):
    """LLM provider with key rotation and provider fallback.

    Wraps multiple LLMProvider instances and manages API key rotation
    via an APIKeyPool. On rate limit errors, automatically switches to
    the next available key. When all keys for a provider are exhausted,
    falls back to other providers (ping-pong). Successful requests reset
    provider health state.
    """

    def __init__(
        self,
        providers: Dict[str, LLMProvider],
        key_pool: APIKeyPool,
        rate_limiters: Dict[str, SlidingWindowRateLimiter],
        retry_engine: RetryEngine,
        max_profile_rotations: int = 3,
    ):
        self._providers = providers
        self._key_pool = key_pool
        self._rate_limiters = rate_limiters
        self._retry_engine = retry_engine
        self._max_profile_rotations = max_profile_rotations
        self._current_profile: Optional[str] = None
        self._rotation_count = 0
        self._logger = get_logger()
        self._token_estimator = EmaTokenEstimator()

    def _estimate_tokens(self, messages: List[Dict], tools: Optional[List[Dict]]) -> int:
        return self._token_estimator.estimate(messages, tools)

    def _apply_tpm_truncation(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]],
        estimated_tokens: int,
        limiter,  # SlidingWindowRateLimiter or None
        profile_name: str,
    ) -> tuple:
        """Apply TPM-based context truncation if needed.

        Returns (local_messages, local_estimated_tokens).
        """
        local_messages = messages
        local_estimated_tokens = estimated_tokens
        if limiter is not None and limiter.tpm_limit > 0:
            truncation = truncate_messages_for_tpm(
                messages, tools, limiter.tpm_limit, self._token_estimator
            )
            if truncation.rounds_removed > 0:
                self._logger.warning(
                    "RateControl",
                    "Context truncated for TPM | profile={}, rounds_removed={}, tokens={}/{}".format(
                        profile_name,
                        truncation.rounds_removed,
                        truncation.original_tokens,
                        truncation.truncated_tokens,
                    ),
                    event_type="context_truncated",
                    data={
                        "profile": profile_name,
                        "rounds_removed": truncation.rounds_removed,
                        "original_tokens": truncation.original_tokens,
                        "truncated_tokens": truncation.truncated_tokens,
                    },
                )
            local_messages = truncation.messages
            local_estimated_tokens = truncation.truncated_tokens
        return local_messages, local_estimated_tokens

    async def chat(
        self,
        messages: List[Dict],
        tools: List[Dict],
        agent_label: str = "Root",
        task_id: str = "unknown",
        depth: int = 0,
    ) -> LLMResponse:
        estimated_tokens = self._estimate_tokens(messages, tools)
        max_iterations = max(
            50,
            len(self._providers) * (self._max_profile_rotations + 1) * 2,
        )

        for iteration in range(max_iterations):
            profile_name = self._key_pool.acquire_key()
            self._current_profile = profile_name

            provider = self._providers.get(profile_name)
            if provider is None:
                raise RuntimeError(
                    "No provider found for profile: {}".format(profile_name)
                )

            # Extract provider name from composite key for limiter lookup.
            # Rate limiters are keyed by provider name (e.g., "aliyun"),
            # while profile_name is a composite key (e.g., "aliyun/deepseek-v4-pro").
            provider_name = profile_name.split("/", 1)[0]

            limiter = self._rate_limiters.get(provider_name)

            local_messages, local_estimated_tokens = self._apply_tpm_truncation(
                messages, tools, estimated_tokens, limiter, profile_name
            )

            reservation_id = 0
            if limiter is not None:
                reservation_id = await limiter.acquire(estimated_tokens=local_estimated_tokens)

            try:
                result = await provider.chat(
                    messages=local_messages,
                    tools=tools,
                    agent_label=agent_label,
                    task_id=task_id,
                    depth=depth,
                )

                self._key_pool.report_success(profile_name)
                self._rotation_count = 0

                if limiter is not None:
                    usage = provider.get_last_usage()
                    if usage is not None:
                        prompt_tokens, completion_tokens = usage
                        total_tokens = prompt_tokens + completion_tokens
                        await limiter.record_usage(total_tokens, reservation_id=reservation_id)
                        self._token_estimator.feedback(estimated_tokens, total_tokens)

                self._logger.info(
                    agent_label,
                    "Fallback provider success | profile={}".format(profile_name),
                    task_id=task_id,
                    state="running",
                    depth=depth,
                    event_type="fallback_provider_success",
                    data={"profile": profile_name},
                )

                return result

            except asyncio.CancelledError:
                if limiter is not None and reservation_id > 0:
                    await limiter.release_reserved(reservation_id)
                raise

            except Exception as e:
                error_class = self._retry_engine.classify_error(e)

                if error_class == ErrorClass.NON_RETRYABLE:
                    self._logger.warning(
                        agent_label,
                        "Non-retryable error from provider | profile={} error={}".format(
                            profile_name, str(e)[:200]
                        ),
                        task_id=task_id,
                        state="error",
                        depth=depth,
                        event_type="fallback_non_retryable",
                        data={
                            "profile": profile_name,
                            "error_type": type(e).__name__,
                            "error_message": str(e)[:500],
                        },
                    )
                    if limiter is not None and reservation_id > 0:
                        await limiter.release_reserved(reservation_id)
                    raise LLMProviderError(e) from e

                if error_class == ErrorClass.RETRYABLE:
                    self._logger.warning(
                        agent_label,
                        "Retryable error from provider | profile={} error={}".format(
                            profile_name, str(e)[:200]
                        ),
                        task_id=task_id,
                        state="running",
                        depth=depth,
                        event_type="fallback_retryable",
                        data={
                            "profile": profile_name,
                            "error_type": type(e).__name__,
                            "error_message": str(e)[:500],
                        },
                    )
                    if limiter is not None and reservation_id > 0:
                        await limiter.release_reserved(reservation_id)
                    raise

                if error_class == ErrorClass.RATE_LIMITED:
                    retry_after = self._retry_engine.extract_retry_after(e)
                    self._key_pool.report_rate_limited(profile_name, retry_after)
                    if limiter is not None and reservation_id > 0:
                        await limiter.release_reserved(reservation_id)
                    self._rotation_count += 1

                    self._logger.warning(
                        "RateControl",
                        "Rate limited on profile | profile={} rotation={}/{}".format(
                            profile_name,
                            self._rotation_count,
                            self._max_profile_rotations,
                        ),
                        event_type="fallback_rate_limited",
                        data={
                            "profile": profile_name,
                            "rotation_count": self._rotation_count,
                            "max_rotations": self._max_profile_rotations,
                            "retry_after_ms": retry_after,
                        },
                    )

                    if self._rotation_count > self._max_profile_rotations:
                        self._rotation_count = 0
                        self._logger.warning(
                            "RateControl",
                            "Provider fallback (ping-pong) | exhausted_profile={}".format(
                                profile_name
                            ),
                            event_type="provider_fallback",
                            data={
                                "exhausted_profile": profile_name,
                                "max_rotations": self._max_profile_rotations,
                            },
                        )

                    continue

        raise LLMProviderError(
            RuntimeError(
                "Fallback provider exceeded maximum iterations ({})".format(
                    max_iterations
                )
            )
        )

    async def stream_chat(
        self,
        messages: List[Dict],
        tools: List[Dict],
        agent_label: str = "Root",
        task_id: str = "unknown",
        depth: int = 0,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream a chat request with key rotation and provider fallback.

        Follows the same safety mechanism pattern as chat():
        acquire_key -> rate_limiter -> stream_chat -> report.
        On rate limit: rotate key and continue loop.
        On other errors: release reservation and raise.
        """
        estimated_tokens = self._estimate_tokens(messages, tools)
        max_iterations = max(
            50,
            len(self._providers) * (self._max_profile_rotations + 1) * 2,
        )

        for iteration in range(max_iterations):
            profile_name = self._key_pool.acquire_key()
            self._current_profile = profile_name

            provider = self._providers.get(profile_name)
            if provider is None:
                raise RuntimeError(
                    "No provider found for profile: {}".format(profile_name)
                )

            provider_name = profile_name.split("/", 1)[0]

            limiter = self._rate_limiters.get(provider_name)

            local_messages, local_estimated_tokens = self._apply_tpm_truncation(
                messages, tools, estimated_tokens, limiter, profile_name
            )

            reservation_id = 0
            if limiter is not None:
                reservation_id = await limiter.acquire(estimated_tokens=local_estimated_tokens)

            try:
                async for chunk in provider.stream_chat(
                    messages=local_messages,
                    tools=tools,
                    agent_label=agent_label,
                    task_id=task_id,
                    depth=depth,
                ):
                    yield chunk

                self._key_pool.report_success(profile_name)
                self._rotation_count = 0

                if limiter is not None:
                    usage = provider.get_last_usage()
                    if usage is not None:
                        prompt_tokens, completion_tokens = usage
                        total_tokens = prompt_tokens + completion_tokens
                        await limiter.record_usage(total_tokens, reservation_id=reservation_id)
                        self._token_estimator.feedback(estimated_tokens, total_tokens)

                self._logger.info(
                    agent_label,
                    "Fallback stream success | profile={}".format(profile_name),
                    task_id=task_id,
                    state="running",
                    event_type="fallback_stream_success",
                    data={"profile": profile_name},
                )

                return

            except asyncio.CancelledError:
                if limiter is not None and reservation_id > 0:
                    await limiter.release_reserved(reservation_id)
                raise

            except LLMProviderError as e:
                error_class = self._retry_engine.classify_error(e.original_error)

                if error_class == ErrorClass.NON_RETRYABLE:
                    self._logger.warning(
                        agent_label,
                        "Non-retryable stream error | profile={} error={}".format(
                            profile_name, str(e)[:200]
                        ),
                        task_id=task_id,
                        state="error",
                        event_type="fallback_stream_non_retryable",
                        data={
                            "profile": profile_name,
                            "error_type": type(e.original_error).__name__,
                            "error_message": str(e)[:500],
                        },
                    )
                    if limiter is not None and reservation_id > 0:
                        await limiter.release_reserved(reservation_id)
                    raise

                if error_class == ErrorClass.RETRYABLE:
                    if limiter is not None and reservation_id > 0:
                        await limiter.release_reserved(reservation_id)
                    raise

                if error_class == ErrorClass.RATE_LIMITED:
                    retry_after = self._retry_engine.extract_retry_after(e.original_error)
                    self._key_pool.report_rate_limited(profile_name, retry_after)
                    if limiter is not None and reservation_id > 0:
                        await limiter.release_reserved(reservation_id)
                    self._rotation_count += 1

                    self._logger.warning(
                        "RateControl",
                        "Rate limited on stream | profile={} rotation={}/{}".format(
                            profile_name,
                            self._rotation_count,
                            self._max_profile_rotations,
                        ),
                        event_type="fallback_stream_rate_limited",
                        data={
                            "profile": profile_name,
                            "rotation_count": self._rotation_count,
                        },
                    )

                    if self._rotation_count > self._max_profile_rotations:
                        self._rotation_count = 0

                    continue

        raise LLMProviderError(
            RuntimeError(
                "Fallback stream provider exceeded maximum iterations ({})".format(
                    max_iterations
                )
            )
        )

    def get_active_provider_info(self) -> Dict:
        return {
            "current_profile": self._current_profile,
            "pool_status": self._key_pool.get_cooldown_status(),
        }
