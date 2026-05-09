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
                # [==================== LOG: control ====================]
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
                # [==================== END LOG ============================]
            local_messages = truncation.messages
            local_estimated_tokens = truncation.truncated_tokens
        return local_messages, local_estimated_tokens

    async def _release_reservation(self, limiter, reservation_id: int) -> None:
        if limiter is not None and reservation_id > 0:
            await limiter.release_reserved(reservation_id)

    async def _record_success(
        self,
        profile_name: str,
        provider,
        limiter,
        reservation_id: int,
        estimated_tokens: int,
        agent_label: str,
        task_id: str,
        depth: int,
        mode: str,
    ) -> None:
        self._key_pool.report_success(profile_name)
        self._rotation_count = 0

        if limiter is not None:
            usage = provider.get_last_usage()
            if usage is not None:
                prompt_tokens, completion_tokens = usage
                total_tokens = prompt_tokens + completion_tokens
                await limiter.record_usage(total_tokens, reservation_id=reservation_id)
                self._token_estimator.feedback(estimated_tokens, total_tokens)

        is_chat = mode == "chat"
        log_kwargs = dict(
            task_id=task_id,
            state="running",
            event_type="fallback_provider_success" if is_chat else "fallback_stream_success",
            data={"profile": profile_name},
        )
        if is_chat:
            log_kwargs["depth"] = depth
        # [==================== LOG: lifecycle ==================]
        self._logger.info(
            agent_label,
            "Fallback {} success | profile={}".format(
                "provider" if is_chat else "stream", profile_name
            ),
            **log_kwargs,
        )
        # [==================== END LOG ============================]

    async def _handle_provider_error(
        self,
        error: Exception,
        original_error: Exception,
        limiter,
        reservation_id: int,
        profile_name: str,
        agent_label: str,
        task_id: str,
        depth: int,
        mode: str,
    ) -> str:
        """Classify and handle a provider error.

        Returns an action for the caller: ``"continue"`` to retry with the
        next key, ``"raise"`` to re-raise, or ``"raise_wrapped"`` to wrap
        in ``LLMProviderError`` first (chat mode, non-retryable only).
        """
        error_class = self._retry_engine.classify_error(original_error)
        is_chat = mode == "chat"

        if error_class == ErrorClass.NON_RETRYABLE:
            kwargs = dict(
                task_id=task_id,
                state="error",
                event_type="fallback_non_retryable" if is_chat else "fallback_stream_non_retryable",
                data={
                    "profile": profile_name,
                    "error_type": type(original_error).__name__,
                    "error_message": str(error)[:500],
                },
            )
            if is_chat:
                kwargs["depth"] = depth
            # [==================== LOG: error ======================]
            self._logger.warning(
                agent_label,
                "Non-retryable {} | profile={} error={}".format(
                    "error from provider" if is_chat else "stream error",
                    profile_name, str(error)[:200],
                ),
                **kwargs,
            )
            # [==================== END LOG ============================]
            await self._release_reservation(limiter, reservation_id)
            return "raise_wrapped" if is_chat else "raise"

        if error_class == ErrorClass.RETRYABLE:
            if is_chat:
                # [==================== LOG: error ======================]
                self._logger.warning(
                    agent_label,
                    "Retryable error from provider | profile={} error={}".format(
                        profile_name, str(error)[:200]
                    ),
                    task_id=task_id,
                    state="running",
                    depth=depth,
                    event_type="fallback_retryable",
                    data={
                        "profile": profile_name,
                        "error_type": type(error).__name__,
                        "error_message": str(error)[:500],
                    },
                )
                # [==================== END LOG ============================]
            await self._release_reservation(limiter, reservation_id)
            return "raise"

        if error_class == ErrorClass.RATE_LIMITED:
            retry_after = self._retry_engine.extract_retry_after(original_error)
            self._key_pool.report_rate_limited(profile_name, retry_after)
            await self._release_reservation(limiter, reservation_id)
            self._rotation_count += 1

            rl_data = {"profile": profile_name, "rotation_count": self._rotation_count}
            if is_chat:
                rl_data["max_rotations"] = self._max_profile_rotations
                rl_data["retry_after_ms"] = retry_after
            # [==================== LOG: error ======================]
            self._logger.warning(
                "RateControl",
                "Rate limited on {} | profile={} rotation={}/{}".format(
                    "profile" if is_chat else "stream",
                    profile_name, self._rotation_count, self._max_profile_rotations,
                ),
                event_type="fallback_rate_limited" if is_chat else "fallback_stream_rate_limited",
                data=rl_data,
            )
            # [==================== END LOG ============================]

            if self._rotation_count > self._max_profile_rotations:
                self._rotation_count = 0
                if is_chat:
                    # [==================== LOG: control ====================]
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
                    # [==================== END LOG ============================]

            return "continue"

        return "raise"

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

                await self._record_success(
                    profile_name, provider, limiter, reservation_id,
                    estimated_tokens, agent_label, task_id, depth, "chat",
                )
                return result

            except asyncio.CancelledError:
                await self._release_reservation(limiter, reservation_id)
                raise

            except Exception as e:
                action = await self._handle_provider_error(
                    e, e, limiter, reservation_id,
                    profile_name, agent_label, task_id, depth, "chat",
                )
                if action == "continue":
                    continue
                if action == "raise_wrapped":
                    raise LLMProviderError(e) from e
                raise

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

                await self._record_success(
                    profile_name, provider, limiter, reservation_id,
                    estimated_tokens, agent_label, task_id, depth, "stream",
                )
                return

            except asyncio.CancelledError:
                await self._release_reservation(limiter, reservation_id)
                raise

            except LLMProviderError as e:
                action = await self._handle_provider_error(
                    e, e.original_error, limiter, reservation_id,
                    profile_name, agent_label, task_id, depth, "stream",
                )
                if action == "continue":
                    continue
                raise

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
