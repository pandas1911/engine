"""Fallback LLM Provider with key rotation and provider fallback (ping-pong).

This module provides a FallbackLLMProvider that wraps multiple LLMProvider
instances, automatically rotating between API keys on rate limit errors and
falling back between providers when all keys are exhausted.
"""

import asyncio
from typing import Dict, List, Optional

from engine.providers.llm_provider import BaseLLMProvider, LLMProvider, LLMProviderError
from engine.providers.provider_models import LLMResponse, ErrorClass
from engine.safety import APIKeyPool, SlidingWindowRateLimiter, AdaptivePacer, RetryEngine
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
        pacers: Dict[str, AdaptivePacer],
        retry_engine: RetryEngine,
        max_profile_rotations: int = 3,
    ):
        self._providers = providers
        self._key_pool = key_pool
        self._rate_limiters = rate_limiters
        self._pacers = pacers
        self._retry_engine = retry_engine
        self._max_profile_rotations = max_profile_rotations
        self._current_profile: Optional[str] = None
        self._rotation_count = 0
        self._logger = get_logger()

    @staticmethod
    def _estimate_tokens(messages: List[Dict], tools: Optional[List[Dict]]) -> int:
        total_chars = sum(len(str(m)) for m in messages)
        total_chars += sum(len(str(t)) for t in (tools or []))
        return max(1, total_chars // 3)

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

            # Extract provider name from composite key for limiter/pacer lookup.
            # Rate limiters and pacers are keyed by provider name (e.g., "aliyun"),
            # while profile_name is a composite key (e.g., "aliyun/deepseek-v4-pro").
            provider_name = profile_name.split("/", 1)[0]

            limiter = self._rate_limiters.get(provider_name)
            reservation_id = 0
            if limiter is not None:
                reservation_id = await limiter.acquire(estimated_tokens=estimated_tokens)

            pacer = self._pacers.get(provider_name)
            if pacer is not None:
                await pacer.wait_if_needed()

            try:
                result = await provider.chat(
                    messages=messages,
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

                if pacer is not None:
                    snapshot = provider.get_rate_limit_snapshot()
                    if snapshot is not None:
                        pacer.update_from_snapshot(snapshot)

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
    ) -> None:
        profile_name = self._key_pool.acquire_key()
        self._current_profile = profile_name

        provider = self._providers.get(profile_name)
        if provider is None:
            raise RuntimeError(
                "No provider found for profile: {}".format(profile_name)
            )

        await provider.stream_chat(
            messages=messages,
            tools=tools,
            agent_label=agent_label,
            task_id=task_id,
        )

    def get_active_provider_info(self) -> Dict:
        return {
            "current_profile": self._current_profile,
            "pool_status": self._key_pool.get_cooldown_status(),
        }
