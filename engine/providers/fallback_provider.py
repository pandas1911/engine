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
        """Initialize the fallback provider.

        Args:
            providers: Mapping from profile_name to LLMProvider instance.
            key_pool: APIKeyPool managing multiple API keys and cooldowns.
            rate_limiters: Per-profile sliding window rate limiters.
            pacers: Per-profile adaptive pacers for request throttling.
            retry_engine: RetryEngine for error classification.
            max_profile_rotations: Maximum key rotations before provider fallback.
        """
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
        """Estimate token count from messages and tools.

        Uses a simple heuristic of characters divided by 3.

        Args:
            messages: List of message dictionaries.
            tools: Optional list of tool definitions.

        Returns:
            Estimated token count (at least 1).
        """
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
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Send a chat request with key rotation and provider fallback.

        Attempts to call the current provider. On rate limit, rotates to
        the next available key. When all keys are exhausted, falls back
        to other providers (ping-pong behavior).

        Args:
            messages: List of message dictionaries with 'role' and 'content'.
            tools: List of tool definitions available to the LLM.
            agent_label: Label for the agent making the request.
            task_id: ID of the task being executed.
            depth: Nesting depth of the calling agent.
            temperature: Optional temperature parameter for the LLM.
            max_tokens: Optional max_tokens parameter for the LLM.

        Returns:
            LLMResponse with content or tool_calls.

        Raises:
            LLMProviderError: On non-retryable errors or when all providers fail.
        """
        estimated_tokens = self._estimate_tokens(messages, tools)
        max_iterations = max(
            50,
            len(self._providers) * (self._max_profile_rotations + 1) * 2,
        )

        for iteration in range(max_iterations):
            # 1. Get active profile from key pool
            profile = self._key_pool.acquire_key()
            profile_name = profile.name
            self._current_profile = profile_name

            provider = self._providers.get(profile_name)
            if provider is None:
                raise RuntimeError(
                    "No provider found for profile: {}".format(profile_name)
                )

            # 2. Apply rate limiting if configured for this profile
            limiter = self._rate_limiters.get(profile_name)
            if limiter is not None:
                await limiter.acquire(estimated_tokens=estimated_tokens)

            # 3. Apply pacing if configured for this profile
            pacer = self._pacers.get(profile_name)
            if pacer is not None:
                await pacer.wait_if_needed()

            try:
                # 4. Execute the chat request
                result = await provider.chat(
                    messages=messages,
                    tools=tools,
                    agent_label=agent_label,
                    task_id=task_id,
                    depth=depth,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                # 5. Success path
                self._key_pool.report_success(profile_name)
                self._rotation_count = 0

                # Record usage if limiter and usage data are available
                if limiter is not None:
                    usage = provider.get_last_usage()
                    if usage is not None:
                        prompt_tokens, completion_tokens = usage
                        total_tokens = prompt_tokens + completion_tokens
                        limiter.record_usage(total_tokens)

                # Update pacer with rate limit snapshot from provider
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

            except Exception as e:
                error_class = self._retry_engine.classify_error(e)

                if error_class == ErrorClass.NON_RETRYABLE:
                    # 7. Non-retryable errors raise immediately
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
                    raise LLMProviderError(e) from e

                if error_class == ErrorClass.RETRYABLE:
                    # 8. Retryable (non-rate-limit) errors raise for caller to handle
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
                    raise

                if error_class == ErrorClass.RATE_LIMITED:
                    # 6. Rate limited: report and rotate
                    retry_after = self._retry_engine.extract_retry_after(e)
                    self._key_pool.report_rate_limited(profile_name, retry_after)
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

                    # Check if we should do provider fallback (ping-pong)
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

        # 9. Safety net: infinite loop protection
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
        """Stream a chat request to the active provider.

        Delegates to the current provider's stream_chat method.

        Args:
            messages: List of message dictionaries with 'role' and 'content'.
            tools: List of tool definitions available to the LLM.
            agent_label: Label for the agent making the request.
            task_id: ID of the task being executed.
        """
        profile = self._key_pool.acquire_key()
        profile_name = profile.name
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
        """Return information about the current provider and pool status.

        Returns:
            Dictionary containing current profile name and pool cooldown status.
        """
        return {
            "current_profile": self._current_profile,
            "pool_status": self._key_pool.get_cooldown_status(),
        }
