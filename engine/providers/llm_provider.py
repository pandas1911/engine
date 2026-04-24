"""LLM Provider implementations for the Agent system.

This module provides abstract and concrete LLM provider implementations.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from engine.config import Config
from engine.logging import get_logger
from engine.safety import RetryEngine
from .provider_models import LLMResponse, ToolCall, ErrorClass


class LLMProviderError(Exception):
    """Unified LLM provider exception wrapper.

    Regardless of whether the underlying SDK is openai, anthropic, or custom,
    all LLM call exceptions are wrapped as LLMProviderError.
    """
    def __init__(self, original_error: Exception):
        self.original_error = original_error
        super().__init__(str(original_error))


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
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
        """Send a chat request to the LLM.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            tools: List of tool definitions available to the LLM
            agent_label: Label for the agent making the request
            task_id: ID of the task being executed
            depth: Nesting depth of the calling agent
            temperature: Optional temperature parameter for the LLM
            max_tokens: Optional max_tokens parameter for the LLM

        Returns:
            LLMResponse with content or tool_calls
        """
        pass

    @abstractmethod
    async def stream_chat(
        self,
        messages: List[Dict],
        tools: List[Dict],
        agent_label: str = "Root",
        task_id: str = "unknown",
    ) -> None:
        """Stream a chat request to the LLM.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            tools: List of tool definitions available to the LLM
            agent_label: Label for the agent making the request
            task_id: ID of the task being executed

        Raises:
            NotImplementedError: Streaming is not yet implemented
        """
        pass


class LLMProvider(BaseLLMProvider):
    """Universal LLM provider using OpenAI-compatible API."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        config: Config,
        retry_engine: Optional[RetryEngine] = None,
    ):
        """Initialize LLM provider.

        Args:
            api_key: API key for the LLM service
            base_url: Base URL for the API endpoint
            model: Model name to use
            config: Configuration for global settings (strip_thinking, retry, etc.)
            retry_engine: Optional RetryEngine. When None, created from config.
        """
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=0,  # Disable SDK built-in retry — we handle retry ourselves
        )
        self.model = model
        self.strip_thinking = config.strip_thinking
        self._retry_engine = retry_engine or RetryEngine(
            max_attempts=config.llm_retry_max_attempts,
            base_delay=config.llm_retry_base_delay,
        )
        self._last_snapshot = None
        self._last_usage = None

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
        """Send a chat request to the LLM with rate limiting and retry."""
        params: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if tools:
            params["tools"] = tools
        if temperature is not None:
            params["temperature"] = temperature
        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        logger = get_logger()
        msg_roles = [m.get("role", "?") for m in messages]
        logger.info(
            agent_label,
            "Sending LLM API request | model={}, message_count={}, has_tools={}, tool_count={}".format(
                self.model, len(messages), bool(tools), len(tools) if tools else 0
            ),
            task_id=task_id, state="running", depth=depth,
            event_type="llm_api_request",
            data={"model": self.model, "message_count": len(messages),
                  "message_roles": msg_roles, "has_tools": bool(tools),
                  "tool_count": len(tools) if tools else 0}
        )

        last_error: Optional[Exception] = None

        for attempt in range(1, self._retry_engine.max_attempts + 1):
            try:
                response = await self.client.chat.completions.create(**params)

                self._extract_rate_limit_headers(response)
                self._extract_usage(response)

                # --- Success path ---
                choice = response.choices[0]
                message = choice.message

                tool_calls = []
                if message.tool_calls:
                    import json
                    for tc in message.tool_calls:
                        args = tc.function.arguments
                        if isinstance(args, str):
                            args = json.loads(args)
                        tool_calls.append(
                            ToolCall(
                                name=tc.function.name,
                                arguments=args,
                                call_id=tc.id,
                            )
                        )

                content = message.content or ""
                content = self._strip_thinking(content)

                if tool_calls:
                    for tc in tool_calls:
                        logger.tool(
                            agent_label,
                            "LLM returned tool call | tool=\"{}\", call_id={}".format(tc.name, tc.call_id),
                            task_id=task_id, state="running", depth=depth,
                            tool_name=tc.name,
                            data={"call_id": tc.call_id, "arguments": tc.arguments}
                        )
                elif content.strip():
                    logger.info(
                        agent_label,
                        "LLM returned text response | content_length={}, thinking_stripped={}".format(
                            len(content), self.strip_thinking
                        ),
                        task_id=task_id, state="running", depth=depth,
                        event_type="llm_text_response",
                        data={"content_length": len(content),
                              "thinking_stripped": self.strip_thinking,
                              "content": content}
                    )

                return LLMResponse(content=content, tool_calls=tool_calls)

            except Exception as e:
                last_error = e

                error_class = self._retry_engine.classify_error(e)

                if error_class == ErrorClass.NON_RETRYABLE:
                    raise LLMProviderError(e) from e

                if attempt < self._retry_engine.max_attempts:
                    retry_after = self._retry_engine.extract_retry_after(e)
                    delay = self._retry_engine.compute_delay(attempt, retry_after)

                    logger.warning(
                        agent_label,
                        "LLM API call failed, retrying | attempt={}/{}, wait={:.2f}s, error_type={}, error=\"{}\"".format(
                            attempt, self._retry_engine.max_attempts, delay,
                            type(e).__name__, str(e)[:200]
                        ),
                        task_id=task_id, state="running", depth=depth,
                        event_type="llm_retry_attempt",
                        data={
                            "attempt": attempt,
                            "max_attempts": self._retry_engine.max_attempts,
                            "wait_seconds": delay,
                            "error_type": type(e).__name__,
                            "error_message": str(e)[:500],
                        }
                    )

                    await asyncio.sleep(delay)
                    continue

        # All retries exhausted
        logger.error(
            agent_label,
            "LLM API call failed after {} attempts | error_type={}, error=\"{}\"".format(
                self._retry_engine.max_attempts, type(last_error).__name__, str(last_error)
            ),
            task_id=task_id, state="error", depth=depth,
            event_type="llm_api_error",
            data={
                "error_type": type(last_error).__name__,
                "error_message": str(last_error),
                "model": self.model,
                "attempts": self._retry_engine.max_attempts,
            }
        )
        raise LLMProviderError(last_error) from last_error

    def _extract_rate_limit_headers(self, response) -> None:
        """Extract rate limit info from response headers (best-effort, provider-agnostic)."""
        try:
            headers = {}
            if hasattr(response, 'headers'):
                headers = response.headers
            elif hasattr(response, 'raw_response') and hasattr(response.raw_response, 'headers'):
                headers = response.raw_response.headers

            if not headers:
                self._last_snapshot = None
                return

            def _safe_int(val, default=None):
                try:
                    return int(val)
                except (TypeError, ValueError):
                    return default

            remaining_rpm = _safe_int(headers.get('x-ratelimit-remaining-requests'))
            remaining_tpm = _safe_int(headers.get('x-ratelimit-remaining-tokens'))
            limit_rpm = _safe_int(headers.get('x-ratelimit-limit-requests'))
            limit_tpm = _safe_int(headers.get('x-ratelimit-limit-tokens'))

            if remaining_rpm is None:
                remaining_rpm = _safe_int(headers.get('ratelimit-remaining'))
            if limit_rpm is None:
                limit_rpm = _safe_int(headers.get('ratelimit-limit'))

            if remaining_rpm is not None or remaining_tpm is not None:
                from engine.providers.provider_models import RateLimitSnapshot
                self._last_snapshot = RateLimitSnapshot(
                    remaining_rpm=remaining_rpm,
                    remaining_tpm=remaining_tpm,
                    limit_rpm=limit_rpm,
                    limit_tpm=limit_tpm,
                )
            else:
                self._last_snapshot = None
        except Exception:
            self._last_snapshot = None

    def _extract_usage(self, response) -> None:
        """Extract token usage from response."""
        try:
            usage = getattr(response, 'usage', None)
            if usage:
                self._last_usage = (
                    getattr(usage, 'prompt_tokens', 0) or 0,
                    getattr(usage, 'completion_tokens', 0) or 0,
                )
        except Exception:
            self._last_usage = None

    def get_rate_limit_snapshot(self):
        """Return last known rate limit snapshot from response headers."""
        return self._last_snapshot

    def get_last_usage(self):
        """Return (prompt_tokens, completion_tokens) from last response."""
        return self._last_usage

    def _strip_thinking(self, content: Optional[str]) -> str:
        """Remove thinking tags from content.

        Args:
            content: Content string that may contain thinking tags

        Returns:
            Content with thinking tags removed
        """
        if not content:
            return ""
        if not self.strip_thinking:
            return content
        import re

        content = re.sub(
            r"<think[^>]*>.*?</think\s*>", "", content, flags=re.DOTALL | re.IGNORECASE
        )
        return content.strip()

    async def stream_chat(
        self,
        messages: List[Dict],
        tools: List[Dict],
        agent_label: str = "Root",
        task_id: str = "unknown",
    ) -> None:
        """Stream a chat request to the LLM.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            tools: List of tool definitions available to the LLM
            agent_label: Label for the agent making the request
            task_id: ID of the task being executed

        Raises:
            NotImplementedError: Streaming is not yet implemented
        """
        raise NotImplementedError("Streaming not yet implemented")


__all__ = ["BaseLLMProvider", "LLMProvider", "LLMProviderError"]
