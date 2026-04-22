"""LLM Provider implementations for the Agent system.

This module provides abstract and concrete LLM provider implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from engine.config import Config
from engine.logging import get_logger
from .provider_models import LLMResponse, ToolCall


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

    def __init__(self, config: Config):
        """Initialize LLM provider.

        Args:
            config: Configuration containing API credentials
        """
        self.client = AsyncOpenAI(
            api_key=config.api_key, base_url=config.base_url
        )
        self.model = config.model
        self.strip_thinking = config.strip_thinking

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
        try:
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
                data={"model": self.model, "message_count": len(messages), "message_roles": msg_roles, "has_tools": bool(tools), "tool_count": len(tools) if tools else 0}
            )

            response = await self.client.chat.completions.create(**params)
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
                logger = get_logger()
                for tc in tool_calls:
                    logger.tool(
                        agent_label,
                        "LLM returned tool call | tool=\"{}\", call_id={}".format(tc.name, tc.call_id),
                        task_id=task_id, state="running", depth=depth,
                        tool_name=tc.name,
                        data={"call_id": tc.call_id, "arguments": tc.arguments}
                    )
            elif content.strip():
                logger = get_logger()
                logger.info(
                    agent_label,
                    "LLM returned text response | content_length={}, thinking_stripped={}".format(
                        len(content), self.strip_thinking
                    ),
                    task_id=task_id, state="running", depth=depth,
                    event_type="llm_text_response",
                    data={"content_length": len(content), "thinking_stripped": self.strip_thinking, "content": content}
                )

            return LLMResponse(content=content, tool_calls=tool_calls)

        except Exception as e:
            logger = get_logger()
            logger.error(
                agent_label,
                "LLM API call failed | error_type={}, error=\"{}\", model={}".format(
                    type(e).__name__, str(e), self.model
                ),
                task_id=task_id, state="error", depth=depth,
                event_type="llm_api_error",
                data={"error_type": type(e).__name__, "error_message": str(e), "model": self.model, "message_count": len(messages)}
            )
            raise LLMProviderError(e) from e

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
