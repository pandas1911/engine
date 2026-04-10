"""LLM Provider implementations for the Agent system.

This module provides abstract and concrete LLM provider implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from src.config import Config
from src.models import LLMResponse, ToolCall


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def chat(
        self,
        messages: List[Dict],
        tools: List[Dict],
        agent_label: str = "Root",
        task_id: str = "unknown",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Send a chat request to the LLM.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            tools: List of tool definitions available to the LLM
            agent_label: Label for the agent making the request
            task_id: ID of the task being executed
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
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Send a chat request to the LLM.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            tools: List of tool definitions available to the LLM
            agent_label: Label for the agent making the request
            task_id: ID of the task being executed
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

            last_user_msg = ""
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if (
                        "[子代理" not in content
                        and "[System Instructions]" not in content
                    ):
                        last_user_msg = (
                            content[:50] + "..." if len(content) > 50 else content
                        )
                        break

            print(
                "[{agent_label}|{task_id}] → LLM ({msg_count} msgs)".format(
                    agent_label=agent_label, task_id=task_id, msg_count=len(messages)
                )
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

            display_id = "[{agent_label}|{task_id}]".format(
                agent_label=agent_label, task_id=task_id
            )

            if tool_calls:
                for tc in tool_calls:
                    task = tc.arguments.get("task", "")
                    label = tc.arguments.get("label", "sub")
                    task_preview = task[:50] + "..." if len(task) > 50 else task
                    print(
                        "{display_id} ← Tool: {tool_name}(task='{task_preview}', label='{label}')".format(
                            display_id=display_id,
                            tool_name=tc.name,
                            task_preview=task_preview,
                            label=label,
                        )
                    )
            elif content.strip():
                preview = content[:300].replace("\n", " ")
                if len(content) > 300:
                    preview += "..."
                print("{display_id} ← Response: {preview}".format(
                    display_id=display_id, preview=preview
                ))

            return LLMResponse(content=content, tool_calls=tool_calls)

        except Exception as e:
            print(
                "[{agent_label}|{task_id}] Error: {error_type}: {error}".format(
                    agent_label=agent_label,
                    task_id=task_id,
                    error_type=type(e).__name__,
                    error=e,
                )
            )
            raise

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


class MockLLMProvider(BaseLLMProvider):
    """Mock LLM provider for testing."""

    def __init__(self):
        self.call_count = 0

    async def chat(
        self,
        messages: List[Dict],
        tools: List[Dict],
        agent_label: str = "Root",
        task_id: str = "unknown",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Send a chat request (mock implementation).

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            tools: List of tool definitions available to the LLM
            agent_label: Label for the agent making the request
            task_id: ID of the task being executed
            temperature: Optional temperature parameter for the LLM
            max_tokens: Optional max_tokens parameter for the LLM

        Returns:
            LLMResponse with content or tool_calls
        """
        self.call_count += 1

        last_content = messages[-1].get("content", "") if messages else ""
        has_spawn_tool = any(
            t.get("function", {}).get("name") == "spawn" for t in tools
        )
        has_subagent_result = (
            "[子代理完成]" in last_content or "[子代理结果汇总]" in last_content
        )
        has_wake_notification = "[唤醒通知]" in last_content

        if self.call_count == 1 and has_spawn_tool:
            return LLMResponse(
                tool_calls=[
                    ToolCall(
                        name="spawn",
                        arguments={"task": "分析代码结构", "label": "analyzer"},
                        call_id="call_1",
                    )
                ]
            )

        if has_subagent_result or has_wake_notification:
            return LLMResponse(
                content="综合所有子代理的结果分析：代码结构清晰，建议增加单元测试。具体发现包括：1）模块划分合理 2）依赖关系清晰 3）命名规范统一。"
            )

        if self._is_subagent(messages):
            depth = self._get_depth(messages)
            return LLMResponse(
                content="[子Agent完成-深度{depth}] 分析完成：发现3个主要模块，依赖关系清晰。".format(
                    depth=depth
                )
            )

        return LLMResponse(content="处理完成")

    def _is_subagent(self, messages: List[Dict]) -> bool:
        """Check if messages indicate subagent context."""
        for msg in messages:
            if msg.get("role") == "system" and "Subagent Context" in msg.get(
                "content", ""
            ):
                return True
        return False

    def _get_depth(self, messages: List[Dict]) -> int:
        """Extract depth from system message."""
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if "Depth:" in content:
                    try:
                        return int(content.split("Depth:")[1].split("/")[0].strip())
                    except (ValueError, IndexError):
                        pass
        return 0

    async def stream_chat(
        self,
        messages: List[Dict],
        tools: List[Dict],
        agent_label: str = "Root",
        task_id: str = "unknown",
    ) -> None:
        """Stream a chat request (mock implementation).

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            tools: List of tool definitions available to the LLM
            agent_label: Label for the agent making the request
            task_id: ID of the task being executed

        Raises:
            NotImplementedError: Streaming is not yet implemented
        """
        raise NotImplementedError("Streaming not yet implemented")


__all__ = ["BaseLLMProvider", "LLMProvider", "MockLLMProvider"]
