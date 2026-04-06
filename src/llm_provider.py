"""LLM Provider implementations for the Agent system.

This module provides abstract and concrete LLM provider implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from openai import AsyncOpenAI

from src.config import Config
from src.models import LLMResponse, ToolCall


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def chat(
        self, messages: List[Dict], tools: List[Dict], **kwargs
    ) -> LLMResponse:
        """Send a chat request to the LLM.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            tools: List of tool definitions available to the LLM
            **kwargs: Additional parameters (e.g., temperature, max_tokens)

        Returns:
            LLMResponse with content or tool_calls
        """
        pass


class MiniMaxProvider(LLMProvider):
    """MiniMax LLM provider using OpenAI-compatible API."""

    def __init__(self, config: Config):
        """Initialize MiniMax provider.

        Args:
            config: Configuration containing API credentials
        """
        self.config = config
        self.client = AsyncOpenAI(
            api_key=config.openai_api_key, base_url=config.openai_base_url
        )
        self.model = config.openai_model

    async def chat(
        self,
        messages: List[Dict],
        tools: List[Dict],
        agent_label: str = "Root",
        task_id: str = "unknown",
        **kwargs,
    ) -> LLMResponse:
        try:
            converted_messages = self._convert_system_messages_for_minimax(messages)
            params: Dict[str, Any] = {
                "model": self.model,
                "messages": converted_messages,
            }
            if tools:
                params["tools"] = tools
            params.update(kwargs)

            last_user_msg = ""
            for msg in reversed(converted_messages):
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

            print(f"[{agent_label}|{task_id}] → LLM ({len(converted_messages)} msgs)")

            response = await self.client.chat.completions.create(**params)
            choice = response.choices[0]
            message = choice.message

            tool_calls = []
            if message.tool_calls:
                for tc in message.tool_calls:
                    import json

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

            display_id = f"[{agent_label}|{task_id}]"

            if tool_calls:
                for tc in tool_calls:
                    task = tc.arguments.get("task", "")
                    label = tc.arguments.get("label", "sub")
                    task_preview = task[:50] + "..." if len(task) > 50 else task
                    print(
                        f"{display_id} ← Tool: {tc.name}(task='{task_preview}', label='{label}')"
                    )
            elif content.strip():
                preview = content[:120].replace("\n", " ")
                if len(content) > 120:
                    preview += "..."
                print(f"{display_id} ← Response: {preview}")

            return LLMResponse(content=content, tool_calls=tool_calls)

        except Exception as e:
            print(f"[{agent_label}|{task_id}] Error: {type(e).__name__}: {e}")
            raise

    def _strip_thinking(self, content: str) -> str:
        if not content:
            return ""
        import re

        content = re.sub(
            r"<think[^>]*>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE
        )

        lines = content.split("\n")
        filtered_lines = []
        skip_patterns = [
            "the user is asking",
            "i should",
            "let me",
            "i need to",
            "i will",
            "i can",
        ]

        for line in lines:
            stripped = line.strip().lower()
            if any(stripped.startswith(p) for p in skip_patterns):
                continue
            if stripped and not stripped.startswith("`"):
                filtered_lines.append(line)

        return "\n".join(filtered_lines).strip()

    def _convert_system_messages_for_minimax(self, messages: List[Dict]) -> List[Dict]:
        """MiniMax API rejects 'system' role messages.

        Convert system messages by prepending their content to the first user
        message, or adding as a user message if no user messages exist.
        """
        non_system_messages = []
        system_instructions = []

        for msg in messages:
            if msg.get("role") == "system":
                system_instructions.append(msg.get("content", ""))
            else:
                non_system_messages.append(msg)

        if not system_instructions:
            return non_system_messages

        combined_system = "\n\n".join(system_instructions)

        if non_system_messages and non_system_messages[0].get("role") == "user":
            first_user_msg = non_system_messages[0]
            non_system_messages[0] = {
                **first_user_msg,
                "content": f"[System Instructions]\n{combined_system}\n\n[User Message]\n{first_user_msg.get('content', '')}",
            }
        else:
            non_system_messages.insert(0, {"role": "user", "content": combined_system})

        return non_system_messages


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing."""

    def __init__(self):
        self.call_count = 0

    async def chat(
        self, messages: List[Dict], tools: List[Dict], **kwargs
    ) -> LLMResponse:
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
                content=f"[子Agent完成-深度{depth}] 分析完成：发现3个主要模块，依赖关系清晰。"
            )

        return LLMResponse(content="处理完成")

    def _is_subagent(self, messages: List[Dict]) -> bool:
        for msg in messages:
            if msg.get("role") == "system" and "Subagent Context" in msg.get(
                "content", ""
            ):
                return True
        return False

    def _get_depth(self, messages: List[Dict]) -> int:
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if "Depth:" in content:
                    try:
                        return int(content.split("Depth:")[1].split("/")[0].strip())
                    except (ValueError, IndexError):
                        pass
        return 0


__all__ = ["LLMProvider", "MiniMaxProvider", "MockLLMProvider"]
