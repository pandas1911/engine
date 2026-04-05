"""Agent core implementation with multi-level nesting support.

This module provides the Agent class with async callbacks and error propagation.
"""

import asyncio
import json
import uuid
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from src.models import AgentState, LLMResponse, Session, ToolCall
from src.config import Config
from src.registry import SubagentRegistry
from src.tools import SpawnTool

if TYPE_CHECKING:
    from src.llm_provider import LLMProvider


class Agent:
    """Core Agent class with multi-level nesting support."""

    def __init__(
        self,
        session: Session,
        config: Config,
        registry: Optional[SubagentRegistry] = None,
        llm_provider: Optional["LLMProvider"] = None,
        tools: Optional[List[Any]] = None,
        task_id: Optional[str] = None,
        parent_task_id: Optional[str] = None,
        label: Optional[str] = None,
    ):
        self.session = session
        self.config = config
        self.registry = registry or SubagentRegistry()
        self.llm = llm_provider or MockLLMProvider()
        self.tools = tools or []
        self.task_id = task_id or f"task_{uuid.uuid4().hex[:8]}"
        self.parent_task_id = parent_task_id
        self.label = label or ("Root" if session.depth == 0 else f"Sub:{session.depth}")
        self.state = AgentState.IDLE
        self._final_result: Optional[str] = None
        self._completion_event = asyncio.Event()

        if session.depth < config.max_depth:
            self.tools.append(
                SpawnTool(
                    self._create_child_agent, self.registry, self.task_id, self.label
                )
            )

    def _create_child_agent(
        self,
        session: Session,
        config: Config,
        registry: SubagentRegistry,
        parent_task_id: str,
        task_id: Optional[str] = None,
    ) -> "Agent":
        """Create a child agent.

        Args:
            session: Child agent's session
            config: Configuration
            registry: Registry instance
            parent_task_id: Parent's task ID
            task_id: Optional task ID for the child (auto-generated if not provided)

        Returns:
            New Agent instance
        """
        return Agent(
            session,
            config,
            registry,
            self.llm,
            [],
            task_id=task_id,
            parent_task_id=parent_task_id,
        )

    async def run(self, message: Optional[str] = None) -> str:
        if message:
            self.session.add_message("user", message)

        self.state = AgentState.RUNNING
        print(f"[{self.label}|{self.task_id}] → Processing")

        spawned_any = await self._process_tool_calls()
        if spawned_any:
            print(f"[{self.label}|{self.task_id}] → Waiting for subagents")
            self.state = AgentState.CALLBACK_PENDING
            await self.registry.mark_ended_with_pending_descendants(self.task_id)
            return "[等待子代理回调...]"
        else:
            print(f"[{self.label}|{self.task_id}] ✓ Completed")
            await self._finish_and_notify()

        return self._final_result or "[无回复]"

    async def _process_tool_calls(self) -> bool:
        """Process tool calls from the LLM.

        Returns:
            True if any child agents were spawned
        """
        spawned = False

        while True:
            if self.llm is None:
                break

            response = await self.llm.chat(
                messages=self.session.get_messages(),
                tools=self._get_tool_schemas(),
                agent_label=self.label,
                task_id=self.task_id,
            )

            if not response.has_tool_calls():
                if response.content:
                    self.session.add_message("assistant", response.content)
                    if not spawned:
                        self._final_result = response.content
                break

            tool_calls_for_msg = [
                {
                    "id": tc.call_id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments)
                        if isinstance(tc.arguments, dict)
                        else tc.arguments,
                    },
                }
                for tc in response.tool_calls
            ]
            self.session.add_message(
                "assistant", response.content or "", tool_calls=tool_calls_for_msg
            )

            for tool_call in response.tool_calls:
                result = await self._execute_tool(tool_call)
                self.session.add_message("tool", result, tool_call_id=tool_call.call_id)

                if tool_call.name == "spawn":
                    spawned = True

        return spawned

    async def _execute_tool(self, tool_call: ToolCall) -> str:
        """Execute a single tool call.

        Args:
            tool_call: The tool call to execute

        Returns:
            Tool execution result
        """
        tool = next((t for t in self.tools if t.name == tool_call.name), None)

        if not tool:
            return f"[错误] 未知工具: {tool_call.name}"

        context = {
            "session": self.session,
            "config": self.config,
            "registry": self.registry,
            "parent_agent": self,
            "parent_task_id": self.task_id,
        }

        try:
            return await tool.execute(tool_call.arguments, context)
        except Exception as e:
            return f"[工具错误] {str(e)}"

    async def _on_subagent_complete(self, child_task_id: str, result: str):
        """Callback when a child agent completes successfully.

        Args:
            child_task_id: The completed child task ID
            result: The child's result
        """
        display_id = f"[{self.label}|{self.task_id}]"
        print(f"{display_id} ← Subagent complete: {child_task_id}")

        self.session.add_message(
            "user",
            f"[子代理{child_task_id}完成] : {result}",
            is_subagent_result=True,
            child_task_id=child_task_id,
        )

        pending_children = self.registry.count_pending_for_parent(self.task_id)

        if pending_children > 0:
            print(f"{display_id} → Waiting {pending_children} more subagents")
            return

        print(f"{display_id} → All subagents done, continuing")
        await self._continue_processing()

    async def _on_subagent_error(self, child_task_id: str, error_message: str):
        display_id = f"[{self.label}|{self.task_id}]"
        print(f"{display_id} ✗ Subagent error: {child_task_id}: {error_message}")

        self.session.add_message(
            "user", f"[子代理错误] {child_task_id}: {error_message}"
        )

        pending_children = self.registry.count_pending_for_parent(self.task_id)

        if pending_children > 0:
            print(
                f"{display_id} → Waiting {pending_children} more subagents (after error)"
            )
            return

        print(f"{display_id} → All subagents done (with errors), continuing")
        await self._continue_processing()

    async def _on_descendant_wake(self, descendant_task_id: str, result: str):
        display_id = f"[{self.label}|{self.task_id}]"
        print(f"{display_id} ← Woken by descendant: {descendant_task_id}")

        self.state = AgentState.RUNNING

        await self._check_all_descendants_complete()

    async def _check_all_descendants_complete(self):
        display_id = f"[{self.label}|{self.task_id}]"
        pending_children = self.registry.count_pending_for_parent(self.task_id)

        if pending_children > 0:
            print(
                f"{display_id} → Woken but {pending_children} subagents still pending"
            )
            return

        child_results = self.registry.collect_child_results(self.task_id)

        if not child_results:
            await self._finish_and_notify()
            return

        print(f"{display_id} ← Collected {len(child_results)} child results")

        findings_prompt = (
            "子代理目前已经全部完成并返回结果，基于以下子代理结果给出综合回复:\n"
        )
        for task_id, result in child_results.items():
            findings_prompt += f"\n[{task_id}]\n{result}\n"

        self.session.add_message("system", findings_prompt)

        await self._continue_processing()

    async def _continue_processing(self):
        """Continue processing after all children complete."""
        self.state = AgentState.RUNNING

        child_results = self.registry.collect_child_results(self.task_id)

        if child_results:
            findings_prompt = (
                "子代理目前已经全部完成并返回结果，基于以下子代理结果给出综合回复:\n"
            )
            for task_id, result in child_results.items():
                findings_prompt += f"\n[{task_id}]\n{result}\n"

            self.session.add_message("system", findings_prompt)

        if self.llm:
            final_response = await self.llm.chat(
                messages=self.session.get_messages(),
                tools=[],
                agent_label=self.label,
                task_id=self.task_id,
            )

            if final_response.content:
                self.session.add_message("assistant", final_response.content)
                self._final_result = final_response.content

        await self._finish_and_notify()

    async def _finish_and_notify(self):
        display_id = f"[{self.label}|{self.task_id}]"
        result_preview = (
            (self._final_result[:100] + "...")
            if self._final_result and len(self._final_result) > 100
            else (self._final_result or "None")
        )
        print(f"{display_id} ✓ Done: {result_preview}")

        self.state = AgentState.COMPLETED
        self._completion_event.set()

        if self.parent_task_id:
            await self.registry.complete(
                self.task_id, self._final_result or "[完成但无输出]"
            )

    def _get_tool_schemas(self) -> List[Dict]:
        """Get tool schemas for LLM.

        Returns:
            List of tool schema dictionaries
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self.tools
        ]


class MockLLMProvider:
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


async def run_agent(task: str, config: Optional[Config] = None) -> tuple[str, Session]:
    """Run root Agent."""
    from src.config import Config as ConfigCls

    cfg = config or ConfigCls()

    root_session = Session(id=f"root_{uuid.uuid4().hex[:8]}", depth=0)
    root_session.add_message("system", "你是主Agent，可以派生子Agent并行处理任务。")

    agent = Agent(root_session, cfg)
    result = await agent.run(task)

    return result, root_session


__all__ = [
    "Agent",
    "MockLLMProvider",
    "run_agent",
]
