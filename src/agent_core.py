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
from src.tools.base import ToolRegistry
from src.tools.builtin.spawn import SpawnTool
from src.llm_provider import MockLLMProvider

if TYPE_CHECKING:
    from src.llm_provider import LLMProvider


class Agent:
    """Core Agent class with multi-level nesting support."""

    MAX_TOOL_ITERATIONS = 20

    def __init__(
        self,
        session: Session,
        config: Config,
        registry: Optional[SubagentRegistry] = None,
        llm_provider: Optional["LLMProvider"] = None,
        tools: Optional[List[Any]] = None,
        tool_registry: Optional[ToolRegistry] = None,
        task_id: Optional[str] = None,
        parent_task_id: Optional[str] = None,
        label: Optional[str] = None,
    ):
        self.session = session
        self.config = config
        self.registry = registry or SubagentRegistry()
        self.llm = llm_provider or MockLLMProvider()
        self.task_id = task_id or f"task_{uuid.uuid4().hex[:8]}"
        self.parent_task_id = parent_task_id
        self.label = label or (
            "Root" if session.depth == 0 else f"Sub Depth:{session.depth}"
        )
        self.state = AgentState.IDLE
        self._final_result: Optional[str] = None
        self._completion_event = asyncio.Event()
        self.display_id = f"[{self.label}|{self.task_id}]"

        # Use provided registry or create new one
        self._tool_registry = tool_registry or ToolRegistry()

        # Backward compat: register any tools passed as list
        if tools:
            for tool in tools:
                self._tool_registry.register(tool)

        # SpawnTool conditional injection
        if session.depth < config.max_depth:
            self._tool_registry.register_spawn(
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
            tool_registry=self._tool_registry.clone(),
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
        iteration = 0

        while iteration < self.MAX_TOOL_ITERATIONS:
            iteration += 1

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

        if iteration >= self.MAX_TOOL_ITERATIONS:
            print(
                f"{self.display_id} ⚠ Tool call limit reached ({self.MAX_TOOL_ITERATIONS} iterations)"
            )
            self._final_result = self._final_result or "[达到工具调用上限]"

        return spawned

    async def _execute_tool(self, tool_call: ToolCall) -> str:
        """Execute a single tool call.

        Args:
            tool_call: The tool call to execute

        Returns:
            Tool execution result
        """
        tool = self._tool_registry.get(tool_call.name)

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
        print(f"{self.display_id} ← Subagent complete: {child_task_id}")

        # self.session.add_message(
        #     "user",
        #     f"[子代理完成 {child_task_id}] : {result}",
        #     is_subagent_result=True,
        #     child_task_id=child_task_id,
        # )

        pending_children = self.registry.count_pending_for_parent(self.task_id)

        if pending_children > 0:
            print(f"{self.display_id} → Waiting {pending_children} more subagents")
            return

        print(f"{self.display_id} → All subagents done, continuing")
        await self._continue_processing()

    async def _on_subagent_error(self, child_task_id: str, error_message: str):
        print(f"{self.display_id} ✗ Subagent error: {child_task_id}: {error_message}")

        self.session.add_message(
            "user", f"[子代理错误] {child_task_id}: {error_message}"
        )

        pending_children = self.registry.count_pending_for_parent(self.task_id)

        if pending_children > 0:
            print(
                f"{self.display_id} → Waiting {pending_children} more subagents (after error)"
            )
            return

        print(f"{self.display_id} → All subagents done (with errors), continuing")
        await self._continue_processing()

    async def _on_descendant_wake(self, descendant_task_id: str, result: str):
        print(f"{self.display_id} ← Woken by descendant: {descendant_task_id}")

        self.state = AgentState.RUNNING
        pending_children = self.registry.count_pending_for_parent(self.task_id)

        if pending_children > 0:
            print(
                f"{self.display_id} → Woken but {pending_children} subagents still pending"
            )
            return

        print(f"{self.display_id} ← All descendants complete, continuing")
        await self._continue_processing()

    async def _continue_processing(self):
        """Continue processing after all children complete."""
        self.state = AgentState.RUNNING

        child_results = self.registry.collect_child_results(self.task_id)

        if child_results:
            print(
                f"{self.display_id} ← Collected {len(child_results)} descendant results"
            )
            findings_prompt = (
                "以下是你派出的所有子代理的执行报告，每个子代理都独立完成了自己的任务。\n"
            )
            for task_id, result in child_results.items():
                task = self.registry.get_task(task_id)
                task_desc = task.task_description if task else "未知任务"
                findings_prompt += (
                    f"【{task_id}】\n任务: {task_desc}\n结果: {result}\n\n"
                )
        else:
            print(f"{self.display_id} ← No descendant results collected")
            findings_prompt = "子代理目前已经全部完成任务，但是并未获得任何结果。"

        self.session.add_message("system", findings_prompt)

        spawned_any = await self._process_tool_calls()

        if spawned_any:
            print(f"{self.display_id} → Re-waiting for new subagents")
            self.state = AgentState.CALLBACK_PENDING
            await self.registry.mark_ended_with_pending_descendants(self.task_id)
        else:
            await self._finish_and_notify()

    async def _finish_and_notify(self):
        result_preview = (
            (self._final_result[:100] + "...")
            if self._final_result and len(self._final_result) > 100
            else (self._final_result or "None")
        )
        print(f"{self.display_id} ✓ Done: {result_preview}")

        self.state = AgentState.COMPLETED
        self._completion_event.set()

        if self.parent_task_id:
            await self.registry.complete(
                self.task_id, self._final_result or "[完成但无输出]"
            )

    def _get_tool_schemas(self) -> List[Dict]:
        return self._tool_registry.get_schemas()


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
    "run_agent",
]
