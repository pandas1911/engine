"""Agent core implementation with multi-level nesting support.

This module provides the Agent class with async callbacks and error propagation.
"""

import asyncio
import json
import uuid
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from src.models import AgentState, LLMResponse, QueueEvent, Session, ToolCall
from src.config import Config
from src.registry import SubagentRegistry
from src.state_machine import AgentStateMachine
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
        self.state_machine = AgentStateMachine(AgentState.IDLE)
        self._final_result: Optional[str] = None
        self._completion_event = asyncio.Event()
        self.display_id = f"[{self.label}|{self.task_id}]"
        self._event_queue: List[
            QueueEvent
        ] = []  # Deferred event queue (native list, Swift Array equivalent)
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

    @property
    def state(self) -> AgentState:
        return self.state_machine.current_state

    def pop_event(self) -> Optional[QueueEvent]:
        """Pop the next event from the queue, or None if empty."""
        if self._event_queue:
            return self._event_queue.pop(0)
        return None

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

        self.state_machine.trigger("start")
        print(f"[{self.label}|{self.task_id}] → Processing")

        spawned_any = await self._process_tool_calls()

        # Drain deferred events before branching decision
        await self._drain_events()

        result = await self._after_drain(spawned_any)
        if result is not None:
            return result

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

    async def _resume_from_children(self, child_results: Dict[str, str]):
        """Continue processing after all children complete."""
        if self.state_machine.current_state == AgentState.WAITING_FOR_CHILDREN:
            self.state_machine.trigger("children_settled")

        # child_results is now passed as parameter (aggregated in registry.complete())

        if child_results:
            print(
                f"{self.display_id} ← Collected {len(child_results)} descendant results"
            )
            findings_prompt = "以下是你派出的所有子代理的执行报告，每个子代理都独立完成了自己的任务。\n"
            for task_id, result in child_results.items():
                task = self.registry.get_task(task_id)
                task_desc = task.task_description if task else "未知任务"
                findings_prompt += (
                    f"【{task_id}】\n任务: {task_desc}\n结果: {result}\n\n"
                )
        else:
            print(f"{self.display_id} ← No descendant results collected")
            findings_prompt = "子代理目前已经全部完成任务，但是并未获得任何结果。"

        self.session.add_message("user", findings_prompt)

        spawned_any = await self._process_tool_calls()

        # Drain deferred events
        await self._drain_events()

        await self._after_drain(spawned_any)

    async def _drain_events(self):
        """Process queued events after tool loop completes."""
        if self.state_machine.current_state == AgentState.COMPLETED:
            return

        event = self.pop_event()
        if event is None:
            return

        await self._resume_from_children(event.child_results)

    async def _after_drain(self, spawned_any: bool) -> Optional[str]:
        """Branch after _drain_events(). State machine is source of truth.

        _drain_events() may recursively call _resume_from_children(), which
        can spawn children or finish on its own. So after drain returns, the
        local spawned_any may be stale — check state machine first.

        Returns str if fully handled (caller should return it), None otherwise.
        """
        state = self.state_machine.current_state

        # Drain already triggered full completion chain
        if state == AgentState.COMPLETED:
            return self._final_result or "[无回复]"

        # Drain's inner _resume_from_children already spawned and transitioned
        if state == AgentState.WAITING_FOR_CHILDREN:
            return "[等待子代理回调...]"

        # State is still RUNNING — drain did nothing, trust local spawned_any
        if spawned_any:
            print(f"{self.display_id} → Waiting for subagents")
            self.state_machine.trigger("spawn_children")
            return "[等待子代理回调...]"
        else:
            print(f"{self.display_id} ✓ Completed")
            await self._finish_and_notify()
            return None

    async def _finish_and_notify(self):
        result_preview = (
            (self._final_result[:100] + "...")
            if self._final_result and len(self._final_result) > 100
            else (self._final_result or "None")
        )
        print(f"{self.display_id} ✓ Done: {result_preview}")

        self.state_machine.trigger("finish")
        self._completion_event.set()

        if self.parent_task_id:
            await self.registry.complete(
                self.task_id, self._final_result or "[完成但无输出]"
            )

    def _get_tool_schemas(self) -> List[Dict]:
        return self._tool_registry.get_schemas()


__all__ = [
    "Agent",
]
