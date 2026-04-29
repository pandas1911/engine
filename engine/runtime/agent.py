"""Agent implementation with multi-level nesting support.

This module provides the Agent class with async callbacks and error propagation.
"""

import asyncio
import json
import uuid
from typing import Dict, List, Optional, TYPE_CHECKING

from .agent_models import AgentState, ErrorCategory, AgentError, Session
from engine.providers.provider_models import ToolCall
from engine.config import Config
from engine.time import TimeProvider
from .task_registry import AgentTaskRegistry
from engine.subagent.events import AgentEvent, ChildCompletionEvent
from .state import AgentStateMachine
from engine.tools.pack import ToolPack
from engine.providers.llm_provider import LLMProviderError
from engine.logging import get_logger
from engine.safety import LaneConcurrencyQueue

if TYPE_CHECKING:
    from engine.providers.llm_provider import LLMProvider


class Agent:
    """Core Agent class with multi-level nesting support.

    # Implements: Drainable protocol (engine.subagent.protocol)
    """

    MAX_TOOL_ITERATIONS = 15

    def __init__(
        self,
        session: Session,
        config: Config,
        llm_provider: "LLMProvider",
        task_registry: Optional[AgentTaskRegistry] = None,
        tool_pack: Optional[ToolPack] = None,
        task_id: Optional[str] = None,
        parent_task_id: Optional[str] = None,
        label: Optional[str] = None,
        lane_queue: Optional[LaneConcurrencyQueue] = None,
    ):
        self.session = session
        self.config = config
        self.task_registry = task_registry or AgentTaskRegistry()
        self.llm = llm_provider
        self.task_id = task_id or f"task_{uuid.uuid4().hex[:8]}"
        self.parent_task_id = parent_task_id
        self.label = label or (
            "Root" if session.depth == 0 else f"Sub Depth:{session.depth}"
        )
        self.state_machine = AgentStateMachine(AgentState.IDLE)
        self._final_result: Optional[str] = None
        self._completion_event = asyncio.Event()
        self._error_info: Optional[AgentError] = None
        self.display_id = f"[{self.label}|{self.task_id}]"
        self._lane_queue = lane_queue
        self._time_provider = TimeProvider(timezone_override=config.user_timezone)
        self._event_queue: List[
            AgentEvent
        ] = []  # Deferred event queue (native list, Swift Array equivalent)
        self._tool_pack = tool_pack or ToolPack([])

        get_logger().info(
            self.label,
            "Agent instance created | session_id={}, depth={}, parent_task_id={}, spawn_enabled={}, tool_count={}, max_iterations={}".format(
                self.session.id, self.session.depth,
                self.parent_task_id or "None (root)",
                self.session.depth < self.config.max_depth,
                len(self._tool_pack), self.MAX_TOOL_ITERATIONS,
            ),
            task_id=self.task_id, state=self.state.value, depth=self.session.depth,
            event_type="agent_init",
            data={
                "session_id": self.session.id,
                "parent_task_id": self.parent_task_id,
                "spawn_enabled": self.session.depth < self.config.max_depth,
                "tool_count": len(self._tool_pack),
            },
        )

    @property
    def state(self) -> AgentState:
        return self.state_machine.current_state

    @property
    def result(self) -> Optional[str]:
        return self._final_result

    def pop_event(self) -> Optional[AgentEvent]:
        """Pop the next event from the queue, or None if empty."""
        if self._event_queue:
            return self._event_queue.pop(0)
        return None

    @property
    def event_queue(self) -> List[AgentEvent]:
        """Read-only access to event queue."""
        return self._event_queue

    @property
    def lane_queue(self) -> Optional[LaneConcurrencyQueue]:
        """Read-only access to lane queue."""
        return self._lane_queue

    @property
    def tool_pack(self) -> Optional[ToolPack]:
        """Read-only access to tool pack."""
        return self._tool_pack

    async def run(self, message: Optional[str] = None, *, trigger: str = "start") -> str:
        if message:
            message = self._time_provider.inject_timestamp(message)
            self.session.add_message("user", message)

        if self.state_machine.can_trigger(trigger):
            prev_state = self.state
            self.state_machine.trigger(trigger)
            if trigger == "start":
                get_logger().info(
                    self.label,
                    "Agent run started | incoming_message_length={}, session_message_count={}".format(
                        len(message) if message else 0,
                        len(self.session.messages),
                    ),
                    task_id=self.task_id, state=self.state.value, depth=self.session.depth,
                    event_type="agent_run_start",
                    data={
                        "message_length": len(message) if message else 0,
                        "session_msg_count": len(self.session.messages),
                        "log_message": message or "",
                    },
                )
            elif trigger == "children_settled":
                get_logger().info(
                    self.label,
                    "Agent resuming from children | incoming_message_length={}, session_message_count={}".format(
                        len(message) if message else 0,
                        len(self.session.messages),
                    ),
                    task_id=self.task_id, state=self.state.value, depth=self.session.depth,
                    event_type="agent_resume",
                    data={
                        "message_length": len(message) if message else 0,
                        "session_msg_count": len(self.session.messages),
                    },
                )
            get_logger().state_change(
                self.label, prev_state.value, self.state.value, trigger,
                task_id=self.task_id, state=self.state.value, depth=self.session.depth)

        try:
            return await self._execute_cycle()
        except Exception as e:
            await self._abort(e)
            return self._final_result

    async def _process_tool_calls(self) -> None:
        """Process tool calls from the LLM."""
        iteration = 0
        warning_injected = False

        while iteration < self.MAX_TOOL_ITERATIONS:
            iteration += 1

            if iteration == 1:
                get_logger().info(
                    self.label,
                    "Tool call loop starting | max_iterations={}".format(self.MAX_TOOL_ITERATIONS),
                    task_id=self.task_id, state=self.state.value, depth=self.session.depth,
                    event_type="tool_loop_start",
                    data={"max_iterations": self.MAX_TOOL_ITERATIONS},
                )

            if self.llm is None:
                break

            if (
                not warning_injected
                and self.config.summary_warning_reserve > 0
                and iteration == self.MAX_TOOL_ITERATIONS - self.config.summary_warning_reserve
            ):
                self.session.add_message("user", self._build_summary_warning(iteration))
                warning_injected = True
                get_logger().info(
                    self.label,
                    "Iteration limit warning injected | remaining_iterations={}".format(
                        self.config.summary_warning_reserve
                    ),
                    task_id=self.task_id, state=self.state.value, depth=self.session.depth,
                    event_type="summary_warning",
                    data={
                        "remaining_iterations": self.config.summary_warning_reserve,
                        "current_iteration": iteration,
                        "max_iterations": self.MAX_TOOL_ITERATIONS,
                    },
                )

            get_logger().info(
                self.label,
                "Sending request to LLM | iteration={}/{}, message_count={}".format(
                    iteration, self.MAX_TOOL_ITERATIONS, len(self.session.messages)
                ),
                task_id=self.task_id, state=self.state.value, depth=self.session.depth,
                event_type="llm_request",
                data={
                    "iteration": iteration,
                    "max_iterations": self.MAX_TOOL_ITERATIONS,
                    "message_count": len(self.session.messages),
                },
            )

            response = await self.llm.chat(
                messages=self.session.get_messages(),
                tools=self._get_tool_schemas(),
                agent_label=self.label,
                task_id=self.task_id,
                depth=self.session.depth,
            )

            if not response.has_tool_calls():
                get_logger().info(
                    self.label,
                    "LLM returned text response (no tool calls) | content_length={}".format(
                        len(response.content) if response.content else 0,
                    ),
                    task_id=self.task_id, state=self.state.value, depth=self.session.depth,
                    event_type="llm_text_response",
                    data={
                        "content_length": len(response.content) if response.content else 0,
                        "content": response.content or "",
                    },
                )
                if response.content:
                    self.session.add_message("assistant", response.content)
                    self._final_result = response.content
                else:
                    self._final_result = "[WARNING] No text response was generated by the agent."
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

            tool_names = [tc.name for tc in response.tool_calls]
            get_logger().info(
                self.label,
                "LLM requested {} tool call(s) | tools={}, iteration={}/{}".format(
                    len(response.tool_calls), tool_names, iteration, self.MAX_TOOL_ITERATIONS
                ),
                task_id=self.task_id, state=self.state.value, depth=self.session.depth,
                event_type="llm_tool_calls",
                data={
                    "tool_count": len(response.tool_calls),
                    "tool_names": tool_names,
                    "iteration": iteration,
                },
            )

            for tool_call in response.tool_calls:
                result = await self._execute_tool(tool_call)
                # Prevent empty content — some LLM APIs reject content="" with 422
                safe_result = result or "[Tool returned empty content]"
                self.session.add_message("tool", safe_result, tool_call_id=tool_call.call_id)

        if iteration >= self.MAX_TOOL_ITERATIONS:
            get_logger().error(
                self.label,
                "Tool call loop hit iteration limit ({}) | has_final_result={}".format(
                    self.MAX_TOOL_ITERATIONS, self._final_result is not None
                ),
                task_id=self.task_id, state=self.state.value, depth=self.session.depth,
                event_type="iteration_limit",
                data={
                    "max_iterations": self.MAX_TOOL_ITERATIONS,
                    "has_final_result": self._final_result is not None,
                },
            )

            if self.config.emergency_summary_enabled:
                self._final_result = await self._emergency_summarize()
            else:
                self._final_result = self._final_result or "[WARNING] Maximum tool call iterations reached."

        return

    def _build_summary_warning(self, current_iteration: int) -> str:
        """Build the warning message injected when approaching iteration limit.

        Args:
            current_iteration: Current iteration number

        Returns:
            Warning message string to be added as a user message.
        """
        remaining = self.MAX_TOOL_ITERATIONS - current_iteration
        return (
            "[System Notice] You have {} tool call iteration(s) remaining. "
            "Please stop making tool calls and provide your final comprehensive answer "
            "based on all data you have collected so far. Do NOT make any more tool calls."
        ).format(remaining)

    async def _emergency_summarize(self) -> str:
        """Force a final summary when iteration limit is reached without a text response.

        Calls LLM one more time WITHOUT tools, forcing a text-only response.
        Uses condensed message history to control token usage.

        Returns:
            Summary text, or fallback warning if summarization fails.
        """
        get_logger().info(
            self.label,
            "Triggering emergency summary | context_messages={}".format(
                self.config.emergency_summary_context_messages
            ),
            task_id=self.task_id, state=self.state.value, depth=self.session.depth,
            event_type="emergency_summary_start",
            data={
                "context_messages": self.config.emergency_summary_context_messages,
                "total_session_messages": len(self.session.messages),
            },
        )

        condensed = self._build_condensed_messages(
            self.config.emergency_summary_context_messages
        )

        condensed.append({
            "role": "user",
            "content": (
                "[System] You have exhausted all available tool call iterations. "
                "You MUST now provide a comprehensive final answer based on all the "
                "data and results you have gathered. Structure your answer clearly."
            ),
        })

        try:
            response = await self.llm.chat(
                messages=condensed,
                tools=[],  # Empty tools — force text-only response
                agent_label=self.label,
                task_id=self.task_id,
                depth=self.session.depth,
            )

            result = response.content or "[WARNING] Emergency summary returned empty."

            get_logger().info(
                self.label,
                "Emergency summary generated | result_length={}".format(len(result)),
                task_id=self.task_id, state=self.state.value, depth=self.session.depth,
                event_type="emergency_summary_complete",
                data={
                    "result_length": len(result),
                    "result": result,
                },
            )

            self.session.add_message("assistant", result)

            return result

        except Exception as e:
            get_logger().error(
                self.label,
                "Emergency summary failed | error_type={}, error={}".format(
                    type(e).__name__, str(e)
                ),
                task_id=self.task_id, state=self.state.value, depth=self.session.depth,
                event_type="emergency_summary_failed",
                data={
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                },
            )
            return "[WARNING] Maximum tool call iterations reached (summary generation failed)."

    def _build_condensed_messages(self, keep_count: int) -> List[Dict]:
        """Build a condensed message list for emergency summary.

        Strategy: always keep ALL system messages (there may be multiple) +
        last N non-system messages (most recent data).
        If keep_count is 0, returns the full session unchanged.

        Args:
            keep_count: Number of recent non-system messages to keep.
                        0 means keep everything (use full session).

        Returns:
            Condensed list of message dicts ready for LLM chat.
        """
        all_msgs = self.session.get_messages()

        if keep_count <= 0:
            return list(all_msgs)

        system_msgs = [m for m in all_msgs if m.get("role") == "system"]
        non_system_msgs = [m for m in all_msgs if m.get("role") != "system"]

        recent_non_system = non_system_msgs[-keep_count:]

        condensed = list(system_msgs)
        condensed.extend(recent_non_system)

        return condensed

    async def _execute_cycle(self) -> str:
        """Core execution: tool calls → drain events → decide next state.

        Linear flow (no recursion, no outer loop). Every path ends with return.
        Shared by run() for both start and children_settled triggers.
        """
        if self.state != AgentState.RUNNING:
            return self._final_result or ""

        await self._process_tool_calls()

        # Drain ALL queued events iteratively (replaces recursive drain_events chain)
        while True:
            if self.state in (AgentState.COMPLETED, AgentState.ERROR):
                return self._final_result or ""

            event = self.pop_event()
            if event is None:
                break

            if isinstance(event, ChildCompletionEvent):
                get_logger().info(
                    self.label,
                    "Draining deferred event | trigger_task_id={}, child_result_count={}".format(
                        getattr(event, 'trigger_task_id', 'N/A'), len(getattr(event, 'child_results', {}))
                    ),
                    task_id=self.task_id, state=self.state.value, depth=self.session.depth,
                    event_type="drain_event",
                    data={
                        "trigger_task_id": getattr(event, 'trigger_task_id', 'N/A'),
                        "child_result_count": len(getattr(event, 'child_results', {})),
                        "error": getattr(event, 'error', False),
                    },
                )
                formatted = self._time_provider.inject_timestamp(event.formatted_prompt)
                self.session.add_message("user", formatted)
                await self._process_tool_calls()
                # Loop continues — processes any events queued during _process_tool_calls

        # All events drained — decide next state using registry as single source of truth
        if self._has_pending_children():
            get_logger().info(
                self.label,
                "Spawned child agents detected, transitioning to WAITING_FOR_CHILDREN",
                task_id=self.task_id, state=self.state.value, depth=self.session.depth,
                event_type="waiting_for_children",
            )
            prev_state = self.state
            self.state_machine.trigger("spawn_children")
            get_logger().state_change(
                self.label, prev_state.value, self.state.value, "spawn_children",
                task_id=self.task_id, state=self.state.value, depth=self.session.depth)
            return "[Waiting for sub-agents to report back...]"
        else:
            get_logger().info(
                self.label,
                "Agent finished without spawning children, proceeding to finalize",
                task_id=self.task_id, state=self.state.value, depth=self.session.depth,
                event_type="agent_direct_complete",
            )
            await self._finish_and_notify()
            return self._final_result or ""

    async def _execute_tool(self, tool_call: ToolCall) -> str:
        """Execute a single tool call.

        Args:
            tool_call: The tool call to execute

        Returns:
            Tool execution result
        """
        tool = self._tool_pack.get(tool_call.name)

        get_logger().tool(
            self.label,
            "Executing tool '{}' | call_id={}".format(tool_call.name, tool_call.call_id),
            task_id=self.task_id, state=self.state.value, depth=self.session.depth,
            tool_name=tool_call.name,
            data={"call_id": tool_call.call_id, "arguments": tool_call.arguments},
        )

        if not tool:
            get_logger().error(
                self.label,
                "Tool '{}' not found in registry | call_id={}".format(tool_call.name, tool_call.call_id),
                task_id=self.task_id, state=self.state.value, depth=self.session.depth,
                event_type="tool_not_found",
                data={
                    "tool_name": tool_call.name,
                    "call_id": tool_call.call_id,
                },
            )
            return f"[ERROR] Tool not found: '{tool_call.name}' is not registered."

        context = {
            "session": self.session,
            "parent_agent": self,
        }

        try:
            result = await tool.execute(tool_call.arguments, context)
            get_logger().tool(
                self.label,
                "Tool '{}' completed | result_length={}".format(tool_call.name, len(result)),
                task_id=self.task_id, state=self.state.value, depth=self.session.depth,
                tool_name=tool_call.name,
                data={
                    "result_length": len(result),
                    "result": result,
                },
            )
            return result
        except Exception as e:
            get_logger().error(
                self.label,
                "Tool '{}' execution failed | error_type={}, error=\"{}\"".format(
                    tool_call.name, type(e).__name__, str(e)
                ),
                task_id=self.task_id, state=self.state.value, depth=self.session.depth,
                event_type="tool_error",
                data={
                    "tool_name": tool_call.name,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                },
            )
            return f"[ERROR] Tool '{tool_call.name}' execution failed: {type(e).__name__}: {str(e)}"

    def _has_pending_children(self) -> bool:
        task = self.task_registry.get_task(self.task_id)
        if not task or not task.child_task_ids:
            return False
        return self.task_registry.count_pending_children(self.task_id) > 0

    def _classify_error(self, error: Exception) -> ErrorCategory:
        """Classify error category based on exception type."""
        if isinstance(error, LLMProviderError):
            return ErrorCategory.LLM_ERROR
        return ErrorCategory.INTERNAL_ERROR

    async def _abort(self, error: Exception) -> None:
        """Unified crash handler. Never throws."""
        # Step 1: Log (best effort)
        try:
            get_logger().error(
                self.label,
                "Agent aborted | error_type={}, error=\"{}\"".format(
                    type(error).__name__, str(error)
                ),
                task_id=self.task_id, state=self.state.value, depth=self.session.depth,
                event_type="agent_abort",
                data={
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                },
            )
        except Exception:
            pass

        # Step 2: Set _final_result + _error_info
        if self.state != AgentState.ERROR:
            self._final_result = "[ERROR] Agent aborted: {} - {}".format(
                type(error).__name__, str(error)
            )
            category = self._classify_error(error)
            self._error_info = AgentError(
                category=category,
                message=str(error),
                exception_type=type(error).__name__,
            )

        # Step 3: State transition (only in non-terminal states)
        transitioned_to_error = False
        if self.state not in (AgentState.COMPLETED, AgentState.ERROR):
            try:
                if self.state_machine.can_trigger("error"):
                    prev_state = self.state
                    self.state_machine.trigger("error")
                    transitioned_to_error = True
                    try:
                        get_logger().state_change(
                            self.label, prev_state.value, self.state.value, "error",
                            task_id=self.task_id, state=self.state.value, depth=self.session.depth)
                    except Exception:
                        pass
            except Exception:
                pass

        # Step 4: _completion_event.set() (best effort)
        try:
            self._completion_event.set()
        except Exception:
            pass

        # Step 5: Notify parent (only when actually transitioned to ERROR)
        if self.parent_task_id and transitioned_to_error:
            try:
                await self.task_registry.complete(
                    self.task_id, self._final_result, error=True
                )
            except Exception:
                pass

    async def abort(self, error: Exception) -> None:
        """Public abort interface (Drainable protocol)."""
        await self._abort(error)

    async def _finish_and_notify(self):
        get_logger().info(
            self.label,
            "Agent finalizing | result_length={}, has_parent={}, session_message_count={}".format(
                len(self._final_result) if self._final_result else 0,
                self.parent_task_id is not None,
                len(self.session.messages),
            ),
            task_id=self.task_id, state=self.state.value, depth=self.session.depth,
            event_type="agent_finalize",
            data={
                "result_length": len(self._final_result) if self._final_result else 0,
                "has_parent": self.parent_task_id is not None,
                "session_msg_count": len(self.session.messages),
                "result": self._final_result or "",
            },
        )
        prev_state = self.state
        self.state_machine.trigger("finish")
        get_logger().state_change(
            self.label, prev_state.value, self.state.value, "finish",
            task_id=self.task_id, state=self.state.value, depth=self.session.depth)
        self._completion_event.set()

        # Release SpawnTool cached SubAgentManager for this agent
        self._tool_pack.release_spawn(self.task_id)

        if self.parent_task_id:
            await self.task_registry.complete(
                self.task_id, self._final_result
            )

    def _get_tool_schemas(self) -> List[Dict]:
        return self._tool_pack.get_schemas(self.session)


__all__ = [
    "Agent",
]
