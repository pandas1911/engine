"""Agent implementation with multi-level nesting support.

This module provides the Agent class with async callbacks and error propagation.
"""

import asyncio
import json
import uuid
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .agent_models import AgentState, ErrorCategory, AgentError, Session
from engine.providers.provider_models import ToolCall
from engine.config import Config
from .task_registry import AgentTaskRegistry
from engine.subagent.events import AgentEvent, ChildCompletionEvent
from engine.subagent.manager import SubAgentManager
from .state import AgentStateMachine
from engine.tools.base import ToolRegistry
from engine.providers.llm_provider import LLMProviderError
from engine.logging.agent_log import AgentLogHelper

if TYPE_CHECKING:
    from engine.providers.llm_provider import LLMProvider


class Agent:
    """Core Agent class with multi-level nesting support.

    # Implements: Drainable protocol (engine.subagent.protocol)
    """

    MAX_TOOL_ITERATIONS = 20

    def __init__(
        self,
        session: Session,
        config: Config,
        llm_provider: "LLMProvider",
        task_registry: Optional[AgentTaskRegistry] = None,
        tools: Optional[List[Any]] = None,
        tool_registry: Optional[ToolRegistry] = None,
        task_id: Optional[str] = None,
        parent_task_id: Optional[str] = None,
        label: Optional[str] = None,
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
        self._log = AgentLogHelper(
            label=self.label,
            task_id=self.task_id,
            state_getter=lambda: self.state_machine.current_state,
            depth_getter=lambda: self.session.depth,
        )
        self._event_queue: List[
            AgentEvent
        ] = []  # Deferred event queue (native list, Swift Array equivalent)
        # SubAgentManager handles child spawning and notification
        self._subagent_mgr = SubAgentManager(
            task_registry=self.task_registry,
            event_queue=self._event_queue,
            drainable=self,
            agent_task_id=self.task_id,
            parent_label=self.label,
            config=self.config,
        )
        # Use provided registry or create new one
        self._tool_registry = tool_registry or ToolRegistry()

        # Backward compat: register any tools passed as list
        if tools:
            for tool in tools:
                self._tool_registry.register(tool)

        # SpawnTool conditional injection
        if session.depth < config.max_depth:
            spawn_tool = self._subagent_mgr.create_spawn_tool()
            self._tool_registry.register_spawn(spawn_tool)

        self._log.info(
            "agent_init",
            "Agent instance created | session_id={}, depth={}, parent_task_id={}, spawn_enabled={}, tool_count={}, max_iterations={}".format(
                self.session.id, self.session.depth,
                self.parent_task_id or "None (root)",
                self.session.depth < self.config.max_depth,
                len(self._tool_registry), self.MAX_TOOL_ITERATIONS,
            ),
            session_id=self.session.id,
            parent_task_id=self.parent_task_id,
            spawn_enabled=self.session.depth < self.config.max_depth,
            tool_count=len(self._tool_registry),
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

    def _create_child_agent(
        self,
        session: Session,
        config: Config,
        task_registry: AgentTaskRegistry,
        parent_task_id: str,
        task_id: Optional[str] = None,
        label: Optional[str] = None,
    ) -> "Agent":
        """Create a child agent.

        Args:
            session: Child agent's session
            config: Configuration
            task_registry: AgentTaskRegistry instance
            parent_task_id: Parent's task ID
            task_id: Optional task ID for the child (auto-generated if not provided)
            label: Optional label for the child agent

        Returns:
            New Agent instance
        """
        return Agent(
            session=session,
            config=config,
            llm_provider=self.llm,
            task_registry=task_registry,
            tool_registry=self._tool_registry.clone(),
            task_id=task_id,
            parent_task_id=parent_task_id,
            label=label,
        )

    async def run(self, message: Optional[str] = None, *, trigger: str = "start") -> str:
        if message:
            self.session.add_message("user", message)

        if self.state_machine.can_trigger(trigger):
            prev_state = self.state
            self.state_machine.trigger(trigger)
            if trigger == "start":
                self._log.info(
                    "agent_run_start",
                    "Agent run started | incoming_message_length={}, session_message_count={}".format(
                        len(message) if message else 0,
                        len(self.session.messages),
                    ),
                    message_length=len(message) if message else 0,
                    session_msg_count=len(self.session.messages),
                    log_message=message or "",
                )
            elif trigger == "children_settled":
                self._log.info(
                    "agent_resume",
                    "Agent resuming from children | incoming_message_length={}, session_message_count={}".format(
                        len(message) if message else 0,
                        len(self.session.messages),
                    ),
                    message_length=len(message) if message else 0,
                    session_msg_count=len(self.session.messages),
                )
            self._log.state_change(prev_state.value, self.state.value, trigger, trigger_location="run()")

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
                self._log.info(
                    "tool_loop_start",
                    "Tool call loop starting | max_iterations={}".format(self.MAX_TOOL_ITERATIONS),
                    max_iterations=self.MAX_TOOL_ITERATIONS,
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
                self._log.info(
                    "summary_warning",
                    "Iteration limit warning injected | remaining_iterations={}".format(
                        self.config.summary_warning_reserve
                    ),
                    remaining_iterations=self.config.summary_warning_reserve,
                    current_iteration=iteration,
                    max_iterations=self.MAX_TOOL_ITERATIONS,
                )

            self._log.info(
                "llm_request",
                "Sending request to LLM | iteration={}/{}, message_count={}".format(
                    iteration, self.MAX_TOOL_ITERATIONS, len(self.session.messages)
                ),
                iteration=iteration,
                max_iterations=self.MAX_TOOL_ITERATIONS,
                message_count=len(self.session.messages),
            )

            response = await self.llm.chat(
                messages=self.session.get_messages(),
                tools=self._get_tool_schemas(),
                agent_label=self.label,
                task_id=self.task_id,
                depth=self.session.depth,
            )

            if not response.has_tool_calls():
                self._log.info(
                    "llm_text_response",
                    "LLM returned text response (no tool calls) | content_length={}".format(
                        len(response.content) if response.content else 0,
                    ),
                    content_length=len(response.content) if response.content else 0,
                    content=response.content or "",
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
            self._log.info(
                "llm_tool_calls",
                "LLM requested {} tool call(s) | tools={}, iteration={}/{}".format(
                    len(response.tool_calls), tool_names, iteration, self.MAX_TOOL_ITERATIONS
                ),
                tool_count=len(response.tool_calls),
                tool_names=tool_names,
                iteration=iteration,
            )

            for tool_call in response.tool_calls:
                result = await self._execute_tool(tool_call)
                self.session.add_message("tool", result, tool_call_id=tool_call.call_id)

        if iteration >= self.MAX_TOOL_ITERATIONS:
            self._log.error(
                "iteration_limit",
                "Tool call loop hit iteration limit ({}) | has_final_result={}".format(
                    self.MAX_TOOL_ITERATIONS, self._final_result is not None
                ),
                max_iterations=self.MAX_TOOL_ITERATIONS,
                has_final_result=self._final_result is not None,
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
        self._log.info(
            "emergency_summary_start",
            "Triggering emergency summary | context_messages={}".format(
                self.config.emergency_summary_context_messages
            ),
            context_messages=self.config.emergency_summary_context_messages,
            total_session_messages=len(self.session.messages),
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

            self._log.info(
                "emergency_summary_complete",
                "Emergency summary generated | result_length={}".format(len(result)),
                result_length=len(result),
                result=result,
            )

            self.session.add_message("assistant", result)

            return result

        except Exception as e:
            self._log.error(
                "emergency_summary_failed",
                "Emergency summary failed | error_type={}, error={}".format(
                    type(e).__name__, str(e)
                ),
                error_type=type(e).__name__,
                error_message=str(e),
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
                self._log.info(
                    "drain_event",
                    "Draining deferred event | trigger_task_id={}, child_result_count={}".format(
                        getattr(event, 'trigger_task_id', 'N/A'), len(getattr(event, 'child_results', {}))
                    ),
                    trigger_task_id=getattr(event, 'trigger_task_id', 'N/A'),
                    child_result_count=len(getattr(event, 'child_results', {})),
                    error=getattr(event, 'error', False),
                )
                self.session.add_message("user", event.formatted_prompt)
                await self._process_tool_calls()
                # Loop continues — processes any events queued during _process_tool_calls

        # All events drained — decide next state using registry as single source of truth
        if self._has_pending_children():
            self._log.info(
                "waiting_for_children",
                "Spawned child agents detected, transitioning to WAITING_FOR_CHILDREN",
            )
            prev_state = self.state
            self.state_machine.trigger("spawn_children")
            self._log.state_change(prev_state.value, self.state.value, "spawn_children", trigger_location="_execute_cycle()")
            return "[Waiting for sub-agents to report back...]"
        else:
            self._log.info(
                "agent_direct_complete",
                "Agent finished without spawning children, proceeding to finalize",
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
        tool = self._tool_registry.get(tool_call.name)

        self._log.tool(
            tool_call.name,
            "Executing tool '{}' | call_id={}, arguments={}".format(
                tool_call.name, tool_call.call_id,
                json.dumps(tool_call.arguments, ensure_ascii=False) if isinstance(tool_call.arguments, dict) else tool_call.arguments
            ),
            call_id=tool_call.call_id,
            arguments=tool_call.arguments,
        )

        if not tool:
            self._log.error(
                "tool_not_found",
                "Tool '{}' not found in registry | call_id={}".format(tool_call.name, tool_call.call_id),
                tool_name=tool_call.name,
                call_id=tool_call.call_id,
            )
            return f"[ERROR] Tool not found: '{tool_call.name}' is not registered."

        context = {
            "session": self.session,
            "config": self.config,
            "task_registry": self.task_registry,
            "parent_agent": self,
            "parent_task_id": self.task_id,
            "agent_factory": self._create_child_agent,
        }

        try:
            result = await tool.execute(tool_call.arguments, context)
            self._log.tool(
                tool_call.name,
                "Tool '{}' completed | result_length={}".format(tool_call.name, len(result)),
                result_length=len(result),
                result=result,
            )
            return result
        except Exception as e:
            self._log.error(
                "tool_error",
                "Tool '{}' execution failed | error_type={}, error=\"{}\"".format(
                    tool_call.name, type(e).__name__, str(e)
                ),
                tool_name=tool_call.name,
                error_type=type(e).__name__,
                error_message=str(e),
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
            self._log.error(
                "agent_abort",
                "Agent aborted | error_type={}, error=\"{}\"".format(
                    type(error).__name__, str(error)
                ),
                error_type=type(error).__name__,
                error_message=str(error),
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
                        self._log.state_change(
                            prev_state.value, self.state.value, "error",
                            trigger_location="_abort()",
                        )
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
        self._log.info(
            "agent_finalize",
            "Agent finalizing | result_length={}, has_parent={}, session_message_count={}".format(
                len(self._final_result) if self._final_result else 0,
                self.parent_task_id is not None,
                len(self.session.messages),
            ),
            result_length=len(self._final_result) if self._final_result else 0,
            has_parent=self.parent_task_id is not None,
            session_msg_count=len(self.session.messages),
            result=self._final_result or "",
        )
        prev_state = self.state
        self.state_machine.trigger("finish")
        self._log.state_change(prev_state.value, self.state.value, "finish", trigger_location="_finish_and_notify()")
        self._completion_event.set()

        if self.parent_task_id:
            await self.task_registry.complete(
                self.task_id, self._final_result
            )

    def _get_tool_schemas(self) -> List[Dict]:
        return self._tool_registry.get_schemas()


__all__ = [
    "Agent",
]
