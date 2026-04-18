"""Agent core implementation with multi-level nesting support.

This module provides the Agent class with async callbacks and error propagation.
"""

import asyncio
import json
import uuid
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from engine.models import AgentState, ErrorCategory, AgentError, LLMResponse, Session, ToolCall
from engine.config import Config
from engine.subagent.registry import SubagentRegistry
from engine.subagent.events import AgentEvent, ChildCompletionEvent
from engine.subagent.manager import SubAgentManager
from engine.state_machine import AgentStateMachine
from engine.tools.base import ToolRegistry
from engine.llm_provider import MockLLMProvider, LLMProviderError
from engine.logger import get_logger

if TYPE_CHECKING:
    from engine.llm_provider import LLMProvider
    from engine.subagent.models import CollectedChildResult


class Agent:
    """Core Agent class with multi-level nesting support.

    # Implements: Drainable protocol (engine.subagent.protocol)
    """

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
        self._error_info: Optional[AgentError] = None
        self.display_id = f"[{self.label}|{self.task_id}]"
        self._event_queue: List[
            AgentEvent
        ] = []  # Deferred event queue (native list, Swift Array equivalent)
        self._child_counter = 0  # Child agent naming counter
        # SubAgentManager handles child spawning and notification
        self._subagent_mgr = SubAgentManager(
            registry=self.registry,
            event_queue=self._event_queue,
            drainable=self,
            agent_task_id=self.task_id,
            parent_label=self.label,
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

        logger = get_logger()
        logger.info(
            self.label,
            "Agent instance created | session_id={}, depth={}, parent_task_id={}, spawn_enabled={}, tool_count={}, max_iterations={}".format(
                self.session.id, self.session.depth,
                self.parent_task_id or "None (root)",
                self.session.depth < self.config.max_depth,
                len(self._tool_registry), self.MAX_TOOL_ITERATIONS,
            ),
            task_id=self.task_id, state=self.state.value, depth=self.session.depth,
            event_type="agent_init",
            data={
                "session_id": self.session.id,
                "parent_task_id": self.parent_task_id,
                "spawn_enabled": self.session.depth < self.config.max_depth,
                "tool_count": len(self._tool_registry),
            }
        )

    @property
    def state(self) -> AgentState:
        return self.state_machine.current_state

    def pop_event(self) -> Optional[AgentEvent]:
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
        label: Optional[str] = None,
    ) -> "Agent":
        """Create a child agent.

        Args:
            session: Child agent's session
            config: Configuration
            registry: Registry instance
            parent_task_id: Parent's task ID
            task_id: Optional task ID for the child (auto-generated if not provided)
            label: Optional label for the child agent

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
            label=label,
        )

    async def run(self, message: Optional[str] = None) -> str:
        if message:
            self.session.add_message("user", message)

        prev_state = self.state
        self.state_machine.trigger("start")
        logger = get_logger()
        message_preview = (message[:200] + "...") if message and len(message) > 200 else (message or "None")
        logger.info(
            self.label,
            "Agent run started | incoming_message_length={}, session_message_count={}, message_preview=\"{}\"".format(
                len(message) if message else 0,
                len(self.session.messages), message_preview,
            ),
            task_id=self.task_id, state=self.state.value, depth=self.session.depth,
            event_type="agent_run_start",
            data={
                "message_length": len(message) if message else 0,
                "session_msg_count": len(self.session.messages),
                "message_preview": message_preview,
            }
        )
        logger.state_change(self.label, prev_state.value, self.state.value, "start", task_id=self.task_id, depth=self.session.depth, data={"trigger_location": "run()"})

        try:
            spawned_any = await self._process_tool_calls()

            # Drain deferred events before branching decision
            await self.drain_events()

            result = await self._after_drain(spawned_any)
            if result is not None:
                return result

            return self._final_result
        except Exception as e:
            await self._abort(e)
            return self._final_result

    async def _process_tool_calls(self) -> bool:
        """Process tool calls from the LLM.

        Returns:
            True if any child agents were spawned
        """
        spawned = False
        iteration = 0

        while iteration < self.MAX_TOOL_ITERATIONS:
            iteration += 1

            if iteration == 1:
                logger = get_logger()
                logger.info(
                    self.label,
                    "Tool call loop starting | max_iterations={}".format(self.MAX_TOOL_ITERATIONS),
                    task_id=self.task_id, state=self.state.value, depth=self.session.depth,
                    event_type="tool_loop_start",
                    data={"max_iterations": self.MAX_TOOL_ITERATIONS}
                )

            if self.llm is None:
                break

            logger = get_logger()
            logger.info(
                self.label,
                "Sending request to LLM | iteration={}/{}, message_count={}".format(
                    iteration, self.MAX_TOOL_ITERATIONS, len(self.session.messages)
                ),
                task_id=self.task_id, state=self.state.value, depth=self.session.depth,
                event_type="llm_request",
                data={"iteration": iteration, "max_iterations": self.MAX_TOOL_ITERATIONS, "message_count": len(self.session.messages)}
            )

            response = await self.llm.chat(
                messages=self.session.get_messages(),
                tools=self._get_tool_schemas(),
                agent_label=self.label,
                task_id=self.task_id,
                depth=self.session.depth,
            )

            if not response.has_tool_calls():
                content_preview = (response.content[:500] + "...") if response.content and len(response.content) > 500 else (response.content or "None")
                logger = get_logger()
                logger.info(
                    self.label,
                    "LLM returned text response (no tool calls) | content_length={}, content_preview=\"{}\"".format(
                        len(response.content) if response.content else 0, content_preview
                    ),
                    task_id=self.task_id, state=self.state.value, depth=self.session.depth,
                    event_type="llm_text_response",
                    data={"content_length": len(response.content) if response.content else 0, "content_preview": content_preview}
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
            logger = get_logger()
            logger.info(
                self.label,
                "LLM requested {} tool call(s) | tools={}, iteration={}/{}".format(
                    len(response.tool_calls), tool_names, iteration, self.MAX_TOOL_ITERATIONS
                ),
                task_id=self.task_id, state=self.state.value, depth=self.session.depth,
                event_type="llm_tool_calls",
                data={"tool_count": len(response.tool_calls), "tool_names": tool_names, "iteration": iteration}
            )

            for tool_call in response.tool_calls:
                result = await self._execute_tool(tool_call)
                self.session.add_message("tool", result, tool_call_id=tool_call.call_id)

                if tool_call.name == "spawn":
                    spawned = True

        if iteration >= self.MAX_TOOL_ITERATIONS:
            logger = get_logger()
            logger.error(
                self.label,
                "Tool call loop hit iteration limit ({}) | has_final_result={}".format(
                    self.MAX_TOOL_ITERATIONS, self._final_result is not None
                ),
                task_id=self.task_id, state=self.state.value, depth=self.session.depth,
                event_type="iteration_limit",
                data={"max_iterations": self.MAX_TOOL_ITERATIONS, "has_final_result": self._final_result is not None}
            )
            self._final_result = self._final_result or "[WARNING] Maximum tool call iterations reached."

        return spawned

    async def _execute_tool(self, tool_call: ToolCall) -> str:
        """Execute a single tool call.

        Args:
            tool_call: The tool call to execute

        Returns:
            Tool execution result
        """
        tool = self._tool_registry.get(tool_call.name)

        logger = get_logger()
        logger.tool(
            self.label,
            "Executing tool '{}' | call_id={}, arguments={}".format(
                tool_call.name, tool_call.call_id,
                json.dumps(tool_call.arguments, ensure_ascii=False) if isinstance(tool_call.arguments, dict) else tool_call.arguments
            ),
            task_id=self.task_id, state=self.state.value, depth=self.session.depth,
            tool_name=tool_call.name,
            data={"call_id": tool_call.call_id, "arguments": tool_call.arguments}
        )

        if not tool:
            logger.error(
                self.label,
                "Tool '{}' not found in registry | call_id={}".format(tool_call.name, tool_call.call_id),
                task_id=self.task_id, state=self.state.value, depth=self.session.depth,
                event_type="tool_not_found",
                data={"tool_name": tool_call.name, "call_id": tool_call.call_id}
            )
            return f"[ERROR] Tool not found: '{tool_call.name}' is not registered."

        context = {
            "session": self.session,
            "config": self.config,
            "registry": self.registry,
            "parent_agent": self,
            "parent_task_id": self.task_id,
            "agent_factory": self._create_child_agent,
        }

        try:
            result = await tool.execute(tool_call.arguments, context)
            result_preview = (result[:500] + "...") if len(result) > 500 else result
            logger.tool(
                self.label,
                "Tool '{}' completed | result_length={}, result_preview=\"{}\"".format(
                    tool_call.name, len(result), result_preview
                ),
                task_id=self.task_id, state=self.state.value, depth=self.session.depth,
                tool_name=tool_call.name,
                data={"result_length": len(result), "result_preview": result_preview}
            )
            return result
        except Exception as e:
            logger.error(
                self.label,
                "Tool '{}' execution failed | error_type={}, error=\"{}\"".format(
                    tool_call.name, type(e).__name__, str(e)
                ),
                task_id=self.task_id, state=self.state.value, depth=self.session.depth,
                event_type="tool_error",
                data={"tool_name": tool_call.name, "error_type": type(e).__name__, "error_message": str(e)}
            )
            return f"[ERROR] Tool '{tool_call.name}' execution failed: {type(e).__name__}: {str(e)}"

    async def resume_from_children(self, formatted_prompt: str, child_results: Optional[Dict[str, "CollectedChildResult"]] = None):
        """Continue processing after all children complete. Accepts pre-formatted prompt from SubAgentManager."""
        try:
            if self.state_machine.current_state == AgentState.WAITING_FOR_CHILDREN:
                prev_state = self.state
                self.state_machine.trigger("children_settled")
                logger = get_logger()
                logger.state_change(self.label, prev_state.value, self.state.value, "children_settled", task_id=self.task_id, depth=self.session.depth, data={"trigger_location": "resume_from_children()"})

            logger = get_logger()
            if child_results:
                result_summaries = {}
                for tid, info in child_results.items():
                    result_summaries[tid] = {
                        "task_description": info.task_description,
                        "result_length": len(info.result),
                        "result_preview": info.result[:200] + "..." if len(info.result) > 200 else info.result,
                    }
                logger.info(
                    self.label,
                    "Resuming from children | collected {} child result(s), child_task_ids={}".format(
                        len(child_results), list(child_results.keys())
                    ),
                    task_id=self.task_id, state=self.state.value, depth=self.session.depth,
                    event_type="children_results",
                    data={"child_count": len(child_results), "child_task_ids": list(child_results.keys()), "results_summary": result_summaries}
                )
            else:
                logger.info(
                    self.label,
                    "Resuming from children | no child results collected",
                    task_id=self.task_id, state=self.state.value, depth=self.session.depth,
                    event_type="children_results",
                    data={"child_count": 0}
                )

            self.session.add_message("user", formatted_prompt)

            spawned_any = await self._process_tool_calls()

            # Drain deferred events
            await self.drain_events()

            await self._after_drain(spawned_any)
        except Exception as e:
            await self._abort(e)

    async def drain_events(self):
        """Process queued events after tool loop completes. Public for Drainable protocol."""
        if self.state_machine.current_state == AgentState.COMPLETED:
            return

        event = self.pop_event()
        if event is None:
            return

        logger = get_logger()
        logger.info(
            self.label,
            "Draining deferred event | trigger_task_id={}, child_result_count={}".format(
                getattr(event, 'trigger_task_id', 'N/A'), len(getattr(event, 'child_results', {}))
            ),
            task_id=self.task_id, state=self.state.value, depth=self.session.depth,
            event_type="drain_event",
            data={"trigger_task_id": getattr(event, 'trigger_task_id', 'N/A'), "child_result_count": len(getattr(event, 'child_results', {})), "error": getattr(event, 'error', False)}
        )

        if isinstance(event, ChildCompletionEvent):
            await self.resume_from_children(event.formatted_prompt, event.child_results)

    async def _after_drain(self, spawned_any: bool) -> Optional[str]:
        """Branch after drain_events(). State machine is source of truth.

        drain_events() may recursively call resume_from_children(), which
        can spawn children or finish on its own. So after drain returns, the
        local spawned_any may be stale — check state machine first.

        Returns str if fully handled (caller should return it), None otherwise.
        """
        state = self.state_machine.current_state

        # Drain already triggered full completion chain
        if state == AgentState.COMPLETED:
            return self._final_result

        # Drain's inner resume_from_children already spawned and transitioned
        if state == AgentState.WAITING_FOR_CHILDREN:
            return "[Waiting for sub-agents to report back...]"

        # State is still RUNNING — drain did nothing, trust local spawned_any
        if spawned_any:
            logger = get_logger()
            logger.info(
                self.label,
                "Spawned child agents detected, transitioning to WAITING_FOR_CHILDREN",
                task_id=self.task_id, state=self.state.value, depth=self.session.depth,
                event_type="waiting_for_children",
            )
            prev_state = self.state
            self.state_machine.trigger("spawn_children")
            logger.state_change(self.label, prev_state.value, self.state.value, "spawn_children", task_id=self.task_id, depth=self.session.depth, data={"trigger_location": "_after_drain()"})
            return "[Waiting for sub-agents to report back...]"
        else:
            if self._has_pending_children():
                logger = get_logger()
                logger.info(
                    self.label,
                    "No new spawns but pending children exist, waiting for them to complete",
                    task_id=self.task_id, state=self.state.value, depth=self.session.depth,
                    event_type="waiting_for_pending_children",
                )
                prev_state = self.state
                self.state_machine.trigger("spawn_children")
                logger.state_change(self.label, prev_state.value, self.state.value, "spawn_children", task_id=self.task_id, depth=self.session.depth, data={"trigger_location": "_after_drain() - pending children"})
                return "[Waiting for sub-agents to report back...]"

            logger = get_logger()
            logger.info(
                self.label,
                "Agent finished without spawning children, proceeding to finalize",
                task_id=self.task_id, state=self.state.value, depth=self.session.depth,
                event_type="agent_direct_complete",
            )
            await self._finish_and_notify()
            return None

    def _has_pending_children(self) -> bool:
        task = self.registry.get_task(self.task_id)
        if not task or not task.child_task_ids:
            return False
        return self.registry.count_pending_children(self.task_id) > 0

    def _classify_error(self, error: Exception) -> ErrorCategory:
        """Classify error category based on exception type."""
        if isinstance(error, LLMProviderError):
            return ErrorCategory.LLM_ERROR
        return ErrorCategory.INTERNAL_ERROR

    async def _abort(self, error: Exception) -> None:
        """Unified crash handler. Never throws."""
        # Step 1: Log (best effort)
        try:
            logger = get_logger()
            logger.error(
                self.label,
                "Agent aborted | error_type={}, error=\"{}\"".format(
                    type(error).__name__, str(error)
                ),
                task_id=self.task_id,
                state=self.state.value,
                depth=self.session.depth,
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
                        logger = get_logger()
                        logger.state_change(
                            self.label, prev_state.value, self.state.value, "error",
                            task_id=self.task_id, depth=self.session.depth,
                            data={"trigger_location": "_abort()"},
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
                await self.registry.complete(
                    self.task_id, self._final_result, error=True
                )
            except Exception:
                pass

    async def abort(self, error: Exception) -> None:
        """Public abort interface (Drainable protocol)."""
        await self._abort(error)

    async def _finish_and_notify(self):
        logger = get_logger()
        result_preview = (self._final_result[:300] + "...") if self._final_result and len(self._final_result) > 300 else self._final_result
        logger.info(
            self.label,
            "Agent finalizing | result_length={}, has_parent={}, session_message_count={}, result_preview=\"{}\"".format(
                len(self._final_result) if self._final_result else 0,
                self.parent_task_id is not None,
                len(self.session.messages), result_preview
            ),
            task_id=self.task_id, state=self.state.value, depth=self.session.depth,
            event_type="agent_finalize",
            data={"result_length": len(self._final_result) if self._final_result else 0, "has_parent": self.parent_task_id is not None, "session_msg_count": len(self.session.messages), "result_preview": result_preview}
        )
        prev_state = self.state
        self.state_machine.trigger("finish")
        logger.state_change(self.label, prev_state.value, self.state.value, "finish", task_id=self.task_id, depth=self.session.depth, data={"trigger_location": "_finish_and_notify()"})
        self._completion_event.set()

        if self.parent_task_id:
            await self.registry.complete(
                self.task_id, self._final_result
            )

    def _get_tool_schemas(self) -> List[Dict]:
        return self._tool_registry.get_schemas()


__all__ = [
    "Agent",
]
