"""SubAgent Manager - Orchestrates child spawning, execution, gate checks, and parent notification.

This module consolidates logic from:
- SpawnTool.execute() and _run_child_agent() (engine/tools/builtin/spawn.py)
- Registry.complete() gate checks + Branch A/B (engine/registry.py)
- Agent.run() with trigger="children_settled" prompt formatting (engine/agent_core.py)
"""

import asyncio
import json
import re
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from engine.runtime.agent_models import AgentState, Session
from engine.config import Config, get_config
from engine.logging import get_logger
from engine.providers.provider_models import Lane
from engine.safety import LaneConcurrencyQueue, ResultTruncator
from .events import AgentEvent, ChildCompletionEvent
from .subagent_models import CollectedChildResult
from engine.prompts import (
    DEPTH_LIMIT_REJECTION,
    get_subagent_system_prompt,
    get_spawn_confirmation,
    get_concurrency_timeout_rejection,
    get_child_results_prompt,
    get_child_results_empty_warning,
)
from engine.runtime.task_registry import CompleteInfo, AgentTaskRegistry
from engine.time import TimeProvider

if TYPE_CHECKING:
    from .protocol import Drainable


class SubAgentManager:
    """Orchestrates child agent lifecycle: spawn, run, gate-check, notify.

    Each Agent instance owns one SubAgentManager. The manager:
    - Spawns child agents with shared llm_provider and tool_pack
    - Constructs child Agents directly (no factory closure)
    - Runs child agents in background tasks (from SpawnTool._run_child_agent)
    - Handles child completion via handler chain (from Registry.complete gate checks)
    - Formats child results for parent consumption (from Agent.run with trigger="children_settled")
    """

    def __init__(
        self,
        task_registry: AgentTaskRegistry,
        event_queue: List[AgentEvent],
        drainable: "Drainable",
        agent_task_id: str,
        parent_label: str,
        config: Optional[Config] = None,
        lane_queue: Optional["LaneConcurrencyQueue"] = None,
        llm_provider=None,
        tool_pack=None,
        root_streaming_handler=None,
    ):
        """
        Args:
            task_registry: AgentTaskRegistry instance (shared across agents)
            event_queue: Agent's _event_queue (List[AgentEvent])
            drainable: Drainable protocol (the Agent)
            agent_task_id: THIS agent's task_id
            parent_label: Display label for logging
            config: Runtime configuration (used for result truncation limits)
            lane_queue: Optional lane-based concurrency queue
            llm_provider: Shared LLM provider for child agent construction
            tool_pack: Shared ToolPack for child agent construction
            root_streaming_handler: Optional SSEStreamingHandler from the root agent for sub-agent streaming
        """
        self._task_registry = task_registry
        self._event_queue = event_queue
        self._drainable = drainable
        self._agent_task_id = agent_task_id
        self._parent_label = parent_label
        self._config = config
        self._lane_queue = lane_queue
        self._llm_provider = llm_provider
        self._tool_pack = tool_pack
        self._root_streaming_handler = root_streaming_handler
        self._time_provider = TimeProvider(
            timezone_override=config.user_timezone if config else None
        )
        self._child_counter = 0
        # Register handler: when any child of this agent completes,
        # task_registry routes the callback here
        self._task_registry.register_handler(agent_task_id, self._on_child_complete)

    # ------------------------------------------------------------------
    # spawn() — migrated from SpawnTool.execute() (spawn.py lines 55-152)
    # ------------------------------------------------------------------

    async def spawn(
        self,
        task_desc: str,
        label: str,
        parent_session: Session,
    ) -> str:
        """Create a child agent and start it in the background.

        Args:
            task_desc: Task description for the child agent.
            label: Short descriptive label for the child.
            parent_session: The parent agent's session (used for depth tracking).

        Returns:
            Confirmation string on success, or error message on failure.
        """
        config = get_config()
        if parent_session.depth >= config.max_depth:
            logger = get_logger()
            logger.error(
                self._parent_label,
                "Spawn rejected: maximum nesting depth reached | current_depth={}, max_depth={}".format(
                    parent_session.depth, config.max_depth
                ),
                task_id=self._agent_task_id, state="running", depth=parent_session.depth,
                event_type="spawn_depth_limit",
                data={"current_depth": parent_session.depth, "max_depth": config.max_depth}
            )
            return DEPTH_LIMIT_REJECTION.format(depth=parent_session.depth, max_depth=config.max_depth)

        # Global concurrency gate — acquire BEFORE register
        lane_slot = None
        if self._lane_queue is not None:
            try:
                lane_slot = await self._lane_queue.acquire(
                    Lane.SUBAGENT,
                    timeout=config.spawn_timeout,
                )
            except TimeoutError:
                status = self._lane_queue.get_status().get(Lane.SUBAGENT)
                active = status.active if status else "?"
                max_conc = status.max_concurrent if status else "?"
                logger = get_logger()
                logger.error(
                    self._parent_label,
                    "Spawn rejected: lane concurrency limit reached | active={}/{}, timed out after {}s | task_desc=\"{}\"".format(
                        active, max_conc, config.spawn_timeout, task_desc,
                    ),
                    task_id=self._agent_task_id, state="running", depth=parent_session.depth,
                    event_type="spawn_lane_concurrency_limit",
                    data={
                        "active_count": active,
                        "max_concurrent": max_conc,
                        "spawn_timeout": config.spawn_timeout,
                        "task_description": task_desc,
                        "label": label,
                    }
                )
                return get_concurrency_timeout_rejection(
                    task_desc=task_desc,
                    label=label,
                    active=active,
                    max_concurrent=max_conc,
                    timeout=config.spawn_timeout,
                )

        task_id = f"task_{uuid.uuid4().hex[:8]}"

        child_session = Session(
            id=f"sess_{uuid.uuid4().hex[:8]}",
            depth=parent_session.depth + 1,
            parent_id=parent_session.id,
        )

        parent_label = self._parent_label
        can_spawn = child_session.depth < config.max_depth and config.is_tool_enabled("spawn")

        self._child_counter += 1
        child_index = self._child_counter
        child_depth = child_session.depth
        path_index = self._build_path_index(parent_label, child_index)
        display_name = "Sub-{}(d:{})".format(path_index, child_depth)
        llm_label = label

        logger = get_logger()
        logger.info(
            self._parent_label,
            "Child agent spawned successfully | child_task_id={}, depth={}, can_spawn={}".format(
                task_id, child_depth, can_spawn,
            ),
            task_id=self._agent_task_id, state="running", depth=parent_session.depth,
            event_type="spawn_created",
            data={
                "child_task_id": task_id, "child_label": display_name,
                "child_session_id": child_session.id, "child_depth": child_depth,
                "max_depth": config.max_depth, "can_spawn_further": can_spawn,
                "task_description": task_desc, "llm_label": llm_label,
            }
        )

        system_prompt = get_subagent_system_prompt(
            parent_label=parent_label,
            task_desc=task_desc,
            depth=child_session.depth,
            max_depth=config.max_depth,
            can_spawn=can_spawn,
            task_id=task_id,
            label=display_name,
        )

        env_block = self._time_provider.format_system_env_block()
        system_prompt = f"{system_prompt}\n\n{env_block}"

        child_session.add_message("system", system_prompt)

        await self._task_registry.register(
            task_id=task_id,
            session_id=child_session.id,
            description=task_desc,
            parent_agent=None,
            parent_task_id=self._agent_task_id,
            depth=child_session.depth,
        )

        # Create SubAgentStreamingWrapper for depth-1 children (root agent's direct children)
        child_streaming_handler = None
        if self._root_streaming_handler is not None and parent_session.depth == 0:
            from engine.streaming_handler import SubAgentStreamingWrapper
            child_streaming_handler = SubAgentStreamingWrapper(
                parent=self._root_streaming_handler,
                task_id=task_id,
            )
            # Emit subagent_start event
            start_part_id = self._root_streaming_handler._next_part_id()
            self._root_streaming_handler.emit("subagent_start", {
                "part_id": start_part_id,
                "task_id": task_id,
                "label": label,
                "description": task_desc,
                "parent_task_id": self._agent_task_id,
            })

        from engine.runtime.agent import Agent

        child_agent = Agent(
            session=child_session,
            config=config,
            llm_provider=self._llm_provider,
            task_registry=self._task_registry,
            tool_pack=self._tool_pack,
            task_id=task_id,
            parent_task_id=self._agent_task_id,
            label=display_name,
            lane_queue=self._lane_queue,
            streaming_handler=child_streaming_handler,
        )

        await self._task_registry.set_agent(task_id, child_agent)

        asyncio.create_task(
            self._run_child(child_agent, task_id, task_desc, child_session.depth, display_name, lane_slot=lane_slot)
        )

        return get_spawn_confirmation(task_id=task_id, label=label)

    # ------------------------------------------------------------------
    # _build_path_index() — generates dotted path index for child labels
    # ------------------------------------------------------------------

    @staticmethod
    def _build_path_index(parent_label: str, child_index: int) -> str:
        """Build a dotted path index from parent label and child counter.

        Root → "1", "2" ...
        Sub-1(d:1) → "1.1", "1.2" ...
        Sub-2.1(d:2) → "2.1.1", "2.1.2" ...

        Args:
            parent_label: Parent agent's display label (e.g. "Root", "Sub-1(d:1)").
            child_index: This child's sequential index within the parent.

        Returns:
            Dotted path index string.
        """
        match = re.match(r"Sub-(.+?)\(d:\d+\)", parent_label)
        if match:
            return "{}.{}".format(match.group(1), child_index)
        # Root or unrecognized label — start fresh
        return str(child_index)

    # ------------------------------------------------------------------
    # _run_child() — migrated from SpawnTool._run_child_agent() (spawn.py lines 154-244)
    # ------------------------------------------------------------------

    async def _run_child(
        self,
        agent: Any,
        task_id: str,
        task_desc: str,
        depth: int,
        display_name: Optional[str] = None,
        lane_slot: Optional[Any] = None,
    ) -> None:
        """Run child agent in background task. On completion, registry.complete() fires the handler chain.

        Args:
            agent: The child agent instance.
            task_id: Child's task ID.
            task_desc: Task description for child.
            depth: Nesting depth of the child.
            display_name: Display name for the child agent.
            lane_slot: Optional LaneSlot from LaneConcurrencyQueue for auto-release.
        """
        if lane_slot is not None:
            async with lane_slot:
                await self._execute_child(agent, task_id, task_desc, depth, display_name)
        else:
            await self._execute_child(agent, task_id, task_desc, depth, display_name)

    async def _execute_child(
        self,
        agent: Any,
        task_id: str,
        task_desc: str,
        depth: int,
        display_name: Optional[str] = None,
    ) -> None:
        """Execute child agent run and log based on final state.

        Args:
            agent: The child agent instance.
            task_id: Child's task ID.
            task_desc: Task description for child.
            depth: Nesting depth of the child.
            display_name: Display name for the child agent.
        """
        try:
            logger = get_logger()
            logger.info(
                display_name or "Sub",
                "Child agent starting background execution",
                task_id=task_id, state="idle", depth=depth,
                event_type="child_run_start",
                data={"task_description": task_desc}
            )
            await agent.run(task_desc)
        except Exception as e:
            # Safety net — agent.run() should catch all exceptions internally via _abort()
            # If we reach here, _abort() or run() has a bug
            await agent.abort(e)
            logger = get_logger()
            logger.error(
                display_name or "Sub",
                "UNEXPECTED: child agent leaked exception | error_type={}, error=\"{}\"".format(
                    type(e).__name__, str(e)),
                task_id=task_id, state="error", depth=depth,
                event_type="child_run_unhandled",
                data={"error_type": type(e).__name__, "error_message": str(e)},
            )
            if self._root_streaming_handler is not None:
                self._root_streaming_handler.emit("subagent_error", {
                    "task_id": task_id,
                    "message": str(agent._final_result or "Unknown error"),
                })
            return

        # Log based on final state (registry.complete handled internally by agent)
        state = agent.state
        if state == AgentState.COMPLETED:
            logger = get_logger()
            logger.info(
                display_name or "Sub",
                "Child agent completed | result_length={}".format(
                    len(agent.result) if agent.result else 0),
                task_id=task_id, state="completed", depth=depth,
                event_type="child_run_success",
                data={"result_length": len(agent.result) if agent.result else 0,
                      "result": agent.result or ""},
            )
        elif state == AgentState.ERROR:
            logger = get_logger()
            logger.error(
                display_name or "Sub",
                "Child agent aborted | error={}".format(agent.result),
                task_id=task_id, state="error", depth=depth,
                event_type="child_run_abort",
                data={"error_result": agent._final_result},
            )
        elif state == AgentState.WAITING_FOR_CHILDREN:
            logger = get_logger()
            logger.info(
                display_name or "Sub",
                "Child agent waiting for sub-agents | state={}".format(state.value),
                task_id=task_id, state=state.value, depth=depth,
                event_type="child_run_waiting",
            )
        else:
            logger = get_logger()
            logger.info(
                display_name or "Sub",
                "Child agent in unexpected state | state={}".format(state.value),
                task_id=task_id, state=state.value, depth=depth,
                event_type="child_run_unexpected_state",
                data={"state": state.value},
            )

        if self._root_streaming_handler is not None:
            if state == AgentState.COMPLETED:
                self._root_streaming_handler.emit("subagent_done", {
                    "task_id": task_id, "success": True,
                })
            elif state == AgentState.ERROR:
                self._root_streaming_handler.emit("subagent_error", {
                    "task_id": task_id,
                    "message": str(agent._final_result or "Unknown error"),
                })

    # ------------------------------------------------------------------
    # _on_child_complete() — migrated from Registry.complete() (registry.py lines 167-243)
    # ------------------------------------------------------------------

    async def _on_child_complete(self, task_id: str, info: CompleteInfo) -> None:
        """Handler called by task_registry when a child completes. Gate checks + notification.

        Args:
            task_id: The completing child's task ID.
            info: Completion info from the registry (pending counts, parent).
        """
        _ct = self._task_registry.get_task(task_id)
        _child_label = (
            getattr(_ct.agent, "label", None)
            if (_ct and _ct.agent)
            else None
        ) or "Child({})".format(task_id[:8])
        _child_depth = _ct.depth if _ct else 0

        # [Gate 1] Still have pending children → return
        if info.pending_children > 0:
            logger = get_logger()
            logger.info(
                _child_label,
                "Task completed but has pending children | task_id={}, pending_children={}".format(
                    task_id, info.pending_children
                ),
                task_id=task_id, state="running",
                depth=_child_depth,
                event_type="registry_complete_blocked_children",
                data={"pending_children": info.pending_children, "result_length": 0}
            )
            return

        # [Gate 2] Parent doesn't exist or not registered → return
        if not (info.parent_task_id and self._task_registry.get_task(info.parent_task_id)):
            return

        # [Gate 3] Still have pending siblings → return
        if info.pending_siblings > 0:
            logger = get_logger()
            logger.info(
                _child_label,
                "Task completed but has pending siblings | task_id={}, parent_task_id={}, pending_siblings={}".format(
                    task_id, info.parent_task_id, info.pending_siblings
                ),
                task_id=task_id, state="running",
                depth=_child_depth,
                event_type="registry_complete_blocked_siblings",
                data={"parent_task_id": info.parent_task_id, "pending_siblings": info.pending_siblings}
            )
            return

        # All gates passed → collect results and notify parent
        child_results = await self._task_registry.collect_and_cleanup(self._agent_task_id)

        parent_state = self._drainable.state
        branch = (
            "A (direct resume)" if parent_state == AgentState.WAITING_FOR_CHILDREN
            else "B (enqueue)" if parent_state == AgentState.RUNNING
            else "C (re-propagate)" if parent_state == AgentState.COMPLETED
            else "unknown"
        )
        child_ids = list(child_results.keys())

        # Build per-child result summaries for logging
        result_summaries = {}
        if child_results:
            for tid, info in child_results.items():
                result_summaries[tid] = {
                    "task_description": info.task_description,
                    "result_length": len(info.result),
                    "result": info.result,
                }

        logger = get_logger()
        logger.info(
            _child_label,
            "All children completed, notifying parent | task_id={}, parent_task_id={}, branch={}, parent_state={}, child_count={}".format(
                task_id, self._agent_task_id, branch, parent_state.value, len(child_results)
            ),
            task_id=task_id, state="running", depth=_child_depth,
            event_type="registry_notify_parent",
            data={
                "parent_task_id": self._agent_task_id,
                "parent_state": parent_state.value,
                "branch": branch,
                "child_count": len(child_results),
                "child_ids": child_ids,
                "results_summary": result_summaries,
            }
        )

        formatted = self._format_child_results(child_results)

        # [Branch A] Parent waiting for children → direct resume, bypass queue
        if parent_state == AgentState.WAITING_FOR_CHILDREN:
            asyncio.create_task(
                self._drainable.run(formatted, trigger="children_settled")
            )

        # [Branch B] Parent still running → enqueue for self-drain
        elif parent_state == AgentState.RUNNING:
            event = ChildCompletionEvent(child_results=child_results, formatted_prompt=formatted)
            self._event_queue.append(event)

        # [Branch C] Parent already completed → re-awaken it to process child results,
        # then its re-completion naturally re-propagates to the grandparent.
        # Root agent (depth=0) is also re-awakened: its _final_result will be updated
        # even though delegate() may have already returned (known behavior).
        elif parent_state == AgentState.COMPLETED:
            parent_task = self._task_registry.get_task(self._agent_task_id)
            if not (parent_task and parent_task.result):
                return

            parent_depth = self._task_registry.get_task_depth(self._agent_task_id)

            # Guard: respect max_reawaken_depth config.
            # If exceeded, accept data loss and stop — do NOT re-propagate stale result.
            max_reawaken_depth = (self._config.max_reawaken_depth
                                  if self._config else 3)
            # Use the agent tree depth as a natural bound:
            # each re-awaken level has strictly decreasing depth,
            # so depth itself limits the recursion chain length.
            if parent_depth > max_reawaken_depth:
                logger = get_logger()
                logger.warning(
                    _child_label,
                    "Re-awaken depth exceeds limit, child result discarded | parent_task_id={}, depth={}, max={}".format(
                        self._agent_task_id, parent_depth, max_reawaken_depth),
                    task_id=task_id, state="completed",
                    depth=_child_depth,
                    event_type="registry_reawaken_depth_exceeded",
                    data={"parent_task_id": self._agent_task_id,
                          "depth": parent_depth,
                          "max_reawaken_depth": max_reawaken_depth},
                )
                return

            # Re-awaken: fire the parent agent again with formatted child results.
            # The parent transitions COMPLETED → RUNNING via "reawaken" trigger,
            # processes child results through LLM, then _finish_and_notify()
            # naturally calls task_registry.complete() which triggers grandparent handler.
            logger = get_logger()
            logger.info(
                self._parent_label,
                "Re-awakening completed parent with child results | parent_task_id={}, child_count={}, depth={}".format(
                    self._agent_task_id, len(child_results), parent_depth),
                task_id=self._agent_task_id, state="completed",
                depth=parent_depth,
                event_type="registry_reawaken_parent",
                data={
                    "parent_task_id": self._agent_task_id,
                    "child_count": len(child_results),
                    "child_ids": child_ids,
                    "depth": parent_depth,
                },
            )
            asyncio.create_task(
                self._drainable.run(formatted, trigger="reawaken")
            )

    # ------------------------------------------------------------------
    # _format_child_results() — formats child results into a JSON prompt
    # ------------------------------------------------------------------

    def _format_child_results(self, child_results: Dict[str, CollectedChildResult]) -> str:
        """Format child results into a JSON prompt for the parent agent.

        Args:
            child_results: Mapping from child task ID to collected result.

        Returns:
            Formatted string ready to be injected as a user message.
        """
        if not child_results:
            return get_child_results_empty_warning()

        max_len = self._config.max_result_length if self._config else 4000

        entries = []
        for task_id, info in child_results.items():
            truncated = ResultTruncator.truncate(info.result, max_len)
            if len(info.result) > max_len:
                logger = get_logger()
                logger.warning(
                    "SubAgentManager",
                    "Child result truncated | task_id={}, original={} chars, limit={} chars".format(
                        task_id, len(info.result), max_len
                    ),
                    event_type="truncation",
                    task_id=task_id,
                    data={"original_length": len(info.result), "max_length": max_len},
                )
            entries.append(json.dumps({
                "task_id": task_id,
                "task": info.task_description,
                "result": truncated,
            }, ensure_ascii=False))

        return get_child_results_prompt("\n".join(entries))


