"""SubAgent Manager - Orchestrates child spawning, execution, gate checks, and parent notification.

This module consolidates logic from:
- SpawnTool.execute() and _run_child_agent() (engine/tools/builtin/spawn.py)
- Registry.complete() gate checks + Branch A/B (engine/registry.py)
- Agent.run() with trigger="children_settled" prompt formatting (engine/agent_core.py)
"""

import asyncio
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from engine.runtime.agent_models import AgentState, Session
from engine.config import Config, get_config
from engine.logging import get_logger
from engine.providers.provider_models import Lane
from engine.safety import LaneConcurrencyQueue
from engine.safety.token_estimator import ResultTruncator
from .events import AgentEvent, ChildCompletionEvent
from engine.prompts import (
    get_subagent_system_prompt,
    get_spawn_confirmation,
    get_concurrency_timeout_rejection,
)
from engine.runtime.task_registry import CompleteInfo, AgentTaskRegistry
from engine.time import TimeProvider

if TYPE_CHECKING:
    from .protocol import Drainable
    from .subagent_models import ChildCompletionNotification

_SUMMARY_MAX_LENGTH = 6000


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
        session_store=None,
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
            session_store: Optional SessionStore for persisting child sessions to disk
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
        self._session_store = session_store
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
        if parent_session.depth >= 1:
            # [===================== LOG: error ======================]
            logger = get_logger()
            logger.error(
                self._parent_label,
                "Spawn rejected: sub-agents cannot spawn further children | current_depth={}".format(
                    parent_session.depth
                ),
                task_id=self._agent_task_id, state="running", depth=parent_session.depth,
                event_type="spawn_depth_limit",
                data={"current_depth": parent_session.depth}
            )
            # [====================== END LOG =======================]
            return "Sub-agents cannot spawn further children. Please complete your current task."

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
                # [===================== LOG: error ======================]
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
                # [====================== END LOG =======================]
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
        can_spawn = False

        self._child_counter += 1
        child_index = self._child_counter
        child_depth = child_session.depth
        display_name = "Sub-{}".format(child_index)
        llm_label = label

        # [=================== LOG: lifecycle ====================]
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
                "can_spawn_further": can_spawn,
                "task_description": task_desc, "llm_label": llm_label,
            }
        )
        # [==================== END LOG ==========================]

        system_prompt = get_subagent_system_prompt(
            parent_label=parent_label,
            task_desc=task_desc,
            depth=child_session.depth,
            can_spawn=can_spawn,
            task_id=task_id,
            label=display_name,
        )

        env_block = self._time_provider.format_system_env_block()
        system_prompt = f"{system_prompt}\n\n{env_block}"

        child_session.add_message("system", system_prompt)

        if self._session_store is not None:
            self._session_store.create_file(task_id, child_session)
            child_session._on_message_added = lambda msg, tid=task_id: self._session_store.append_line(tid, msg)

        await self._task_registry.register(
            task_id=task_id,
            session_id=child_session.id,
            description=task_desc,
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
            # [=================== EMIT: subagent ====================]
            start_part_id = self._root_streaming_handler._next_part_id()
            self._root_streaming_handler.emit("subagent_start", {
                "part_id": start_part_id,
                "task_id": task_id,
                "label": label,
                "description": task_desc,
                "parent_task_id": self._agent_task_id,
            })
            # [====================== END EMIT =======================]

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
            streaming_handler=child_streaming_handler,
        )

        await self._task_registry.set_agent(task_id, child_agent)

        asyncio.create_task(
            self._run_child(child_agent, task_id, task_desc, child_session.depth, display_name, lane_slot=lane_slot)
        )

        return get_spawn_confirmation(task_id=task_id, label=label)

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
            # [=================== LOG: lifecycle ====================]
            logger = get_logger()
            logger.info(
                display_name or "Sub",
                "Child agent starting background execution",
                task_id=task_id, state="idle", depth=depth,
                event_type="child_run_start",
                data={"task_description": task_desc}
            )
            # [==================== END LOG ==========================]
            await agent.run(task_desc)
        except Exception as e:
            # Safety net — agent.run() should catch all exceptions internally via _abort()
            # If we reach here, _abort() or run() has a bug
            await agent.abort(e)
            # [===================== LOG: error ======================]
            logger = get_logger()
            logger.error(
                display_name or "Sub",
                "UNEXPECTED: child agent leaked exception | error_type={}, error=\"{}\"".format(
                    type(e).__name__, str(e)),
                task_id=task_id, state="error", depth=depth,
                event_type="child_run_unhandled",
                data={"error_type": type(e).__name__, "error_message": str(e)},
            )
            # [====================== END LOG =======================]
            # [=================== EMIT: subagent ====================]
            if self._root_streaming_handler is not None:
                self._root_streaming_handler.emit("subagent_error", {
                    "task_id": task_id,
                    "message": str(agent._final_result or "Unknown error"),
                })
            # [====================== END EMIT =======================]
            return

        # Log based on final state (registry.complete handled internally by agent)
        state = agent.state
        if state == AgentState.COMPLETED:
            # [=================== LOG: lifecycle ====================]
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
            # [==================== END LOG ==========================]
        elif state == AgentState.ERROR:
            # [===================== LOG: error ======================]
            logger = get_logger()
            logger.error(
                display_name or "Sub",
                "Child agent aborted | error={}".format(agent.result),
                task_id=task_id, state="error", depth=depth,
                event_type="child_run_abort",
                data={"error_result": agent._final_result},
            )
            # [====================== END LOG =======================]
        elif state == AgentState.WAITING_FOR_CHILDREN:
            # [=================== LOG: lifecycle ====================]
            logger = get_logger()
            logger.info(
                display_name or "Sub",
                "Child agent waiting for sub-agents | state={}".format(state.value),
                task_id=task_id, state=state.value, depth=depth,
                event_type="child_run_waiting",
            )
            # [==================== END LOG ==========================]
        else:
            # [=================== LOG: lifecycle ====================]
            logger = get_logger()
            logger.info(
                display_name or "Sub",
                "Child agent in unexpected state | state={}".format(state.value),
                task_id=task_id, state=state.value, depth=depth,
                event_type="child_run_unexpected_state",
                data={"state": state.value},
            )
            # [==================== END LOG ==========================]

        # [=================== EMIT: subagent ====================]
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
        # [====================== END EMIT =======================]

    # ------------------------------------------------------------------
    # _on_child_complete() — migrated from Registry.complete() (registry.py lines 167-243)
    # ------------------------------------------------------------------

    async def _on_child_complete(self, task_id: str, info: CompleteInfo) -> None:
        """Handler called by task_registry when a child completes. Per-child immediate wake.

        Each child independently triggers its own notification to the parent.
        No sibling gates — the parent is woken immediately for every completing child.

        Args:
            task_id: The completing child's task ID.
            info: Completion info from the registry (pending counts, parent).
        """
        # [Gate] Parent doesn't exist or not registered → return
        if not (info.parent_task_id and self._task_registry.get_task(info.parent_task_id)):
            return

        child_task = self._task_registry.get_task(task_id)
        if not child_task:
            return

        notification = self._build_child_notification(task_id, child_task)

        parent_state = self._drainable.state

        # [=================== LOG: lifecycle ====================]
        get_logger().info(
            getattr(child_task.agent, "label", None) or "Child",
            "Child completed, notifying parent | task_id={}, parent_task_id={}, branch={}, parent_state={}".format(
                task_id, self._agent_task_id,
                "A" if parent_state == AgentState.WAITING_FOR_CHILDREN else "B" if parent_state == AgentState.RUNNING else "skip",
                parent_state.value,
            ),
            task_id=task_id, state="completed", depth=child_task.depth,
            event_type="child_notify_parent",
            data={
                "parent_task_id": self._agent_task_id,
                "parent_state": parent_state.value,
                "child_status": notification.status,
            },
        )
        # [==================== END LOG ==========================]

        # [Branch A] Parent waiting for children → direct resume
        if parent_state == AgentState.WAITING_FOR_CHILDREN:
            formatted = notification.to_prompt()
            asyncio.create_task(
                self._drainable.run(formatted, trigger="children_settled")
            )

        # [Branch B] Parent still running → enqueue for self-drain
        elif parent_state == AgentState.RUNNING:
            event = ChildCompletionEvent(notification=notification)
            self._event_queue.append(event)

        # Parent in COMPLETED/ERROR/IDLE → skip

    # ------------------------------------------------------------------
    # _build_child_notification() — builds ChildCompletionNotification for one child
    # ------------------------------------------------------------------

    def _build_child_notification(self, task_id: str, child_task) -> "ChildCompletionNotification":
        """Build a ChildCompletionNotification for a single completing child.

        Args:
            task_id: The child's task ID.
            child_task: The AgentTask instance for the child.

        Returns:
            A ChildCompletionNotification with extracted label, status, and summary.
        """
        from .subagent_models import ChildCompletionNotification

        if child_task.result is not None:
            status = "completed"
        elif child_task.agent is not None and child_task.agent.state == AgentState.ERROR:
            status = "error"
        else:
            status = "completed"

        label = (
            getattr(child_task.agent, "label", None)
            if child_task.agent
            else None
        ) or task_id

        summary = ""
        if child_task.agent is not None and child_task.agent.session is not None:
            for msg in reversed(child_task.agent.session.messages):
                if msg.role == "assistant" and msg.content:
                    content = msg.content.strip()
                    if not content.startswith("<think"):
                        summary = content
                        break

        if not summary:
            if status == "error" and child_task.agent is not None:
                summary = getattr(child_task.agent, "_final_result", None) or "Unknown error"
            elif child_task.result:
                summary = child_task.result

        summary = ResultTruncator.truncate(summary, _SUMMARY_MAX_LENGTH)

        session_file = "{}.jsonl".format(task_id)

        # Release agent reference — callback already persists all messages in real-time
        child_task.agent = None

        return ChildCompletionNotification(
            task_id=task_id,
            label=label,
            task=child_task.task_description,
            status=status,
            summary=summary,
            session_file=session_file,
        )


