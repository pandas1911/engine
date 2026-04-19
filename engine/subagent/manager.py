"""SubAgent Manager - Orchestrates child spawning, execution, gate checks, and parent notification.

This module consolidates logic from:
- SpawnTool.execute() and _run_child_agent() (engine/tools/builtin/spawn.py)
- Registry.complete() gate checks + Branch A/B (engine/registry.py)
- Agent.resume_from_children() prompt formatting (engine/agent_core.py)
"""

import asyncio
import json
import uuid
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from engine.models import AgentState, Session
from engine.config import Config
from engine.logger import get_logger
from engine.safety import ResultTruncator
from engine.subagent.events import AgentEvent, ChildCompletionEvent
from engine.subagent.models import CollectedChildResult
from engine.subagent.registry import CompleteInfo, SubagentRegistry

if TYPE_CHECKING:
    from engine.subagent.spawn import SpawnTool
    from engine.subagent.protocol import Drainable


class SubAgentManager:
    """Orchestrates child agent lifecycle: spawn, run, gate-check, notify.

    Each Agent instance owns one SubAgentManager. The manager:
    - Spawns child agents (logic migrated from SpawnTool.execute)
    - Runs child agents in background tasks (from SpawnTool._run_child_agent)
    - Handles child completion via handler chain (from Registry.complete gate checks)
    - Formats child results for parent consumption (from Agent.resume_from_children)
    """

    def __init__(
        self,
        registry: SubagentRegistry,
        event_queue: List[AgentEvent],
        drainable: "Drainable",
        agent_task_id: str,
        parent_label: str,
        config: Optional[Config] = None,
    ):
        """
        Args:
            registry: SubagentRegistry instance (shared across agents)
            event_queue: Agent's _event_queue (List[AgentEvent])
            drainable: Drainable protocol (the Agent)
            agent_task_id: THIS agent's task_id
            parent_label: Display label for logging
            config: Runtime configuration (used for result truncation limits)
        """
        self._registry = registry
        self._event_queue = event_queue
        self._drainable = drainable
        self._agent_task_id = agent_task_id
        self._parent_label = parent_label
        self._config = config
        self._child_counter = 0
        # Register handler: when any child of this agent completes,
        # registry routes the callback here
        self._registry.register_handler(agent_task_id, self._on_child_complete)

    # ------------------------------------------------------------------
    # spawn() — migrated from SpawnTool.execute() (spawn.py lines 55-152)
    # ------------------------------------------------------------------

    async def spawn(
        self,
        task_desc: str,
        label: str,
        parent_session: Session,
        config: Config,
        agent_factory: Callable,
    ) -> str:
        """Create a child agent and start it in the background.

        Args:
            task_desc: Task description for the child agent.
            label: Short descriptive label for the child.
            parent_session: The parent agent's session (used for depth tracking).
            config: Runtime configuration.
            agent_factory: Callable that creates a child Agent.

        Returns:
            Confirmation string on success, or error message on failure.
        """
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
            return f"[Spawn Failed] Maximum nesting depth reached (current: {parent_session.depth}/{config.max_depth}). Please complete the task at the current level — no further child agents can be spawned."

        task_id = f"task_{uuid.uuid4().hex[:8]}"

        child_session = Session(
            id=f"sess_{uuid.uuid4().hex[:8]}",
            depth=parent_session.depth + 1,
            parent_id=parent_session.id,
        )

        parent_label = self._parent_label
        can_spawn = child_session.depth < config.max_depth

        self._child_counter += 1
        child_index = self._child_counter
        child_depth = child_session.depth
        display_name = "Sub-{}(d:{})".format(child_index, child_depth)
        llm_label = label

        logger = get_logger()
        logger.info(
            self._parent_label,
            "Child agent spawned successfully | child_label=\"{}\", child_task_id={}, depth={}, can_spawn={}, task_description=\"{}\"".format(
                display_name, task_id, child_depth, can_spawn,
                task_desc[:300] + "..." if len(task_desc) > 300 else task_desc
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

        system_prompt = f"""# Subagent Context

You are a **subagent** spawned by the {parent_label} for a specific task.

## Your Role
- You were created to handle: {task_desc}
- Complete this task. That's your entire purpose.
- You are NOT the {parent_label}. Don't try to be.

## Rules
1. **Stay focused** - Do your assigned task, nothing else
2. **Complete the task** - Your final message will be automatically reported to the {parent_label}
3. **Be ephemeral** - You may be terminated after task completion. That's fine.
4. **Trust push-based completion** - Descendant results are auto-announced back to you

## Output Format

Your final response is delivered **verbatim** to your parent agent. Every token enters the parent's context — be ruthless about brevity.

### Principles
1. **No filler** — No greetings, no "Here is...", no "I have completed...", no meta-commentary. Start with the result.
2. **No reasoning traces** — The parent needs your conclusion, not how you got there.
3. **No repetition** — If the parent gave you information in the task description, do not echo it back.

### Structure by Task Type
Adapt your output to the task. Use what fits, drop what doesn't:

- **Find / Retrieve** → Bullet list of key findings
- **Build / Modify** → The output
- **Analyze / Judge** → Conclusion first, then brief supporting points (includes yes/no questions — answer on line 1)
- **Execute** → One line per action: what you did + result

A one-line summary at the top is encouraged when the result is complex — skip it for simple answers.

## Sub-Agent Spawning
{"You CAN spawn your own sub-agents." if can_spawn else "You are a leaf worker and CANNOT spawn further sub-agents."}

## Session Context
- Label: {label}
- Depth: {child_session.depth}/{config.max_depth}
- Your task ID: {task_id}"""

        child_session.add_message("system", system_prompt)

        await self._registry.register(
            task_id=task_id,
            session_id=child_session.id,
            description=task_desc,
            parent_agent=None,
            parent_task_id=self._agent_task_id,
            depth=child_session.depth,
        )

        child_agent = agent_factory(
            child_session, config, self._registry, self._agent_task_id, task_id,
            label=display_name,
        )

        await self._registry.set_agent(task_id, child_agent)

        asyncio.create_task(
            self._run_child(child_agent, task_id, task_desc, child_session.depth, display_name)
        )

        return f"""━━━━ Spawned Task ━━━━
Task ID: {task_id}
Agent Label: {label}

Sub-agent is now executing in the background. Upon completion, you will be automatically re-activated and receive a full result report. You may proceed with other independent tasks or simply end your current turn."""

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
    ) -> None:
        """Run child agent in background task. On completion, registry.complete() fires the handler chain.

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
                "Child agent starting background execution | task=\"{}\"".format(task_desc[:200]),
                task_id=task_id, state="idle", depth=depth,
                event_type="child_run_start",
                data={"task_description": task_desc}
            )
            await agent.run(task_desc)
        except Exception as e:
            # Safety net — agent.run() should catch all exceptions internally via _abort()
            # If we reach here, _abort() or run() has a bug
            await agent._abort(e)
            logger = get_logger()
            logger.error(
                display_name or "Sub",
                "UNEXPECTED: child agent leaked exception | error_type={}, error=\"{}\"".format(
                    type(e).__name__, str(e)),
                task_id=task_id, state="error", depth=depth,
                event_type="child_run_unhandled",
                data={"error_type": type(e).__name__, "error_message": str(e)},
            )
            return

        # Log based on final state (registry.complete handled internally by agent)
        state = agent.state_machine.current_state
        if state == AgentState.COMPLETED:
            result_preview = (
                (agent._final_result[:500] + "...")
                if agent._final_result and len(agent._final_result) > 500
                else (agent._final_result or "None")
            )
            logger = get_logger()
            logger.info(
                display_name or "Sub",
                "Child agent completed | result_length={}".format(
                    len(agent._final_result) if agent._final_result else 0),
                task_id=task_id, state="completed", depth=depth,
                event_type="child_run_success",
                data={"result_length": len(agent._final_result) if agent._final_result else 0,
                      "result_preview": result_preview},
            )
        elif state == AgentState.ERROR:
            logger = get_logger()
            logger.error(
                display_name or "Sub",
                "Child agent aborted | error={}".format(agent._final_result),
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

    # ------------------------------------------------------------------
    # _on_child_complete() — migrated from Registry.complete() (registry.py lines 167-243)
    # ------------------------------------------------------------------

    async def _on_child_complete(self, task_id: str, info: CompleteInfo) -> None:
        """Handler called by registry when a child completes. Gate checks + notification.

        Args:
            task_id: The completing child's task ID.
            info: Completion info from the registry (pending counts, parent).
        """
        _ct = self._registry.get_task(task_id)
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
        if not (info.parent_task_id and self._registry.get_task(info.parent_task_id)):
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
        child_results = await self._registry.collect_and_cleanup(self._agent_task_id)

        parent_state = self._drainable.state
        branch = (
            "A (direct resume)" if parent_state == AgentState.WAITING_FOR_CHILDREN
            else "B (enqueue)" if parent_state == AgentState.RUNNING
            else "C (re-propagate)" if parent_state == AgentState.COMPLETED
            else "unknown"
        )
        child_ids = list(child_results.keys())
        logger = get_logger()
        logger.info(
            _child_label,
            "All gates passed, notifying parent | task_id={}, parent_task_id={}, branch={}, parent_state={}, child_count={}".format(
                task_id, self._agent_task_id, branch, parent_state.value, len(child_results)
            ),
            task_id=task_id, state="running", depth=_child_depth,
            event_type="registry_notify_parent",
            data={"parent_task_id": self._agent_task_id, "parent_state": parent_state.value, "branch": branch, "child_result_count": len(child_results), "child_ids": child_ids}
        )

        formatted = self._format_child_results(child_results)

        # [Branch A] Parent waiting for children → direct resume, bypass queue
        if parent_state == AgentState.WAITING_FOR_CHILDREN:
            asyncio.create_task(
                self._drainable.resume_from_children(formatted, child_results)
            )

        # [Branch B] Parent still running → enqueue for self-drain
        elif parent_state == AgentState.RUNNING:
            event = ChildCompletionEvent(child_results=child_results, formatted_prompt=formatted)
            self._event_queue.append(event)

        # [Branch C] Parent already completed → its earlier notification was blocked
        # by Gate 1 (pending_children > 0). Now that this last child is done,
        # re-propagate the notification upward to the grandparent.
        elif parent_state == AgentState.COMPLETED:
            parent_task = self._registry.get_task(self._agent_task_id)
            if parent_task and parent_task.result:
                logger = get_logger()
                logger.info(
                    self._parent_label,
                    "Parent already completed, re-propagating notification to grandparent | parent_task_id={}".format(
                        self._agent_task_id),
                    task_id=self._agent_task_id, state="completed",
                    depth=self._registry.get_task_depth(self._agent_task_id),
                    event_type="registry_repropagate_completed_parent",
                    data={"parent_task_id": self._agent_task_id},
                )
                await self._registry.complete(self._agent_task_id, parent_task.result)

    # ------------------------------------------------------------------
    # _format_child_results() — migrated from Agent.resume_from_children() (agent_core.py lines 383-396)
    # ------------------------------------------------------------------

    def _format_child_results(self, child_results: Dict[str, CollectedChildResult]) -> str:
        """Format child results into a JSON prompt for the parent agent.

        Args:
            child_results: Mapping from child task ID to collected result.

        Returns:
            Formatted string ready to be injected as a user message.
        """
        if not child_results:
            return "[WARNING] All sub-agents have completed their tasks, but no results were collected."

        max_len = self._config.max_result_length if self._config else 4000
        findings_prompt = "All sub-agents have completed their tasks. Below are their results.\n\n"

        for task_id, info in child_results.items():
            truncated = ResultTruncator.truncate(info.result, max_len)
            entry = {
                "task_id": task_id,
                "task": info.task_description,
                "result": truncated,
            }
            findings_prompt += json.dumps(entry, ensure_ascii=False) + "\n"

        return findings_prompt

    # ------------------------------------------------------------------
    # create_spawn_tool() — factory for the thin SpawnTool wrapper
    # ------------------------------------------------------------------

    def create_spawn_tool(self) -> "SpawnTool":
        """Create a SpawnTool instance bound to this manager.

        Returns:
            A SpawnTool that delegates to this manager's spawn() method.
        """
        from engine.subagent.spawn import SpawnTool
        return SpawnTool(self)
