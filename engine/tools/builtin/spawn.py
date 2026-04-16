"""Spawn tool for creating child agents."""

import asyncio
import uuid
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

from engine.config import Config
from engine.logger import get_logger
from engine.models import Session
from engine.tools.base import Tool

if TYPE_CHECKING:
    from engine.registry import SubagentRegistry


class SpawnTool(Tool):
    """Tool for spawning child agents asynchronously."""

    name = "spawn"
    description = (
        "Spawn a child agent to asynchronously execute an independent task. "
        "Multiple calls per turn are supported to dispatch multiple child agents in parallel.\n"
        "[Execution] Returns immediately: a confirmation with task_id on success, or an error message on failure. "
        "After calling, the parent agent may continue with other tasks or end the current turn — no need to wait for child results. "
        "Child agents will proactively report their results back to the parent once completed in the background.\n"
        "[Use Cases] Best suited for breaking down a task into independent subtasks and dispatching them in parallel to child agents."
    )
    parameters = {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "The task description assigned to the child agent. The child agent will treat it as its sole objective and execute independently, so ensure it includes sufficient context and clear completion criteria.",
            },
            "label": {
                "type": "string",
                "description": "A descriptive label for the child agent, used for log tracing and result identification (optional). A short name that summarizes the task is recommended.",
            },
        },
        "required": ["task"],
    }

    def __init__(
        self,
        agent_factory: Callable,
        registry: "SubagentRegistry",
        parent_task_id: str,
        parent_label: str = "Root",
    ):
        self.agent_factory = agent_factory
        self.registry = registry
        self.parent_task_id = parent_task_id
        self.parent_label = parent_label

    async def execute(self, arguments: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Execute spawn tool to create a child agent."""
        parent_session = context["session"]
        config = context["config"]

        if parent_session.depth >= config.max_depth:
            logger = get_logger()
            logger.error(
                self.parent_label,
                "Spawn rejected: maximum nesting depth reached | current_depth={}, max_depth={}".format(
                    parent_session.depth, config.max_depth
                ),
                task_id=self.parent_task_id, state="running", depth=parent_session.depth,
                event_type="spawn_depth_limit",
                data={"current_depth": parent_session.depth, "max_depth": config.max_depth}
            )
            return f"[Spawn Failed] Maximum nesting depth reached (current: {parent_session.depth}/{config.max_depth}). Please complete the task at the current level — no further child agents can be spawned."

        task_desc = arguments.get("task", "")
        label = arguments.get("label", "subagent")
        task_id = f"task_{uuid.uuid4().hex[:8]}"

        child_session = Session(
            id=f"sess_{uuid.uuid4().hex[:8]}",
            depth=parent_session.depth + 1,
            parent_id=parent_session.id,
        )

        parent_label = self.parent_label
        can_spawn = child_session.depth < config.max_depth

        logger = get_logger()
        parent_agent = context["parent_agent"]
        parent_agent._child_counter += 1
        child_index = parent_agent._child_counter
        child_depth = child_session.depth
        agent_display_name = "Sub-{}(d:{})".format(child_index, child_depth)
        llm_label = arguments.get("label", None)

        logger.info(
            self.parent_label,
            "Child agent spawned successfully | child_label=\"{}\", child_task_id={}, depth={}, can_spawn={}, task_description=\"{}\"".format(
                agent_display_name, task_id, child_depth, can_spawn,
                task_desc[:300] + "..." if len(task_desc) > 300 else task_desc
            ),
            task_id=self.parent_task_id, state="running", depth=parent_session.depth,
            event_type="spawn_created",
            data={
                "child_task_id": task_id, "child_label": agent_display_name,
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

## Sub-Agent Spawning
{"You CAN spawn your own sub-agents." if can_spawn else "You are a leaf worker and CANNOT spawn further sub-agents."}

## Session Context
- Label: {label}
- Depth: {child_session.depth}/{config.max_depth}
- Your task ID: {task_id}"""

        child_session.add_message("system", system_prompt)

        await self.registry.register(
            task_id=task_id,
            session_id=child_session.id,
            description=task_desc,
            parent_agent=context["parent_agent"],
            parent_task_id=self.parent_task_id,
            depth=child_session.depth,
        )

        asyncio.create_task(
            self._run_child_agent(child_session, task_desc, task_id, config, agent_display_name)
        )

        return f"""━━━━ Spawned Task ━━━━
Task ID: {task_id}
Agent Label: {label}

Sub-agent is now executing in the background. Upon completion, you will be automatically re-activated and receive a full result report. You may proceed with other independent tasks or simply end your current turn."""

    async def _run_child_agent(
        self, child_session: Session, task_desc: str, task_id: str, config: Config,
        agent_display_name: Optional[str] = None,
    ):
        """Run child agent and handle completion.

        Args:
            child_session: Child agent's session
            task_desc: Task description for child
            task_id: Child's task ID
            config: Configuration object
            agent_display_name: Display name for the child agent

        Note:
            On completion, registry.complete() will automatically notify parent.
            On error, registry.complete() is called with error=True.
        """
        agent = self.agent_factory(
            child_session, config, self.registry, self.parent_task_id, task_id,
            label=agent_display_name,
        )
        await self.registry.set_agent(task_id, agent)
        logger = get_logger()
        try:
            logger.info(
                agent_display_name or "Sub",
                "Child agent starting background execution | task=\"{}\"".format(task_desc[:200]),
                task_id=task_id, state="idle", depth=child_session.depth,
                event_type="child_run_start",
                data={"task_description": task_desc}
            )
            result = await agent.run(task_desc)
            result_preview = (result[:500] + "...") if result and len(result) > 500 else (result or "None")
            logger.info(
                agent_display_name or "Sub",
                "Child agent completed successfully | result_length={}".format(len(result) if result else 0),
                task_id=task_id, state="completed", depth=child_session.depth,
                event_type="child_run_success",
                data={"result_length": len(result) if result else 0, "result_preview": result_preview}
            )
        except Exception as e:
            logger.error(
                agent_display_name or "Sub",
                "Child agent execution failed | error_type={}, error=\"{}\"".format(type(e).__name__, str(e)),
                task_id=task_id, state="error", depth=child_session.depth,
                event_type="child_run_error",
                data={"error_type": type(e).__name__, "error_message": str(e), "task_description": task_desc}
            )
            agent.state_machine.trigger("error")
            error_result = (
                f"[ERROR] Child agent execution failed.\n\n"
                f"Task: {task_desc}\n"
                f"Error Type: {type(e).__name__}\n"
                f"Error Details: {str(e)}\n\n"
                f"The child agent has been terminated. Please decide whether to re-spawn a new child agent or adjust the task strategy based on the error information above."
            )
            await self.registry.complete(task_id, error_result, error=True)
