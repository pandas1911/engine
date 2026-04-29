"""Spawn tool for creating child agents."""

import asyncio
from typing import Any, Dict, TYPE_CHECKING

from engine.tools.base import Tool
from engine.logging import get_logger
from engine.config import get_config

if TYPE_CHECKING:
    from engine.subagent.manager import SubAgentManager


class SpawnTool(Tool):
    """Spawn tool — creates child agents via SubAgentManager.

    SubAgentManager is lazily created per-agent on first execute() call,
    with parameters extracted from the parent_agent in context.
    """

    name = "spawn"
    description = (
        "Spawn a child sub-agent to handle a specific task in parallel. "
        "The child agent runs asynchronously and results are automatically "
        "delivered back to you when all children complete. "
        "Use this to parallelize work — spawn multiple children for independent tasks."
    )
    parameters = {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "Clear, specific task description for the child agent. "
                               "Be concise but complete — this is all the child sees.",
            },
            "label": {
                "type": "string",
                "description": "Short label for the child agent (used in logs and identification).",
            },
        },
        "required": ["task"],
    }

    def __init__(self):
        self._managers: Dict[str, "SubAgentManager"] = {}
        self._lock = asyncio.Lock()

    async def execute(self, arguments: Dict[str, Any], context: Dict[str, Any]) -> str:
        session = context["session"]
        parent_agent = context["parent_agent"]
        config = get_config()
        task_id = parent_agent.task_id

        # Runtime depth safety net
        if session.depth >= config.max_depth:
            logger = get_logger()
            logger.error(
                parent_agent.label,
                "Spawn rejected: maximum nesting depth reached | "
                "current_depth={}, max_depth={}".format(session.depth, config.max_depth),
                task_id=task_id, state="running", depth=session.depth,
                event_type="spawn_depth_limit",
                data={"current_depth": session.depth, "max_depth": config.max_depth},
            )
            return (
                "[Spawn Failed] Maximum nesting depth reached "
                "(current: {}/{}). No further child agents can be spawned."
            ).format(session.depth, config.max_depth)

        # Lazy init SubAgentManager with lock for concurrency safety
        async with self._lock:
            if task_id not in self._managers:
                from engine.subagent.manager import SubAgentManager
                self._managers[task_id] = SubAgentManager(
                    task_registry=parent_agent.task_registry,
                    event_queue=parent_agent.event_queue,
                    drainable=parent_agent,
                    agent_task_id=task_id,
                    parent_label=parent_agent.label,
                    config=config,
                    lane_queue=parent_agent.lane_queue,
                    llm_provider=parent_agent.llm,
                    tool_pack=parent_agent.tool_pack,
                )

        mgr = self._managers[task_id]
        task_desc = arguments.get("task", "")
        label = arguments.get("label", "subagent")
        return await mgr.spawn(task_desc, label, session)

    def release(self, agent_task_id: str) -> None:
        """Release cached SubAgentManager for completed agent."""
        self._managers.pop(agent_task_id, None)
