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
    with parameters extracted from the agent in context.
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
        agent = context["agent"]
        config = get_config()
        task_id = agent.task_id

        # Runtime depth safety net — depth=1 is the architectural limit
        if session.depth >= 1:
            logger = get_logger()
            logger.error(
                agent.label,
                "Spawn rejected: sub-agents cannot spawn further children | "
                "current_depth={}".format(session.depth),
                task_id=task_id, state="running", depth=session.depth,
                event_type="spawn_depth_limit",
                data={"current_depth": session.depth},
            )
            return (
                "[Spawn Failed] Sub-agents cannot spawn further children. "
                "Please complete the task at the current level."
            )

        # Lazy init SubAgentManager with lock for concurrency safety
        async with self._lock:
            if task_id not in self._managers:
                from engine.subagent.manager import SubAgentManager
                self._managers[task_id] = SubAgentManager(
                    task_registry=agent.task_registry,
                    event_queue=agent.event_queue,
                    drainable=agent,
                    agent_task_id=task_id,
                    parent_label=agent.label,
                    config=config,
                    lane_queue=agent.lane_queue,
                    llm_provider=agent.llm,
                    tool_pack=agent.tool_pack,
                    root_streaming_handler=agent.streaming_handler,
                    session_store=getattr(agent, "session_store", None),
                )

        mgr = self._managers[task_id]
        task_desc = arguments.get("task", "")
        label = arguments.get("label", "unknown")
        return await mgr.spawn(task_desc, label, session)

    def release(self, agent_task_id: str) -> None:
        """Release cached SubAgentManager for completed agent."""
        self._managers.pop(agent_task_id, None)
