"""Spawn tool for creating child agents."""

from typing import Any, Dict, TYPE_CHECKING

from engine.tools.base import Tool

if TYPE_CHECKING:
    from .manager import SubAgentManager


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

    def __init__(self, subagent_mgr: "SubAgentManager"):
        self._subagent_mgr = subagent_mgr

    async def execute(self, arguments: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Execute spawn tool to create a child agent."""
        session = context["session"]
        config = context["config"]
        agent_factory = context.get("agent_factory")

        task_desc = arguments.get("task", "")
        label = arguments.get("label", "subagent")

        return await self._subagent_mgr.spawn(
            task_desc, label, session, config, agent_factory
        )
