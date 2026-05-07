"""List children tool — show all child agents spawned by the current agent.

Returns a human-readable summary of every child task including task_id,
label, status, message count, and task description.
"""

from typing import Any, Dict

from engine.tools.base import Tool


class ListChildrenTool(Tool):
    """List all child agents spawned by the current agent."""

    name = "list_children"

    description = (
        "Lists all child agents spawned by the current agent. "
        "Returns each child's task_id, label, status (completed / error / running / unknown), "
        "message count, and task description. "
        "No parameters required."
    )

    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    _MAX_DESC_LEN: int = 80

    async def execute(self, arguments: Dict[str, Any], context: Dict[str, Any]) -> str:
        agent = context.get("agent") or context.get("parent_agent")
        task_id = context.get("task_id")

        if not agent or not task_id:
            return "No child agents found."

        task_registry = getattr(agent, "task_registry", None)
        if task_registry is None:
            return "No child agents found."

        current_task = task_registry.get_task(task_id)
        if current_task is None:
            return "No child agents found."

        child_ids = current_task.child_task_ids

        # Path 1: Registry-based (in-memory child_task_ids still available)
        if child_ids:
            lines: list[str] = []
            for child_id in child_ids:
                child_task = task_registry.get_task(child_id)
                if child_task is None:
                    continue

                label = self._resolve_label(child_task, child_id)
                status = self._classify_status(child_task)
                msg_count = self._count_messages(child_task, agent)
                desc = self._truncate_description(child_task.task_description)

                lines.append(
                    "[{}] task_id={} | status={} | messages={} | task: {}".format(
                        label, child_id, status, msg_count, desc
                    )
                )

            header = "Child agents ({} total):".format(len(lines))
            return header + "\n" + "\n".join(lines)

        # Path 2: Disk-based fallback (multi-turn: registry lost but files exist)
        session_store = getattr(agent, "session_store", None)
        if session_store is not None:
            try:
                children = session_store.list_children()
                if children:
                    lines = []
                    for child in children:
                        status = "completed" if child.message_count >= 0 else "unknown"
                        lines.append(
                            "[{}] task_id={} | status={} | messages={} | task: {}".format(
                                child.task_id, child.task_id, status,
                                child.message_count, "(on disk)"
                            )
                        )
                    header = "Child agents ({} total, from disk):".format(len(lines))
                    return header + "\n" + "\n".join(lines)
            except Exception:
                pass

        return "No child agents have been spawned yet."

    @staticmethod
    def _classify_status(child_task: Any) -> str:
        """Determine the display status of a child task.

        Priority:
          1. result is not None  -> "completed"
          2. agent exists and state is ERROR -> "error"
          3. agent exists        -> "running"
          4. otherwise           -> "unknown"
        """
        if child_task.result is not None:
            return "completed"

        agent = child_task.agent
        if agent is not None:
            state = getattr(agent, "state", None)
            if state is not None and state.value == "error":
                return "error"
            return "running"

        return "unknown"

    @staticmethod
    def _resolve_label(child_task: Any, fallback_id: str) -> str:
        agent = child_task.agent
        if agent is not None:
            label = getattr(agent, "label", None)
            if label:
                return label
        return fallback_id

    @staticmethod
    def _count_messages(child_task: Any, parent_agent: Any = None) -> int:
        agent = child_task.agent
        if agent is not None:
            session = getattr(agent, "session", None)
            if session is not None:
                messages = getattr(session, "messages", None)
                if messages is not None:
                    return len(messages)

        # Disk fallback: try read_child_session via available session_store
        session_store = getattr(agent, "session_store", None) or getattr(
            parent_agent, "session_store", None
        )
        if session_store is not None:
            try:
                task_id = getattr(child_task, "task_id", None) or getattr(
                    child_task, "id", None
                )
                if task_id:
                    stored = session_store.read_child_session(task_id)
                    if stored is not None:
                        messages = getattr(stored, "messages", None)
                        if messages is not None:
                            return len(messages)
            except Exception:
                pass

        return 0

    def _truncate_description(self, description: str) -> str:
        if len(description) <= self._MAX_DESC_LEN:
            return description
        return description[: self._MAX_DESC_LEN - 3] + "..."
