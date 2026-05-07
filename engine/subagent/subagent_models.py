"""Subagent-specific data models."""

from typing import Any, Optional, Set
from dataclasses import dataclass, field


@dataclass
class AgentTask:
    """A task for an agent execution (root or sub)."""

    task_id: str
    session_id: str
    task_description: str
    parent_agent: Any  # Forward reference to Agent
    parent_task_id: Optional[str] = None
    result: Optional[str] = None
    depth: int = 0
    child_task_ids: Set[str] = field(default_factory=set)
    ended_at: Optional[float] = None
    agent: Optional[Any] = None  # Reference to the agent instance for this task


@dataclass
class ChildCompletionNotification:
    """Structured notification sent to parent when a single child completes."""
    task_id: str
    label: str                    # e.g. "Sub-1(d:1)"
    task: str                     # Original task description
    status: str                   # "completed" | "error"
    summary: str                  # Free-form summary from child's last assistant message
    session_file: str             # Relative filename: "{task_id}.json"

    def to_prompt(self) -> str:
        """Format this notification as a user message for the parent agent."""
        return (
            "[Child Agent Report] {label} ({task_id}) has completed:\n"
            "- Status: {status}\n"
            "- Task: {task}\n"
            "- Summary: {summary}\n"
            "\n"
            "Use `read_session` with task_id=\"{task_id}\" to inspect the full session."
        ).format(
            label=self.label,
            task_id=self.task_id,
            status=self.status,
            task=self.task,
            summary=self.summary,
        )
