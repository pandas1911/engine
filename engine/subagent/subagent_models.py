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
class CollectedChildResult:
    """Complete information collected from child agent results, replacing downstream dependency on _tasks."""

    task_description: str
    result: str
