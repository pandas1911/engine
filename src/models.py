"""Agent system data models.

This module contains all data models for the Agent system.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field


class AgentState(Enum):
    """Agent execution states."""

    IDLE = "idle"
    RUNNING = "running"
    WAITING_FOR_CHILDREN = "waiting_for_children"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class Message:
    """A message in a conversation session."""

    role: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        """Convert message to dictionary format."""
        result = {"role": self.role, "content": self.content}
        if self.role == "tool" and "tool_call_id" in self.metadata:
            result["tool_call_id"] = self.metadata["tool_call_id"]
        if self.role == "assistant" and "tool_calls" in self.metadata:
            result["tool_calls"] = self.metadata["tool_calls"]
        return result


@dataclass
class Session:
    """A conversation session with the Agent."""

    id: str
    depth: int = 0
    parent_id: Optional[str] = None
    messages: List[Message] = field(default_factory=list)

    def add_message(self, role: str, content: str, **metadata):
        """Add a message to the session."""
        self.messages.append(Message(role, content, metadata))

    def get_messages(self) -> List[Dict]:
        """Get all messages as dictionaries."""
        return [m.to_dict() for m in self.messages]


@dataclass
class ToolCall:
    """A tool call from the LLM."""

    name: str
    arguments: Dict[str, Any]
    call_id: str


@dataclass
class LLMResponse:
    """Response from an LLM provider."""

    content: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)

    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls."""
        return len(self.tool_calls) > 0


@dataclass
class SubagentTask:
    """A task for a subagent execution."""

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
class QueueEvent:
    trigger_task_id: str  # Trigger child task_id (debug/log)
    child_results: Dict[str, str]  # All child task_id → result aggregation
    error: bool


__all__ = [
    "AgentState",
    "Message",
    "Session",
    "ToolCall",
    "LLMResponse",
    "SubagentTask",
    "QueueEvent",
]
