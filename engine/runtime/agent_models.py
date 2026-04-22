"""Agent runtime data models."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from engine.subagent.subagent_models import CollectedChildResult


class AgentState(Enum):
    """Agent execution states."""

    IDLE = "idle"
    RUNNING = "running"
    WAITING_FOR_CHILDREN = "waiting_for_children"
    COMPLETED = "completed"
    ERROR = "error"


class ErrorCategory(Enum):
    """Error categories — extend as needed."""
    LLM_ERROR = "llm_error"
    INTERNAL_ERROR = "internal_error"


@dataclass
class AgentError:
    """Structured error information for programmatic consumers."""
    category: ErrorCategory
    message: str
    exception_type: Optional[str] = None

    def __str__(self) -> str:
        if self.exception_type:
            return "[{}] {} - {}".format(self.category.value, self.exception_type, self.message)
        return "[{}] {}".format(self.category.value, self.message)


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
class QueueEvent:
    trigger_task_id: str  # Trigger child task_id (debug/log)
    child_results: Dict[str, CollectedChildResult]  # All child task_id → enriched result
    error: bool


@dataclass
class AgentResult:
    """Result from agent execution."""

    content: str
    session: Session
    success: bool
    error: Optional[AgentError] = None


__all__ = [
    "AgentState",
    "ErrorCategory",
    "AgentError",
    "Message",
    "Session",
    "QueueEvent",
    "AgentResult",
]
