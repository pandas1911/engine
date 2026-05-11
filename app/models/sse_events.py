"""SSE event type definitions for Part-based frontend streaming."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class StreamEvent:
    """Base SSE event."""
    type: str
    data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentStartEvent(StreamEvent):
    """Agent begins execution."""
    type: str = "agent_start"

@dataclass
class PartNewEvent(StreamEvent):
    """New content Part starts."""
    type: str = "part_new"
    data: Dict[str, Any] = field(default_factory=lambda: {"part_id": 0, "part_type": "text", "text": ""})

@dataclass
class PartDeltaEvent(StreamEvent):
    """Incremental text appended to an open Part."""
    type: str = "part_delta"
    data: Dict[str, Any] = field(default_factory=lambda: {"part_id": 0, "text": ""})

@dataclass
class PartCloseEvent(StreamEvent):
    """Content Part ends."""
    type: str = "part_close"
    data: Dict[str, Any] = field(default_factory=lambda: {"part_id": 0})

@dataclass
class ToolCallStartEvent(StreamEvent):
    """Tool execution begins."""
    type: str = "tool_call_start"
    data: Dict[str, Any] = field(default_factory=lambda: {"part_id": 0, "tool_name": "", "arguments": {}, "call_id": ""})

@dataclass
class ToolCallResultEvent(StreamEvent):
    """Tool execution completes."""
    type: str = "tool_call_result"
    data: Dict[str, Any] = field(default_factory=lambda: {"part_id": 0, "tool_name": "", "result": "", "call_id": ""})

@dataclass
class DoneEvent(StreamEvent):
    """Agent execution complete."""
    type: str = "done"
    data: Dict[str, Any] = field(default_factory=lambda: {"success": True, "session_id": ""})

@dataclass
class ErrorEvent(StreamEvent):
    """Error occurred during execution."""
    type: str = "error"
    data: Dict[str, Any] = field(default_factory=lambda: {"message": "", "session_id": ""})


# --- Sub-agent events ---

@dataclass
class SubAgentStartEvent(StreamEvent):
    """Sub-agent begins execution."""
    type: str = "subagent_start"
    data: Dict[str, Any] = field(default_factory=lambda: {"part_id": 0, "task_id": "", "label": "", "description": ""})

@dataclass
class SubAgentPartNewEvent(StreamEvent):
    """New content Part starts within a sub-agent."""
    type: str = "subagent_part_new"
    data: Dict[str, Any] = field(default_factory=lambda: {"part_id": 0, "task_id": "", "part_type": "text", "text": ""})

@dataclass
class SubAgentPartDeltaEvent(StreamEvent):
    """Incremental text appended to a sub-agent Part."""
    type: str = "subagent_part_delta"
    data: Dict[str, Any] = field(default_factory=lambda: {"part_id": 0, "task_id": "", "text": ""})

@dataclass
class SubAgentPartCloseEvent(StreamEvent):
    """Sub-agent content Part ends."""
    type: str = "subagent_part_close"
    data: Dict[str, Any] = field(default_factory=lambda: {"part_id": 0, "task_id": ""})

@dataclass
class SubAgentToolStartEvent(StreamEvent):
    """Tool execution begins within a sub-agent."""
    type: str = "subagent_tool_start"
    data: Dict[str, Any] = field(default_factory=lambda: {"part_id": 0, "task_id": "", "tool_name": "", "arguments": {}, "call_id": ""})

@dataclass
class SubAgentToolResultEvent(StreamEvent):
    """Tool execution completes within a sub-agent."""
    type: str = "subagent_tool_result"
    data: Dict[str, Any] = field(default_factory=lambda: {"part_id": 0, "task_id": "", "tool_name": "", "result": "", "call_id": ""})

@dataclass
class SubAgentDoneEvent(StreamEvent):
    """Sub-agent execution complete."""
    type: str = "subagent_done"
    data: Dict[str, Any] = field(default_factory=lambda: {"task_id": "", "success": True})

@dataclass
class SubAgentErrorEvent(StreamEvent):
    """Error occurred during sub-agent execution."""
    type: str = "subagent_error"
    data: Dict[str, Any] = field(default_factory=lambda: {"task_id": "", "message": ""})

@dataclass
class WaitingForChildrenEvent(StreamEvent):
    type: str = "waiting_for_children"
    data: Dict[str, Any] = field(default_factory=lambda: {"session_id": ""})
