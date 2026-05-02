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
