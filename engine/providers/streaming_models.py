"""Streaming event types and SSE schema definitions."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ── LLM-level streaming chunk ──────────────────────────────

@dataclass
class StreamChunk:
    """A single chunk yielded by stream_chat()."""
    delta_text: str = ""           # Response text delta
    thinking_text: str = ""        # Thinking/reasoning text delta
    tool_calls: Optional[List[Any]] = None  # Partial or complete tool call deltas
    finish_reason: Optional[str] = None
    thinking_source: Optional[str] = None   # "tag_parser" | "reasoning_content" | "reasoning_details" | None


# ── SSE-level event types ──────────────────────────────────

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
class ThinkingDeltaEvent(StreamEvent):
    """Incremental thinking/reasoning text."""
    type: str = "thinking_delta"
    data: Dict[str, Any] = field(default_factory=lambda: {"text": ""})

@dataclass
class TextDeltaEvent(StreamEvent):
    """Incremental response text."""
    type: str = "text_delta"
    data: Dict[str, Any] = field(default_factory=lambda: {"text": ""})

@dataclass
class ToolCallStartEvent(StreamEvent):
    """Tool execution begins."""
    type: str = "tool_call_start"
    data: Dict[str, Any] = field(default_factory=lambda: {"tool_name": "", "arguments": {}})

@dataclass
class ToolCallResultEvent(StreamEvent):
    """Tool execution completes."""
    type: str = "tool_call_result"
    data: Dict[str, Any] = field(default_factory=lambda: {"tool_name": "", "result": ""})

@dataclass
class DoneEvent(StreamEvent):
    """Agent execution complete."""
    type: str = "done"
    data: Dict[str, Any] = field(default_factory=lambda: {"success": True, "content": ""})

@dataclass
class ErrorEvent(StreamEvent):
    """Error occurred during execution."""
    type: str = "error"
    data: Dict[str, Any] = field(default_factory=lambda: {"message": ""})
