"""Provider data models for LLM interactions."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field


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


__all__ = [
    "ToolCall",
    "LLMResponse",
]
