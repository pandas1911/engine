"""LLM streaming chunk types."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class StreamChunk:
    """A single chunk yielded by stream_chat()."""
    delta_text: str = ""           # Response text delta
    thinking_text: str = ""        # Thinking/reasoning text delta
    tool_calls: Optional[List[Any]] = None  # Partial or complete tool call deltas
    finish_reason: Optional[str] = None
    thinking_source: Optional[str] = None   # "tag_parser" | "reasoning_content" | "reasoning_details" | None
