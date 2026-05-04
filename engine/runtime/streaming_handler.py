"""Streaming response handler for Agent.

Defines the StreamingHandler Protocol and SSEStreamingHandler implementation
that encapsulates ALL streaming-specific concerns: part lifecycle events,
LLM response data accumulation (content + tool_calls), and event emission.
"""

from __future__ import annotations

import json
from typing import Any, Callable, List, Optional, Protocol, runtime_checkable

from engine.providers.chunk_types import StreamChunk
from engine.providers.provider_models import ToolCall


@runtime_checkable
class StreamingHandler(Protocol):
    """Protocol for handling streaming responses from an Agent.

    Implementations own ALL streaming-specific state and logic:
    - Part lifecycle (part_new/part_delta/part_close events)
    - LLM response data accumulation (content text + tool_call buffers)
    - Tool execution events (tool_start/tool_end)
    - Generic event passthrough (agent_done, error)

    The handler IS the streaming path — without a handler, the Agent uses
    the non-streaming llm.chat() API directly.
    """

    def emit(self, event_name: str, data: Any) -> None:
        """Emit a generic event (e.g. agent_done, error)."""
        ...

    def on_chunk(self, chunk: StreamChunk) -> None:
        """Process a streaming chunk.

        Manages part_new/part_delta/part_close events internally.
        Also accumulates content text and tool_call buffers for get_content()/get_tool_calls().
        """
        ...

    def get_content(self) -> str:
        """Return the accumulated text content across all chunks."""
        ...

    def get_tool_calls(self) -> List[ToolCall]:
        """Return the parsed tool calls accumulated from chunks.

        Should be called after the stream ends (after the last chunk with finish_reason).
        """
        ...

    def on_tool_start(self, tool_name: str, arguments: dict, call_id: str) -> int:
        """Emit tool_start event and return the assigned part_id."""
        ...

    def on_tool_end(self, tool_name: str, result: str, call_id: str, part_id: int) -> None:
        """Emit tool_end event."""
        ...

    def reset(self) -> None:
        """Reset per-call state (part IDs + data buffers). Does NOT reset the counter."""
        ...


class SSEStreamingHandler:
    """Concrete StreamingHandler for SSE (Server-Sent Events) streaming.

    Wraps a Callable[[str, Any], None] callback and manages:
    - Part lifecycle state machine (reasoning, text, tool parts)
    - LLM response data accumulation (content + tool_calls)
    - Event emission via the callback

    The _part_counter is monotonic across the handler's lifetime, ensuring
    globally unique part IDs even across multiple _get_llm_response calls.
    """

    def __init__(
        self,
        callback: Callable[[str, Any], None],
        allocate_part_id: Optional[Callable[[], int]] = None,
    ) -> None:
        self._callback = callback
        self._allocate_part_id = allocate_part_id
        self._part_counter: int = 0
        self._active_reasoning_part_id: Optional[int] = None
        self._active_text_part_id: Optional[int] = None
        self._collected_content: str = ""
        self._tool_call_buffers: dict[int, dict] = {}
        self._collected_tool_calls: List[ToolCall] = []

    def _next_part_id(self) -> int:
        if self._allocate_part_id is not None:
            return self._allocate_part_id()
        self._part_counter += 1
        return self._part_counter

    def emit(self, event_name: str, data: Any) -> None:
        self._callback(event_name, data)

    def on_chunk(self, chunk: StreamChunk) -> None:
        thinking = chunk.thinking_text or ""
        delta = chunk.delta_text or ""

        if thinking:
            if self._active_reasoning_part_id is None:
                part_id = self._next_part_id()
                self._active_reasoning_part_id = part_id
                self.emit("part_new", {
                    "part_id": part_id,
                    "part_type": "reasoning",
                    "text": thinking,
                })
            else:
                self.emit("part_delta", {
                    "part_id": self._active_reasoning_part_id,
                    "text": thinking,
                })

        # Close reasoning part when transitioning to text
        if delta and self._active_reasoning_part_id is not None:
            self.emit("part_close", {
                "part_id": self._active_reasoning_part_id,
            })
            self._active_reasoning_part_id = None

        if delta:
            if self._active_text_part_id is None:
                part_id = self._next_part_id()
                self._active_text_part_id = part_id
                self.emit("part_new", {
                    "part_id": part_id,
                    "part_type": "text",
                    "text": delta,
                })
            else:
                self.emit("part_delta", {
                    "part_id": self._active_text_part_id,
                    "text": delta,
                })

        if chunk.finish_reason:
            if self._active_reasoning_part_id is not None:
                self.emit("part_close", {
                    "part_id": self._active_reasoning_part_id,
                })
                self._active_reasoning_part_id = None
            if self._active_text_part_id is not None:
                self.emit("part_close", {
                    "part_id": self._active_text_part_id,
                })
                self._active_text_part_id = None

        # Accumulate content
        self._collected_content += chunk.delta_text or ""

        # Accumulate tool call deltas
        if chunk.tool_calls:
            for tc_delta in chunk.tool_calls:
                idx = tc_delta.index
                if idx not in self._tool_call_buffers:
                    self._tool_call_buffers[idx] = {"name": "", "arguments": "", "call_id": ""}
                if hasattr(tc_delta, 'id') and tc_delta.id:
                    self._tool_call_buffers[idx]["call_id"] = tc_delta.id
                if hasattr(tc_delta, 'function') and tc_delta.function:
                    if tc_delta.function.name:
                        self._tool_call_buffers[idx]["name"] += tc_delta.function.name
                    if tc_delta.function.arguments:
                        self._tool_call_buffers[idx]["arguments"] += tc_delta.function.arguments

        # Parse complete tool calls on finish
        if chunk.finish_reason:
            self._collected_tool_calls.clear()
            for idx in sorted(self._tool_call_buffers.keys()):
                buf = self._tool_call_buffers[idx]
                args: dict = {}
                if buf["arguments"]:
                    try:
                        args = json.loads(buf["arguments"])
                    except json.JSONDecodeError:
                        args = {"raw": buf["arguments"]}
                self._collected_tool_calls.append(ToolCall(
                    name=buf["name"],
                    arguments=args,
                    call_id=buf["call_id"],
                ))

    def get_content(self) -> str:
        return self._collected_content

    def get_tool_calls(self) -> List[ToolCall]:
        return self._collected_tool_calls

    def on_tool_start(self, tool_name: str, arguments: dict, call_id: str) -> int:
        if tool_name == "spawn":
            return 0
        part_id = self._next_part_id()
        self.emit("tool_start", {
            "tool_name": tool_name,
            "arguments": arguments,
            "call_id": call_id,
            "part_id": part_id,
        })
        return part_id

    def on_tool_end(self, tool_name: str, result: str, call_id: str, part_id: int) -> None:
        self.emit("tool_end", {
            "tool_name": tool_name,
            "result": result,
            "call_id": call_id,
            "part_id": part_id,
        })

    def reset(self) -> None:
        self._active_reasoning_part_id = None
        self._active_text_part_id = None
        self._collected_content = ""
        self._tool_call_buffers = {}
        self._collected_tool_calls = []


class SubAgentStreamingWrapper:
    """StreamingHandler wrapper that namespaces events for sub-agent streams.

    Wraps a parent SSEStreamingHandler's emit callback and:
    - Prefixes all event names with ``subagent_``
    - Injects ``task_id`` into every emitted data dict
    - Converts ``agent_done`` -> ``subagent_done`` and ``error`` -> ``subagent_error``
    - Uses a shared part counter via the ``allocate_part_id`` callback
    - Mirrors SSEStreamingHandler's accumulation logic exactly
    """

    def __init__(
        self,
        emit: Callable[[str, Any], None],
        task_id: str,
        allocate_part_id: Callable[[], int],
    ) -> None:
        self._emit = emit
        self._task_id = task_id
        self._allocate_part_id = allocate_part_id
        self._active_reasoning_part_id: Optional[int] = None
        self._active_text_part_id: Optional[int] = None
        self._collected_content: str = ""
        self._tool_call_buffers: dict[int, dict] = {}
        self._collected_tool_calls: List[ToolCall] = []

    def emit(self, event_name: str, data: Any) -> None:
        if event_name == "agent_done":
            namespaced = "subagent_done"
        elif event_name == "error":
            namespaced = "subagent_error"
        else:
            namespaced = f"subagent_{event_name}"
        if isinstance(data, dict):
            data = {**data, "task_id": self._task_id}
        self._emit(namespaced, data)

    def on_chunk(self, chunk: StreamChunk) -> None:
        thinking = chunk.thinking_text or ""
        delta = chunk.delta_text or ""

        if thinking:
            if self._active_reasoning_part_id is None:
                part_id = self._allocate_part_id()
                self._active_reasoning_part_id = part_id
                self.emit("part_new", {
                    "part_id": part_id,
                    "part_type": "reasoning",
                    "text": thinking,
                })
            else:
                self.emit("part_delta", {
                    "part_id": self._active_reasoning_part_id,
                    "text": thinking,
                })

        # Close reasoning part when transitioning to text
        if delta and self._active_reasoning_part_id is not None:
            self.emit("part_close", {
                "part_id": self._active_reasoning_part_id,
            })
            self._active_reasoning_part_id = None

        if delta:
            if self._active_text_part_id is None:
                part_id = self._allocate_part_id()
                self._active_text_part_id = part_id
                self.emit("part_new", {
                    "part_id": part_id,
                    "part_type": "text",
                    "text": delta,
                })
            else:
                self.emit("part_delta", {
                    "part_id": self._active_text_part_id,
                    "text": delta,
                })

        if chunk.finish_reason:
            if self._active_reasoning_part_id is not None:
                self.emit("part_close", {
                    "part_id": self._active_reasoning_part_id,
                })
                self._active_reasoning_part_id = None
            if self._active_text_part_id is not None:
                self.emit("part_close", {
                    "part_id": self._active_text_part_id,
                })
                self._active_text_part_id = None

        # Accumulate content
        self._collected_content += chunk.delta_text or ""

        # Accumulate tool call deltas
        if chunk.tool_calls:
            for tc_delta in chunk.tool_calls:
                idx = tc_delta.index
                if idx not in self._tool_call_buffers:
                    self._tool_call_buffers[idx] = {"name": "", "arguments": "", "call_id": ""}
                if hasattr(tc_delta, 'id') and tc_delta.id:
                    self._tool_call_buffers[idx]["call_id"] = tc_delta.id
                if hasattr(tc_delta, 'function') and tc_delta.function:
                    if tc_delta.function.name:
                        self._tool_call_buffers[idx]["name"] += tc_delta.function.name
                    if tc_delta.function.arguments:
                        self._tool_call_buffers[idx]["arguments"] += tc_delta.function.arguments

        # Parse complete tool calls on finish
        if chunk.finish_reason:
            self._collected_tool_calls.clear()
            for idx in sorted(self._tool_call_buffers.keys()):
                buf = self._tool_call_buffers[idx]
                args: dict = {}
                if buf["arguments"]:
                    try:
                        args = json.loads(buf["arguments"])
                    except json.JSONDecodeError:
                        args = {"raw": buf["arguments"]}
                self._collected_tool_calls.append(ToolCall(
                    name=buf["name"],
                    arguments=args,
                    call_id=buf["call_id"],
                ))

    def get_content(self) -> str:
        return self._collected_content

    def get_tool_calls(self) -> List[ToolCall]:
        return self._collected_tool_calls

    def on_tool_start(self, tool_name: str, arguments: dict, call_id: str) -> int:
        if tool_name == "spawn":
            return 0
        part_id = self._allocate_part_id()
        self.emit("tool_start", {
            "tool_name": tool_name,
            "arguments": arguments,
            "call_id": call_id,
            "part_id": part_id,
        })
        return part_id

    def on_tool_end(self, tool_name: str, result: str, call_id: str, part_id: int) -> None:
        self.emit("tool_end", {
            "tool_name": tool_name,
            "result": result,
            "call_id": call_id,
            "part_id": part_id,
        })

    def reset(self) -> None:
        self._active_reasoning_part_id = None
        self._active_text_part_id = None
        self._collected_content = ""
        self._tool_call_buffers = {}
        self._collected_tool_calls = []
