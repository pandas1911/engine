"""Streaming response handler for Agent.

Defines BaseStreamingHandler and two concrete implementations:
SSEStreamingHandler (root agent SSE streaming) and SubAgentStreamingWrapper
(sub-agent streaming with event namespacing).
"""

from __future__ import annotations

import json
from typing import Any, Callable, List, Optional

from engine.providers.chunk_types import StreamChunk
from engine.providers.provider_models import ToolCall


class BaseStreamingHandler:
    """Shared streaming logic: part lifecycle, content accumulation, tool call buffering.
    
    Subclasses MUST implement:
    - emit(event_name, data): Event dispatch strategy
    
    Part ID allocation is unified via _next_part_id():
    - If allocate_part_id callback provided → delegate to it
    - Otherwise → increment local _part_counter (fallback)
    """

    def __init__(self, allocate_part_id: Optional[Callable[[], int]] = None):
        self._allocate_part_id = allocate_part_id
        self._part_counter: int = 0
        self._active_reasoning_part_id: Optional[int] = None
        self._active_text_part_id: Optional[int] = None
        self._collected_content: str = ""
        self._tool_call_buffers: dict[int, dict] = {}
        self._collected_tool_calls: List[ToolCall] = []

    def emit(self, event_name: str, data: Any) -> None:
        raise NotImplementedError

    def _next_part_id(self) -> int:
        if self._allocate_part_id is not None:
            return self._allocate_part_id()
        self._part_counter += 1
        return self._part_counter

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


class SSEStreamingHandler(BaseStreamingHandler):
    """SSE streaming handler for root agents.

    Extends BaseStreamingHandler with callback-based emit and spawn suppression.
    """

    def __init__(
        self,
        callback: Callable[[str, Any], None],
        allocate_part_id: Optional[Callable[[], int]] = None,
    ) -> None:
        super().__init__(allocate_part_id)
        self._callback = callback

    def emit(self, event_name: str, data: Any) -> None:
        self._callback(event_name, data)

    def on_tool_start(self, tool_name: str, arguments: dict, call_id: str) -> int:
        """Override to add spawn suppression for root agent."""
        if tool_name == "spawn":
            return 0
        return super().on_tool_start(tool_name, arguments, call_id)


class SubAgentStreamingWrapper(BaseStreamingHandler):
    """Streaming handler wrapper that namespaces events for sub-agent streams.

    Wraps a parent BaseStreamingHandler and:
    - Passes through ``subagent_*`` events without re-prefixing
    - Maps both ``agent_done`` and ``subagent_done`` to ``subagent_done``
    - Maps both ``error`` and ``subagent_error`` to ``subagent_error``
    - Prefixes other events with ``subagent_``
    - Injects ``task_id`` only when not already present in data dict
    - No spawn suppression (inherits base class behavior)
    - Uses parent's part ID counter via _next_part_id()
    """

    def __init__(
        self,
        parent: "BaseStreamingHandler",
        task_id: str,
    ) -> None:
        super().__init__(allocate_part_id=parent._next_part_id)
        self._parent = parent
        self._task_id = task_id

    def emit(self, event_name: str, data: Any) -> None:
        if event_name in ("agent_done", "subagent_done"):
            namespaced = "subagent_done"
        elif event_name in ("error", "subagent_error"):
            namespaced = "subagent_error"
        elif event_name.startswith("subagent_"):
            namespaced = event_name
        else:
            namespaced = f"subagent_{event_name}"
        if isinstance(data, dict):
            if "task_id" not in data:
                data = {**data, "task_id": self._task_id}
        self._parent.emit(namespaced, data)
