"""Unit tests for SSEStreamingHandler."""

import pytest
from unittest.mock import MagicMock

from engine.providers.chunk_types import StreamChunk
from engine.providers.provider_models import ToolCall
from engine.streaming_handler import SSEStreamingHandler, BaseStreamingHandler


def _make_handler():
    events = []
    handler = SSEStreamingHandler(lambda e, d: events.append((e, d)))
    return handler, events


def _make_func_delta(name=None, arguments=None):
    fd = MagicMock()
    fd.name = name
    fd.arguments = arguments
    return fd


def _make_tc_delta(index=0, call_id=None, func_name=None, func_args=None):
    tc = MagicMock()
    tc.index = index
    tc.id = call_id
    tc.function = _make_func_delta(name=func_name, arguments=func_args)
    return tc


# 1. Protocol conformance
def test_protocol_conformance():
    handler, _ = _make_handler()
    assert isinstance(handler, BaseStreamingHandler)


# 2. Part lifecycle — reasoning only
def test_reasoning_only_lifecycle():
    handler, events = _make_handler()
    handler.on_chunk(StreamChunk(thinking_text="hmm"))
    handler.on_chunk(StreamChunk(thinking_text=" more", finish_reason="stop"))

    assert events == [
        ("part_new", {"part_id": 1, "part_type": "reasoning", "text": "hmm"}),
        ("part_delta", {"part_id": 1, "text": " more"}),
        ("part_close", {"part_id": 1}),
    ]


# 3. Part lifecycle — text only
def test_text_only_lifecycle():
    handler, events = _make_handler()
    handler.on_chunk(StreamChunk(delta_text="Hello"))
    handler.on_chunk(StreamChunk(delta_text=" World", finish_reason="stop"))

    assert events == [
        ("part_new", {"part_id": 1, "part_type": "text", "text": "Hello"}),
        ("part_delta", {"part_id": 1, "text": " World"}),
        ("part_close", {"part_id": 1}),
    ]


# 4. Reasoning→text transition in same chunk
def test_reasoning_to_text_same_chunk():
    handler, events = _make_handler()
    handler.on_chunk(StreamChunk(thinking_text="hmm", delta_text="answer"))

    assert events == [
        ("part_new", {"part_id": 1, "part_type": "reasoning", "text": "hmm"}),
        ("part_close", {"part_id": 1}),
        ("part_new", {"part_id": 2, "part_type": "text", "text": "answer"}),
    ]


# 5. Reasoning→text across chunks
def test_reasoning_to_text_across_chunks():
    handler, events = _make_handler()
    handler.on_chunk(StreamChunk(thinking_text="hmm"))
    handler.on_chunk(StreamChunk(delta_text="answer"))

    assert events == [
        ("part_new", {"part_id": 1, "part_type": "reasoning", "text": "hmm"}),
        ("part_close", {"part_id": 1}),
        ("part_new", {"part_id": 2, "part_type": "text", "text": "answer"}),
    ]


# 6. Multiple reasoning/text cycles (with finish between)
def test_multiple_reasoning_text_cycles():
    handler, events = _make_handler()

    # Cycle 1: reasoning -> text -> finish
    handler.on_chunk(StreamChunk(thinking_text="think1"))
    handler.on_chunk(StreamChunk(delta_text="text1", finish_reason="stop"))

    # Reset simulates new _get_llm_response call
    handler.reset()

    # Cycle 2: reasoning -> text -> finish
    handler.on_chunk(StreamChunk(thinking_text="think2"))
    handler.on_chunk(StreamChunk(delta_text="text2", finish_reason="stop"))

    assert events == [
        # Cycle 1
        ("part_new", {"part_id": 1, "part_type": "reasoning", "text": "think1"}),
        ("part_close", {"part_id": 1}),
        ("part_new", {"part_id": 2, "part_type": "text", "text": "text1"}),
        ("part_close", {"part_id": 2}),
        # Cycle 2 — counter keeps going
        ("part_new", {"part_id": 3, "part_type": "reasoning", "text": "think2"}),
        ("part_close", {"part_id": 3}),
        ("part_new", {"part_id": 4, "part_type": "text", "text": "text2"}),
        ("part_close", {"part_id": 4}),
    ]


# 7. Tool events
def test_tool_events():
    handler, events = _make_handler()
    part_id = handler.on_tool_start("search", {"q": "test"}, "call_abc")
    handler.on_tool_end("search", "results", "call_abc", part_id)

    assert events == [
        ("tool_start", {
            "tool_name": "search",
            "arguments": {"q": "test"},
            "call_id": "call_abc",
            "part_id": part_id,
        }),
        ("tool_end", {
            "tool_name": "search",
            "result": "results",
            "call_id": "call_abc",
            "part_id": part_id,
        }),
    ]


# 8. Part ID monotonicity
def test_part_id_monotonicity():
    handler, events = _make_handler()

    # Text part -> id 1
    handler.on_chunk(StreamChunk(delta_text="hi", finish_reason="stop"))
    # Tool event -> id 2
    pid = handler.on_tool_start("t", {}, "c1")
    assert pid == 2
    handler.reset()

    # After reset, counter keeps going -> id 3
    handler.on_chunk(StreamChunk(delta_text="yo", finish_reason="stop"))

    part_ids = [ev[1]["part_id"] for ev in events]
    assert part_ids == [1, 1, 2, 3, 3]


# 9. Reset clears part state but not counter
def test_reset_clears_part_state_not_counter():
    handler, events = _make_handler()

    handler.on_chunk(StreamChunk(thinking_text="hmm"))
    handler.on_chunk(StreamChunk(delta_text="answer"))
    handler.reset()
    # Same content types after reset should create NEW parts, not deltas
    handler.on_chunk(StreamChunk(thinking_text="new think"))
    handler.on_chunk(StreamChunk(delta_text="new answer"))

    new_events = [ev for ev in events if ev[0] == "part_new"]
    assert len(new_events) == 4
    # part_ids: 1(reasoning), 2(text from transition), 3(reasoning after reset), 4(text after reset)
    assert [ev[1]["part_id"] for ev in new_events] == [1, 2, 3, 4]


# 10. No-op with no-op callback
def test_noop_callback():
    handler = SSEStreamingHandler(lambda e, d: None)
    handler.on_chunk(StreamChunk(delta_text="Hello", thinking_text="hmm"))
    handler.on_chunk(StreamChunk(delta_text=" World", finish_reason="stop"))
    assert handler.get_content() == "Hello World"


# 11. emit() passes through to callback
def test_emit_passthrough():
    handler, events = _make_handler()
    handler.emit("agent_done", {"status": "ok"})
    handler.emit("error", {"message": "oops"})

    assert events == [
        ("agent_done", {"status": "ok"}),
        ("error", {"message": "oops"}),
    ]


# 12. Empty chunks produce no events
def test_empty_chunks():
    handler, events = _make_handler()
    handler.on_chunk(StreamChunk())
    handler.on_chunk(StreamChunk(delta_text="", thinking_text=""))
    assert events == []
    assert handler.get_content() == ""


# 13. Multiple finish_reason — no double-close
def test_no_double_close_on_multiple_finish():
    handler, events = _make_handler()
    handler.on_chunk(StreamChunk(delta_text="hi", finish_reason="stop"))
    handler.on_chunk(StreamChunk(finish_reason="stop"))

    close_events = [ev for ev in events if ev[0] == "part_close"]
    assert len(close_events) == 1


# 14. Content accumulation
def test_content_accumulation():
    handler, _ = _make_handler()
    handler.on_chunk(StreamChunk(delta_text="Hello "))
    handler.on_chunk(StreamChunk(delta_text="World"))
    handler.on_chunk(StreamChunk(delta_text="!", finish_reason="stop"))
    assert handler.get_content() == "Hello World!"


# 15. Tool call accumulation
def test_tool_call_accumulation():
    handler, _ = _make_handler()

    tc1 = _make_tc_delta(index=0, call_id="call_1", func_name="search", func_args='{"q":')
    tc2 = _make_tc_delta(index=0, call_id=None, func_name=None, func_args='"test"}')

    handler.on_chunk(StreamChunk(tool_calls=[tc1]))
    handler.on_chunk(StreamChunk(tool_calls=[tc2], finish_reason="stop"))

    calls = handler.get_tool_calls()
    assert len(calls) == 1
    assert calls[0].name == "search"
    assert calls[0].call_id == "call_1"
    assert calls[0].arguments == {"q": "test"}


# 16. Tool call JSON parsing — valid and invalid
def test_tool_call_json_parsing():
    handler, _ = _make_handler()

    # Valid JSON
    tc_valid = _make_tc_delta(index=0, call_id="c1", func_name="fn", func_args='{"a":1}')
    handler.on_chunk(StreamChunk(tool_calls=[tc_valid], finish_reason="stop"))
    assert handler.get_tool_calls()[0].arguments == {"a": 1}

    handler.reset()

    # Invalid JSON
    tc_bad = _make_tc_delta(index=0, call_id="c2", func_name="fn", func_args="not json")
    handler.on_chunk(StreamChunk(tool_calls=[tc_bad], finish_reason="stop"))
    assert handler.get_tool_calls()[0].arguments == {"raw": "not json"}


# 17. Reset clears data buffers
def test_reset_clears_data_buffers():
    handler, _ = _make_handler()
    handler.on_chunk(StreamChunk(delta_text="some text"))
    tc = _make_tc_delta(index=0, call_id="c1", func_name="fn", func_args='{}')
    handler.on_chunk(StreamChunk(tool_calls=[tc], finish_reason="stop"))

    assert handler.get_content() == "some text"
    assert len(handler.get_tool_calls()) == 1

    handler.reset()
    assert handler.get_content() == ""
    assert handler.get_tool_calls() == []


# 18. Tool calls only appear after finish_reason
def test_tool_calls_only_after_finish():
    handler, _ = _make_handler()

    tc = _make_tc_delta(index=0, call_id="c1", func_name="search", func_args='{"q":"x"}')
    handler.on_chunk(StreamChunk(tool_calls=[tc]))

    # Before finish_reason: tool calls accumulated in buffer but not parsed
    assert handler.get_tool_calls() == []

    handler.on_chunk(StreamChunk(finish_reason="stop"))

    calls = handler.get_tool_calls()
    assert len(calls) == 1
    assert calls[0].name == "search"
