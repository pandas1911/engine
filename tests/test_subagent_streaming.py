"""Unit tests for SubAgentStreamingWrapper and sub-agent streaming behavior."""

from engine.providers.chunk_types import StreamChunk
from engine.runtime.streaming_handler import SSEStreamingHandler, SubAgentStreamingWrapper


def _make_handler():
    events = []
    handler = SSEStreamingHandler(lambda e, d: events.append((e, d)))
    return handler, events


def _make_wrapper(task_id="task_test"):
    events = []
    handler, _ = _make_handler()
    wrapper = SubAgentStreamingWrapper(
        emit=lambda e, d: events.append((e, d)),
        task_id=task_id,
        allocate_part_id=handler._next_part_id,
    )
    return wrapper, events, handler


# 1. Part ID uniqueness across root and wrapper sharing same counter
def test_part_id_uniqueness():
    handler, _ = _make_handler()
    wrapper = SubAgentStreamingWrapper(
        emit=lambda e, d: None,
        task_id="task_test",
        allocate_part_id=handler._next_part_id,
    )

    root_ids = [handler._next_part_id() for _ in range(3)]
    wrapper_ids = [wrapper._allocate_part_id() for _ in range(3)]
    extra_root = handler._next_part_id()

    all_ids = root_ids + wrapper_ids + [extra_root]
    assert len(set(all_ids)) == 7
    assert set(all_ids) == {1, 2, 3, 4, 5, 6, 7}


# 2. Wrapper namespaces all events with subagent_ prefix and injects task_id
def test_wrapper_namespaces_events():
    wrapper, events, _ = _make_wrapper(task_id="task_test")

    wrapper.emit("part_new", {"part_id": 1, "part_type": "text", "text": "hi"})
    assert events[-1] == ("subagent_part_new", {
        "part_id": 1, "part_type": "text", "text": "hi", "task_id": "task_test",
    })

    wrapper.emit("agent_done", {"success": True})
    assert events[-1] == ("subagent_done", {"success": True, "task_id": "task_test"})

    for ev_name, _ in events:
        assert ev_name.startswith("subagent_"), f"Event '{ev_name}' lacks subagent_ prefix"


# 3. Wrapper converts error -> subagent_error (never raw error)
def test_wrapper_emits_subagent_error_not_error():
    wrapper, events, _ = _make_wrapper(task_id="task_err")

    wrapper.emit("error", {"message": "oops"})

    assert events[-1] == ("subagent_error", {"message": "oops", "task_id": "task_err"})
    error_events = [e for e in events if e[0] == "error"]
    assert len(error_events) == 0


# 4. Wrapper converts agent_done -> subagent_done (never raw done)
def test_wrapper_emits_subagent_done_not_done():
    wrapper, events, _ = _make_wrapper(task_id="task_done")

    wrapper.emit("agent_done", {"success": True})

    assert events[-1] == ("subagent_done", {"success": True, "task_id": "task_done"})
    done_events = [e for e in events if e[0] == "done"]
    assert len(done_events) == 0


# 5. Spawn tool suppression on SSEStreamingHandler
def test_spawn_tool_suppression():
    handler, events = _make_handler()

    result = handler.on_tool_start("spawn", {"task": "test"}, "call_1")
    assert result == 0
    assert len(events) == 0

    pid = handler.on_tool_start("web_search", {"query": "test"}, "call_2")
    assert pid == 1
    assert len(events) == 1
    assert events[0][0] == "tool_start"


# 6. Wrapper on_chunk produces namespaced events with task_id
def test_wrapper_on_chunk_produces_namespaced_events():
    wrapper, events, _ = _make_wrapper(task_id="chunk_test")

    wrapper.on_chunk(StreamChunk(thinking_text="hmm"))
    wrapper.on_chunk(StreamChunk(delta_text="answer"))

    assert len(events) == 3

    assert events[0][0] == "subagent_part_new"
    assert events[0][1]["part_type"] == "reasoning"
    assert events[0][1]["task_id"] == "chunk_test"

    assert events[1][0] == "subagent_part_close"
    assert events[1][1]["task_id"] == "chunk_test"

    assert events[2][0] == "subagent_part_new"
    assert events[2][1]["part_type"] == "text"
    assert events[2][1]["task_id"] == "chunk_test"

    for _, data in events:
        assert "task_id" in data


# 7. Wrapper tool events are namespaced
def test_wrapper_tool_events():
    wrapper, events, _ = _make_wrapper(task_id="tool_test")

    part_id = wrapper.on_tool_start("search", {"q": "test"}, "call_1")
    wrapper.on_tool_end("search", "results", "call_1", part_id)

    assert events[0][0] == "subagent_tool_start"
    assert events[0][1]["task_id"] == "tool_test"
    assert events[0][1]["part_id"] == part_id

    assert events[1][0] == "subagent_tool_end"
    assert events[1][1]["task_id"] == "tool_test"
    assert events[1][1]["part_id"] == part_id


# 8. Wrapper reset clears state but counter keeps incrementing
def test_wrapper_reset_clears_state_not_counter():
    wrapper, events, handler = _make_wrapper(task_id="reset_test")

    wrapper.on_chunk(StreamChunk(thinking_text="hmm"))
    wrapper.reset()
    wrapper.on_chunk(StreamChunk(thinking_text="new think"))

    new_events = [e for e in events if e[0] == "subagent_part_new"]
    assert len(new_events) == 2
    assert new_events[0][1]["part_id"] == 1
    assert new_events[1][1]["part_id"] == 2


# 9. Wrapper content accumulation
def test_wrapper_content_accumulation():
    wrapper, _, _ = _make_wrapper()

    wrapper.on_chunk(StreamChunk(delta_text="Hello "))
    wrapper.on_chunk(StreamChunk(delta_text="World"))

    assert wrapper.get_content() == "Hello World"


# 10. Backward compat: handler without allocate_part_id works identically
def test_backward_compat_handler_without_allocate_part_id():
    events = []
    handler = SSEStreamingHandler(lambda e, d: events.append((e, d)))

    assert handler._next_part_id() == 1
    assert handler._next_part_id() == 2
    assert handler._next_part_id() == 3

    handler.on_chunk(StreamChunk(delta_text="hi"))
    handler.on_chunk(StreamChunk(delta_text=" there", finish_reason="stop"))

    assert events == [
        ("part_new", {"part_id": 4, "part_type": "text", "text": "hi"}),
        ("part_delta", {"part_id": 4, "text": " there"}),
        ("part_close", {"part_id": 4}),
    ]


# 11. Two wrappers sharing same root counter produce non-overlapping part IDs
def test_concurrent_subagents_unique_ids():
    handler, _ = _make_handler()
    events_a = []
    events_b = []
    wrapper_a = SubAgentStreamingWrapper(
        emit=lambda e, d: events_a.append((e, d)),
        task_id="task_a",
        allocate_part_id=handler._next_part_id,
    )
    wrapper_b = SubAgentStreamingWrapper(
        emit=lambda e, d: events_b.append((e, d)),
        task_id="task_b",
        allocate_part_id=handler._next_part_id,
    )

    ids_a = [wrapper_a._allocate_part_id() for _ in range(3)]
    ids_b = [wrapper_b._allocate_part_id() for _ in range(3)]
    extra = handler._next_part_id()

    all_ids = ids_a + ids_b + [extra]
    assert len(set(all_ids)) == len(all_ids), f"Overlapping IDs: {all_ids}"
    assert sorted(all_ids) == [1, 2, 3, 4, 5, 6, 7]


# 12. Full event sequence through wrapper produces correct order
def test_subagent_event_sequence():
    wrapper, events, _ = _make_wrapper(task_id="seq_test")

    # Thinking chunk → subagent_part_new (reasoning)
    wrapper.on_chunk(StreamChunk(thinking_text="hmm"))
    # More thinking → subagent_part_delta
    wrapper.on_chunk(StreamChunk(thinking_text=" more"))
    # Text chunk closes reasoning, opens text → subagent_part_close + subagent_part_new
    wrapper.on_chunk(StreamChunk(delta_text="answer"))
    # Tool start → subagent_tool_start
    part_id = wrapper.on_tool_start("search", {"q": "test"}, "call_1")
    # Tool end → subagent_tool_end
    wrapper.on_tool_end("search", "results", "call_1", part_id)
    # Agent done → subagent_done
    wrapper.emit("agent_done", {"success": True})

    event_names = [e[0] for e in events]
    assert event_names == [
        "subagent_part_new",
        "subagent_part_delta",
        "subagent_part_close",
        "subagent_part_new",
        "subagent_tool_start",
        "subagent_tool_end",
        "subagent_done",
    ]

    for _, data in events:
        assert "task_id" in data
        assert data["task_id"] == "seq_test"


# 13. Wrapper allows spawn tool calls through (no longer suppressed)
def test_spawn_tool_suppression_in_wrapper():
    wrapper, events, _ = _make_wrapper(task_id="spawn_test")

    result = wrapper.on_tool_start("spawn", {"task": "sub-task"}, "call_x")
    assert result > 0
    assert len(events) == 1
    assert events[0][0] == "subagent_tool_start"
    assert events[0][1]["tool_name"] == "spawn"

    pid = wrapper.on_tool_start("web_search", {"query": "test"}, "call_y")
    assert pid > 0
    assert len(events) == 2
    assert events[1][0] == "subagent_tool_start"
    assert events[1][1]["tool_name"] == "web_search"


# 14. spawn.py uses "unknown" as the default label
def test_label_default_is_unknown():
    import inspect
    from engine.subagent import spawn as spawn_mod

    source = inspect.getsource(spawn_mod.SpawnTool.execute)
    assert '"unknown"' in source, 'Default label "unknown" not found in SpawnHandler.execute'


# 15. Wrapper on_tool_start("spawn", ...) emits events normally (no longer suppressed)
def test_wrapper_spawn_tool_suppression():
    wrapper, events, _ = _make_wrapper(task_id="suppress_test")

    result = wrapper.on_tool_start("spawn", {"task": "anything"}, "call_spawn_1")
    assert result > 0
    assert len(events) == 1
    assert events[0][0] == "subagent_tool_start"
    assert events[0][1]["tool_name"] == "spawn"
