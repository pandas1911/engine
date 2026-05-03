"""Unit tests for Agent event callback pattern."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from engine.runtime.agent import Agent
from engine.runtime.agent_models import Session
from engine.config import Config
from engine.providers.chunk_types import StreamChunk
from engine.providers.provider_models import LLMResponse, ToolCall
from engine.runtime.streaming_handler import SSEStreamingHandler


def _create_test_agent(event_callback=None):
    session = Session(id="test-session", depth=0)
    config = Config()
    llm = MagicMock()
    handler = SSEStreamingHandler(event_callback) if event_callback else None
    return Agent(
        session=session,
        config=config,
        llm_provider=llm,
        streaming_handler=handler,
    )


async def _async_iter(items):
    for item in items:
        yield item


def test_emit_noop_without_callback():
    agent = _create_test_agent(event_callback=None)
    agent._emit("test_event", {"key": "value"})


def test_emit_calls_callback():
    callback = MagicMock()
    agent = _create_test_agent(event_callback=callback)
    agent._emit("test_event", {"key": "value"})
    callback.assert_called_once_with("test_event", {"key": "value"})


def test_emit_multiple_events_in_order():
    callback = MagicMock()
    agent = _create_test_agent(event_callback=callback)
    agent._emit("first", {"n": 1})
    agent._emit("second", {"n": 2})
    agent._emit("third", {"n": 3})
    assert callback.call_count == 3
    callback.assert_any_call("first", {"n": 1})
    callback.assert_any_call("second", {"n": 2})
    callback.assert_any_call("third", {"n": 3})


@pytest.mark.asyncio
async def test_delegate_passes_callback():
    callback = MagicMock()

    async def fake_delegate(*args, **kwargs):
        assert kwargs.get("event_callback") is callback
        from engine.runtime.agent_models import AgentResult, Session
        return AgentResult(content="ok", session=Session(id="s", depth=0), success=True)

    with patch("engine.runner.delegate", side_effect=fake_delegate) as mock_d:
        from engine.runner import delegate
        result = await delegate(
            task_description="test",
            event_callback=callback,
        )
        assert result.success
        mock_d.assert_called_once()


@pytest.mark.asyncio
async def test_callback_none_uses_chat_not_stream():
    agent = _create_test_agent(event_callback=None)
    agent.llm.chat = AsyncMock(return_value=LLMResponse(content="hi"))

    response = await agent._get_llm_response()

    agent.llm.chat.assert_called_once()
    agent.llm.stream_chat.assert_not_called()
    assert response.content == "hi"


@pytest.mark.asyncio
async def test_callback_present_uses_stream():
    callback = MagicMock()
    agent = _create_test_agent(event_callback=callback)

    chunks = [
        StreamChunk(delta_text="Hello ", thinking_text=""),
        StreamChunk(delta_text="World", thinking_text="", finish_reason="stop"),
    ]
    agent.llm.stream_chat = MagicMock(return_value=_async_iter(chunks))

    response = await agent._get_llm_response()

    agent.llm.stream_chat.assert_called_once()
    agent.llm.chat.assert_not_called()
    assert response.content == "Hello World"


@pytest.mark.asyncio
async def test_get_llm_response_rebuilds_tool_calls():
    callback = MagicMock()
    agent = _create_test_agent(event_callback=callback)

    func_delta_1 = MagicMock()
    func_delta_1.name = "search"
    func_delta_1.arguments = '{"q":'

    func_delta_2 = MagicMock()
    func_delta_2.name = None
    func_delta_2.arguments = '"test"}'

    tc1 = MagicMock()
    tc1.index = 0
    tc1.id = "call_abc"
    tc1.function = func_delta_1

    tc2 = MagicMock()
    tc2.index = 0
    tc2.id = None
    tc2.function = func_delta_2

    chunks = [
        StreamChunk(delta_text="", thinking_text="", tool_calls=[tc1]),
        StreamChunk(delta_text="", thinking_text="", tool_calls=[tc2], finish_reason="stop"),
    ]
    agent.llm.stream_chat = MagicMock(return_value=_async_iter(chunks))

    response = await agent._get_llm_response()

    assert len(response.tool_calls) == 1
    tc = response.tool_calls[0]
    assert tc.name == "search"
    assert tc.call_id == "call_abc"
    assert tc.arguments == {"q": "test"}


@pytest.mark.asyncio
async def test_get_llm_response_emits_chunks():
    callback = MagicMock()
    agent = _create_test_agent(event_callback=callback)

    chunks = [
        StreamChunk(delta_text="", thinking_text="hmm"),
        StreamChunk(delta_text="answer"),
    ]
    agent.llm.stream_chat = MagicMock(return_value=_async_iter(chunks))

    await agent._get_llm_response()

    assert callback.call_count == 3
    callback.assert_any_call("part_new", {"part_id": 1, "part_type": "reasoning", "text": "hmm"})
    callback.assert_any_call("part_close", {"part_id": 1})
    callback.assert_any_call("part_new", {"part_id": 2, "part_type": "text", "text": "answer"})


@pytest.mark.asyncio
async def test_dual_content_chunk_produces_correct_parts():
    """When a single chunk has both thinking and text, reasoning closes before text opens."""
    callback = MagicMock()
    agent = _create_test_agent(event_callback=callback)

    chunks = [
        StreamChunk(delta_text="answer", thinking_text="hmm"),
    ]
    agent.llm.stream_chat = MagicMock(return_value=_async_iter(chunks))

    await agent._get_llm_response()

    assert callback.call_count == 3
    # Order: part_new(reasoning) -> part_close(reasoning) -> part_new(text)
    calls = [c.args for c in callback.call_args_list]
    assert calls[0] == ("part_new", {"part_id": 1, "part_type": "reasoning", "text": "hmm"})
    assert calls[1] == ("part_close", {"part_id": 1})
    assert calls[2] == ("part_new", {"part_id": 2, "part_type": "text", "text": "answer"})


@pytest.mark.asyncio
async def test_empty_chunks_produce_no_part_events():
    """Chunks with no content should not create any Part events."""
    callback = MagicMock()
    agent = _create_test_agent(event_callback=callback)

    chunks = [
        StreamChunk(delta_text="", thinking_text=""),
        StreamChunk(delta_text="", thinking_text=""),
    ]
    agent.llm.stream_chat = MagicMock(return_value=_async_iter(chunks))

    await agent._get_llm_response()

    assert callback.call_count == 0


@pytest.mark.asyncio
async def test_finish_reason_closes_active_parts():
    """finish_reason should emit part_close for any active Parts."""
    callback = MagicMock()
    agent = _create_test_agent(event_callback=callback)

    chunks = [
        StreamChunk(delta_text="Hello"),
        StreamChunk(delta_text=" World", finish_reason="stop"),
    ]
    agent.llm.stream_chat = MagicMock(return_value=_async_iter(chunks))

    await agent._get_llm_response()

    calls = [c.args for c in callback.call_args_list]
    # part_new(text, id=1) -> part_delta(id=1) -> part_close(id=1)
    assert calls[0] == ("part_new", {"part_id": 1, "part_type": "text", "text": "Hello"})
    assert calls[1] == ("part_delta", {"part_id": 1, "text": " World"})
    assert calls[2] == ("part_close", {"part_id": 1})


@pytest.mark.asyncio
async def test_tool_events_include_part_id():
    """Tool start/end events should include part_id."""
    callback = MagicMock()
    agent = _create_test_agent(event_callback=callback)

    func_delta = MagicMock()
    func_delta.name = "search"
    func_delta.arguments = '{"q":"test"}'

    tc = MagicMock()
    tc.index = 0
    tc.id = "call_abc"
    tc.function = func_delta

    chunks = [
        StreamChunk(delta_text="", thinking_text="", tool_calls=[tc]),
        StreamChunk(delta_text="", thinking_text="", tool_calls=[tc], finish_reason="stop"),
    ]
    agent.llm.stream_chat = MagicMock(return_value=_async_iter(chunks))
    agent.llm.chat = AsyncMock(return_value=LLMResponse(content="done"))

    mock_tool = MagicMock()
    mock_tool.name = "search"
    agent._tool_pack = MagicMock()
    agent._tool_pack.get.return_value = mock_tool
    mock_tool.execute = AsyncMock(return_value="search results")

    await agent._process_tool_calls()

    tool_start_calls = [c for c in callback.call_args_list if c.args[0] == "tool_start"]
    tool_end_calls = [c for c in callback.call_args_list if c.args[0] == "tool_end"]
    assert len(tool_start_calls) >= 1
    assert len(tool_end_calls) >= 1
    assert "part_id" in tool_start_calls[0].args[1]
    assert "part_id" in tool_end_calls[0].args[1]
    assert tool_start_calls[0].args[1]["part_id"] == tool_end_calls[0].args[1]["part_id"]
