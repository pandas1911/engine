"""Unit tests for Agent event callback pattern."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from engine.runtime.agent import Agent
from engine.runtime.agent_models import Session
from engine.config import Config
from engine.providers.chunk_types import StreamChunk
from engine.providers.provider_models import LLMResponse, ToolCall


def _create_test_agent(event_callback=None):
    session = Session(id="test-session", depth=0)
    config = Config()
    llm = MagicMock()
    return Agent(
        session=session,
        config=config,
        llm_provider=llm,
        event_callback=event_callback,
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

    assert callback.call_count == 2
    callback.assert_any_call("llm_chunk", {"thinking_text": "hmm", "delta_text": ""})
    callback.assert_any_call("llm_chunk", {"thinking_text": "", "delta_text": "answer"})
