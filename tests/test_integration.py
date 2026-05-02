"""End-to-end integration tests for the chat API.

Tests the complete HTTP → SSE stream → event verification → session persistence
flow using mocked delegate() to avoid real LLM API calls.
"""

import json
from typing import Any
import pytest
import pytest_asyncio
from unittest.mock import patch
from httpx import AsyncClient, ASGITransport

from app.main import app
from app._state import set_streaming
from engine.runtime.agent_models import AgentResult, Session


async def _mock_delegate_success(task_description, **kwargs):
    event_callback = kwargs.get("event_callback")
    if event_callback:
        event_callback("llm_chunk", {"thinking_text": "Let me think...", "delta_text": ""})
        event_callback("llm_chunk", {"thinking_text": "", "delta_text": "Hello "})
        event_callback("llm_chunk", {"thinking_text": "", "delta_text": "World"})
        event_callback("agent_done", {"success": True, "content": "Hello World"})
    return AgentResult(
        content="Hello World",
        session=kwargs.get("session", Session(id="test-session-1", depth=0)),
        success=True,
    )


async def _mock_delegate_with_tools(task_description, **kwargs):
    event_callback = kwargs.get("event_callback")
    if event_callback:
        event_callback("llm_chunk", {"thinking_text": "", "delta_text": ""})
        event_callback("tool_start", {"tool_name": "web_search", "arguments": {"query": "Python"}, "call_id": "call_123"})
        event_callback("tool_end", {"tool_name": "web_search", "result": "Python is great", "call_id": "call_123"})
        event_callback("llm_chunk", {"thinking_text": "", "delta_text": "Based on search results..."})
        event_callback("agent_done", {"success": True, "content": "Based on search results..."})
    return AgentResult(
        content="Based on search results...",
        session=kwargs.get("session", Session(id="test-session-2", depth=0)),
        success=True,
    )


async def _mock_delegate_error(task_description, **kwargs):
    event_callback = kwargs.get("event_callback")
    if event_callback:
        event_callback("error", {"message": "Test error"})
    return AgentResult(
        content="",
        session=kwargs.get("session", Session(id="test-session-3", depth=0)),
        success=False,
        error=None,
    )


@pytest_asyncio.fixture(autouse=True)
async def _reset_state():
    set_streaming(False)
    yield
    set_streaming(False)


@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


def _parse_sse_events(raw: str) -> list[dict[str, Any]]:
    """Parse raw SSE text into structured events."""
    events = []
    current_event = None
    for line in raw.split("\n"):
        line = line.strip()
        if line.startswith("event: "):
            current_event = line[7:]
        elif line.startswith("data: ") and current_event is not None:
            events.append({"event": current_event, "data": json.loads(line[6:])})
            current_event = None
    return events


@pytest.mark.asyncio
async def test_health_endpoint(client):
    """GET /api/health returns status ok."""
    response = await client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
@patch("engine.runner.delegate")
async def test_chat_sse_stream_format(mock_delegate, client):
    """POST /api/chat returns SSE stream with correct event types and ordering."""
    mock_delegate.side_effect = _mock_delegate_success

    response = await client.post("/api/chat", json={"message": "Hello"})
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")

    events = _parse_sse_events(response.text)

    event_types = [e["event"] for e in events]
    # agent_start is emitted by _event_generator, not delegate
    assert event_types[0] == "agent_start"
    assert "thinking_delta" in event_types
    assert "text_delta" in event_types
    assert event_types[-1] == "done"

    thinking_idx = event_types.index("thinking_delta")
    text_idx = event_types.index("text_delta")
    assert thinking_idx < text_idx


@pytest.mark.asyncio
@patch("engine.runner.delegate")
async def test_chat_thinking_deltas(mock_delegate, client):
    """Thinking delta events carry reasoning text."""
    mock_delegate.side_effect = _mock_delegate_success

    response = await client.post("/api/chat", json={"message": "Hello"})
    assert response.status_code == 200

    events = _parse_sse_events(response.text)
    thinking_events = [e for e in events if e["event"] == "thinking_delta"]
    assert len(thinking_events) >= 1
    assert thinking_events[0]["data"]["text"] == "Let me think..."


@pytest.mark.asyncio
@patch("engine.runner.delegate")
async def test_chat_tool_calls(mock_delegate, client):
    """Tool call events appear in correct start -> result sequence with matching call_id."""
    mock_delegate.side_effect = _mock_delegate_with_tools

    response = await client.post("/api/chat", json={"message": "Search Python"})
    assert response.status_code == 200

    events = _parse_sse_events(response.text)
    event_types = [e["event"] for e in events]

    assert "tool_call_start" in event_types
    assert "tool_call_result" in event_types

    assert event_types.index("tool_call_start") < event_types.index("tool_call_result")

    tool_start = next(e for e in events if e["event"] == "tool_call_start")
    assert tool_start["data"]["tool_name"] == "web_search"
    assert tool_start["data"]["call_id"] == "call_123"

    tool_result = next(e for e in events if e["event"] == "tool_call_result")
    assert tool_result["data"]["call_id"] == "call_123"


@pytest.mark.asyncio
@patch("engine.runner.delegate")
async def test_chat_error_handling(mock_delegate, client):
    """Error events are surfaced when delegate emits them."""
    mock_delegate.side_effect = _mock_delegate_error

    response = await client.post("/api/chat", json={"message": "Hello"})
    assert response.status_code == 200

    events = _parse_sse_events(response.text)
    error_events = [e for e in events if e["event"] == "error"]
    assert len(error_events) >= 1
    assert "Test error" in error_events[0]["data"]["message"]


@pytest.mark.asyncio
@patch("engine.runner.delegate")
async def test_chat_multi_turn_session(mock_delegate, client):
    """Sending session_id on a second request continues the same session."""
    mock_delegate.side_effect = _mock_delegate_success

    r1 = await client.post("/api/chat", json={"message": "Hello"})
    assert r1.status_code == 200

    events1 = _parse_sse_events(r1.text)
    done1 = next(e for e in events1 if e["event"] == "done")
    session_id = done1["data"]["session_id"]
    assert session_id

    r2 = await client.post(
        "/api/chat",
        json={"message": "Follow-up", "session_id": session_id},
    )
    assert r2.status_code == 200

    events2 = _parse_sse_events(r2.text)
    done2 = next(e for e in events2 if e["event"] == "done")
    assert done2["data"]["session_id"] == session_id


@pytest.mark.asyncio
async def test_chat_concurrent_rejection(client):
    """Concurrent requests are rejected with HTTP 429."""
    set_streaming(True)

    response = await client.post("/api/chat", json={"message": "Hello"})
    assert response.status_code == 429
    body = response.json()
    assert "error" in body
