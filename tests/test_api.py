"""Backend API endpoint tests."""
import pytest
import pytest_asyncio
from unittest.mock import patch
from httpx import AsyncClient, ASGITransport

from app.main import app
from app._state import set_streaming


async def _mock_delegate_simple(task_description, **kwargs):
    """Mock delegate() that simulates engine events via callback."""
    event_callback = kwargs.get("event_callback")
    if event_callback:
        event_callback("llm_chunk", {"thinking_text": "", "delta_text": "Response"})
        event_callback("agent_done", {"success": True, "content": "Response"})
    from engine.runtime.agent_models import AgentResult
    return AgentResult(content="Response", success=True)


@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture(autouse=True)
def reset_streaming():
    """Reset streaming state between tests."""
    set_streaming(False)
    yield
    set_streaming(False)


# ── Health ──

@pytest.mark.asyncio
async def test_health_returns_ok(client):
    response = await client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


# ── Chat SSE ──

@pytest.mark.asyncio
@patch("engine.runner.delegate")
async def test_chat_returns_sse(mock_delegate, client):
    mock_delegate.side_effect = _mock_delegate_simple
    response = await client.post("/api/chat", json={"message": "Hi"})
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")


@pytest.mark.asyncio
@patch("engine.runner.delegate")
async def test_chat_includes_session_id_in_done(mock_delegate, client):
    mock_delegate.side_effect = _mock_delegate_simple
    response = await client.post("/api/chat", json={"message": "Hi"})
    assert response.status_code == 200
    # _event_generator overrides session_id with chat_{uuid}, verify it's present
    assert "chat_" in response.text
    assert "session_id" in response.text


# ── Concurrent request ──

@pytest.mark.asyncio
async def test_concurrent_request_rejected_429(client):
    set_streaming(True)
    response = await client.post("/api/chat", json={"message": "Hi"})
    assert response.status_code == 429
    assert "already being processed" in response.json()["error"]


# ── Request validation ──

@pytest.mark.asyncio
async def test_chat_missing_message_returns_422(client):
    response = await client.post("/api/chat", json={})
    assert response.status_code == 422


# ── Session CRUD ──

@pytest.mark.asyncio
async def test_get_session_not_found(client):
    response = await client.get("/api/sessions/nonexistent-id-12345")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_delete_session_not_found(client):
    response = await client.delete("/api/sessions/nonexistent-id-12345")
    assert response.status_code == 404


# ── CORS ──

@pytest.mark.asyncio
async def test_cors_headers_present(client):
    response = await client.get(
        "/api/health",
        headers={"Origin": "http://localhost:3000"},
    )
    assert response.status_code == 200
    assert "access-control-allow-origin" in response.headers


# ── Truncation logic ──

@pytest.mark.asyncio
async def test_truncation_removes_complete_turns():
    """Test that _truncate_session removes oldest turns correctly."""
    from app.routers.chat import _truncate_session
    from engine.runtime.agent_models import Session

    session = Session(id="trunc-test", depth=0)
    # System message (preserved)
    session.add_message("system", "System prompt")
    # Turn 1: user + assistant
    session.add_message("user", "Q1")
    session.add_message("assistant", "A1")
    # Turn 2: user + assistant (with tool calls)
    session.add_message("user", "Q2")
    session.add_message("assistant", "A2")
    session.add_message("tool", "result")
    session.add_message("assistant", "A2 final")
    # Turn 3: user + assistant
    session.add_message("user", "Q3")
    session.add_message("assistant", "A3")

    # 8 non-system messages, limit is 20 — no truncation needed
    _truncate_session(session)
    assert len(session.messages) == 9  # system + 8 non-system

    # Now test with smaller limit
    import app.routers.chat as chat_mod
    original_max = chat_mod.MAX_MESSAGES
    chat_mod.MAX_MESSAGES = 4

    try:
        _truncate_session(session)
        # Should have removed turns until <= 4 non-system messages remain
        non_system = [m for m in session.messages if m.role != "system"]
        assert len(non_system) <= 4
        # System message must be preserved
        assert session.messages[0].role == "system"
        # Latest turn should be preserved
        assert session.messages[-1].content == "A3"
    finally:
        chat_mod.MAX_MESSAGES = original_max
