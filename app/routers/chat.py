"""Chat SSE endpoint."""
import json
import uuid
from typing import Optional

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from app.session_store import SessionStore
from app._state import is_streaming, set_streaming
from engine.runtime.agent_models import Session

router = APIRouter()
session_store = SessionStore()

MAX_MESSAGES = 20


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


def _find_turn_boundaries(messages):
    """Identify turn start indices. Each turn starts with a 'user' message."""
    turn_starts = []
    for i, m in enumerate(messages):
        if m.role == "user":
            turn_starts.append(i)
    return turn_starts


def _truncate_session(session):
    """Remove oldest complete turns when non-system messages exceed limit."""
    non_system_count = sum(1 for m in session.messages if m.role != "system")
    if non_system_count <= MAX_MESSAGES:
        return

    turns = _find_turn_boundaries(session.messages)

    while non_system_count > MAX_MESSAGES and len(turns) > 1:
        start_idx = turns[0]
        end_idx = turns[1] if len(turns) > 1 else len(session.messages)

        removed_count = end_idx - start_idx
        del session.messages[start_idx:end_idx]
        non_system_count -= removed_count

        turns = _find_turn_boundaries(session.messages)


async def _event_generator(request: Request, chat_req: ChatRequest):
    """Consume delegate_stream() and yield SSE events."""
    from engine.runner import delegate_stream

    session = None
    if chat_req.session_id:
        session = session_store.load(chat_req.session_id)
    if session:
        session.add_message("user", chat_req.message)
        _truncate_session(session)
    else:
        session = Session(id=f"chat_{uuid.uuid4().hex[:8]}", depth=0)

    session_id = session.id

    try:
        async for event in delegate_stream(
            task_description=chat_req.message,
            session=session,
        ):
            if await request.is_disconnected():
                break
            # Inject session_id into key events so the frontend can persist it
            if event.type in ("agent_start", "done"):
                event.data["session_id"] = session_id
            yield {
                "event": event.type,
                "data": json.dumps(event.data),
            }
    except Exception as e:
        yield {
            "event": "error",
            "data": json.dumps({"message": str(e), "session_id": session_id}),
        }
    finally:
        session_store.save(session)
        set_streaming(False)


@router.post("/chat")
async def chat_endpoint(request: Request, chat_req: ChatRequest):
    """Stream chat response via SSE."""
    if is_streaming():
        return JSONResponse(
            status_code=429,
            content={"error": "A request is already being processed"},
        )

    set_streaming(True)
    return EventSourceResponse(
        _event_generator(request, chat_req),
        media_type="text/event-stream",
    )
