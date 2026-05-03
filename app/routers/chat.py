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
    """Consume delegate() events via callback and yield SSE frames."""
    import asyncio
    from engine.runner import delegate

    session = None
    if chat_req.session_id:
        session = session_store.load(chat_req.session_id)
    if session:
        _truncate_session(session)
    else:
        session = Session(id=f"chat_{uuid.uuid4().hex[:8]}", depth=0)

    session_id = session.id
    event_queue: asyncio.Queue = asyncio.Queue()
    done_event = asyncio.Event()

    def on_engine_event(event_name: str, data: dict) -> None:
        if event_name == "part_new":
            event_queue.put_nowait({
                "event": "part_new",
                "data": json.dumps({
                    "part_id": data["part_id"],
                    "part_type": data["part_type"],
                    "text": data.get("text", ""),
                }),
            })
        elif event_name == "part_delta":
            event_queue.put_nowait({
                "event": "part_delta",
                "data": json.dumps({
                    "part_id": data["part_id"],
                    "text": data.get("text", ""),
                }),
            })
        elif event_name == "part_close":
            event_queue.put_nowait({
                "event": "part_close",
                "data": json.dumps({
                    "part_id": data["part_id"],
                }),
            })
        elif event_name == "tool_start":
            event_queue.put_nowait({
                "event": "tool_call_start",
                "data": json.dumps({
                    "part_id": data.get("part_id", 0),
                    "tool_name": data["tool_name"],
                    "arguments": data.get("arguments", {}),
                    "call_id": data.get("call_id", ""),
                }),
            })
        elif event_name == "tool_end":
            event_queue.put_nowait({
                "event": "tool_call_result",
                "data": json.dumps({
                    "part_id": data.get("part_id", 0),
                    "tool_name": data["tool_name"],
                    "result": data.get("result", ""),
                    "call_id": data.get("call_id", ""),
                }),
            })
        elif event_name == "agent_done":
            event_queue.put_nowait({
                "event": "done",
                "data": json.dumps({
                    "success": data.get("success", True),
                    "session_id": session_id,
                }),
            })
            done_event.set()
        elif event_name == "error":
            event_queue.put_nowait({
                "event": "error",
                "data": json.dumps({
                    "message": data.get("message", "Unknown error"),
                    "session_id": session_id,
                }),
            })
            done_event.set()

    async def run_delegate():
        try:
            await delegate(
                task_description=chat_req.message,
                session=session,
                event_callback=on_engine_event,
            )
        except Exception as e:
            event_queue.put_nowait({
                "event": "error",
                "data": json.dumps({"message": str(e), "session_id": session_id}),
            })
            done_event.set()

    delegate_task = asyncio.create_task(run_delegate())

    try:
        yield {
            "event": "agent_start",
            "data": json.dumps({"session_id": session_id}),
        }

        while not done_event.is_set() or not event_queue.empty():
            if await request.is_disconnected():
                delegate_task.cancel()
                break
            try:
                event = await asyncio.wait_for(event_queue.get(), timeout=1.0)
                yield event
            except asyncio.TimeoutError:
                yield {"comment": "keepalive"}
                if delegate_task.done() and event_queue.empty():
                    break
    finally:
        session_store.save(session)
        set_streaming(False)
        if not delegate_task.done():
            delegate_task.cancel()


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
