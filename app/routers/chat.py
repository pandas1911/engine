"""Chat SSE endpoint."""
import json
import uuid
from typing import Optional

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from engine.runner import Engine
from engine import get_logger
from app._state import is_streaming, set_active_session, get_active_session, clear_active_session
from engine.runtime.agent_models import Session

logger = get_logger()

router = APIRouter()

MAX_MESSAGES = 50


def _get_session_store():
    """Get the shared SessionStore from Engine's Infrastructure."""
    return Engine.get()._infra.session_store


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
    """Consume SessionManager events via callback and yield SSE frames."""
    import asyncio

    session = None
    if chat_req.session_id:
        session = _get_session_store().load(chat_req.session_id)
    if session:
        _truncate_session(session)
    else:
        session = Session(id=f"chat_{uuid.uuid4().hex[:8]}", depth=0)
        removed = _get_session_store().cleanup_old_sessions(max_sessions=3)
        if removed:
            logger.info("ChatRouter", "Session cleanup: removed {} old session(s)".format(removed))

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
        elif event_name == "waiting_for_children":
            event_queue.put_nowait({
                "event": "waiting_for_children",
                "data": json.dumps({"session_id": data.get("session_id", "")}),
            })
        elif event_name == "turn_start":
            event_queue.put_nowait({
                "event": "turn_start",
                "data": json.dumps({"trigger": data.get("trigger", "user_message")}),
            })
        elif event_name == "subagent_start":
            event_queue.put_nowait({
                "event": "subagent_start",
                "data": json.dumps({
                    "part_id": data.get("part_id", 0),
                    "task_id": data.get("task_id", ""),
                    "label": data.get("label", ""),
                    "description": data.get("description", ""),
                    "parent_task_id": data.get("parent_task_id", ""),
                }),
            })
        elif event_name == "subagent_part_new":
            event_queue.put_nowait({
                "event": "subagent_part_new",
                "data": json.dumps({
                    "part_id": data["part_id"],
                    "task_id": data.get("task_id", ""),
                    "part_type": data["part_type"],
                    "text": data.get("text", ""),
                }),
            })
        elif event_name == "subagent_part_delta":
            event_queue.put_nowait({
                "event": "subagent_part_delta",
                "data": json.dumps({
                    "part_id": data["part_id"],
                    "task_id": data.get("task_id", ""),
                    "text": data.get("text", ""),
                }),
            })
        elif event_name == "subagent_part_close":
            event_queue.put_nowait({
                "event": "subagent_part_close",
                "data": json.dumps({
                    "part_id": data["part_id"],
                    "task_id": data.get("task_id", ""),
                }),
            })
        elif event_name == "subagent_tool_start":
            event_queue.put_nowait({
                "event": "subagent_tool_start",
                "data": json.dumps({
                    "part_id": data.get("part_id", 0),
                    "task_id": data.get("task_id", ""),
                    "tool_name": data["tool_name"],
                    "arguments": data.get("arguments", {}),
                    "call_id": data.get("call_id", ""),
                }),
            })
        elif event_name == "subagent_tool_end":
            event_queue.put_nowait({
                "event": "subagent_tool_result",
                "data": json.dumps({
                    "part_id": data.get("part_id", 0),
                    "task_id": data.get("task_id", ""),
                    "tool_name": data["tool_name"],
                    "result": data.get("result", ""),
                    "call_id": data.get("call_id", ""),
                }),
            })
        elif event_name == "subagent_done":
            event_queue.put_nowait({
                "event": "subagent_done",
                "data": json.dumps({
                    "task_id": data.get("task_id", ""),
                    "success": data.get("success", True),
                }),
            })
        elif event_name == "subagent_error":
            event_queue.put_nowait({
                "event": "subagent_error",
                "data": json.dumps({
                    "task_id": data.get("task_id", ""),
                    "message": data.get("message", "Unknown error"),
                }),
            })

    engine = Engine.get()
    mgr = engine.create_session(
        session=session,
        event_callback=on_engine_event,
    )

    async def run_delegate():
        try:
            await mgr.start(chat_req.message)
        except Exception as e:
            event_queue.put_nowait({
                "event": "error",
                "data": json.dumps({"message": str(e), "session_id": session_id}),
            })
            done_event.set()

    delegate_task = asyncio.create_task(run_delegate())

    set_active_session(
        session_id=mgr.session.id,
        session_manager=mgr,
        event_queue=mgr._event_queue,
        done_event=done_event,
        delegate_task=delegate_task,
    )

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
        _get_session_store().save(session)
        clear_active_session()
        mgr.unregister()
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

    return EventSourceResponse(
        _event_generator(request, chat_req),
        media_type="text/event-stream",
    )


@router.post("/chat/abort")
async def abort_endpoint():
    active = get_active_session()
    if active is None:
        return JSONResponse(status_code=404, content={"error": "No active session"})

    # Cancel the delegate asyncio task — triggers CancelledError at next await point.
    # The _event_generator's finally block will handle cleanup:
    #   - Save session via _get_session_store().save(session)
    #   - Call clear_active_session()
    #   - Call mgr.unregister()
    delegate_task = active.get("delegate_task")
    if delegate_task and not delegate_task.done():
        delegate_task.cancel()

    return {"status": "aborted"}
