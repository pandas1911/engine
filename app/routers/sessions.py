"""Session management endpoints."""
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from engine.runner import Engine

router = APIRouter()


def _get_session_store():
    """Get the shared SessionStore from Engine's Infrastructure."""
    return Engine.get()._infra.session_store


def _truncate_title(text: str, max_len: int = 50) -> str:
    if len(text) <= max_len:
        return text
    truncated = text[:max_len]
    last_space = truncated.rfind(" ")
    if last_space > 0:
        truncated = truncated[:last_space]
    return truncated


@router.get("/sessions")
async def list_sessions():
    store = _get_session_store()
    session_ids = store.list_sessions()
    results = []

    for sid in session_ids:
        try:
            session = store.load(sid)
        except Exception:
            continue
        if session is None:
            continue

        messages = session.messages

        title = "New Session"
        for msg in messages:
            if msg.role == "user":
                title = _truncate_title(msg.content)
                break

        last_active = None
        if messages:
            last_active = str(messages[-1].timestamp)

        message_count = sum(1 for msg in messages if msg.role != "system")

        results.append({
            "id": session.id,
            "title": title,
            "last_active": last_active,
            "message_count": message_count,
        })

    results.sort(key=lambda s: s["last_active"] or "", reverse=True)

    return {"sessions": results}


@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    store = _get_session_store()
    session = store.load(session_id)
    if session is None:
        return JSONResponse(status_code=404, content={"error": "Session not found"})
    data = {
        "id": session.id,
        "depth": session.depth,
        "parent_id": session.parent_id,
        "messages": session.get_messages(),
    }
    return data


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    deleted = _get_session_store().delete(session_id)
    if not deleted:
        return JSONResponse(status_code=404, content={"error": "Session not found"})
    return {"status": "deleted"}
