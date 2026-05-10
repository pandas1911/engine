"""Session management endpoints."""
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from engine.runner import Engine

router = APIRouter()


def _get_session_store():
    """Get the shared SessionStore from Engine's Infrastructure."""
    return Engine.get()._infra.session_store


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
