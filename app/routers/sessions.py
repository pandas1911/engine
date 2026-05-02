"""Session management endpoints."""
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.session_store import SessionStore

router = APIRouter()
session_store = SessionStore()


@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    session = session_store.load(session_id)
    if session is None:
        return JSONResponse(status_code=404, content={"error": "Session not found"})
    data = SessionStore._serialize_session(session)
    return data


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    deleted = session_store.delete(session_id)
    if not deleted:
        return JSONResponse(status_code=404, content={"error": "Session not found"})
    return {"status": "deleted"}
