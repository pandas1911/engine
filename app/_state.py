"""Module-level state for active session tracking."""

from typing import Any, Dict, Optional

_active_session: Optional[Dict[str, Any]] = None


def is_streaming() -> bool:
    return _active_session is not None


def set_active_session(
    session_id: str,
    session_manager: Any,
    event_queue: Any,
    done_event: Any,
    delegate_task: Any = None,
) -> None:
    global _active_session
    _active_session = {
        "session_id": session_id,
        "session_manager": session_manager,
        "event_queue": event_queue,
        "done_event": done_event,
        "delegate_task": delegate_task,
    }


def get_active_session() -> Optional[Dict[str, Any]]:
    return _active_session


def clear_active_session() -> None:
    global _active_session
    _active_session = None


# Backward compat
def set_streaming(value: bool) -> None:
    global _active_session
    if not value:
        _active_session = None
