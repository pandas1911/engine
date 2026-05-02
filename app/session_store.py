"""Session serialization and JSON file persistence."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from engine.runtime.agent_models import Message, Session


_DEFAULT_STORAGE_DIR = "./sessions"


class SessionStore:
    """JSON file-based session persistence."""

    def __init__(self, storage_dir: str = _DEFAULT_STORAGE_DIR):
        self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)

    def save(self, session: Session) -> Path:
        """Save session to JSON file. Returns the file path."""
        data = self._serialize_session(session)
        path = self._storage_dir / f"{session.id}.json"
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        return path

    def load(self, session_id: str) -> Optional[Session]:
        """Load session from JSON file. Returns None if not found or corrupted."""
        path = self._storage_dir / f"{session_id}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            return self._deserialize_session(data)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None

    def delete(self, session_id: str) -> bool:
        """Delete a session file. Returns True if deleted, False if not found."""
        path = self._storage_dir / f"{session_id}.json"
        if path.exists():
            path.unlink()
            return True
        return False

    def list_sessions(self) -> List[str]:
        """List available session IDs from storage directory."""
        if not self._storage_dir.exists():
            return []
        return [
            p.stem for p in self._storage_dir.glob("*.json")
            if p.is_file()
        ]

    # ── Serialization helpers ──

    @staticmethod
    def _serialize_session(session: Session) -> Dict[str, Any]:
        """Convert Session to JSON-safe dict."""
        return {
            "id": session.id,
            "depth": session.depth,
            "parent_id": session.parent_id,
            "messages": [
                SessionStore._serialize_message(m) for m in session.messages
            ],
        }

    @staticmethod
    def _serialize_message(message: Message) -> Dict[str, Any]:
        """Convert Message to JSON-safe dict."""
        return {
            "role": message.role,
            "content": message.content,
            "metadata": message.metadata,
            "timestamp": message.timestamp.isoformat() if isinstance(message.timestamp, datetime) else str(message.timestamp),
        }

    @staticmethod
    def _deserialize_session(data: Dict[str, Any]) -> Session:
        """Convert JSON dict back to Session."""
        session = Session(
            id=data["id"],
            depth=data.get("depth", 0),
            parent_id=data.get("parent_id"),
        )
        for msg_data in data.get("messages", []):
            msg = SessionStore._deserialize_message(msg_data)
            session.messages.append(msg)
        return session

    @staticmethod
    def _deserialize_message(data: Dict[str, Any]) -> Message:
        """Convert JSON dict back to Message."""
        timestamp = data.get("timestamp", "")
        if isinstance(timestamp, str) and timestamp:
            try:
                timestamp = datetime.fromisoformat(timestamp)
            except ValueError:
                timestamp = datetime.now()
        else:
            timestamp = datetime.now()

        return Message(
            role=data["role"],
            content=data.get("content", ""),
            metadata=data.get("metadata", {}),
            timestamp=timestamp,
        )
