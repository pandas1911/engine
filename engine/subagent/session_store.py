"""Session file persistence layer for sub-agent sessions.

Directory layout:
    sessions/{root_session_id}/
        main.json           <- root agent session
        task_abc123.json     <- child session (named by task_id)
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from engine.logging import get_logger


@dataclass
class ChildSessionInfo:
    """Metadata about a child session file."""
    task_id: str
    file_path: str
    message_count: int
    file_size_bytes: int


class SessionStore:
    """Manages session persistence as JSON files on disk.

    All methods require create_root() to have been called first.
    Thread safety: designed for single-process asyncio (no file locking).
    """

    def __init__(self, root_dir: str):
        """Args:
            root_dir: Base directory where sessions/ will be created.
        """
        self._root_dir = Path(root_dir)
        self._sessions_dir: Optional[Path] = None

    def create_root(self, root_session_id: str) -> Path:
        """Create the session directory for a root conversation.

        Returns the path to sessions/{root_session_id}/.
        """
        self._sessions_dir = self._root_dir / root_session_id
        self._sessions_dir.mkdir(parents=True, exist_ok=True)
        return self._sessions_dir

    @property
    def sessions_dir(self) -> Optional[Path]:
        return self._sessions_dir

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def save_main_session(self, session) -> None:
        """Persist root agent session to main.json."""
        self._write_session("main", session)

    def save_child_session(self, task_id: str, session) -> None:
        """Persist child session to {task_id}.json."""
        self._write_session(task_id, session)

    def append_message(
        self, task_id: str, session, role: str, content: str, **metadata
    ) -> None:
        """Append a message to in-memory session AND persist to disk.

        This is the real-time hook called during child execution to ensure
        partial sessions are recoverable after a crash.
        """
        session.add_message(role, content, **metadata)
        self._write_session(task_id, session)

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def read_child_session(self, task_id: str):
        """Read and deserialize a child session from disk.

        Returns None if file does not exist or is corrupted.
        """
        from engine.runtime.agent_models import Session, Message

        file_path = self._sessions_dir / "{}.json".format(task_id)
        if not file_path.exists():
            return None

        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            get_logger().warning(
                "SessionStore",
                "Failed to read session file | task_id={}, error={}".format(task_id, e),
                task_id=task_id,
            )
            return None

        return self._deserialize_session(data)

    def list_children(self) -> List[ChildSessionInfo]:
        """List all child session files with metadata.

        Only lists files matching task_*.json pattern. Does NOT read
        full session content -- extracts message_count from JSON key.
        """
        if not self._sessions_dir or not self._sessions_dir.exists():
            return []

        result = []
        for f in self._sessions_dir.glob("task_*.json"):
            try:
                raw = f.read_text(encoding="utf-8")
                data = json.loads(raw)
                message_count = len(data.get("messages", []))
            except (json.JSONDecodeError, OSError):
                message_count = -1  # Corrupted file marker

            result.append(ChildSessionInfo(
                task_id=f.stem,
                file_path=str(f),
                message_count=message_count,
                file_size_bytes=f.stat().st_size,
            ))
        return result

    def get_child_file_path(self, task_id: str) -> str:
        """Return absolute path for a child's session file."""
        return str((self._sessions_dir / "{}.json".format(task_id)).resolve())

    # ------------------------------------------------------------------
    # Internal: serialization
    # ------------------------------------------------------------------

    def _write_session(self, name: str, session) -> None:
        """Atomic write: serialize to temp file, then rename."""
        target = self._sessions_dir / "{}.json".format(name)
        tmp = target.with_suffix(".tmp")
        try:
            payload = self._serialize_session(session)
            tmp.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp.rename(target)  # Atomic on POSIX
        except Exception:
            # Clean up temp file on failure
            try:
                tmp.unlink()
            except OSError:
                pass
            raise

    @staticmethod
    def _serialize_session(session) -> Dict:
        """Serialize Session to a JSON-compatible dict."""
        return {
            "id": session.id,
            "depth": session.depth,
            "parent_id": session.parent_id,
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "metadata": m.metadata,
                    "timestamp": m.timestamp.isoformat() if m.timestamp else None,
                }
                for m in session.messages
            ],
        }

    @staticmethod
    def _deserialize_session(data: Dict):
        """Deserialize a dict back to Session with Message objects."""
        from engine.runtime.agent_models import Session, Message

        messages = []
        for m_data in data.get("messages", []):
            ts_str = m_data.get("timestamp")
            msg = Message(
                role=m_data["role"],
                content=m_data["content"],
                metadata=m_data.get("metadata", {}),
            )
            if ts_str:
                try:
                    msg.timestamp = datetime.fromisoformat(ts_str)
                except (ValueError, TypeError):
                    pass  # Keep default timestamp
            messages.append(msg)

        return Session(
            id=data["id"],
            depth=data.get("depth", 0),
            parent_id=data.get("parent_id"),
            messages=messages,
        )
