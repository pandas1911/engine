"""Unified session persistence layer with JSONL append format.

Supports both JSONL (new) and JSON (legacy) file formats.
Directory layout:
    sessions/{root_session_id}/
        main.jsonl           <- root agent session
        task_abc123.jsonl     <- child session (named by task_id)
"""

import json
import shutil
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
    """Manages session persistence as JSONL files on disk.

    JSONL format:
        Line 1: {"id":"chat_xxx","depth":0,"parent_id":null}   <- session header
        Line 2+: {"role":"user","content":"...","metadata":{},...}  <- messages

    All engine-facing methods require create_root() to have been called first.
    App-facing methods (save/load/delete/list_sessions) manage their own paths.
    Thread safety: designed for single-process asyncio (no file locking).
    """

    def __init__(self, root_dir: str = "./sessions"):
        self._root_dir = Path(root_dir)
        self._sessions_dir: Optional[Path] = None

    # ------------------------------------------------------------------
    # Session directory management
    # ------------------------------------------------------------------

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
    # Write operations (engine-facing)
    # ------------------------------------------------------------------

    def create_file(self, name: str, session) -> None:
        """Create a JSONL file with header line. Atomic write (tmp + rename).

        Call this once when creating a new session file, before any append_line calls.
        """
        target = self._sessions_dir / "{}.jsonl".format(name)
        tmp = target.with_suffix(".tmp")
        try:
            header = json.dumps({
                "id": session.id,
                "depth": session.depth,
                "parent_id": session.parent_id,
            }, ensure_ascii=False)
            tmp.write_text(header + "\n", encoding="utf-8")
            tmp.rename(target)
        except Exception:
            try:
                tmp.unlink()
            except OSError:
                pass
            raise

    def append_line(self, name: str, message) -> None:
        """Append a single message as one JSON line.

        Called by Session._on_message_added callback for real-time persistence.
        """
        target = self._sessions_dir / "{}.jsonl".format(name)
        line = json.dumps(self._serialize_message(message), ensure_ascii=False)
        with open(target, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def rewrite_file(self, name: str, session) -> None:
        """Full rewrite: header + all messages. Used for final checkpoint/compaction.

        Atomic write via tmp + rename.
        """
        target = self._sessions_dir / "{}.jsonl".format(name)
        tmp = target.with_suffix(".tmp")
        try:
            lines = []
            # Header line
            lines.append(json.dumps({
                "id": session.id,
                "depth": session.depth,
                "parent_id": session.parent_id,
            }, ensure_ascii=False))
            # Message lines
            for m in session.messages:
                lines.append(json.dumps(
                    self._serialize_message(m), ensure_ascii=False
                ))
            tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
            tmp.rename(target)
        except Exception:
            try:
                tmp.unlink()
            except OSError:
                pass
            raise

    # ------------------------------------------------------------------
    # Read operations (engine-facing)
    # ------------------------------------------------------------------

    def _find_file(self, name: str) -> Optional[Path]:
        """Find session file, preferring .jsonl over legacy .json."""
        jsonl_path = self._sessions_dir / "{}.jsonl".format(name)
        if jsonl_path.exists():
            return jsonl_path
        json_path = self._sessions_dir / "{}.json".format(name)
        if json_path.exists():
            return json_path
        return None

    def read_session_file(self, name: str):
        """Read session from .jsonl or legacy .json.

        Returns a Session object, or None if not found or corrupted.
        """
        file_path = self._find_file(name)
        if file_path is None:
            return None
        if file_path.suffix == ".jsonl":
            return self._read_jsonl(file_path)
        return self._read_legacy_json(file_path)

    def _read_jsonl(self, file_path: Path):
        """Parse JSONL: line 1 = header, lines 2+ = messages. Skip malformed lines."""
        from engine.runtime.agent_models import Session

        try:
            raw = file_path.read_text(encoding="utf-8")
        except OSError as e:
            get_logger().warning(
                "SessionStore",
                "Failed to read JSONL file | path={}, error={}".format(file_path, e),
            )
            return None

        lines = raw.splitlines()
        if not lines:
            return None

        # Parse header
        try:
            header = json.loads(lines[0])
        except json.JSONDecodeError:
            get_logger().warning(
                "SessionStore",
                "Malformed header in JSONL | path={}".format(file_path),
            )
            return None

        # Parse messages
        messages = []
        for line in lines[1:]:
            if not line.strip():
                continue
            try:
                msg_data = json.loads(line)
                messages.append(self._deserialize_message(msg_data))
            except (json.JSONDecodeError, KeyError, TypeError):
                # Skip malformed lines for crash recovery
                continue

        return Session(
            id=header["id"],
            depth=header.get("depth", 0),
            parent_id=header.get("parent_id"),
            messages=messages,
        )

    def _read_legacy_json(self, file_path: Path):
        """Parse old-format .json for backward compatibility."""
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            get_logger().warning(
                "SessionStore",
                "Failed to read legacy JSON | path={}, error={}".format(file_path, e),
            )
            return None

        return self._deserialize_legacy_session(data)

    def read_child_session(self, task_id: str):
        """Read a child session by task_id.

        Returns a Session object, or None if not found.
        """
        return self.read_session_file(task_id)

    def list_children(self) -> List[ChildSessionInfo]:
        """List all child session files with metadata.

        Searches for both task_*.jsonl and task_*.json files.
        Does NOT read full session content.
        """
        if not self._sessions_dir or not self._sessions_dir.exists():
            return []

        result = []
        seen_stems = set()

        # Prefer .jsonl files, then fall back to .json
        for ext, is_jsonl in [(".jsonl", True), (".json", False)]:
            for f in self._sessions_dir.glob("task_*{}".format(ext)):
                stem = f.stem
                if stem in seen_stems:
                    continue
                seen_stems.add(stem)

                message_count = -1
                try:
                    raw = f.read_text(encoding="utf-8")
                    if is_jsonl:
                        line_count = len([line for line in raw.splitlines() if line.strip()])
                        message_count = max(0, line_count - 1)  # First line is header
                    else:
                        data = json.loads(raw)
                        message_count = len(data.get("messages", []))
                except (json.JSONDecodeError, OSError):
                    pass  # Keep -1 as corrupted file marker

                result.append(ChildSessionInfo(
                    task_id=stem,
                    file_path=str(f),
                    message_count=message_count,
                    file_size_bytes=f.stat().st_size,
                ))

        return result

    # ------------------------------------------------------------------
    # App-facing API (replaces app/session_store.py)
    # ------------------------------------------------------------------

    def save(self, session) -> Path:
        """Save session to sessions/{id}/main.jsonl (full rewrite).

        Ensures the session directory exists, then performs an atomic rewrite.
        """
        session_dir = self._root_dir / session.id
        session_dir.mkdir(parents=True, exist_ok=True)

        # Temporarily swap _sessions_dir for rewrite_file
        prev = self._sessions_dir
        self._sessions_dir = session_dir
        try:
            self.rewrite_file("main", session)
        finally:
            self._sessions_dir = prev

        return session_dir / "main.jsonl"

    def load(self, session_id: str):
        """Load from sessions/{id}/main.jsonl or main.json.

        Returns a Session object, or None if not found.
        """
        session_dir = self._root_dir / session_id
        if not session_dir.exists():
            return None

        prev = self._sessions_dir
        self._sessions_dir = session_dir
        try:
            return self.read_session_file("main")
        finally:
            self._sessions_dir = prev

    def delete(self, session_id: str) -> bool:
        """Delete session directory.

        Returns True if deleted, False if not found.
        """
        session_dir = self._root_dir / session_id
        if session_dir.exists() and session_dir.is_dir():
            shutil.rmtree(session_dir)
            return True
        return False

    def list_sessions(self) -> List[str]:
        """List all session IDs (directories with main.jsonl or main.json)."""
        if not self._root_dir.exists():
            return []

        result = []
        for d in self._root_dir.iterdir():
            if not d.is_dir():
                continue
            if (d / "main.jsonl").exists() or (d / "main.json").exists():
                result.append(d.name)
        return sorted(result)

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize_message(message) -> Dict:
        """Convert Message to a JSON-safe dict."""
        ts = None
        if message.timestamp is not None:
            ts = message.timestamp.isoformat() if hasattr(message.timestamp, "isoformat") else str(message.timestamp)
        return {
            "role": message.role,
            "content": message.content,
            "metadata": message.metadata,
            "timestamp": ts,
        }

    @staticmethod
    def _deserialize_message(data: Dict):
        """Parse dict back to Message, with timestamp fallback."""
        from engine.runtime.agent_models import Message

        ts_str = data.get("timestamp")
        timestamp = None
        if isinstance(ts_str, str) and ts_str:
            try:
                timestamp = datetime.fromisoformat(ts_str)
            except (ValueError, TypeError):
                timestamp = datetime.now()
        else:
            timestamp = datetime.now()

        return Message(
            role=data["role"],
            content=data.get("content", ""),
            metadata=data.get("metadata", {}),
            timestamp=timestamp,
        )

    @staticmethod
    def _deserialize_legacy_session(data: Dict):
        """Deserialize a legacy .json dict back to Session with Messages."""
        from engine.runtime.agent_models import Session

        messages = []
        for m_data in data.get("messages", []):
            messages.append(SessionStore._deserialize_message(m_data))

        return Session(
            id=data["id"],
            depth=data.get("depth", 0),
            parent_id=data.get("parent_id"),
            messages=messages,
        )
