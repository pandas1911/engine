"""Read session tool — inspect a child agent's session content.

Provides three scopes:
  - full:    all messages (thinking wrapped in tags)
  - summary: last assistant message only
  - last_n:  N most recent messages

Data source priority:
  1. Live session (in-memory via child_task.agent.session)
  2. Persisted file (SessionStore.read_child_session)

Supports both .jsonl and .json file formats.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from engine.runtime.agent_models import Message
from engine.safety.token_estimator import ResultTruncator
from engine.tools.base import Tool


class ReadSessionTool(Tool):
    """Read the session content of a child agent."""

    name = "read_session"

    description = (
        "Read the session content of a child agent. "
        "Returns formatted text suitable for LLM consumption. "
        "Use 'full' scope for all messages, 'summary' for the final answer, "
        "or 'last_n' for the most recent N messages."
    )

    parameters = {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "The child agent's task_id to inspect.",
            },
            "scope": {
                "type": "string",
                "enum": ["full", "summary", "last_n"],
                "description": (
                    "'full' — all messages, 'summary' — final answer only, "
                    "'last_n' — recent N messages"
                ),
            },
            "count": {
                "type": "integer",
                "description": (
                    "Number of recent messages when scope='last_n'. Default: 10."
                ),
            },
        },
        "required": ["task_id", "scope"],
    }

    _FULL_MAX_CHARS: int = 8000
    _SUMMARY_MAX_CHARS: int = 15000

    async def execute(
        self, arguments: Dict[str, Any], context: Dict[str, Any]
    ) -> str:
        task_id = arguments.get("task_id")
        scope = arguments.get("scope")

        if not task_id:
            return "Error: 'task_id' parameter is required."

        if scope not in ("full", "summary", "last_n"):
            return "Error: 'scope' must be one of: full, summary, last_n."

        # Resolve the parent agent (backward compat: both keys)
        agent = context.get("agent") or context.get("parent_agent")
        if not agent:
            return "Error: no agent context available."

        # Resolve messages from live session or persisted file
        messages = self._resolve_messages(task_id, agent, context)
        if messages is None:
            return "Error: no session found for task_id='{}'.".format(task_id)

        if not messages:
            return "No displayable messages found for task_id='{}'.".format(task_id)

        # Apply the requested scope
        if scope == "summary":
            return self._format_summary(messages)
        elif scope == "last_n":
            count = arguments.get("count", 10)
            if not isinstance(count, int) or count <= 0:
                count = 10
            subset = messages[-count:]
            return self._format_messages(subset)
        else:  # full
            return self._format_messages(messages)

    # ------------------------------------------------------------------
    # Message resolution
    # ------------------------------------------------------------------

    def _resolve_messages(
        self, task_id: str, agent: Any, context: Dict[str, Any]
    ) -> Optional[List[Message]]:
        """Get messages from live session or persisted file.

        Path 1: Live registry lookup (in-memory session).
        Path 2: SessionStore disk fallback.
        """
        messages: Optional[List[Message]] = None

        # Path 1: live session from registry
        task_registry = getattr(agent, "task_registry", None)
        if task_registry is not None:
            child_task = task_registry.get_task(task_id)
            if child_task is not None:
                child_agent = getattr(child_task, "agent", None)
                if child_agent is not None:
                    session = getattr(child_agent, "session", None)
                    if session is not None:
                        msgs = getattr(session, "messages", None)
                        if msgs:
                            messages = list(msgs)

        # Path 2: SessionStore disk fallback
        if messages is None:
            session_store = getattr(agent, "session_store", None)
            if session_store is not None:
                try:
                    stored = session_store.read_child_session(task_id)
                    if stored is not None:
                        msgs = getattr(stored, "messages", None)
                        if msgs:
                            messages = list(msgs)
                except Exception:
                    pass

                # Try .jsonl fallback when .json read yields nothing
                if messages is None:
                    messages = self._try_read_jsonl(session_store, task_id)

        return messages

    @staticmethod
    def _try_read_jsonl(session_store: Any, task_id: str) -> Optional[List[Message]]:
        """Attempt to read a .jsonl session file when .json is unavailable."""
        sessions_dir = getattr(session_store, "sessions_dir", None)
        if sessions_dir is None:
            return None

        jsonl_path = Path(sessions_dir) / "{}.jsonl".format(task_id)
        if not jsonl_path.exists():
            return None

        try:
            msgs: List[Message] = []
            for line in jsonl_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                msgs.append(Message(
                    role=data.get("role", "unknown"),
                    content=data.get("content", ""),
                    metadata=data.get("metadata", {}),
                ))
            return msgs if msgs else None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Thinking / noise filter
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_thinking(messages: List[Message]) -> List[Message]:
        """Remove thinking, reasoning, and system messages.

        Filters out:
          - role == 'reasoning'
          - role == 'system'
          - assistant messages whose content starts with '<think'
        """
        result: List[Message] = []
        for msg in messages:
            if msg.role in ("reasoning", "system"):
                continue
            if msg.role == "assistant" and msg.content.startswith("<think"):
                continue
            result.append(msg)
        return result

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def _format_summary(self, messages: List[Message]) -> str:
        """Return the last non-thinking assistant message with generous limit."""
        for msg in reversed(messages):
            if msg.role == "assistant":
                return ResultTruncator.truncate(msg.content, self._SUMMARY_MAX_CHARS)
        # No assistant message found — fall back to last message
        last = messages[-1]
        return ResultTruncator.truncate(
            "[{}] {}".format(last.role, last.content), self._SUMMARY_MAX_CHARS
        )

    def _format_messages(self, messages: List[Message]) -> str:
        """Format messages with tool enrichment and thinking tags."""
        # Build tool_call lookup from assistant messages
        tool_call_lookup: Dict[str, tuple] = {}
        for msg in messages:
            if msg.role == "assistant" and "tool_calls" in msg.metadata:
                for tc in msg.metadata["tool_calls"]:
                    tc_id = tc.get("id")
                    if tc_id:
                        func = tc.get("function", {})
                        tool_call_lookup[tc_id] = (
                            func.get("name", "unknown"),
                            func.get("arguments", {}),
                        )

        lines: List[str] = []
        for msg in messages:
            if msg.role == "system":
                continue

            if msg.role == "reasoning" or (
                msg.role == "assistant" and msg.content.startswith("<think")
            ):
                lines.append("[thinking] {} [/thinking]".format(msg.content))
            elif msg.role == "tool":
                tc_id = msg.metadata.get("tool_call_id", "")
                tool_name, tool_args = tool_call_lookup.get(tc_id, ("tool", {}))
                args_str = json.dumps(tool_args, ensure_ascii=False) if tool_args else ""
                header = "{}({})".format(tool_name, args_str) if args_str else tool_name
                lines.append("[tool] {}\n{}".format(header, msg.content))
            else:
                lines.append("[{}] {}".format(msg.role, msg.content))

        text = "\n".join(lines)
        return ResultTruncator.truncate(text, self._FULL_MAX_CHARS)
