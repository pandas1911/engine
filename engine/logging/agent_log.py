"""Agent log helper — binds agent context to eliminate repetitive log boilerplate."""

from typing import Any, Callable, Dict, Optional

from .sink import get_logger


class AgentLogHelper:
    """Binds agent context (label, task_id, state, depth) so each log call
    only specifies what is unique — the event_type, message, and extra data.

    The state_getter and depth_getter are callables evaluated at log-call-time
    (not creation-time), ensuring always-current values for mutable fields.
    """

    def __init__(
        self,
        label: str,
        task_id: str,
        state_getter: Callable,
        depth_getter: Callable,
    ):
        self._label = label
        self._task_id = task_id
        self._state_getter = state_getter
        self._depth_getter = depth_getter

    def info(self, event_type: str, message: str, **extra_data: Any) -> None:
        """Log an info event with bound agent context."""
        get_logger().info(
            self._label,
            message,
            task_id=self._task_id,
            state=self._state_getter().value,
            depth=self._depth_getter(),
            event_type=event_type,
            data=extra_data or None,
        )

    def error(self, event_type: str, message: str, **extra_data: Any) -> None:
        """Log an error event with bound agent context."""
        get_logger().error(
            self._label,
            message,
            task_id=self._task_id,
            state=self._state_getter().value,
            depth=self._depth_getter(),
            event_type=event_type,
            data=extra_data or None,
        )

    def tool(self, tool_name: str, message: str, **extra_data: Any) -> None:
        """Log a tool event with bound agent context."""
        data: Dict[str, Any] = {"tool_name": tool_name}
        data.update(extra_data)
        get_logger().tool(
            self._label,
            message,
            task_id=self._task_id,
            state=self._state_getter().value,
            depth=self._depth_getter(),
            tool_name=tool_name,
            data=data,
        )

    def state_change(
        self,
        from_state: str,
        to_state: str,
        trigger: str,
        **extra: Any,
    ) -> None:
        """Log a state transition with bound agent context."""
        data: Dict[str, Any] = {
            "from_state": from_state,
            "to_state": to_state,
            "trigger": trigger,
        }
        data.update(extra)
        get_logger().state_change(
            self._label,
            from_state,
            to_state,
            trigger,
            task_id=self._task_id,
            state=to_state,
            depth=self._depth_getter(),
            data=data,
        )


__all__ = ["AgentLogHelper"]
