"""Sub-agent protocols.

This module defines the protocols (structural interfaces) used by the sub-agent system.
"""

from typing import TYPE_CHECKING, Dict, Optional, Protocol, runtime_checkable
from engine.models import AgentState

if TYPE_CHECKING:
    from engine.subagent.models import CollectedChildResult


@runtime_checkable
class Drainable(Protocol):
    """Protocol for objects whose event queues can be drained.

    Implementors must expose their current state, allow async event draining,
    and support async abort with an error.
    """

    @property
    def state(self) -> AgentState:
        """Current agent state."""
        ...

    async def drain_events(self) -> None:
        """Process / drain the internal event queue."""
        ...

    async def resume_from_children(
        self,
        formatted_prompt: str,
        child_results: Optional[Dict[str, "CollectedChildResult"]] = None,
    ) -> None:
        """Resume agent processing after all children complete."""
        ...

    async def abort(self, error: Exception) -> None:
        """Abort execution with the given error."""
        ...
