"""Sub-agent protocols.

This module defines the protocols (structural interfaces) used by the sub-agent system.
"""

from typing import TYPE_CHECKING, Dict, Optional, Protocol, runtime_checkable
from engine.models import AgentState

if TYPE_CHECKING:
    from engine.subagent.models import CollectedChildResult


@runtime_checkable
class Drainable(Protocol):
    """Protocol for objects that support async resume and abort.

    Implementors must expose their current state, support resuming from children,
    and support async abort with an error.
    """

    @property
    def state(self) -> AgentState:
        """Current agent state."""
        ...

    @property
    def result(self) -> Optional[str]:
        """The agent's final result, or None if not yet available."""
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
