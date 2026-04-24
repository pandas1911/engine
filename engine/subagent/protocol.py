"""Sub-agent protocols.

This module defines the protocols (structural interfaces) used by the sub-agent system.
"""

from typing import Optional, Protocol, runtime_checkable
from engine.runtime.agent_models import AgentState


@runtime_checkable
class Drainable(Protocol):
    """Protocol for objects that support async run and abort.

    Implementors must expose their current state, support running with
    different triggers (start, children_settled), and support async abort.
    """

    @property
    def state(self) -> AgentState:
        """Current agent state."""
        ...

    @property
    def result(self) -> Optional[str]:
        """The agent's final result, or None if not yet available."""
        ...

    async def run(
        self,
        message: Optional[str] = None,
        *,
        trigger: str = "start",
    ) -> str:
        """Run the agent with optional message and trigger."""
        ...

    async def abort(self, error: Exception) -> None:
        """Abort execution with the given error."""
        ...
