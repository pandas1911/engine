"""Agent state machine implementation.

This module provides a simple state machine for managing agent execution states.
"""

from typing import Dict, Tuple

from engine.models import AgentState


class InvalidTransitionError(Exception):
    """Raised when an invalid state transition is attempted."""

    def __init__(self, current_state: AgentState, event: str) -> None:
        self.current_state = current_state
        self.event = event
        super().__init__(
            f"Invalid transition: cannot trigger '{event}' from state '{current_state.value}'"
        )


class AgentStateMachine:
    """State machine for managing agent execution states.

    Supports transitions between agent states based on events.
    """

    TRANSITIONS: Dict[Tuple[AgentState, str], AgentState] = {
        (AgentState.IDLE, "start"): AgentState.RUNNING,
        (AgentState.RUNNING, "spawn_children"): AgentState.WAITING_FOR_CHILDREN,
        (AgentState.RUNNING, "finish"): AgentState.COMPLETED,
        (AgentState.RUNNING, "error"): AgentState.ERROR,
        (AgentState.WAITING_FOR_CHILDREN, "children_settled"): AgentState.RUNNING,
    }

    def __init__(self, initial_state: AgentState) -> None:
        """Initialize the state machine.

        Args:
            initial_state: The starting state for the agent.
        """
        self._current_state = initial_state

    @property
    def current_state(self) -> AgentState:
        """Get the current state of the agent.

        Returns:
            The current AgentState.
        """
        return self._current_state

    def can_trigger(self, event: str) -> bool:
        """Check if an event can be triggered from the current state.

        Args:
            event: The event to check.

        Returns:
            True if the event can be triggered, False otherwise.
        """
        return (self._current_state, event) in self.TRANSITIONS

    def trigger(self, event: str) -> None:
        """Trigger a state transition.

        Args:
            event: The event to trigger.

        Raises:
            InvalidTransitionError: If the transition is not valid.
        """
        key = (self._current_state, event)
        if key not in self.TRANSITIONS:
            raise InvalidTransitionError(self._current_state, event)
        self._current_state = self.TRANSITIONS[key]
