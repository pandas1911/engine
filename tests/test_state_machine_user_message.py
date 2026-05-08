"""Tests for the 'user_message' state trigger.

Validates the WAITING_FOR_CHILDREN -> RUNNING transition via 'user_message'.
Pure in-memory tests — no mocks, no LLM, no network.
"""

import pytest

from engine.runtime.agent_models import AgentState
from engine.runtime.state import AgentStateMachine, InvalidTransitionError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sm(state: AgentState) -> AgentStateMachine:
    """Create a state machine initialised at *state*."""
    return AgentStateMachine(state)


# ---------------------------------------------------------------------------
# Valid transition
# ---------------------------------------------------------------------------


class TestUserMessageTransition:
    """Tests for the 'user_message' trigger from WAITING_FOR_CHILDREN."""

    def test_user_message_from_waiting(self) -> None:
        """(WAITING_FOR_CHILDREN, 'user_message') -> RUNNING"""
        sm = _sm(AgentState.WAITING_FOR_CHILDREN)
        sm.trigger("user_message")
        assert sm.current_state == AgentState.RUNNING

    def test_user_message_can_trigger_true(self) -> None:
        """can_trigger('user_message') is True from WAITING_FOR_CHILDREN."""
        sm = _sm(AgentState.WAITING_FOR_CHILDREN)
        assert sm.can_trigger("user_message") is True


# ---------------------------------------------------------------------------
# Invalid transitions
# ---------------------------------------------------------------------------


class TestUserMessageRejected:
    """Verify that 'user_message' is rejected from non-WAITING states."""

    def test_user_message_from_running_rejected(self) -> None:
        """RUNNING rejects 'user_message'."""
        sm = _sm(AgentState.RUNNING)
        with pytest.raises(InvalidTransitionError):
            sm.trigger("user_message")

    def test_user_message_from_idle_rejected(self) -> None:
        """IDLE rejects 'user_message'."""
        sm = _sm(AgentState.IDLE)
        with pytest.raises(InvalidTransitionError):
            sm.trigger("user_message")

    def test_user_message_from_completed_rejected(self) -> None:
        """COMPLETED rejects 'user_message'."""
        sm = _sm(AgentState.COMPLETED)
        with pytest.raises(InvalidTransitionError):
            sm.trigger("user_message")

    def test_user_message_error_attributes(self) -> None:
        """InvalidTransitionError carries current_state and event attributes."""
        sm = _sm(AgentState.IDLE)
        with pytest.raises(InvalidTransitionError) as exc_info:
            sm.trigger("user_message")
        assert exc_info.value.current_state == AgentState.IDLE
        assert exc_info.value.event == "user_message"
