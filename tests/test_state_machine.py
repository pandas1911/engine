"""Unit tests for AgentStateMachine.

Validates all 6 valid transitions and multiple invalid transition attempts.
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
# 7 valid transitions (one per TRANSITIONS entry)
# ---------------------------------------------------------------------------


class TestValidTransitions:
    """Each test exercises exactly one entry in the TRANSITIONS dict."""

    def test_start_from_idle(self) -> None:
        """(IDLE, 'start') -> RUNNING"""
        sm = _sm(AgentState.IDLE)
        assert sm.current_state == AgentState.IDLE
        sm.trigger("start")
        assert sm.current_state == AgentState.RUNNING

    def test_spawn_children_from_running(self) -> None:
        """(RUNNING, 'spawn_children') -> WAITING_FOR_CHILDREN"""
        sm = _sm(AgentState.RUNNING)
        sm.trigger("spawn_children")
        assert sm.current_state == AgentState.WAITING_FOR_CHILDREN

    def test_finish_from_running(self) -> None:
        """(RUNNING, 'finish') -> COMPLETED"""
        sm = _sm(AgentState.RUNNING)
        sm.trigger("finish")
        assert sm.current_state == AgentState.COMPLETED

    def test_error_from_running(self) -> None:
        """(RUNNING, 'error') -> ERROR"""
        sm = _sm(AgentState.RUNNING)
        sm.trigger("error")
        assert sm.current_state == AgentState.ERROR

    def test_children_settled_from_waiting(self) -> None:
        """(WAITING_FOR_CHILDREN, 'children_settled') -> RUNNING"""
        sm = _sm(AgentState.WAITING_FOR_CHILDREN)
        sm.trigger("children_settled")
        assert sm.current_state == AgentState.RUNNING

    def test_error_from_waiting(self) -> None:
        """(WAITING_FOR_CHILDREN, 'error') -> ERROR"""
        sm = _sm(AgentState.WAITING_FOR_CHILDREN)
        sm.trigger("error")
        assert sm.current_state == AgentState.ERROR


# ---------------------------------------------------------------------------
# Invalid transitions
# ---------------------------------------------------------------------------


class TestInvalidTransitions:
    """Verify that disallowed events raise InvalidTransitionError."""

    def test_start_from_running(self) -> None:
        """RUNNING rejects 'start' (already started)."""
        sm = _sm(AgentState.RUNNING)
        with pytest.raises(InvalidTransitionError):
            sm.trigger("start")

    def test_finish_from_idle(self) -> None:
        """IDLE rejects 'finish' (never started)."""
        sm = _sm(AgentState.IDLE)
        with pytest.raises(InvalidTransitionError):
            sm.trigger("finish")

    def test_start_from_completed(self) -> None:
        """COMPLETED rejects 'start' (terminal state)."""
        sm = _sm(AgentState.COMPLETED)
        with pytest.raises(InvalidTransitionError):
            sm.trigger("start")

    def test_finish_from_error(self) -> None:
        """ERROR rejects 'finish' (terminal state)."""
        sm = _sm(AgentState.ERROR)
        with pytest.raises(InvalidTransitionError):
            sm.trigger("finish")

    def test_error_from_idle(self) -> None:
        """IDLE rejects 'error' (nothing has started yet)."""
        sm = _sm(AgentState.IDLE)
        with pytest.raises(InvalidTransitionError):
            sm.trigger("error")

    def test_invalid_transition_error_attributes(self) -> None:
        """InvalidTransitionError carries current_state and event attributes."""
        sm = _sm(AgentState.IDLE)
        with pytest.raises(InvalidTransitionError) as exc_info:
            sm.trigger("bogus_event")
        err = exc_info.value
        assert err.current_state == AgentState.IDLE
        assert err.event == "bogus_event"
        # Message should mention both the state and event
        msg = str(err)
        assert "idle" in msg
        assert "bogus_event" in msg


class TestCanTrigger:
    """can_trigger must return True for valid combos and False otherwise."""

    def test_can_trigger_true_for_valid(self) -> None:
        """Every entry in TRANSITIONS should return True from can_trigger."""
        for (state, event), _ in AgentStateMachine.TRANSITIONS.items():
            sm = _sm(state)
            assert sm.can_trigger(event), (
                f"can_trigger({event!r}) should be True from {state.value}"
            )

    def test_can_trigger_false_for_invalid(self) -> None:
        """Representative invalid combos should return False."""
        invalid_combos = [
            (AgentState.IDLE, "finish"),
            (AgentState.RUNNING, "start"),
            (AgentState.RUNNING, "children_settled"),
            (AgentState.COMPLETED, "start"),
            (AgentState.ERROR, "start"),
            (AgentState.ERROR, "finish"),
        ]
        for state, event in invalid_combos:
            sm = _sm(state)
            assert not sm.can_trigger(event), (
                f"can_trigger({event!r}) should be False from {state.value}"
            )


# ---------------------------------------------------------------------------
# Lifecycle / integration-style tests
# ---------------------------------------------------------------------------


class TestLifecycle:
    """Multi-step sequences that exercise the state machine end-to-end."""

    def test_error_halts_lifecycle(self) -> None:
        """An error transition leads to the terminal ERROR state."""
        sm = _sm(AgentState.IDLE)
        sm.trigger("start")
        sm.trigger("error")
        assert sm.current_state == AgentState.ERROR

        # ERROR is terminal — nothing works
        for event in ("start", "finish", "spawn_children",
                       "children_settled", "error"):
            assert not sm.can_trigger(event)

    def test_initial_state_preserved_until_trigger(self) -> None:
        """current_state stays unchanged until trigger is called."""
        for state in AgentState:
            sm = _sm(state)
            assert sm.current_state == state
