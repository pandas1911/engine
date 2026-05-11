"""Sub-agent event types.

This module defines the event types used for child-agent lifecycle communication.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .subagent_models import ChildCompletionNotification


@dataclass
class AgentEvent:
    """Base class for all agent events."""
    pass


@dataclass
class ChildCompletionEvent(AgentEvent):
    """Event emitted when a single child agent completes."""
    notification: "ChildCompletionNotification"
