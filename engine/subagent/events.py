"""Sub-agent event types.

This module defines the event types used for child-agent lifecycle communication.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from engine.subagent.models import CollectedChildResult


@dataclass
class AgentEvent:
    """Base class for all agent events."""
    pass


@dataclass
class ChildCompletionEvent(AgentEvent):
    """Event emitted when all child agents have completed.

    Attributes:
        child_results: Mapping from child task ID to collected result.
        formatted_prompt: The aggregated prompt built from child outputs.
    """
    child_results: Dict[str, CollectedChildResult] = field(default_factory=dict)
    formatted_prompt: str = ""
