"""Provider data models for LLM interactions."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    """A tool call from the LLM."""

    name: str
    arguments: Dict[str, Any]
    call_id: str


@dataclass
class LLMResponse:
    """Response from an LLM provider."""

    content: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)

    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls."""
        return len(self.tool_calls) > 0


class PaceLevel(Enum):
    """Pacing level for provider health."""

    HEALTHY = "healthy"
    PRESSING = "pressing"
    CRITICAL = "critical"


class Lane(Enum):
    """Traffic lane for request routing."""

    MAIN = "main"
    SUBAGENT = "subagent"


class ErrorClass(Enum):
    """Classification of errors for retry decisions."""

    RETRYABLE = "retryable"
    NON_RETRYABLE = "non_retryable"
    RATE_LIMITED = "rate_limited"


@dataclass
class ProviderProfile:
    """Configuration profile for an LLM provider."""

    name: str
    api_key: str
    base_url: str
    model: str
    rpm_limit: float
    tpm_limit: float
    weight: int = 1


@dataclass
class RateLimitSnapshot:
    """Snapshot of rate limit headers from a provider response."""

    remaining_rpm: Optional[int] = None
    remaining_tpm: Optional[int] = None
    limit_rpm: Optional[int] = None
    limit_tpm: Optional[int] = None
    retry_after_ms: Optional[float] = None


@dataclass
class ProviderHealth:
    """Health state for a provider instance."""

    profile_name: str
    consecutive_errors: int = 0
    last_error_time: Optional[float] = None
    cooldown_until: Optional[float] = None
    pace_level: PaceLevel = field(default=PaceLevel.HEALTHY)


__all__ = [
    "ToolCall",
    "LLMResponse",
    "PaceLevel",
    "Lane",
    "ErrorClass",
    "ProviderProfile",
    "RateLimitSnapshot",
    "ProviderHealth",
]
