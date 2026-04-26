"""Provider data models for LLM interactions."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
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
class ProviderConfig:
    """Configuration for an LLM provider instance."""

    name: str
    api_key: str
    base_url: str
    rpm_limit: float = 100
    tpm_limit: float = 100000
    models: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class ProviderParams:
    """Resolved parameters needed to call an LLM provider."""

    api_key: str
    base_url: str
    model: str


def resolve_model_ref(ref: str) -> Tuple[str, str]:
    """Split a 'provider/model' reference into (provider, model).

    Raises:
        ValueError: If the reference does not contain a '/' separator.
    """
    if "/" not in ref:
        raise ValueError(
            f"Invalid model reference '{ref}': expected 'provider/model' format"
        )
    provider, model = ref.split("/", 1)
    return provider, model


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
    "ProviderConfig",
    "ProviderParams",
    "resolve_model_ref",
    "RateLimitSnapshot",
    "ProviderHealth",
]
