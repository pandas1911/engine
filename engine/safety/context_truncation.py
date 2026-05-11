"""TPM-based context truncation for long conversations.

When a single LLM request's estimated tokens exceed the provider's TPM limit,
truncates conversation history by removing complete rounds (oldest first)
until the request fits under the limit.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from engine.safety.token_estimator import EmaTokenEstimator


@dataclass
class TruncationResult:
    """Result of context truncation."""
    messages: List[Dict]
    rounds_removed: int
    original_tokens: int
    truncated_tokens: int


def _find_round_boundaries(messages: List[Dict]) -> List[int]:
    """Return indices of all user messages (round start positions)."""
    return [i for i, m in enumerate(messages) if m.get("role") == "user"]


def truncate_messages_for_tpm(
    messages: List[Dict],
    tools: Optional[List[Dict]],
    tpm_limit: int,
    token_estimator: EmaTokenEstimator,
) -> TruncationResult:
    """Truncate conversation history to fit under TPM limit.

    Removes complete rounds (user → next-user boundary) from oldest first.
    Always preserves:
      - messages[0] if role == "system" (sacred system prompt)
      - The last round (current user input + trailing messages)

    Args:
        messages: List of message dicts in OpenAI format.
        tools: Optional list of tool definition dicts.
        tpm_limit: Provider's tokens-per-minute limit.
        token_estimator: EmaTokenEstimator instance for token counting.

    Returns:
        TruncationResult with (possibly truncated) messages, stats.
        Input list is NEVER mutated.
    """
    original_tokens = token_estimator.estimate(messages, tools)

    # Fast path: already under limit or TPM disabled
    if tpm_limit <= 0 or original_tokens <= tpm_limit:
        return TruncationResult(
            messages=list(messages),
            rounds_removed=0,
            original_tokens=original_tokens,
            truncated_tokens=original_tokens,
        )

    # Find system prompt boundary
    has_system = len(messages) > 0 and messages[0].get("role") == "system"
    system_end = 1 if has_system else 0

    # Find round boundaries (indices of user messages in original messages)
    user_indices = _find_round_boundaries(messages)

    # Need at least 2 user messages to have removable rounds:
    # user_indices[-1] = last round (current input, never removed)
    # user_indices[:-1] = removable rounds
    if len(user_indices) < 2:
        return TruncationResult(
            messages=list(messages),
            rounds_removed=0,
            original_tokens=original_tokens,
            truncated_tokens=original_tokens,
        )

    last_round_start = user_indices[-1]
    removable_count = len(user_indices) - 1  # all except last round

    # Try removing 1 round, then 2, then 3... until under TPM.
    # Always slice from the original messages list to avoid stale indices.
    for rounds_to_remove in range(1, removable_count + 1):
        first_kept = user_indices[rounds_to_remove]
        candidate = list(messages[:system_end]) + list(messages[first_kept:])
        candidate_tokens = token_estimator.estimate(candidate, tools)
        if candidate_tokens <= tpm_limit:
            return TruncationResult(
                messages=candidate,
                rounds_removed=rounds_to_remove,
                original_tokens=original_tokens,
                truncated_tokens=candidate_tokens,
            )

    # All removable rounds removed, keep only system + last round
    candidate = list(messages[:system_end]) + list(messages[last_round_start:])
    candidate_tokens = token_estimator.estimate(candidate, tools)
    return TruncationResult(
        messages=candidate,
        rounds_removed=removable_count,
        original_tokens=original_tokens,
        truncated_tokens=candidate_tokens,
    )
