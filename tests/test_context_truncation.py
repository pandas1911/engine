"""Unit tests for truncate_messages_for_tpm.

Validates round-based truncation, system prompt preservation,
edge cases, and non-mutation — all in-memory, no mocks.
"""

import pytest

from engine.safety.context_truncation import truncate_messages_for_tpm
from engine.safety.token_estimator import EmaTokenEstimator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _estimator(coefficient: float = 3.0) -> EmaTokenEstimator:
    """Create an estimator with a known coefficient for predictable tests."""
    return EmaTokenEstimator(coefficient=coefficient)


def _msg(role: str, content: str) -> dict:
    """Shorthand to create a message dict."""
    return {"role": role, "content": content}


# ---------------------------------------------------------------------------
# 1. Basic truncation — round 1 removed
# ---------------------------------------------------------------------------


def test_basic_truncation() -> None:
    est = _estimator()
    # system + 3 rounds (each round = user + assistant), last user
    messages = [
        _msg("system", "You are helpful."),
        _msg("user", "round 1 question"),
        _msg("assistant", "round 1 answer"),
        _msg("user", "round 2 question"),
        _msg("assistant", "round 2 answer"),
        _msg("user", "round 3 question"),
    ]

    # Compute tokens for system + round2 + round3 (what we expect after removing round 1)
    kept = [messages[0]] + messages[3:]
    kept_tokens = est.estimate(kept, None)

    result = truncate_messages_for_tpm(messages, None, kept_tokens, est)

    assert result.rounds_removed == 1
    assert result.messages[0] == messages[0]  # system preserved
    assert result.messages[1:] == messages[3:]  # round 1 gone
    assert result.original_tokens > result.truncated_tokens
    assert result.truncated_tokens <= kept_tokens


# ---------------------------------------------------------------------------
# 2. No truncation needed — well under limit
# ---------------------------------------------------------------------------


def test_no_truncation_needed() -> None:
    est = _estimator()
    messages = [
        _msg("system", "hi"),
        _msg("user", "hello"),
        _msg("assistant", "hey"),
    ]

    original_tokens = est.estimate(messages, None)
    result = truncate_messages_for_tpm(messages, None, original_tokens + 500, est)

    assert result.rounds_removed == 0
    assert result.messages == messages
    assert result.original_tokens == original_tokens
    assert result.truncated_tokens == original_tokens


# ---------------------------------------------------------------------------
# 3. System prompt alone exceeds TPM — nothing removable
# ---------------------------------------------------------------------------


def test_system_prompt_alone_exceeds_tpm() -> None:
    est = _estimator()
    messages = [
        _msg("system", "a" * 300),
        _msg("user", "hi"),
    ]

    tpm_limit = 1
    result = truncate_messages_for_tpm(messages, None, tpm_limit, est)

    # Only 1 user message => no removable rounds => returned unchanged
    assert result.rounds_removed == 0
    assert result.messages == messages
    assert result.truncated_tokens == result.original_tokens


# ---------------------------------------------------------------------------
# 4. Single round — nothing removable after last round preserved
# ---------------------------------------------------------------------------


def test_single_round_nothing_removable() -> None:
    est = _estimator()
    messages = [
        _msg("system", "sys"),
        _msg("user", "first question"),
        _msg("assistant", "first answer"),
        _msg("user", "current question"),
    ]

    # system + user[0] + assistant[0] + user[1] — only 2 user messages,
    # last is preserved, so only 1 removable round
    # After removing it: system + last user
    kept = [messages[0], messages[3]]
    kept_tokens = est.estimate(kept, None)

    result = truncate_messages_for_tpm(messages, None, kept_tokens, est)

    assert result.rounds_removed == 1
    assert result.messages == kept


# ---------------------------------------------------------------------------
# 5. No user messages — returned unchanged
# ---------------------------------------------------------------------------


def test_no_user_messages() -> None:
    est = _estimator()
    messages = [
        _msg("system", "sys"),
        _msg("assistant", "hello"),
        _msg("assistant", "world"),
    ]

    result = truncate_messages_for_tpm(messages, None, 1, est)

    assert result.rounds_removed == 0
    assert result.messages == messages


# ---------------------------------------------------------------------------
# 6. Consecutive user messages — each is its own round boundary
# ---------------------------------------------------------------------------


def test_consecutive_user_messages() -> None:
    est = _estimator()
    messages = [
        _msg("system", "sys"),
        _msg("user", "question A"),
        _msg("user", "question B"),
        _msg("user", "question C"),
    ]

    # 3 user messages => 2 removable (A, B), 1 preserved (C)
    # Remove both A and B => system + C
    kept = [messages[0], messages[3]]
    kept_tokens = est.estimate(kept, None)

    result = truncate_messages_for_tpm(messages, None, kept_tokens, est)

    assert result.rounds_removed == 2
    assert result.messages == kept


# ---------------------------------------------------------------------------
# 7. Input not mutated
# ---------------------------------------------------------------------------


def test_input_not_mutated() -> None:
    est = _estimator()
    messages = [
        _msg("system", "sys"),
        _msg("user", "q1"),
        _msg("assistant", "a1"),
        _msg("user", "q2"),
    ]

    original_id = id(messages)
    original_copy = list(messages)
    original_content = [m.copy() for m in messages]

    truncate_messages_for_tpm(messages, None, 1, est)

    # Same list object (not replaced)
    assert id(messages) == original_id
    # Contents unchanged
    assert messages == original_copy
    for i, m in enumerate(messages):
        assert m == original_content[i]


# ---------------------------------------------------------------------------
# 8. TPM limit zero or negative — no-op
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tpm_limit", [0, -1, -100])
def test_tpm_limit_zero_or_negative(tpm_limit: int) -> None:
    est = _estimator()
    messages = [
        _msg("system", "sys"),
        _msg("user", "q1"),
        _msg("assistant", "a1"),
        _msg("user", "q2"),
    ]

    result = truncate_messages_for_tpm(messages, None, tpm_limit, est)

    assert result.rounds_removed == 0
    assert result.messages == messages


# ---------------------------------------------------------------------------
# 9. Empty messages list
# ---------------------------------------------------------------------------


def test_empty_messages_list() -> None:
    est = _estimator()
    result = truncate_messages_for_tpm([], None, 100, est)

    assert result.rounds_removed == 0
    assert result.messages == []
    assert result.original_tokens == 1  # max(1, int(0/3.0)) = 1


# ---------------------------------------------------------------------------
# 10. With tools — tool tokens accounted for
# ---------------------------------------------------------------------------


def test_with_tools() -> None:
    est = _estimator()
    messages = [
        _msg("system", "sys"),
        _msg("user", "q1"),
        _msg("assistant", "a1"),
        _msg("user", "q2"),
    ]
    tools = [{"name": "tool_a", "description": "does something useful"}]

    # Compute tokens with tools, set limit so truncation happens
    tokens_with_tools = est.estimate(messages, tools)
    # After removing round 1: system + last user
    kept = [messages[0], messages[3]]
    kept_tokens_with_tools = est.estimate(kept, tools)

    result = truncate_messages_for_tpm(messages, tools, kept_tokens_with_tools, est)

    assert result.rounds_removed == 1
    assert result.messages == kept
    assert result.original_tokens == tokens_with_tools
    assert result.truncated_tokens == kept_tokens_with_tools


# ---------------------------------------------------------------------------
# 11. Last round never removed — even if it exceeds TPM alone
# ---------------------------------------------------------------------------


def test_last_round_never_removed() -> None:
    est = _estimator()
    messages = [
        _msg("system", "sys"),
        _msg("user", "q1"),
        _msg("assistant", "a1"),
        _msg("user", "x" * 600),  # very large last message
    ]

    # Even system + last round exceeds this limit
    system_and_last = [messages[0], messages[3]]
    tokens_after_full_removal = est.estimate(system_and_last, None)

    # Set tpm_limit well below what system + last round needs
    result = truncate_messages_for_tpm(messages, None, 1, est)

    # Should remove the only removable round but keep system + last round
    assert result.rounds_removed == 1
    assert result.messages == system_and_last
    assert result.truncated_tokens == tokens_after_full_removal


# ---------------------------------------------------------------------------
# 12. Multiple rounds removed — 2 out of 3
# ---------------------------------------------------------------------------


def test_multiple_rounds_removed() -> None:
    est = _estimator()
    messages = [
        _msg("system", "sys"),
        _msg("user", "round 1 q"),
        _msg("assistant", "round 1 a"),
        _msg("user", "round 2 q"),
        _msg("assistant", "round 2 a"),
        _msg("user", "round 3 q"),
        _msg("assistant", "round 3 a"),
        _msg("user", "current q"),
    ]

    # 4 user messages => 3 removable, 1 preserved (current q)
    # Remove 2 rounds => system + round3 + current
    kept = [messages[0]] + messages[5:]
    kept_tokens = est.estimate(kept, None)

    result = truncate_messages_for_tpm(
        messages, None, kept_tokens, est
    )

    assert result.rounds_removed == 2
    assert result.messages == kept


# ---------------------------------------------------------------------------
# 13. Interspersed system message — truncated with its round
# ---------------------------------------------------------------------------


def test_interspersed_system_message() -> None:
    est = _estimator()
    messages = [
        _msg("system", "main system prompt"),
        _msg("user", "q1"),
        _msg("system", "intermediate system note"),
        _msg("assistant", "a1"),
        _msg("user", "current q"),
    ]

    # user_indices = [1, 4] => removable round starts at index 1
    # Removing round 1: keep system[0] + everything from index 4 onward
    kept = [messages[0], messages[4]]
    kept_tokens = est.estimate(kept, None)

    result = truncate_messages_for_tpm(messages, None, kept_tokens, est)

    assert result.rounds_removed == 1
    assert result.messages == kept
    # The interspersed system message at index 2 was removed along with its round
    assert all(m["content"] != "intermediate system note" for m in result.messages)
