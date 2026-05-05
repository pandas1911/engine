# TPM-Based Context Truncation

## TL;DR

> **Quick Summary**: When a single LLM request's estimated tokens exceed the provider's TPM limit, automatically truncate conversation history by removing complete rounds (oldest first) until the request fits.
> 
> **Deliverables**:
> - New standalone module `engine/safety/context_truncation.py` with pure truncation function
> - Integration into `FallbackLLMProvider.chat()` and `stream_chat()` (inside iteration loop, per-provider TPM)
> - Unit tests for truncation logic + integration tests for FallbackProvider
> - Updated `docs/codebase-structure.md`
> 
> **Estimated Effort**: Short
> **Parallel Execution**: YES - 3 waves
> **Critical Path**: Task 1 → Task 3 → Task 4 → Task 5

---

## Context

### Original Request
As conversation grows longer, each request sends more context tokens to the LLM. Eventually a single request can exceed the provider's TPM limit, causing the provider to reject it. The current rate limiter caps `estimated_tokens` to prevent deadlock but doesn't actually reduce the request size.

### Interview Summary
**Key Discussions**:
- **Truncation unit**: A "round" = from a user message to the next user message, including all assistant, tool, and interspersed system messages in between.
- **Direction**: Remove from the beginning (oldest first). Keep the first system prompt (sacred), then start cutting from the first user message.
- **Interspersed system prompts**: If a system prompt appears between user messages, it gets truncated along with the round.
- **No completion token reservation**: Use full TPM limit as budget.
- **Integration point**: Inside FallbackProvider's iteration loop, after `acquire_key()` + limiter lookup, using the specific provider's `limiter.tpm_limit`.
- **Standalone function**: Write as a pure function in a new module, not a method on FallbackProvider.
- **Logging**: Warning-level log when truncation occurs, include rounds removed and token counts.
- **Tests**: Write tests after implementation.

### Metis Review
**Identified Gaps** (addressed):
- **Multi-provider TPM mismatch**: User chose per-provider truncation inside the iteration loop.
- **System prompt alone exceeds TPM**: Auto-resolved — return `[system_prompt]` only + log error.
- **Last round must never be truncated**: Auto-resolved — the current round (last user message + trailing messages) is always preserved.
- **tpm_limit <= 0**: Auto-resolved — no-op, return messages unchanged.
- **No user messages in conversation**: Auto-resolved — return messages unchanged.
- **Input mutation safety**: Auto-resolved — `session.get_messages()` returns new list, but truncation function itself also returns a new list for safety.

---

## Work Objectives

### Core Objective
Prevent LLM requests from exceeding provider TPM limits by truncating conversation history before sending.

### Concrete Deliverables
- `engine/safety/context_truncation.py` — standalone truncation function
- Updated `engine/safety/__init__.py` — export new function
- Updated `engine/providers/fallback_provider.py` — call truncation in `chat()` and `stream_chat()`
- `tests/test_context_truncation.py` — unit + integration tests
- Updated `docs/codebase-structure.md` — document new module

### Definition of Done
- [ ] `uv run pytest tests/test_context_truncation.py` → all tests pass
- [ ] `uv run pytest` → no regressions
- [ ] Truncation function is pure (no side effects, no input mutation)
- [ ] System prompt never removed
- [ ] Last round (current user input) never removed

### Must Have
- Round-based truncation from oldest first
- System prompt (messages[0] with role=system) always preserved
- Last round always preserved (current user input)
- Per-provider TPM limit used (inside iteration loop)
- Warning log on truncation with event_type="context_truncated"
- Pure function, returns new list
- Works for both `chat()` and `stream_chat()` paths

### Must NOT Have (Guardrails)
- Do NOT modify `engine/safety/rate_limit.py`
- Do NOT modify `engine/runtime/agent.py` or `engine/runtime/agent_models.py`
- Do NOT modify `app/routers/chat.py` or its `_truncate_session`
- Do NOT add partial message truncation (slicing content within a single message)
- Do NOT add smart round selection (longest first, etc.)
- Do NOT add configurable minimum rounds
- Do NOT add Prometheus metrics or alerting
- Do NOT add retry logic inside truncation function
- Do NOT add completion token reservation
- Do NOT mutate the input messages list

---

## Verification Strategy (MANDATORY)

> **ZERO HUMAN INTERVENTION** - ALL verification is agent-executed. No exceptions.

### Test Decision
- **Infrastructure exists**: YES (pytest, see `tests/test_rate_limiter.py` for pattern)
- **Automated tests**: Tests-after
- **Framework**: pytest

### QA Policy
Every task MUST include agent-executed QA scenarios.
Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

- **Library/Module**: Use Bash (`uv run pytest`) — Run tests, assert pass/fail counts

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately - standalone truncation function):
├── Task 1: Create truncation function [deep]
└── Task 2: Export from __init__.py [quick]

Wave 2 (After Wave 1 - integration):
└── Task 3: Integrate into FallbackProvider [deep]

Wave 3 (After Wave 2 - tests + docs):
├── Task 4: Unit tests for truncation function [deep]
├── Task 5: Integration tests for FallbackProvider [deep]
└── Task 6: Update docs/codebase-structure.md [quick]

Wave FINAL (After ALL tasks — 4 parallel reviews):
├── Task F1: Plan compliance audit (oracle)
├── Task F2: Code quality review (unspecified-high)
├── Task F3: Real manual QA (unspecified-high)
└── Task F4: Scope fidelity check (deep)
-> Present results -> Get explicit user okay

Critical Path: Task 1 → Task 3 → Task 5 → F1-F4
Parallel Speedup: Wave 1 runs 2 tasks in parallel, Wave 3 runs 3 tasks in parallel
```

### Dependency Matrix

| Task | Depends On | Blocks | Wave |
|------|-----------|--------|------|
| 1    | -         | 2, 3, 4 | 1   |
| 2    | 1         | 3, 5   | 1   |
| 3    | 1, 2      | 5, 6   | 2   |
| 4    | 1         | -      | 3   |
| 5    | 3         | -      | 3   |
| 6    | 3         | -      | 3   |

### Agent Dispatch Summary

- **Wave 1**: **2** — T1 → `deep`, T2 → `quick`
- **Wave 2**: **1** — T3 → `deep`
- **Wave 3**: **3** — T4 → `deep`, T5 → `deep`, T6 → `quick`
- **FINAL**: **4** — F1 → `oracle`, F2 → `unspecified-high`, F3 → `unspecified-high`, F4 → `deep`

---

## TODOs

- [ ] 1. Create `engine/safety/context_truncation.py` — standalone truncation function

  **What to do**:
  - Create new file `engine/safety/context_truncation.py`
  - Implement the following code:

  ```python
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
  ```

  **Edge cases to handle**:
  - No user messages at all → return messages unchanged
  - Only 1 user message (current round, nothing removable) → return messages unchanged even if over TPM
  - Consecutive user messages → each is its own round boundary
  - Empty messages list → return empty list
  - `tools=None` or `tools=[]` → works correctly with estimator

  **Must NOT do**:
  - Do NOT mutate the input messages list
  - Do NOT add retry logic
  - Do NOT add partial message content truncation
  - Do NOT log inside this function (caller handles logging)
  - Do NOT import from `engine.providers` or `engine.runtime` (avoid circular deps)

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Core logic function with multiple edge cases to handle correctly
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Task 2)
  - **Parallel Group**: Wave 1
  - **Blocks**: Tasks 2, 3, 4
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `app/routers/chat.py:26-52` — Existing `_find_turn_boundaries()` and `_truncate_session()` for reference on how rounds/turns are identified. Adapt the concept but operate on `List[Dict]` (not `Message` objects).
  - `engine/safety/token_estimator.py:25-29` — `EmaTokenEstimator.estimate(messages, tools)` signature and return format
  - `engine/runtime/agent_models.py:44-59` — `Message.to_dict()` to understand message dict format: `{"role": str, "content": str, "tool_call_id": str (optional), "tool_calls": list (optional)}`

  **API/Type References**:
  - `engine/safety/token_estimator.py:EmaTokenEstimator` — Import this type for the function parameter. Use `estimate(messages, tools) -> int`
  - `typing.List, typing.Optional, typing.Tuple` or a dataclass for return type

  **WHY Each Reference Matters**:
  - `chat.py:26-52`: Shows the existing turn-boundary identification pattern — adapt the logic but for dicts
  - `token_estimator.py:25-29`: Must call `estimate()` correctly with both messages and tools
  - `agent_models.py:44-59`: Must handle all message dict fields (role, content, tool_call_id, tool_calls) correctly — don't strip fields

  **Acceptance Criteria**:

  - [ ] File `engine/safety/context_truncation.py` created
  - [ ] Function `truncate_messages_for_tpm` is pure — returns new list, no side effects
  - [ ] Handles all edge cases listed above
  - [ ] No imports from `engine.providers` or `engine.runtime`

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Basic truncation — messages exceed TPM, rounds removed
    Tool: Bash (uv run pytest)
    Preconditions: truncation function importable
    Steps:
      1. Create messages: [system("prompt"), user("msg1"), assistant("resp1"), user("msg2"), assistant("resp2"), user("current")]
      2. Set tpm_limit low enough that estimated tokens > limit after removing round 1
      3. Call truncate_messages_for_tpm(messages, tools=None, tpm_limit=tpm_limit, token_estimator=estimator)
      4. Assert result contains [system, user("msg2"), assistant("resp2"), user("current")] — round 1 removed
      5. Assert rounds_removed == 1
      6. Assert original_tokens > truncated_tokens
    Expected Result: Oldest round removed, system prompt preserved, last round preserved
    Failure Indicators: System prompt missing, last round missing, input mutated
    Evidence: .sisyphus/evidence/task-1-basic-truncation.txt

  Scenario: No truncation needed — messages already under TPM
    Tool: Bash (uv run pytest)
    Preconditions: truncation function importable
    Steps:
      1. Create messages with estimated tokens well under tpm_limit
      2. Call truncate_messages_for_tpm
      3. Assert result equals original messages
      4. Assert rounds_removed == 0
    Expected Result: Messages returned unchanged
    Failure Indicators: rounds_removed > 0, result differs from input
    Evidence: .sisyphus/evidence/task-1-no-truncation.txt

  Scenario: System prompt alone exceeds TPM — nothing removable
    Tool: Bash (uv run pytest)
    Steps:
      1. Create messages: [system(very_long_content), user("current")]
      2. Set tpm_limit lower than system prompt tokens
      3. Call truncate_messages_for_tpm
      4. Assert result == [system(very_long_content)] (only system prompt retained, no crash)
    Expected Result: Returns just system prompt, no infinite loop
    Evidence: .sisyphus/evidence/task-1-system-exceeds.txt
  ```

  **Commit**: NO (groups with all tasks)

---

- [ ] 2. Export truncation function from `engine/safety/__init__.py`

  **What to do**:
  - Add import of `truncate_messages_for_tpm` (and `TruncationResult` if used) from `engine.safety.context_truncation` in `engine/safety/__init__.py`
  - Add to `__all__` list

  **Must NOT do**:
  - Do NOT modify any existing imports or exports
  - Do NOT change the import order of existing items

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single file, two-line change (import + __all__ update)
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Task 1, though logically depends on Task 1's API)
  - **Parallel Group**: Wave 1
  - **Blocks**: Task 3
  - **Blocked By**: Task 1 (needs to know exact function name / export name)

  **References**:

  **Pattern References**:
  - `engine/safety/__init__.py:1-14` — Current import pattern and `__all__` list to follow exactly

  **WHY Each Reference Matters**:
  - `__init__.py:1-14`: Shows exact format — `from engine.safety.X import Y` then `__all__ = [...]`

  **Acceptance Criteria**:

  - [ ] `from engine.safety import truncate_messages_for_tpm` works without error
  - [ ] `truncate_messages_for_tpm` appears in `engine.safety.__all__`

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Import works from engine.safety
    Tool: Bash
    Steps:
      1. Run: uv run python -c "from engine.safety import truncate_messages_for_tpm; print('OK')"
      2. Assert output contains "OK"
    Expected Result: Import succeeds
    Evidence: .sisyphus/evidence/task-2-import.txt
  ```

  **Commit**: NO (groups with all tasks)

- [ ] 3. Integrate truncation into FallbackLLMProvider

  **What to do**:
  - In `engine/providers/fallback_provider.py`, add import at the top:
    ```python
    from engine.safety.context_truncation import truncate_messages_for_tpm
    ```
  - In `chat()` method, add truncation **inside the iteration loop**, after limiter lookup (line ~79) and **before** `limiter.acquire()` (line ~82). The exact insertion point:

  ```python
  # --- INSERT AFTER line 79 (limiter = self._rate_limiters.get(provider_name)) ---
  # --- AND BEFORE line 82 (reservation_id = await limiter.acquire(...)) ---

  # Truncate context if it exceeds this provider's TPM limit
  local_messages = messages
  local_estimated_tokens = estimated_tokens
  if limiter is not None and limiter.tpm_limit > 0:
      truncation = truncate_messages_for_tpm(
          messages, tools, limiter.tpm_limit, self._token_estimator
      )
      if truncation.rounds_removed > 0:
          self._logger.warning(
              "RateControl",
              "Context truncated for TPM | profile={}, rounds_removed={}, tokens={}/{}".format(
                  profile_name,
                  truncation.rounds_removed,
                  truncation.original_tokens,
                  truncation.truncated_tokens,
              ),
              event_type="context_truncated",
              data={
                  "profile": profile_name,
                  "rounds_removed": truncation.rounds_removed,
                  "original_tokens": truncation.original_tokens,
                  "truncated_tokens": truncation.truncated_tokens,
              },
          )
      local_messages = truncation.messages
      local_estimated_tokens = truncation.truncated_tokens

  reservation_id = 0
  if limiter is not None:
      reservation_id = await limiter.acquire(estimated_tokens=local_estimated_tokens)
  # --- END INSERT ---

  # Then change provider.chat() call (line ~85) to use local_messages:
  result = await provider.chat(
      messages=local_messages,  # <-- changed from messages
      tools=tools,
      ...
  )
  ```

  - In `stream_chat()` method, apply the **identical** truncation block inside the iteration loop, after `limiter = self._rate_limiters.get(provider_name)` (line ~244) and before `limiter.acquire()` (line ~247). Same pattern:
    ```python
    # Same truncation block as above
    local_messages = messages
    local_estimated_tokens = estimated_tokens
    if limiter is not None and limiter.tpm_limit > 0:
        truncation = truncate_messages_for_tpm(...)
        if truncation.rounds_removed > 0:
            self._logger.warning(...)
        local_messages = truncation.messages
        local_estimated_tokens = truncation.truncated_tokens

    # acquire with local_estimated_tokens
    reservation_id = await limiter.acquire(estimated_tokens=local_estimated_tokens)

    # stream with local_messages
    async for chunk in provider.stream_chat(messages=local_messages, ...):
    ```

  - **Remove the old lines** that the new code replaces:
    - Old line 80-82: `reservation_id = 0 / if limiter is not None: / reservation_id = await limiter.acquire(estimated_tokens=estimated_tokens)` — replaced by the new block above
    - Old line 85-91: `provider.chat(messages=messages, ...)` — change `messages` to `local_messages`

  - **Critical**: Truncation happens INSIDE the loop so the correct provider's `limiter.tpm_limit` is used

  **Must NOT do**:
  - Do NOT modify `messages` parameter — create `local_messages` copy
  - Do NOT move truncation outside the iteration loop
  - Do NOT change any existing error handling or key rotation logic
  - Do NOT modify `estimated_tokens` calculation outside the loop (line 58 / line 226) — it's still needed for the initial estimate

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Must understand FallbackProvider's retry loop flow and integrate correctly without breaking key rotation / error handling
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2 (sequential, after Wave 1)
  - **Blocks**: Tasks 4, 5, 6
  - **Blocked By**: Tasks 1, 2

  **References**:

  **Pattern References**:
  - `engine/providers/fallback_provider.py:64-82` — The iteration loop in `chat()` showing where `acquire_key()`, provider lookup, limiter lookup happen. Truncation goes AFTER line 79 (`limiter = self._rate_limiters.get(provider_name)`) and BEFORE line 82 (`await limiter.acquire(...)`)
  - `engine/providers/fallback_provider.py:232-247` — The same pattern in `stream_chat()`. Apply identical truncation logic.
  - `engine/providers/fallback_provider.py:104-112` — Logging pattern to follow for truncation log (use `self._logger.warning()` with `event_type` and `data` dict)

  **API/Type References**:
  - `engine/safety/context_truncation.py:truncate_messages_for_tpm` — The function from Task 1, returns `TruncationResult` with `.messages`, `.rounds_removed`, `.original_tokens`, `.truncated_tokens`
  - `engine/safety/rate_limit.py:370-382` — `SlidingWindowRateLimiter.tpm_limit` and `rpm_limit` properties (accessed as `limiter.tpm_limit`)

  **WHY Each Reference Matters**:
  - `fallback_provider.py:64-82`: This IS the integration point. Must go inside this exact location.
  - `fallback_provider.py:232-247`: Same pattern for stream — must be identical logic.
  - `fallback_provider.py:104-112`: Logging convention — must match existing style.

  **Acceptance Criteria**:

  - [ ] Truncation called inside iteration loop, after limiter lookup, before `acquire()`
  - [ ] Both `chat()` and `stream_chat()` paths have identical truncation logic
  - [ ] `local_messages` used for provider call, original `messages` unchanged
  - [ ] Log emitted with `event_type="context_truncated"` when truncation occurs
  - [ ] No change to error handling, key rotation, or retry logic

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Truncation called when messages exceed TPM
    Tool: Bash (uv run pytest)
    Steps:
      1. Mock FallbackProvider with messages exceeding tpm_limit
      2. Assert truncate_messages_for_tpm is called with correct tpm_limit
      3. Assert provider.chat() receives truncated messages, not original
    Expected Result: Truncation applied, truncated messages forwarded to provider
    Evidence: .sisyphus/evidence/task-3-truncation-called.txt

  Scenario: No truncation when messages fit under TPM
    Tool: Bash (uv run pytest)
    Steps:
      1. Mock FallbackProvider with messages under tpm_limit
      2. Assert provider.chat() receives original messages unchanged
      3. Assert no truncation log emitted
    Expected Result: Messages pass through unchanged
    Evidence: .sisyphus/evidence/task-3-no-truncation.txt

  Scenario: stream_chat also truncates correctly
    Tool: Bash (uv run pytest)
    Steps:
      1. Mock FallbackProvider with messages exceeding tpm_limit
      2. Call stream_chat, assert truncate_messages_for_tpm called
      3. Assert provider.stream_chat() receives truncated messages
    Expected Result: Same truncation behavior in streaming path
    Evidence: .sisyphus/evidence/task-3-stream-truncation.txt
  ```

  **Commit**: NO (groups with all tasks)

- [ ] 4. Unit tests for truncation function

  **What to do**:
  - Create `tests/test_context_truncation.py`
  - Write unit tests for `truncate_messages_for_tpm` covering:
    1. **Basic truncation**: 5 rounds, remove 2 oldest, verify result contains system + remaining rounds
    2. **No truncation needed**: tokens already under limit → unchanged
    3. **System prompt alone exceeds TPM**: only system prompt returned, no crash
    4. **Single round (nothing removable)**: only system + 1 user message → returned unchanged even if over TPM
    5. **No user messages at all**: all system/assistant messages → returned unchanged
    6. **Consecutive user messages**: two user messages back-to-back → each is its own round
    7. **Input not mutated**: verify original list `id()` and content unchanged after call
    8. **tpm_limit <= 0**: no-op, returns unchanged
    9. **Empty messages list**: returns empty list
    10. **With tools**: tools contribute to token estimation, truncation accounts for them
    11. **Last round never removed**: even if last round alone exceeds TPM, it stays
    12. **Multiple rounds removed**: verify correct number of rounds_removed in result
    13. **Interspersed system message**: system message between user messages gets truncated with its round
  - Use concrete message data (no vague fixtures)
  - Follow test pattern from `tests/test_rate_limiter.py` (existing test style in project)

  **Must NOT do**:
  - Do NOT require live LLM calls
  - Do NOT test FallbackProvider integration here (that's Task 5)
  - Do NOT use vague assertions like "verify it works"

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Many edge cases to cover thoroughly with concrete test data
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 5, 6)
  - **Parallel Group**: Wave 3
  - **Blocks**: None
  - **Blocked By**: Task 1

  **References**:

  **Pattern References**:
  - `tests/test_rate_limiter.py:1-30` — Test file structure, imports, helper functions pattern to follow
  - `engine/safety/context_truncation.py` — The function under test (from Task 1)
  - `engine/safety/token_estimator.py:EmaTokenEstimator` — Need to create estimator instances for test inputs

  **WHY Each Reference Matters**:
  - `test_rate_limiter.py:1-30`: Shows project's testing conventions — import style, helper factories, assertion patterns
  - `token_estimator.py`: Must instantiate estimator with known coefficient for predictable test results

  **Acceptance Criteria**:

  - [ ] `uv run pytest tests/test_context_truncation.py` → all tests pass
  - [ ] At least 13 test cases covering all scenarios listed above
  - [ ] No live LLM calls in any test

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: All unit tests pass
    Tool: Bash
    Steps:
      1. Run: uv run pytest tests/test_context_truncation.py -v
      2. Assert all tests pass (exit code 0)
      3. Count test cases >= 13
    Expected Result: All tests pass, comprehensive coverage
    Failure Indicators: Any test failure, fewer than 13 tests
    Evidence: .sisyphus/evidence/task-4-unit-tests.txt

  Scenario: No regressions in existing tests
    Tool: Bash
    Steps:
      1. Run: uv run pytest
      2. Assert all tests pass
    Expected Result: No existing test broken
    Evidence: .sisyphus/evidence/task-4-no-regression.txt
  ```

  **Commit**: NO (groups with all tasks)

---

- [ ] 5. Integration tests for FallbackProvider

  **What to do**:
  - Add integration tests to `tests/test_context_truncation.py` (or a separate `tests/test_fallback_truncation.py` if preferred)
  - Test that FallbackLLMProvider correctly calls truncation and forwards truncated messages:
    1. **chat() path with truncation**: mock provider with high messages → verify `provider.chat()` called with truncated messages, verify truncation log emitted
    2. **stream_chat() path with truncation**: same verification for streaming
    3. **chat() without truncation**: messages under limit → original messages forwarded
    4. **Provider fallback with different TPMs**: primary provider has low TPM (truncation), fallback has high TPM (no truncation) — verify correct per-provider behavior
  - Use `unittest.mock.AsyncMock` to mock LLMProvider and SlidingWindowRateLimiter
  - Verify `estimated_tokens` passed to `limiter.acquire()` reflects truncated amount

  **Must NOT do**:
  - Do NOT require live LLM API calls
  - Do NOT test truncation function logic here (that's Task 4)

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Async mocking, multi-component integration, both chat/stream paths
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 4, 6)
  - **Parallel Group**: Wave 3
  - **Blocks**: None
  - **Blocked By**: Task 3

  **References**:

  **Pattern References**:
  - `engine/providers/fallback_provider.py:50-209` — Full `chat()` implementation, the method under test
  - `engine/providers/fallback_provider.py:211-345` — Full `stream_chat()` implementation
  - `engine/providers/fallback_provider.py:29-45` — Constructor showing all dependencies for mocking

  **API/Type References**:
  - `engine/providers/llm_provider.py:BaseLLMProvider` — Base class for mocking providers
  - `engine/safety/rate_limit.py:SlidingWindowRateLimiter` — For mocking the limiter with specific `tpm_limit`
  - `engine/safety/key_pool.py:APIKeyPool` — For mocking key pool
  - `engine/safety/retry.py:RetryEngine` — For mocking retry engine

  **WHY Each Reference Matters**:
  - `fallback_provider.py:50-209`: This IS the code under test — must understand the full flow to mock correctly
  - Constructor (L29-45): Shows all dependencies needed to construct FallbackLLMProvider in tests

  **Acceptance Criteria**:

  - [ ] All integration tests pass
  - [ ] Tests verify truncated messages forwarded to provider (not original)
  - [ ] Tests verify per-provider TPM used for truncation
  - [ ] No live API calls

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Integration tests pass
    Tool: Bash
    Steps:
      1. Run: uv run pytest tests/test_context_truncation.py -v -k "integration"
      2. Assert all integration tests pass
    Expected Result: All integration tests pass
    Failure Indicators: Any test failure, mock assertion error
    Evidence: .sisyphus/evidence/task-5-integration-tests.txt
  ```

  **Commit**: NO (groups with all tasks)

---

- [ ] 6. Update `docs/codebase-structure.md`

  **What to do**:
  - Read `docs/codebase-structure.md` and find the `engine/safety/` section
  - Add `context_truncation.py` entry describing the new truncation function
  - Note the export in `__init__.py` (if the doc lists exports)
  - Verify all other sections are still accurate after the changes

  **Must NOT do**:
  - Do NOT modify any content unrelated to the new module
  - Do NOT restructure or reorganize existing documentation

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single documentation update, straightforward
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 4, 5)
  - **Parallel Group**: Wave 3
  - **Blocks**: None
  - **Blocked By**: Task 3 (needs to know final function signature)

  **References**:

  **Pattern References**:
  - `docs/codebase-structure.md` — Find the safety section and follow existing entry format

  **WHY Each Reference Matters**:
  - `docs/codebase-structure.md`: Per AGENTS.md rule 4, must be kept in sync with codebase

  **Acceptance Criteria**:

  - [ ] `context_truncation.py` documented in `docs/codebase-structure.md`
  - [ ] Document matches actual code (function name, parameters, behavior)

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Documentation is accurate
    Tool: Bash
    Steps:
      1. Grep docs/codebase-structure.md for "context_truncation"
      2. Assert entry exists and describes truncate_messages_for_tpm
    Expected Result: Documentation present and accurate
    Evidence: .sisyphus/evidence/task-6-docs.txt
  ```

  **Commit**: NO (groups with all tasks)

---

## Final Verification Wave (MANDATORY — after ALL implementation tasks)

> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.

- [ ] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists (read file, run command). For each "Must NOT Have": search codebase for forbidden patterns — reject with file:line if found. Check evidence files exist in .sisyphus/evidence/. Compare deliverables against plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [ ] F2. **Code Quality Review** — `unspecified-high`
  Run `uv run pytest` + any linter. Review all changed files for: unused imports, empty catches, console.log in prod, commented-out code. Check AI slop: excessive comments, over-abstraction, generic names.
  Output: `Tests [N pass/N fail] | Files [N clean/N issues] | VERDICT`

- [ ] F3. **Real Manual QA** — `unspecified-high`
  Start from clean state. Execute EVERY QA scenario from EVERY task — follow exact steps, capture evidence. Test cross-task integration (truncation + rate limiter working together). Test edge cases: empty messages, single message, system prompt only. Save to `.sisyphus/evidence/final-qa/`.
  Output: `Scenarios [N/N pass] | Integration [N/N] | Edge Cases [N tested] | VERDICT`

- [ ] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", read actual diff (git log/diff). Verify 1:1 — everything in spec was built (no missing), nothing beyond spec was built (no creep). Check "Must NOT do" compliance. Detect cross-task contamination. Flag unaccounted changes.
  Output: `Tasks [N/N compliant] | Contamination [CLEAN/N issues] | Unaccounted [CLEAN/N files] | VERDICT`

---

## Commit Strategy

- **Single Commit**: `feat(safety): add TPM-based context truncation for long conversations`
  - `engine/safety/context_truncation.py`, `engine/safety/__init__.py`, `engine/providers/fallback_provider.py`, `tests/test_context_truncation.py`, `docs/codebase-structure.md`
  - Pre-commit: `uv run pytest`

---

## Success Criteria

### Verification Commands
```bash
uv run pytest tests/test_context_truncation.py -v  # Expected: all tests pass
uv run pytest  # Expected: no regressions, all existing tests pass
```

### Final Checklist
- [ ] All "Must Have" present
- [ ] All "Must NOT Have" absent
- [ ] All tests pass
- [ ] docs/codebase-structure.md updated
