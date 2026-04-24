# Time-Aware Prompt System for Engine

## TL;DR

> **Quick Summary**: Implement a two-layer time-awareness system for the engine agent framework. Layer 1 injects static date/timezone info into system prompts (cache-friendly). Layer 2 prepends dynamic timestamps to user messages entering `Agent.run()`.
> 
> **Deliverables**:
> - New `engine/time.py` module with `TimeProvider` class
> - Config field for optional timezone override
> - System prompt `<env>` block injection in `delegate()` and sub-agent creation
> - Message timestamp prefix injection in `Agent.run()`
> - Full test coverage
> 
> **Estimated Effort**: Medium
> **Parallel Execution**: YES - 2 waves
> **Critical Path**: Task 1 → Task 3 → Task 4 → Task 5

---

## Context

### Original Request
Design a time-aware prompt system for the engine agent framework, referencing the OpenCode/OpenClaw analysis report. The system should use a two-layer approach: static date in system prompt (cache-friendly) and dynamic timestamp on user messages.

### Interview Summary
**Key Discussions**:
- Layer 1 (System Prompt): OpenCode style `Today's date: Thu Apr 23 2026` plus timezone — stable, changes daily, doesn't break prompt caching
- Layer 2 (Message Prefix): `[Wed 2026-04-23 14:30 CST]` prepended to user messages — per-request, doesn't affect system prompt cache
- NO Layer 3 (tool for precise time) — not needed now, can add later
- Timezone: Auto-detect from system + optional override via `USER_TIMEZONE` env var
- Injection scope: Only messages entering `Agent.run()` (initial messages, not internal system injections)
- Sub-agents: ALL agents get both layers (root and sub-agents unified)
- Timezone abbreviation: Use IANA timezone name in system prompt, use timezone abbreviation in message prefix

**Research Findings**:
- OpenCode: Simple `Today's date: ${date}` in `<env>` block
- OpenClaw: Layered — timezone only in system prompt, timestamp prefix on messages, tool for precise time
- Key insight from report: Include weekday abbreviation because smaller models can't reliably derive weekday from date

### Metis Review
**Identified Gaps** (addressed):
- No `<env>` block exists currently — will append to existing system prompts
- System prompt is set once at construction — Layer 1 must be injected at that point
- `Agent.run()` has two trigger types (`start` and `children_settled`) — only inject on `trigger="start"`
- `datetime.now()` returns naive datetime — must use `datetime.now(tz=...)` consistently
- Config singleton means timezone is fixed for process lifetime — acceptable, document it
- `zoneinfo.ZoneInfo` (stdlib) for timezone — no third-party deps

---

## Work Objectives

### Core Objective
Give every agent in the engine framework time awareness through a two-layer prompt injection strategy that is cache-friendly and requires zero third-party dependencies.

### Concrete Deliverables
- `engine/time.py` — `TimeProvider` class with timezone resolution, date formatting, timestamp formatting
- `engine/config.py` — `user_timezone: Optional[str]` field + `USER_TIMEZONE` env var mapping
- `engine/runtime/agent.py` — Layer 2 injection in `Agent.run()` for `trigger="start"` messages
- `engine/__init__.py` — Layer 1 `<env>` block appended to system prompt in `delegate()`
- `engine/subagent/manager.py` — Layer 1 `<env>` block appended to sub-agent system prompt in `spawn()`
- `tests/test_time_provider.py` — Unit tests for TimeProvider
- `tests/test_agent_time_injection.py` — Integration tests for agent time injection

### Definition of Done
- [ ] `pytest tests/ -v` passes with 0 failures
- [ ] Root agent system prompt contains `<env>` block with date and timezone
- [ ] Sub-agent system prompt contains `<env>` block with date and timezone
- [ ] User messages entering `Agent.run()` are prefixed with `[Dow YYYY-MM-DD HH:MM TZ]`
- [ ] Internal messages (summary warning, child completion) are NOT timestamped
- [ ] Invalid timezone in env var falls back gracefully to system/UTC

### Must Have
- `TimeProvider` class in `engine/time.py` (standalone, no circular deps)
- Timezone resolution chain: config override → system detection → UTC fallback
- Layer 1: `<env>` block with `Today's date:` and `Time zone:` appended to ALL system prompts
- Layer 2: `[Dow YYYY-MM-DD HH:MM TZ]` prefix on messages entering `Agent.run()` with `trigger="start"`
- Full test coverage for TimeProvider + agent integration
- All code comments in English

### Must NOT Have (Guardrails)
- NO Layer 3 time tool (explicitly out of scope)
- NO time context on `tool` or `assistant` role messages
- NO modification to `LLMProvider` or `llm.chat()` interface
- NO modification to `Session.get_messages()` or `Message.to_dict()` — injection happens BEFORE `add_message()`
- NO third-party timezone dependencies (no `pytz`, no `tzlocal`) — use `zoneinfo` stdlib only
- NO modification to `engine/runtime/agent_models.py` (keep models pure)
- NO dynamic system prompt refreshing during an active session (breaks caching)
- NO time context injection on `children_settled` trigger messages or internal system messages
- NO AI slop: over-commenting, unnecessary abstraction, excessive validation beyond timezone resolution

---

## Verification Strategy

> **ZERO HUMAN INTERVENTION** - ALL verification is agent-executed. No exceptions.

### Test Decision
- **Infrastructure exists**: YES (pytest + pytest-asyncio)
- **Automated tests**: YES (Tests-after — implementation first, then tests verify)
- **Framework**: pytest

### QA Policy
Every task MUST include agent-executed QA scenarios.
Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

- **Library/Module**: Use Bash (`pytest`) — Run tests, assert pass/fail
- **Integration**: Use Bash (`pytest`) — Run integration tests, verify message content

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately - foundation + config):
├── Task 1: Create engine/time.py with TimeProvider [deep]
├── Task 2: Add user_timezone to Config [quick]
└── Task 3: Layer 2 — Inject message timestamp in Agent.run() [quick]

Wave 2 (After Wave 1 - system prompt injection + tests):
├── Task 4: Layer 1 — Inject <env> block in delegate() system prompt [quick]
├── Task 5: Layer 1 — Inject <env> block in SubAgentManager.spawn() [quick]
├── Task 6: Unit tests for TimeProvider [unspecified-high]
└── Task 7: Integration tests for agent time injection [unspecified-high]

Wave FINAL (After ALL tasks — 4 parallel reviews):
├── Task F1: Plan compliance audit (oracle)
├── Task F2: Code quality review (unspecified-high)
├── Task F3: Real manual QA (unspecified-high)
└── Task F4: Scope fidelity check (deep)
-> Present results -> Get explicit user okay

Critical Path: Task 1 → Task 3 → Task 4 → Task 6 → F1-F4
Parallel Speedup: ~50% faster than sequential
Max Concurrent: 3 (Wave 1), 4 (Wave 2)
```

### Dependency Matrix

| Task | Depends On | Blocks | Wave |
|------|-----------|--------|------|
| 1 | - | 3, 4, 5, 6, 7 | 1 |
| 2 | - | 1 | 1 |
| 3 | 1, 2 | 4, 7 | 1 |
| 4 | 1, 2 | 6, 7 | 2 |
| 5 | 1, 2 | 6, 7 | 2 |
| 6 | 1, 2, 4, 5 | F1-F4 | 2 |
| 7 | 3, 4, 5 | F1-F4 | 2 |

### Agent Dispatch Summary

- **Wave 1**: **3** — T1 → `deep`, T2 → `quick`, T3 → `quick`
- **Wave 2**: **4** — T4 → `quick`, T5 → `quick`, T6 → `unspecified-high`, T7 → `unspecified-high`
- **FINAL**: **4** — F1 → `oracle`, F2 → `unspecified-high`, F3 → `unspecified-high`, F4 → `deep`

---

## TODOs

- [ ] 1. Create `engine/time.py` with `TimeProvider` class

  **What to do**:
  - Create a new standalone module `engine/time.py`
  - Implement `TimeProvider` class with the following methods:
    - `__init__(self, timezone_override: Optional[str] = None)` — accepts optional IANA timezone string
    - `resolve_timezone(self) -> str` — resolution chain: override → system detection → "UTC" fallback. Validate override with `zoneinfo.ZoneInfo` (catch `ZoneInfoNotFoundError`), on failure log warning and fall back to system timezone via `datetime.now().astimezone().tzinfo` or `time.tzname`. Return IANA string like `"Asia/Shanghai"`.
    - `format_system_env_block(self) -> str` — returns a multi-line string:
      ```
      <env>
        Today's date: Thu Apr 23 2026
        Time zone: Asia/Shanghai
      </env>
      ```
      Use `datetime.now(tz=timezone_obj).strftime("%a %b %d %Y")` for the date. Use the resolved IANA timezone name.
    - `format_message_timestamp(self, now: Optional[datetime] = None) -> str` — returns string like `[Wed 2026-04-23 14:30 CST]`. Format: `[{weekday_abbr} {YYYY-MM-DD} {HH:MM} {tz_abbr}]`. Use `datetime.now(tz=timezone_obj)` if `now` not provided. Get timezone abbreviation via `now.strftime("%Z")` or `now.tzname()`.
    - `inject_timestamp(self, message: str, now: Optional[datetime] = None) -> str` — prepend `format_message_timestamp()` to message. Skip if message already has a timestamp prefix (regex check for `^\[...\d{4}-\d{2}-\d{2}...\]`). Return `f"{ts} {message}"`.
  - Use only stdlib: `datetime`, `zoneinfo`, `re`, `logging`, `time`. NO third-party deps.
  - All comments and docstrings in English.

  **Must NOT do**:
  - Import from `engine.runtime.agent` or `engine.subagent` (circular dep risk)
  - Add third-party dependencies
  - Over-abstract (one class is enough, no factory/strategy patterns)

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Self-contained module with timezone logic, needs careful stdlib usage
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2, 3)
  - **Blocks**: Tasks 3, 4, 5, 6, 7
  - **Blocked By**: None (can start immediately)

  **References**:

  **Pattern References**:
  - `engine/config.py:8-25` — Dataclass field pattern for simple config values
  - `engine/providers/provider_models.py` — Example of a clean, standalone data module with no circular deps

  **External References**:
  - Python stdlib `zoneinfo.ZoneInfo`: https://docs.python.org/3/library/zoneinfo.html — timezone-aware datetime handling
  - Python `datetime.strftime` format codes: `%a` (weekday abbr), `%Y-%m-%d`, `%H:%M`, `%Z` (tz abbr)

  **Acceptance Criteria**:

  - [ ] `engine/time.py` exists and imports without error: `python -c "from engine.time import TimeProvider"`
  - [ ] `TimeProvider()` with no args resolves to system timezone (not empty, not None)
  - [ ] `TimeProvider("Asia/Shanghai").resolve_timezone()` returns `"Asia/Shanghai"`
  - [ ] `TimeProvider("Invalid/Zone").resolve_timezone()` falls back to system timezone without raising
  - [ ] `format_system_env_block()` returns string containing `<env>`, `Today's date:`, `Time zone:`
  - [ ] `format_message_timestamp()` returns string matching pattern `[... ....-..-.. ..:.. ...]`
  - [ ] `inject_timestamp("hello")` returns `"[...] hello"` format
  - [ ] `inject_timestamp("[Wed 2026-04-23 14:30 CST] hello")` returns unchanged (no double-inject)

  **QA Scenarios:**

  ```
  Scenario: TimeProvider resolves valid timezone override
    Tool: Bash (python -c)
    Preconditions: Python environment with engine module available
    Steps:
      1. Run: python -c "from engine.time import TimeProvider; t = TimeProvider('Asia/Shanghai'); print(t.resolve_timezone())"
      2. Assert output contains "Asia/Shanghai"
    Expected Result: "Asia/Shanghai" printed to stdout
    Failure Indicators: Different timezone printed, or exception raised
    Evidence: .sisyphus/evidence/task-1-valid-tz.txt

  Scenario: TimeProvider falls back on invalid timezone
    Tool: Bash (python -c)
    Preconditions: Python environment available
    Steps:
      1. Run: python -c "from engine.time import TimeProvider; t = TimeProvider('Invalid/Zone'); tz = t.resolve_timezone(); print(tz); assert tz != 'Invalid/Zone', 'Should not return invalid timezone'"
      2. Assert no exception, output is a valid timezone string
    Expected Result: A valid IANA timezone string (system tz or UTC), no exception
    Failure Indicators: ZoneInfoNotFoundError raised, or empty string returned
    Evidence: .sisyphus/evidence/task-1-invalid-tz-fallback.txt

  Scenario: format_system_env_block produces expected structure
    Tool: Bash (python -c)
    Preconditions: Python environment available
    Steps:
      1. Run: python -c "from engine.time import TimeProvider; t = TimeProvider('UTC'); block = t.format_system_env_block(); assert '<env>' in block; assert 'Today' in block; assert 'Time zone:' in block; print(block)"
      2. Assert output contains <env>, Today's date, Time zone
    Expected Result: Multi-line string with <env> block
    Failure Indicators: AssertionError, or missing expected strings
    Evidence: .sisyphus/evidence/task-1-env-block.txt
  ```

  **Commit**: YES (groups with Task 2)
  - Message: `feat(time): add TimeProvider module and timezone config`
  - Files: `engine/time.py`
  - Pre-commit: `python -c "from engine.time import TimeProvider"`

- [ ] 2. Add `user_timezone` field to `Config` dataclass

  **What to do**:
  - In `engine/config.py`, add `user_timezone: Optional[str] = None` field to the `Config` dataclass (after `log_dir`)
  - Add `"USER_TIMEZONE": "user_timezone"` mapping to `ConfigLoader.MAPPING`
  - In `ConfigLoader.load()`, after the existing `log_dir` handling block, add timezone loading:
    ```python
    user_timezone = provider.get("USER_TIMEZONE")
    if user_timezone:
        config.user_timezone = user_timezone
    ```
  - This allows users to set `USER_TIMEZONE=Asia/Shanghai` in `.env` or environment

  **Must NOT do**:
  - Change existing config fields or their defaults
  - Add timezone validation to Config (that's TimeProvider's job)
  - Modify ConfigLoader.REQUIRED_KEYS (timezone is optional)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple dataclass field addition, follows existing pattern exactly
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 3)
  - **Blocks**: Tasks 1 (as data dependency, but can be developed in parallel)
  - **Blocked By**: None (can start immediately)

  **References**:

  **Pattern References**:
  - `engine/config.py:25-26` — Existing optional field pattern (`log_dir: Optional[str] = None`)
  - `engine/config.py:52-57` — Existing MAPPING dict pattern
  - `engine/config.py:83-85` — Existing optional config loading pattern for `log_dir`

  **Acceptance Criteria**:

  - [ ] `Config` dataclass has `user_timezone: Optional[str] = None` field
  - [ ] `ConfigLoader.MAPPING` contains `"USER_TIMEZONE": "user_timezone"`
  - [ ] `ConfigLoader.load()` with `USER_TIMEZONE=Asia/Shanghai` sets `config.user_timezone = "Asia/Shanghai"`
  - [ ] `ConfigLoader.load()` without `USER_TIMEZONE` sets `config.user_timezone = None`

  **QA Scenarios:**

  ```
  Scenario: Config loads USER_TIMEZONE from environment
    Tool: Bash (python -c)
    Preconditions: Python environment available
    Steps:
      1. Run: USER_TIMEZONE=Asia/Shanghai python -c "from engine.config import ConfigLoader, ConfigProvider; class P(ConfigProvider):
          def get(self, k):
              return {'LLM_API_KEY':'k','LLM_BASE_URL':'u','LLM_MODEL':'m','USER_TIMEZONE':'Asia/Shanghai'}.get(k)
          c = ConfigLoader.load(P()); assert c.user_timezone == 'Asia/Shanghai', f'got {c.user_timezone}'; print('PASS')"
      2. Assert "PASS" in output
    Expected Result: "PASS" printed
    Failure Indicators: AssertionError, or config.user_timezone is None
    Evidence: .sisyphus/evidence/task-2-config-tz.txt

  Scenario: Config defaults user_timezone to None when not set
    Tool: Bash (python -c)
    Preconditions: Python environment available
    Steps:
      1. Run: python -c "from engine.config import ConfigLoader, ConfigProvider; class P(ConfigProvider):
          def get(self, k):
              return {'LLM_API_KEY':'k','LLM_BASE_URL':'u','LLM_MODEL':'m'}.get(k)
          c = ConfigLoader.load(P()); assert c.user_timezone is None, f'got {c.user_timezone}'; print('PASS')"
      2. Assert "PASS" in output
    Expected Result: "PASS" printed
    Failure Indicators: AssertionError
    Evidence: .sisyphus/evidence/task-2-config-no-tz.txt
  ```

  **Commit**: YES (groups with Task 1)
  - Message: `feat(time): add TimeProvider module and timezone config`
  - Files: `engine/config.py`
  - Pre-commit: `python -c "from engine.config import Config; c = Config(api_key='k', base_url='u', model='m'); assert c.user_timezone is None"`

- [ ] 3. Layer 2 — Inject message timestamp in `Agent.run()`

  **What to do**:
  - In `engine/runtime/agent.py`, import `TimeProvider` at the top
  - In `Agent.__init__()`, create `self._time_provider = TimeProvider(timezone_override=config.user_timezone)`
  - In `Agent.run()`, modify the message injection at line 157:
    ```python
    async def run(self, message: Optional[str] = None, *, trigger: str = "start") -> str:
        if message:
            # Layer 2: Only inject timestamp on "start" trigger (external user messages)
            if trigger == "start":
                message = self._time_provider.inject_timestamp(message)
            self.session.add_message("user", message)
    ```
  - The `trigger == "start"` check ensures:
    - ✅ Root agent initial message gets timestamp
    - ✅ Sub-agent task descriptions get timestamp (they come through `run(task_desc)`)
    - ❌ `children_settled` resume messages do NOT get timestamp (internal, formatted child results)
    - ❌ Summary warnings in `_build_summary_warning()` do NOT get timestamp (added via separate `add_message()` calls outside `run()`)

  **Must NOT do**:
  - Modify `Session.add_message()` — injection must happen before it
  - Inject timestamps on `trigger="children_settled"` messages
  - Inject timestamps on messages added by `_build_summary_warning()` or `_emergency_summarize()`
  - Modify `Message.to_dict()` or `Session.get_messages()`

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single injection point, 3 lines of code change
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES (but depends on Tasks 1, 2 for import)
  - **Parallel Group**: Wave 1 (with Tasks 1, 2 — but actually needs Task 1 to exist first)
  - **Blocks**: Tasks 7
  - **Blocked By**: Tasks 1, 2

  **References**:

  **Pattern References**:
  - `engine/runtime/agent.py:155-158` — Current `run()` method where message is added to session
  - `engine/runtime/agent.py:341-355` — `_build_summary_warning()` — internal user message that should NOT be timestamped (called outside `run()`)
  - `engine/runtime/agent.py:383-390` — Emergency summary prompt — internal user message that should NOT be timestamped (called outside `run()`)
  - `engine/runtime/agent.py:496` — `event.formatted_prompt` — child completion injected via `run(trigger="children_settled")`, should NOT be timestamped

  **API/Type References**:
  - `engine/time.py:TimeProvider.inject_timestamp(message, now)` — the method to call

  **Acceptance Criteria**:

  - [ ] `Agent.__init__()` creates `self._time_provider` from config
  - [ ] `Agent.run("hello")` with `trigger="start"` adds message with `[...] hello` prefix to session
  - [ ] `Agent.run(formatted, trigger="children_settled")` adds message WITHOUT timestamp prefix
  - [ ] Existing tests still pass (no regression)

  **QA Scenarios:**

  ```
  Scenario: Agent.run() prefixes start trigger messages with timestamp
    Tool: Bash (pytest)
    Preconditions: Tasks 1, 2 completed; mock LLM provider available
    Steps:
      1. Create a test that instantiates Agent with mock LLM, calls agent.run("hello there"), then inspects session.messages
      2. Assert the first user message content starts with "[" (timestamp prefix)
      3. Assert "hello there" is in the message content
    Expected Result: User message has timestamp prefix like "[Wed 2026-04-23 14:30 CST] hello there"
    Failure Indicators: Message content is "hello there" without prefix
    Evidence: .sisyphus/evidence/task-3-start-timestamp.txt

  Scenario: Agent.run() does NOT prefix children_settled messages
    Tool: Bash (pytest)
    Preconditions: Tasks 1, 2 completed; mock LLM provider available
    Steps:
      1. Create test that calls agent.run("child results here", trigger="children_settled")
      2. Assert the user message content is "child results here" (no prefix)
    Expected Result: No timestamp prefix on children_settled messages
    Failure Indicators: Message starts with "["
    Evidence: .sisyphus/evidence/task-3-no-children-settled-prefix.txt
  ```

  **Commit**: YES
  - Message: `feat(agent): inject message timestamps on Agent.run()`
  - Files: `engine/runtime/agent.py`
  - Pre-commit: `python -c "from engine.runtime.agent import Agent"`

- [ ] 4. Layer 1 — Inject `<env>` block in `delegate()` system prompt

  **What to do**:
  - In `engine/__init__.py`, import `TimeProvider` at the top
  - Modify the `delegate()` function:
    ```python
    async def delegate(...) -> AgentResult:
        time_provider = TimeProvider(timezone_override=config.user_timezone if config else None)
        
        session = Session(id=f"root_{uuid.uuid4().hex[:8]}", depth=0)
        
        # Layer 1: Inject static time info into system prompt
        base_system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        env_block = time_provider.format_system_env_block()
        full_system_prompt = f"{base_system_prompt}\n\n{env_block}"
        session.add_message("system", full_system_prompt)
    ```
  - The `<env>` block is appended AFTER the user's custom system prompt (or default one)
  - Since system prompt is set once at construction, the date is accurate for when `delegate()` was called

  **Must NOT do**:
  - Replace the user's system prompt — always append
  - Modify `DEFAULT_SYSTEM_PROMPT` itself
  - Create a separate system message (append to existing one)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 3 lines of change in a single function
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 5, 6, 7)
  - **Blocks**: Tasks 6, 7
  - **Blocked By**: Tasks 1, 2

  **References**:

  **Pattern References**:
  - `engine/__init__.py:67-76` — Current `delegate()` function, specifically lines 74-75 where system prompt is added

  **Acceptance Criteria**:

  - [ ] `delegate()` creates `TimeProvider` with config timezone
  - [ ] Root agent system prompt ends with `<env>` block containing date and timezone
  - [ ] Custom `system_prompt` parameter is preserved (prepended before `<env>` block)
  - [ ] Default system prompt still works (when `system_prompt=None`)

  **QA Scenarios:**

  ```
  Scenario: delegate() system prompt contains <env> block
    Tool: Bash (pytest)
    Preconditions: Tasks 1, 2 completed
    Steps:
      1. Create test that calls delegate() with mock LLM, then inspects session.messages[0]
      2. Assert first message (system) contains "<env>" and "Today's date:"
      3. Assert it also contains the base system prompt text
    Expected Result: System message contains both base prompt and <env> block
    Failure Indicators: <env> block missing, or base prompt missing
    Evidence: .sisyphus/evidence/task-4-delegate-env-block.txt

  Scenario: Custom system prompt is preserved with <env> appended
    Tool: Bash (pytest)
    Preconditions: Tasks 1, 2 completed
    Steps:
      1. Create test with custom system_prompt="My custom prompt"
      2. Assert session.messages[0].content starts with "My custom prompt"
      3. Assert session.messages[0].content contains "<env>"
    Expected Result: Both custom prompt and <env> block present
    Failure Indicators: Custom prompt replaced or <env> missing
    Evidence: .sisyphus/evidence/task-4-custom-prompt.txt
  ```

  **Commit**: YES (groups with Task 5)
  - Message: `feat(prompt): inject static date/timezone in system prompts`
  - Files: `engine/__init__.py`
  - Pre-commit: `python -c "from engine import delegate"`

- [ ] 5. Layer 1 — Inject `<env>` block in `SubAgentManager.spawn()` system prompt

  **What to do**:
  - In `engine/subagent/manager.py`, import `TimeProvider` at the top
  - In `SubAgentManager.__init__()`, store `self._time_provider = TimeProvider(timezone_override=config.user_timezone if config else None)`
  - In `SubAgentManager.spawn()`, after the `system_prompt` f-string is built (around line 225), append the `<env>` block:
    ```python
    # After existing system_prompt f-string...
    env_block = self._time_provider.format_system_env_block()
    system_prompt = f"{system_prompt}\n\n{env_block}"
    
    child_session.add_message("system", system_prompt)
    ```
  - This ensures sub-agents also have time context in their system prompt

  **Must NOT do**:
  - Modify the sub-agent system prompt template structure
  - Create a separate system message for the <env> block
  - Import from `engine.runtime.agent` or `engine.__init__`

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 3 lines of change, follows Task 4 pattern exactly
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 4, 6, 7)
  - **Blocks**: Tasks 6, 7
  - **Blocked By**: Tasks 1, 2

  **References**:

  **Pattern References**:
  - `engine/subagent/manager.py:185-227` — Sub-agent system prompt construction and session.add_message call
  - Task 4 in this plan — identical pattern for root agent (append <env> block to system prompt)

  **Acceptance Criteria**:

  - [ ] `SubAgentManager.__init__()` creates `self._time_provider`
  - [ ] Sub-agent system prompt ends with `<env>` block containing date and timezone
  - [ ] Existing sub-agent prompt template (rules, context) is preserved above <env> block

  **QA Scenarios:**

  ```
  Scenario: Sub-agent system prompt contains <env> block
    Tool: Bash (pytest)
    Preconditions: Tasks 1, 2 completed
    Steps:
      1. Create test that spawns a sub-agent (via manager.spawn() with mocks)
      2. Inspect child_session.messages[0] (system message)
      3. Assert it contains "<env>" and "Today's date:"
      4. Assert it also contains "You are a **subagent**" (original prompt)
    Expected Result: System message has both sub-agent prompt and <env> block
    Failure Indicators: <env> missing, or original sub-agent prompt overwritten
    Evidence: .sisyphus/evidence/task-5-subagent-env-block.txt
  ```

  **Commit**: YES (groups with Task 4)
  - Message: `feat(prompt): inject static date/timezone in system prompts`
  - Files: `engine/subagent/manager.py`
  - Pre-commit: `python -c "from engine.subagent.manager import SubAgentManager"`

- [ ] 6. Unit tests for `TimeProvider`

  **What to do**:
  - Create `tests/test_time_provider.py`
  - Test cases:
    1. `test_resolve_timezone_with_valid_override` — `TimeProvider("Asia/Shanghai").resolve_timezone()` == `"Asia/Shanghai"`
    2. `test_resolve_timezone_with_invalid_override` — `TimeProvider("Invalid/Zone").resolve_timezone()` returns valid fallback (system or UTC), no exception
    3. `test_resolve_timezone_no_override` — `TimeProvider().resolve_timezone()` returns non-empty string
    4. `test_format_system_env_block` — output contains `<env>`, `Today's date:`, `Time zone:`, properly formatted date
    5. `test_format_system_env_block_date_format` — date matches `%a %b %d %Y` pattern (e.g., "Thu Apr 23 2026")
    6. `test_format_message_timestamp` — output matches `\[...\d{4}-\d{2}-\d{2} \d{2}:\d{2} ...\]` pattern
    7. `test_format_message_timestamp_with_fixed_time` — use `datetime(2026, 4, 23, 14, 30, tzinfo=ZoneInfo("Asia/Shanghai"))` as `now` parameter, assert exact output `[Wed 2026-04-23 14:30 ...]`
    8. `test_inject_timestamp_adds_prefix` — `inject_timestamp("hello")` starts with `[` and contains "hello"
    9. `test_inject_timestamp_skips_already_stamped` — `inject_timestamp("[Wed 2026-04-23 14:30 CST] hello")` returns unchanged
    10. `test_inject_timestamp_empty_message` — empty string handled gracefully
    11. `test_different_timezones_produce_different_blocks` — UTC vs Asia/Shanghai produce different `Time zone:` lines

  **Must NOT do**:
  - Require network access or real API keys
  - Use hardcoded dates that will fail in the future (use `now` parameter for deterministic tests)
  - Test internal implementation details — test public API only

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Comprehensive test suite, 10+ test cases, needs careful assertion design
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 4, 5, 7)
  - **Blocks**: F1-F4
  - **Blocked By**: Tasks 1, 2

  **References**:

  **Test References**:
  - `tests/conftest.py` — Existing fixture patterns (Config creation, mock providers)
  - `tests/test_easy_task.py` — Example of test structure using pytest-asyncio

  **Pattern References**:
  - `engine/time.py:TimeProvider` — All public methods to test

  **Acceptance Criteria**:

  - [ ] `pytest tests/test_time_provider.py -v` passes with 0 failures
  - [ ] All 11 test cases listed above are implemented
  - [ ] No tests require network access or external dependencies

  **QA Scenarios:**

  ```
  Scenario: All TimeProvider unit tests pass
    Tool: Bash (pytest)
    Preconditions: engine/time.py exists with TimeProvider class
    Steps:
      1. Run: pytest tests/test_time_provider.py -v
      2. Assert all tests pass (exit code 0)
      3. Assert at least 10 test functions were collected
    Expected Result: All tests pass, 0 failures
    Failure Indicators: Any test FAIL or ERROR
    Evidence: .sisyphus/evidence/task-6-test-results.txt

  Scenario: Tests are deterministic (no flaky date-dependent failures)
    Tool: Bash (pytest)
    Preconditions: Tests written with `now` parameter for time-dependent tests
    Steps:
      1. Run tests twice in succession: pytest tests/test_time_provider.py -v
      2. Run again: pytest tests/test_time_provider.py -v
      3. Assert both runs produce identical pass/fail results
    Expected Result: Both runs pass identically
    Failure Indicators: Different results between runs
    Evidence: .sisyphus/evidence/task-6-deterministic.txt
  ```

  **Commit**: YES (groups with Task 7)
  - Message: `test(time): add TimeProvider unit tests and agent integration tests`
  - Files: `tests/test_time_provider.py`
  - Pre-commit: `pytest tests/test_time_provider.py -v`

- [ ] 7. Integration tests for agent time injection

  **What to do**:
  - Create `tests/test_agent_time_injection.py`
  - Test cases:
    1. `test_root_agent_message_has_timestamp` — Create Agent with mock LLM, call `run("hello")`, verify session.messages contains user message with `[...] hello` prefix
    2. `test_children_settled_no_timestamp` — Create Agent, call `run("results", trigger="children_settled")`, verify NO timestamp prefix
    3. `test_delegate_system_prompt_has_env_block` — Call `delegate()` with mock config, verify session.messages[0] (system) contains `<env>` block
    4. `test_delegate_custom_prompt_preserved_with_env` — Call `delegate()` with custom system_prompt, verify both custom text and `<env>` present
    5. `test_subagent_system_prompt_has_env_block` — Spawn sub-agent via SubAgentManager with mocks, verify child session's system message has `<env>` block
    6. `test_summary_warning_no_timestamp` — Verify `_build_summary_warning()` content does NOT get timestamped (it's added directly via add_message, not through run())
    7. `test_timezone_override_propagates` — Set `config.user_timezone = "Asia/Tokyo"`, verify both system prompt and message timestamp use Asia/Tokyo

  **Must NOT do**:
  - Require network access or real API keys
  - Test across multiple real LLM calls (use mocks)
  - Test sub-agent execution end-to-end (too complex, just test system prompt construction)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Integration tests requiring mock setup, multiple injection points to verify
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 4, 5, 6)
  - **Blocks**: F1-F4
  - **Blocked By**: Tasks 3, 4, 5

  **References**:

  **Test References**:
  - `tests/conftest.py:11-19` — Config fixture pattern
  - `tests/conftest.py:33-41` — Mock LLM provider fixture pattern

  **Pattern References**:
  - `engine/runtime/agent.py:155-158` — Agent.run() injection point
  - `engine/__init__.py:67-76` — delegate() system prompt construction
  - `engine/subagent/manager.py:185-227` — Sub-agent system prompt construction

  **Acceptance Criteria**:

  - [ ] `pytest tests/test_agent_time_injection.py -v` passes with 0 failures
  - [ ] All 7 test cases listed above are implemented
  - [ ] Full test suite still passes: `pytest tests/ -v`

  **QA Scenarios:**

  ```
  Scenario: All integration tests pass
    Tool: Bash (pytest)
    Preconditions: All implementation tasks (1-5) completed
    Steps:
      1. Run: pytest tests/test_agent_time_injection.py -v
      2. Assert all tests pass (exit code 0)
      3. Assert at least 7 test functions collected
    Expected Result: All tests pass
    Failure Indicators: Any FAIL or ERROR
    Evidence: .sisyphus/evidence/task-7-integration-results.txt

  Scenario: Full test suite regression
    Tool: Bash (pytest)
    Preconditions: All implementation and test tasks completed
    Steps:
      1. Run: pytest tests/ -v
      2. Assert 0 failures, 0 errors
      3. Compare test count with pre-implementation baseline (should increase by ~18)
    Expected Result: All existing + new tests pass
    Failure Indicators: Any regression in existing tests
    Evidence: .sisyphus/evidence/task-7-full-suite.txt
  ```

  **Commit**: YES (groups with Task 6)
  - Message: `test(time): add TimeProvider unit tests and agent integration tests`
  - Files: `tests/test_agent_time_injection.py`
  - Pre-commit: `pytest tests/test_agent_time_injection.py -v`

---

## Final Verification Wave (MANDATORY — after ALL implementation tasks)

> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.

- [ ] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists (read file, run command). For each "Must NOT Have": search codebase for forbidden patterns — reject with file:line if found. Check evidence files exist in .sisyphus/evidence/. Compare deliverables against plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [ ] F2. **Code Quality Review** — `unspecified-high`
  Run `pytest tests/ -v` + ruff check. Review all changed files for: `as any`/type ignores, empty catches, print statements in prod, commented-out code, unused imports. Check AI slop: excessive comments, over-abstraction, generic names.
  Output: `Build [PASS/FAIL] | Lint [PASS/FAIL] | Tests [N pass/N fail] | Files [N clean/N issues] | VERDICT`

- [ ] F3. **Real Manual QA** — `unspecified-high`
  Start from clean state. Run ALL test files. Verify: TimeProvider produces correct output with mock timezones, agent messages contain timestamp prefixes, system prompts contain `<env>` blocks, internal messages do NOT have timestamps. Save to `.sisyphus/evidence/final-qa/`.
  Output: `Scenarios [N/N pass] | Integration [N/N] | Edge Cases [N tested] | VERDICT`

- [ ] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", read actual diff. Verify 1:1 — everything in spec was built, nothing beyond spec was built. Check "Must NOT do" compliance. Flag unaccounted changes.
  Output: `Tasks [N/N compliant] | Contamination [CLEAN/N issues] | Unaccounted [CLEAN/N files] | VERDICT`

---

## Commit Strategy

- **Task 1+2**: `feat(time): add TimeProvider module and timezone config` — engine/time.py, engine/config.py
- **Task 3**: `feat(agent): inject message timestamps on Agent.run()` — engine/runtime/agent.py
- **Task 4+5**: `feat(prompt): inject static date/timezone in system prompts` — engine/__init__.py, engine/subagent/manager.py
- **Task 6+7**: `test(time): add TimeProvider unit tests and agent integration tests` — tests/test_time_provider.py, tests/test_agent_time_injection.py

---

## Success Criteria

### Verification Commands
```bash
pytest tests/ -v                                    # Expected: all tests pass, 0 failures
pytest tests/test_time_provider.py -v               # Expected: TimeProvider tests pass
pytest tests/test_agent_time_injection.py -v        # Expected: agent injection tests pass
python -c "from engine.time import TimeProvider; t = TimeProvider(); print(t.format_system_env_block())"  # Expected: prints <env> block with date
```

### Final Checklist
- [ ] All "Must Have" present
- [ ] All "Must NOT Have" absent
- [ ] All tests pass
- [ ] No third-party dependencies added
- [ ] All code comments in English
