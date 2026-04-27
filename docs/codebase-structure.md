# Engine Codebase Structure

> A multi-agent orchestration framework that supports nested sub-agent spawning, multi-provider LLM routing with primary/fallback ordering, and per-provider rate limiting.

---

## Directory Overview

```
engine/
├── engine/                    # Core package
│   ├── __init__.py            # Thin re-export layer (re-exports from runner.py and submodules)
│   ├── runner.py              # delegate(), DEFAULT_SYSTEM_PROMPT, custom tool discovery
│   ├── config.py              # Configuration loading (engine.json)
│   ├── time.py                # Timezone-aware time utilities
│   ├── safety/                # Rate limiting, concurrency, retry, pacing
│   │   ├── __init__.py        # Re-export layer for all safety classes
│   │   ├── concurrency.py     # ConcurrencyLimiter, LaneConcurrencyQueue, LaneSlot, LaneStatus
│   │   ├── rate_limit.py      # SlidingWindowRateLimiter
│   │   ├── token_estimator.py # EmaTokenEstimator — adaptive chars→tokens estimator
│   │   ├── key_pool.py        # APIKeyPool
│   │   ├── retry.py           # RetryEngine
│   │   └── pacing.py          # AdaptivePacer, ResultTruncator, RegistrySizeMonitor
│   ├── runtime/               # Agent execution core
│   │   ├── __init__.py
│   │   ├── agent.py           # Agent class — main execution loop
│   │   ├── agent_models.py    # Data models (Session, Message, AgentResult, etc.)
│   │   ├── state.py           # Agent state machine
│   │   └── task_registry.py   # Task CRUD with handler-based notification
│   ├── providers/             # LLM provider layer
│   │   ├── __init__.py
│   │   ├── llm_provider.py    # BaseLLMProvider / LLMProvider (OpenAI-compatible)
│   │   ├── provider_models.py # Data models (ToolCall, LLMResponse, Lane, etc.)
│   │   └── fallback_provider.py  # Multi-provider fallback with key rotation
│   ├── subagent/              # Sub-agent spawning and lifecycle
│   │   ├── __init__.py
│   │   ├── manager.py         # SubAgentManager — spawn, gate-check, notify
│   │   ├── spawn.py           # SpawnTool — thin tool wrapper
│   │   ├── protocol.py        # Drainable protocol definition
│   │   ├── events.py          # Event types (ChildCompletionEvent)
│   │   └── subagent_models.py # SubagentTask, CollectedChildResult
│   ├── tools/                 # Extensible tool system
│   │   ├── __init__.py
│   │   ├── base.py            # Tool ABC, FunctionTool, ToolRegistry
│   │   ├── builtin/           # Built-in tools (empty, reserved)
│   │   │   └── __init__.py
│   │   └── custom/            # Auto-discovered custom tools (web search, web fetch)
│   │       ├── __init__.py
│   │       └── web_fetch.py   # URL content fetching with HTML→Markdown/Text conversion
│   └── logging/               # Structured logging
│       ├── __init__.py
│       └── sink.py            # Logger, formatters, async file handler
├── tests/                     # Test suite
│   ├── test_easy_task.py      # Simple delegation test
│   └── test_multilayer_subagent.py  # Multi-layer nesting test
├── docs/                      # Documentation
│   └── codebase-structure.md  # This file
├── logs/                      # Runtime log output (JSONL)
├── pyproject.toml             # Project metadata and dependencies
├── engine.json.example        # Example configuration file
├── .env.example               # Example environment variables
├── AGENTS.md                  # Agent collaboration rules
└── .gitignore
```

---

## Module Details

### 1. `engine/__init__.py` — Thin Re-export Layer

A minimal re-export module (12 lines) that re-exports the public API from `runner.py` and submodules. All implementation logic was extracted to `runner.py`.

**Re-exports:**

| Symbol | Source |
|---|---|
| `delegate` | `engine.runner` |
| `DEFAULT_SYSTEM_PROMPT` | `engine.runner` |
| `_discover_custom_tools` | `engine.runner` |
| `_refresh_custom_tools` | `engine.runner` |
| `Tool`, `FunctionTool` | `engine.tools.base` |
| `AgentResult` | `engine.runtime.agent_models` |
| `AgentTaskRegistry` | `engine.runtime.task_registry` |
| `init_logger`, `get_logger`, `stop_logger` | `engine.logging` |

---

### 2. `engine/runner.py` — Delegation Runner

The main entry point containing `delegate()` and all startup orchestration logic. Extracted from the original `engine/__init__.py`.

**Key functions:**

| Function | Description |
|---|---|
| `delegate(task_description, system_prompt?, tools?, config?)` | Main entry point. Creates a root agent session, initializes all infrastructure (providers, rate limiters, pacers, key pool, lane queue), discovers custom tools, and runs the agent loop. Returns `AgentResult`. |
| `_discover_custom_tools()` | Auto-discovers `Tool` subclasses from `engine/tools/custom/*.py` using `importlib` + `inspect`. Results are cached. |
| `_refresh_custom_tools()` | Clears the custom tools cache. |

**Key constants:**

- `DEFAULT_SYSTEM_PROMPT` — The default system prompt for the root agent, which instructs it to decompose tasks, dispatch work in parallel, and synthesize results.

**Startup flow (`delegate()`):**

1. Load config via `get_config()` (auto-discovers `engine.json`)
2. Create `TimeProvider`, inject timezone info into system prompt
3. Initialize logger with configured log directory
4. Iterate `config.providers` dict — for each provider, create `SlidingWindowRateLimiter` and `AdaptivePacer`; for each model under that provider, create an `LLMProvider` keyed by `"provider/model"`
5. Build ordered key list from `config.primary` + `config.fallback`
6. Create shared: `APIKeyPool` (with ordered composite key names), `RetryEngine`, `FallbackLLMProvider`
7. Create `LaneConcurrencyQueue` (MAIN + SUBAGENT lanes)
8. Discover and merge custom tools
9. Create root `Agent`, register in `AgentTaskRegistry`
10. Run the agent, return `AgentResult`

---

### 3. `engine/config.py` — Configuration

Loads runtime configuration from `engine.json`.

**Classes:**

| Class | Description |
|---|---|
| `Config` | Dataclass holding all configuration values (providers dict, primary/fallback model refs, retry settings, concurrency limits, pacing, etc.) |
| `ConfigLoader` | Static methods for discovering and loading `engine.json`. Validates `providers` dict structure, `primary`/`fallback` references, and per-model `model_params` for reserved keys. |

**Config fields:**

| Field | Default | Description |
|---|---|---|
| `providers` | `{}` | `Dict[str, ProviderConfig]` — nested dict keyed by provider name. Each entry defines api_key, base_url, rpm_limit, tpm_limit, and a `models` dict of model_name → model_params |
| `primary` | `""` | Required. Primary model reference in `"provider/model"` format |
| `fallback` | `[]` | Optional list of fallback model references in `"provider/model"` format |
| `strip_thinking` | `True` | Remove `<think/>` tags from LLM responses |
| `max_depth` | `3` | Maximum sub-agent nesting depth |
| `spawn_timeout` | `60.0` | Seconds to wait for a concurrency slot before rejecting spawn |
| `max_result_length` | `3000` | Max chars for child agent results before truncation |
| `summary_warning_reserve` | `2` | Iterations before limit to inject summary warning |
| `emergency_summary_enabled` | `True` | Force text-only LLM call when iteration limit reached |
| `log_dir` | `None` | Directory for JSONL log files (defaults to `logs/`) |
| `llm_retry_max_attempts` | `3` | Max retry attempts per LLM call |
| `llm_retry_base_delay` | `1.0` | Base delay in seconds for exponential backoff |
| `main_lane_concurrency` | `4` | Max concurrent agents in the MAIN lane |
| `subagent_lane_concurrency` | `5` | Max concurrent sub-agents in the SUBAGENT lane |
| `pacing_enabled` | `True` | Enable adaptive request pacing |
| `pacing_min_interval_ms` | `500.0` | Minimum interval between LLM calls |
| `key_rotation_enabled` | `True` | ⚠️ **Unused** — defined in config but not referenced in codebase |
| `fallback_enabled` | `True` | ⚠️ **Unused** — defined in config but not referenced in codebase |
| `cooldown_initial_ms` | `30000.0` | Initial key cooldown on rate limit |
| `cooldown_max_ms` | `300000.0` | Maximum key cooldown |
| `user_timezone` | `None` | Timezone override (env var `USER_TIMEZONE` takes precedence) |

**Config discovery strategy:**

1. Check CWD for `engine.json`
2. Walk upward to find `pyproject.toml`, check that directory for `engine.json`
3. Raise `FileNotFoundError` if not found

---

### 4. `engine/safety/` — Rate Limiting & Safety Guards

A package providing resource protection mechanisms for the agent system. Split into focused sub-modules, with `__init__.py` re-exporting all public classes for backward compatibility.

#### `__init__.py` — Re-export Layer

Re-exports all classes from sub-modules so that `from engine.safety import ...` continues to work without changes.

#### `concurrency.py` — Concurrency Control

| Class | Description |
|---|---|
| `ConcurrencyLimiter` | Asyncio.Semaphore wrapper with observable active count |
| `LaneConcurrencyQueue` | Per-lane (MAIN/SUBAGENT) concurrency control with FIFO queuing via `asyncio.Condition` |
| `LaneSlot` | Async context manager representing a concurrency slot |
| `LaneStatus` | Data class for lane status queries |
| `_LaneState` | Internal state per lane |

#### `rate_limit.py` — Sliding Window Rate Limiter

| Class | Description |
|---|---|
| `SlidingWindowRateLimiter` | Dual RPM/TPM sliding window with event-driven scheduler (no busy waiting) |

**Key flow:** Fast path (capacity available, no waiters) → immediate return. Slow path → enqueue Future, background `_scheduler` task wakes waiters when capacity frees up.

**Deadlock prevention:**

- `acquire()` caps `estimated_tokens` to `tpm_limit` so a single oversized request cannot block forever when estimated > capacity.
- `_scheduler()` includes deadlock detection: when the sliding window is empty but a waiter still cannot proceed (because its estimated request exceeds the full capacity), the scheduler force-releases the waiter to prevent permanent stall.
- `acquire()` wait is bounded by a configurable timeout derived from `2 * window_seconds`, raising `asyncio.TimeoutError` on expiry.
- Private helper `_remove_tpm_entry_by_rid()` consolidates TPM entry cleanup logic.

#### `token_estimator.py` — EMA Token Estimator

| Class | Description |
|---|---|
| `EmaTokenEstimator` | Adaptive chars→tokens estimator using exponential moving average. Replaces fixed chars//3 formula with a self-correcting coefficient (default 3.0, bounds [1.0, 5.0], EMA alpha 0.2). |

**Key methods:**

| Method | Description |
|---|---|
| `estimate(messages, tools)` | Estimate token count using current coefficient |
| `feedback(estimated_tokens, actual_tokens)` | Update coefficient via EMA after observing actual usage |
| `coefficient` (property) | Current coefficient value |

#### `key_pool.py` — API Key Pool

| Class | Description |
|---|---|
| `APIKeyPool` | Multi-key management with staircase cooldown (30s → 60s → 300s). Accepts `names: List[str]` (composite keys like `"provider/model"`). Selection prefers keys with lowest `consecutive_errors` among those not in cooldown. |

**Key methods:**

| Method | Description |
|---|---|
| `acquire_key()` | Returns best available key name (fewest errors, respects insertion order) |
| `report_rate_limited(name)` | Increments errors, applies staircase cooldown |
| `report_success(name)` | Resets error count and cooldown |
| `is_all_in_cooldown()` | Checks if all keys are in cooldown |
| `get_active_names()` | Returns key names not currently in cooldown |

#### `retry.py` — Retry Engine

| Class | Description |
|---|---|
| `RetryEngine` | Error classification (RATE_LIMITED/RETRYABLE/NON_RETRYABLE) with exponential backoff + jitter |
| `T` | TypeVar used for generic retry return type |

#### `pacing.py` — Adaptive Pacing

| Class | Description |
|---|---|
| `AdaptivePacer` | Dynamic throttling transitioning between HEALTHY/PRESSING/CRITICAL pace levels |
| `ResultTruncator` | Static utility for truncating oversized results |
| `RegistrySizeMonitor` | Monitors task registry size and identifies completed tasks to purge |

**Pace levels:** HEALTHY (>50% remaining) → 0ms extra delay. PRESSING (20-50%) → 200ms. CRITICAL (<20%) → 1000ms.

---

### 5. `engine/time.py` — Time Utilities

Timezone-aware time formatting for the agent framework.

| Class | Description |
|---|---|
| `TimeProvider` | Resolves timezone (override → system → UTC), formats env blocks and message timestamps |

**Methods:**

| Method | Description |
|---|---|
| `resolve_timezone()` | Returns timezone string with caching |
| `format_system_env_block()` | Returns `<env>Today's date: ... Time zone: ...</env>` block |
| `format_message_timestamp()` | Returns `[Wed 2026-04-23 14:30 CST]` format |
| `inject_timestamp()` | Prepends timestamp to message unless one already exists |

---

### 6. `engine/runtime/` — Agent Execution Core

#### `agent.py` — Agent Class

The central execution engine. Each agent owns a session, tool registry, state machine, and sub-agent manager.

**State machine:**

```
IDLE → [start] → RUNNING → [finish] → COMPLETED
                   ↓                      ↑
          [spawn_children]    [children_settled]
                   ↓                      ↑
          WAITING_FOR_CHILDREN ──────────┘
                   ↓
               [error] → ERROR
```

**Core loop (`_execute_cycle()`):**

1. Process tool calls iteratively (max 20 iterations)
2. Drain queued events (ChildCompletionEvents from sub-agents)
3. If pending children exist → transition to `WAITING_FOR_CHILDREN`
4. If no pending children → finalize and notify parent

**Key features:**

- **Emergency summary**: When iteration limit is reached without a text response, makes one final LLM call WITHOUT tools to force a summary
- **Summary warning**: Injects a warning message N iterations before the limit
- **Timestamp injection**: All user messages get timezone-aware timestamps

#### `agent_models.py` — Data Models

| Model | Description |
|---|---|
| `AgentState` | Enum: `IDLE`, `RUNNING`, `WAITING_FOR_CHILDREN`, `COMPLETED`, `ERROR` |
| `ErrorCategory` | Enum: `LLM_ERROR`, `INTERNAL_ERROR` |
| `AgentError` | Structured error with category, message, and exception type |
| `Message` | Chat message with role, content, metadata, timestamp. Converts to dict for LLM API |
| `Session` | Conversation container with `add_message()` and `get_messages()` |
| `QueueEvent` | Internal event with trigger_task_id, child_results, error flag |
| `AgentResult` | Final output: content, session, success, optional error |

#### `state.py` — State Machine

`AgentStateMachine` with a static `TRANSITIONS` table mapping `(current_state, event)` → `next_state`. Raises `InvalidTransitionError` on invalid transitions.

#### `task_registry.py` — Task Registry

CRUD for `SubagentTask` entries with handler-based notification.

**Key operations:**

| Operation | Description |
|---|---|
| `register()` | Create a task with cycle detection |
| `store_result()` | Store result, return `CompleteInfo` (pending counts) |
| `complete()` | Store result + notify registered handler |
| `collect_child_results()` | Gather all direct child results |
| `collect_and_cleanup()` | Atomic: collect results, clear children, remove child tasks |
| `get_all_ancestors()` | BFS traversal up the task hierarchy |
| `register_handler()` | Map parent_task_id → completion callback |

---

### 7. `engine/providers/` — LLM Provider Layer

#### `llm_provider.py`

| Class | Description |
|---|---|
| `BaseLLMProvider` | Abstract base with `chat()` and `stream_chat()` |
| `LLMProvider` | OpenAI-compatible implementation using `AsyncOpenAI` |
| `LLMProviderError` | Unified exception wrapper for all LLM errors |

**LLMProvider constructor:**

`LLMProvider(provider_params: ProviderParams, runtime_config: Config, model_params: Optional[Dict[str, Any]] = None)`

- `provider_params` — resolved connection parameters (api_key, base_url, model)
- `runtime_config` — global Config for retry/behavior settings
- `model_params` — optional dict of model-specific kwargs merged into each API call (e.g. temperature, max_tokens)

**LLMProvider features:**

- Per-call retry with configurable max attempts and exponential backoff
- Thinking tag stripping (`<think/>` removal)
- Rate limit header extraction (`x-ratelimit-*`)
- Token usage tracking (prompt_tokens, completion_tokens)
- `model_params` merged into every `chat()` call (reserved keys `model`, `messages`, `tools` are forbidden)

#### `provider_models.py`

| Model | Description |
|---|---|
| `ToolCall` | LLM tool call: name, arguments, call_id |
| `LLMResponse` | LLM response: content + optional tool_calls |
| `PaceLevel` | Enum: `HEALTHY`, `PRESSING`, `CRITICAL` |
| `Lane` | Enum: `MAIN`, `SUBAGENT` |
| `ErrorClass` | Enum: `RETRYABLE`, `NON_RETRYABLE`, `RATE_LIMITED` |
| `ProviderConfig` | Provider entry: name, api_key, base_url, rpm_limit (default 100), tpm_limit (default 100000), models dict (model_name → model_params dict) |
| `ProviderParams` | Resolved call params: api_key, base_url, model |
| `resolve_model_ref()` | Splits `"provider/model"` string on first `/` into `(provider, model)` tuple |
| `RateLimitSnapshot` | Remaining/limit for RPM and TPM |
| `ProviderHealth` | Per-key health: consecutive errors, cooldown, pace level |

#### `fallback_provider.py`

`FallbackLLMProvider` wraps multiple `LLMProvider` instances with automatic key rotation and sequential provider fallback.

**Constructor:** `FallbackLLMProvider(providers: Dict[str, LLMProvider], key_pool: APIKeyPool, rate_limiters: Dict[str, SlidingWindowRateLimiter], pacers: Dict[str, AdaptivePacer], retry_engine: RetryEngine)`

Providers are ordered by the insertion order of the `providers` dict (primary first, then fallbacks). No weight-based selection — ordering is deterministic from config.

**Token estimation:** `_estimate_tokens` proxies to an `EmaTokenEstimator` instance instead of using a fixed chars//3 formula, producing adaptive estimates. After each successful LLM call, `feedback()` updates the estimator with actual token usage so subsequent estimates self-correct.

**Flow:**

1. Acquire key from `APIKeyPool`
2. Apply rate limiting (sliding window) — estimated_tokens is capped to prevent deadlock
3. Apply adaptive pacing
4. Execute chat request
5. On success → record usage, update pacer, report success, feed actual tokens back to estimator
6. On rate limit → report rate limited, rotate key, retry
7. On retryable error → release reservation, propagate
8. On non-retryable error → release reservation, raise

---

### 8. `engine/subagent/` — Sub-Agent System

#### `manager.py` — SubAgentManager

Orchestrates the full child agent lifecycle. Each `Agent` owns one instance.

**Key methods:**

| Method | Description |
|---|---|
| `spawn()` | Create child session, register task, build system prompt, launch `asyncio.create_task` |
| `_run_child()` | Background execution with lane slot management |
| `_on_child_complete()` | Gate-check handler: pending children? pending siblings? → collect results, notify parent |
| `_format_child_results()` | Format collected child results as JSON prompt |
| `create_spawn_tool()` | Factory for `SpawnTool` bound to this manager |

**Gate-check logic (`_on_child_complete()`):**

1. **Gate 1**: Still have pending children → return (wait)
2. **Gate 2**: Parent doesn't exist → return
3. **Gate 3**: Still have pending siblings → return (wait)
4. All gates passed → collect results, determine branch:
   - **Branch A**: Parent in `WAITING_FOR_CHILDREN` → direct resume via `run(trigger="children_settled")`
   - **Branch B**: Parent in `RUNNING` → enqueue `ChildCompletionEvent` for self-drain
   - **Branch C**: Parent already `COMPLETED` → re-propagate notification to grandparent

#### `spawn.py` — SpawnTool

Thin `Tool` wrapper that delegates to `SubAgentManager.spawn()`. Registered in the tool registry as the `spawn` tool, available to agents below `max_depth`.

#### `protocol.py` — Drainable Protocol

`@runtime_checkable` protocol requiring `state`, `result`, `run()`, and `abort()`. The `Agent` class implements this protocol.

#### `events.py` — Event Types

| Event | Description |
|---|---|
| `AgentEvent` | Base event class |
| `ChildCompletionEvent` | Carries `child_results` dict and `formatted_prompt` string |

#### `subagent_models.py` — Data Models

| Model | Description |
|---|---|
| `SubagentTask` | Task entry: task_id, session_id, description, parent references, child_task_ids, result |
| `CollectedChildResult` | Collected output: task_description + result string |

---

### 9. `engine/tools/` — Tool System

#### `base.py`

| Class | Description |
|---|---|
| `Tool` | Abstract base class with `name`, `description`, `parameters`, and async `execute()` |
| `FunctionTool` | Wraps a plain function (sync or async) as a Tool |
| `ToolRegistry` | Central registry: `register()`, `register_spawn()`, `clone()`, `get()`, `get_schemas()` |
| `ToolRegistrationError` | Raised on duplicate/empty tool names |

**Design notes:**

- `spawn` is a system tool managed separately via `register_spawn()`
- `clone()` copies business tools but NOT the spawn tool (each agent registers its own)
- Schemas follow OpenAI function calling format

#### `custom/`

Auto-discovered custom tools directory. Place `Tool` subclasses here and they will be automatically loaded by `_discover_custom_tools()`. Currently contains:

- **`web_search`** (`web_search.py`) — DuckDuckGo HTML search tool with retry-backoff for HTTP 202 rate-limit responses, request pacing, and retry logging.
- **`web_fetch`** (`web_fetch.py`) — URL content fetching tool with configurable format (class variable `DEFAULT_FORMAT`, default: markdown), transient-error retry, Cloudflare handling, and response size limits.

---

### 10. `engine/logging/` — Logging

#### `sink.py`

| Component | Description |
|---|---|
| `LoggerInterface` | ABC defining log methods: `info`, `error`, `warning`, `tool`, `state_change` |
| `LogEntry` | Dataclass: timestamp, level, agent_id, agent_label, depth, state, event_type, message, data, tool_name |
| `TerminalFormatter` | Color-coded terminal output with configurable preview length |
| `JSONFormatter` | JSON serialization of log entries |
| `AsyncFileHandler` | Async JSONL file writer using `asyncio.Queue` + background writer task |
| `Logger` | Main implementation with sync buffer for pre-init logs |

**Global API:**

| Function | Description |
|---|---|
| `get_logger()` | Returns singleton Logger instance |
| `init_logger(log_dir?)` | Initializes file handler, starts async writer |
| `stop_logger()` | Gracefully stops async file writer |

---

### 11. `tests/` — Test Suite

| File | Description |
|---|---|
| `test_easy_task.py` | Tests `delegate()` with a Chinese-language research prompt |
| `test_multilayer_subagent.py` | Tests 3-child × 2-grandchild nesting with random number aggregation |
| `test_rate_limit_safety.py` | Unit tests for rate limiter deadlock prevention, timeout, and EMA token estimator (13 tests) |

Both integration tests use `pytest-asyncio` and call the real `delegate()` function (requires valid `engine.json`).

---

## Data Flow

```
User
  │
  ▼
delegate() (engine/runner.py)
  ├── Config loading (engine.json)
  ├── Provider initialization (providers dict → LLMProviders → primary+fallback ordering)
  ├── Lane queue setup (MAIN:4, SUBAGENT:5)
  ├── Tool discovery (custom tools auto-loaded)
  └── Agent creation & registration
        │
        ▼
  Agent.run()
    ├── State: IDLE → RUNNING
    ├── _execute_cycle()
    │     ├── _process_tool_calls() ─── LLM chat loop (max 20 iterations)
    │     │     ├── LLMProvider.chat() ──→ FallbackLLMProvider.chat()
    │     │     │                               ├── APIKeyPool.acquire_key()
    │     │     │                               ├── SlidingWindowRateLimiter.acquire() ← estimated_tokens capped to tpm_limit
    │     │     │                               ├── AdaptivePacer.wait_if_needed()
    │     │     │                               └── LLMProvider.chat() (OpenAI SDK)
    │     │     ├── Tool execution (ToolRegistry → Tool.execute())
    │     │     └── spawn tool → SubAgentManager.spawn()
    │     │                       ├── LaneConcurrencyQueue.acquire()
    │     │                       ├── Create child session + system prompt
    │     │                       ├── Register in AgentTaskRegistry
    │     │                       └── asyncio.create_task(_run_child)
    │     │
    │     ├── Drain ChildCompletionEvents
    │     └── State decision: WAITING_FOR_CHILDREN or COMPLETED
    │
    ├── _finish_and_notify() → AgentTaskRegistry.complete()
    └── Return AgentResult
```

---

## Key Design Patterns

1. **Lane-based concurrency**: Separate concurrency pools for main agents (lane=MAIN) and sub-agents (lane=SUBAGENT), each with independent limits.

2. **Push-based notification**: When a child completes, `AgentTaskRegistry.complete()` fires a registered handler on the parent's `SubAgentManager`, which handles gate-checks and parent notification — no polling required.

3. **Event-driven rate limiting**: `SlidingWindowRateLimiter` uses a background scheduler task that precisely calculates when capacity will free up, avoiding busy-waiting.

4. **Staircase cooldown**: `APIKeyPool` escalates cooldown (30s → 60s → 300s) on repeated rate limits, with automatic recovery on success.

5. **Self-draining events**: Agents drain their own event queue iteratively, processing `ChildCompletionEvent`s one at a time without recursion.

6. **Tool auto-discovery**: Custom tools in `engine/tools/custom/` are automatically discovered and registered by `engine/runner.py`.

---

## External Dependencies

| Package | Purpose |
|---|---|
| `openai` | OpenAI-compatible API client (used by LLMProvider) |
| `httpx` | Async HTTP client (used by web search and web fetch tools) |
| `markdownify` | HTML-to-Markdown conversion (used by web fetch tool) |
| `python-dotenv` | Environment variable loading from `.env` |
| `pytest` + `pytest-asyncio` | Test framework with async support |
