# Engine Codebase Structure

> A multi-agent orchestration framework that supports nested sub-agent spawning, multi-provider LLM routing with primary/fallback ordering, and per-provider rate limiting.

---

## Directory Overview

```
engine/
├── engine/                    # Core package
│   ├── __init__.py            # Thin re-export layer (re-exports from runner.py and submodules)
│   ├── runner.py              # delegate(), DEFAULT_SYSTEM_PROMPT, ToolPack construction, is_tool_enabled filtering
│   ├── config.py              # Configuration loading (engine.json)
│   ├── prompts.py             # Centralized prompt definitions (pure leaf module, zero engine.* imports)
│   ├── time.py                # Timezone-aware time utilities
│   ├── safety/                # Rate limiting, concurrency, retry, pacing
│   │   ├── __init__.py        # Re-export layer for all safety classes
│   │   ├── concurrency.py     # LaneConcurrencyQueue, LaneSlot, LaneStatus
│   │   ├── rate_limit.py      # SlidingWindowRateLimiter
│   │   ├── token_estimator.py # EmaTokenEstimator — adaptive chars→tokens estimator
│   │   ├── key_pool.py        # APIKeyPool
│   │   ├── retry.py           # RetryEngine
│   │   └── pacing.py          # AdaptivePacer, ResultTruncator, RegistrySizeMonitor
│   ├── runtime/               # Agent execution core
│   │   ├── __init__.py
│   │   ├── agent.py           # Agent class — main execution loop (no SubAgentManager, uses ToolPack), streaming_handler is public attribute
│   │   ├── agent_models.py    # Data models (Session, Message, AgentResult, etc.)
│   │   ├── state.py           # Agent state machine
│   │   ├── task_registry.py   # Task CRUD with handler-based notification
│   │   └── streaming_handler.py  # StreamingHandler Protocol + SSEStreamingHandler + SubAgentStreamingWrapper (sub-agent streaming)
│   ├── providers/             # LLM provider layer
│   │   ├── __init__.py
│   │   ├── llm_provider.py    # BaseLLMProvider / LLMProvider (OpenAI-compatible)
│   │   ├── provider_models.py # Data models (ToolCall, LLMResponse, Lane, etc.)
│   │   ├── fallback_provider.py  # Multi-provider fallback with key rotation
│   │   ├── thinking_strategy.py  # Provider-specific thinking content extraction (strategy pattern)
│   │   ├── thinking_capture.py   # Tag-based thinking capture state machine (for tag_parser strategy)
│   │   └── chunk_types.py        # LLM streaming chunk types (StreamChunk dataclass)
│   ├── subagent/              # Sub-agent spawning and lifecycle
│   │   ├── __init__.py
│   │   ├── manager.py         # SubAgentManager — spawn, gate-check, notify; accepts root_streaming_handler for sub-agent streaming
│   │   ├── spawn.py           # SpawnTool — lazy-caches SubAgentManager per agent, passes root_streaming_handler through
│   │   ├── protocol.py        # Drainable protocol definition
│   │   ├── events.py          # Event types (ChildCompletionEvent)
│   │   └── subagent_models.py # AgentTask, CollectedChildResult
│   ├── tools/                 # Extensible tool system
│   │   ├── __init__.py
│   │   ├── base.py            # Tool ABC, FunctionTool, ToolRegistry (pure storage)
│   │   ├── pack.py            # ToolPack — immutable view over ToolRegistry with depth-aware schema filtering
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
│   ├── test_multilayer_subagent.py  # Multi-layer nesting test
│   ├── test_session_reuse.py  # Session reuse unit tests
│   └── test_subagent_streaming.py  # Sub-agent streaming unit tests
├── app/                       # FastAPI web application
│   ├── main.py                # FastAPI app factory, static file mount
│   ├── _state.py              # Global streaming lock (single-request enforcement)
│   ├── session_store.py       # In-memory session persistence (save/load Session objects)
│   ├── models/
│   │   ├── __init__.py
│   │   └── sse_events.py      # Part-based SSE event dataclasses (StreamEvent + 8 root + 8 sub-agent event types)
│   └── routers/
│       ├── __init__.py
│       ├── chat.py            # POST /chat SSE endpoint with Part-based event mapping (root + sub-agent events)
│       ├── sessions.py        # Session management endpoints
│       └── health.py          # Health check endpoint
├── web/                       # Frontend static files
│   ├── index.html             # Minimal HTML shell
│   ├── styles.css             # CSS styles (extracted from monolithic index.html, includes sub-agent panel styles)
│   ├── app.js                 # Main JS: SSE handling, Part data model, UI logic (root + sub-agent event handling)
│   ├── parts.js               # Part rendering: create/update/close DOM elements (root + sub-agent parts)
│   └── tests/
│       └── subagent-streaming.test.js  # Frontend tests for sub-agent SSE event handling and rendering
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
| `DEFAULT_SYSTEM_PROMPT` | `engine.prompts` |
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
| `delegate(task_description, system_prompt?, tools?, config?, session?)` | Main entry point. Creates a root agent session (or reuses an existing one if `session` is provided), initializes all infrastructure, and runs the agent loop. When `session` is provided, only refreshes the env block (date/timezone) in the existing system message. Returns `AgentResult`. |
| `_discover_custom_tools()` | Auto-discovers `Tool` subclasses from `engine/tools/custom/*.py` using `importlib` + `inspect`. Results are cached. |
| `_refresh_custom_tools()` | Clears the custom tools cache. |
| `_refresh_env_block(session, time_provider)` | Refreshes the date/timezone `<env>` block in the session's first system message. Replaces existing block or appends if absent. |

**Key constants:**

- Prompt definitions have been extracted to `engine/prompts.py` (see Section 4).

**Startup flow (`delegate()`):**

1. Load config via `get_config()` (auto-discovers `engine.json`)
2. Create `TimeProvider`, inject timezone info into system prompt (or refresh env block if session is provided)
3. Initialize logger with configured log directory
4. Iterate `config.providers` dict — for each provider, create `SlidingWindowRateLimiter` and `AdaptivePacer`; for each model under that provider, create an `LLMProvider` keyed by `"provider/model"`
5. Build ordered key list from `config.primary` + `config.fallback`
6. Create shared: `APIKeyPool` (with ordered composite key names), `RetryEngine`, `FallbackLLMProvider`
7. Create `LaneConcurrencyQueue` (MAIN + SUBAGENT lanes)
8. Discover and merge custom tools
9. Filter tools by `config.is_tool_enabled()` — unlisted tools default to enabled
10. Conditionally add `SpawnTool` (if `"spawn"` is enabled)
11. Build `ToolPack` from enabled tools
12. Create root `Agent` with `ToolPack`, register in `AgentTaskRegistry`
13. Run the agent, return `AgentResult`

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
| `tools` | `{}` | `Dict[str, bool]` — tool enable/disable mapping. Unlisted tools default to `True` (enabled). Use `config.is_tool_enabled(name)` to check. |

**Config discovery strategy:**

1. Check CWD for `engine.json`
2. Walk upward to find `pyproject.toml`, check that directory for `engine.json`
3. Raise `FileNotFoundError` if not found

---

### 4. `engine/prompts.py` — Centralized Prompt Definitions

A pure leaf module (zero `engine.*` imports) serving as the single source of truth for all LLM prompt text.

**Static Constants:**

| Constant | Description |
|---|---|
| `BASE_PROMPT` | Root agent base execution strategy |
| `SPAWN_PROMPT` | Root agent sub-agent spawning rules |
| `DEPTH_LIMIT_REJECTION` | Depth limit rejection message template (format string) |

**Dynamic Functions:**

| Function | Description |
|---|---|
| `build_root_system_prompt(include_spawn)` | Assemble root agent prompt (BASE + optional SPAWN) |
| `get_subagent_system_prompt(parent_label, task_desc, depth, max_depth, can_spawn, task_id, label)` | Build sub-agent system prompt |
| `get_summary_warning(remaining_iterations)` | Iteration limit warning message |
| `get_emergency_summary_prompt()` | Emergency summary forcing final answer |
| `get_child_results_prompt(child_results_json)` | Format child results for parent consumption |
| `get_child_results_empty_warning()` | Warning when no child results collected |
| `get_spawn_confirmation(task_id, label)` | Spawn success confirmation message |
| `get_concurrency_timeout_rejection(task_desc, label, active, max_concurrent, timeout)` | Concurrency limit rejection (unified from two templates) |
| `get_runtime_depth_rejection(depth, max_depth)` | Runtime depth safety net rejection |

**Derived values:**
- `DEFAULT_SYSTEM_PROMPT` = `build_root_system_prompt(include_spawn=True)` — backward-compatible alias

---

### 5. `engine/safety/` — Rate Limiting & Safety Guards

A package providing resource protection mechanisms for the agent system. Split into focused sub-modules, with `__init__.py` re-exporting all public classes for backward compatibility.

#### `__init__.py` — Re-export Layer

Re-exports all classes from sub-modules so that `from engine.safety import ...` continues to work without changes.

#### `concurrency.py` — Concurrency Control

| Class | Description |
|---|---|
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

### 6. `engine/time.py` — Time Utilities

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

### 7. `engine/runtime/` — Agent Execution Core

#### `agent.py` — Agent Class

The central execution engine. Each agent owns a session, tool pack, state machine, and event queue. No `SubAgentManager` — spawning is handled by `SpawnTool` within the `ToolPack`.

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

- **ToolPack-based tools**: Agent receives a `ToolPack` (immutable tool view) at construction. Tool schemas are depth-filtered by `ToolPack.get_schemas()` (spawn hidden at max depth).
- **Properties**: `state`, `result`, `event_queue`, `lane_queue`, `tool_pack` — all read-only via properties
- **Streaming handler (public attribute)**: `streaming_handler` is a public attribute (renamed from `_streaming_handler`). The handler is passed through to `SubAgentManager` for sub-agent streaming via the `SpawnTool`.
- **Emergency summary**: When iteration limit is reached without a text response, makes one final LLM call WITHOUT tools to force a summary
- **Summary warning**: Injects a warning message N iterations before the limit
- **Timestamp injection**: All user messages get timezone-aware timestamps
- **Streaming via handler**: Agent delegates all streaming event emission to an optional `StreamingHandler` (received as `streaming_handler: Optional[StreamingHandler]` in constructor). When a handler is present, `_get_llm_response()` uses `handler.reset()` → `handler.on_chunk(chunk)` loop → `handler.get_content()/get_tool_calls()`. When no handler is present (sub-agents), the non-streaming `llm.chat()` path is used. The handler owns all part lifecycle state, content accumulation, and tool call buffering.

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

CRUD for `AgentTask` entries with handler-based notification.
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

#### `streaming_handler.py` — Streaming Response Handler

Defines the `StreamingHandler` Protocol and `SSEStreamingHandler` concrete implementation that encapsulates all streaming-specific concerns extracted from the Agent class. Also provides `SubAgentStreamingWrapper` for streaming sub-agent LLM output back through the root handler.

**Protocol:**

| Protocol | Description |
|---|---|
| `StreamingHandler` | `@runtime_checkable` Protocol with 7 methods for streaming event handling |

**Protocol methods:**

| Method | Description |
|---|---|
| `emit(event_name, data)` | Emit a generic event (agent_done, error) |
| `on_chunk(chunk)` | Process a StreamChunk: manages part events + accumulates content/tool_calls |
| `get_content()` | Return accumulated text content |
| `get_tool_calls()` | Return parsed ToolCall objects (after finish_reason) |
| `on_tool_start(tool_name, arguments, call_id)` | Emit tool_start, return part_id. Silently returns 0 for `tool_name == "spawn"` (no SSE event emitted). |
| `on_tool_end(tool_name, result, call_id, part_id)` | Emit tool_end |
| `reset()` | Reset per-call state (part IDs + data buffers). Does NOT reset counter. |

**Concrete implementations:**

| Class | Description |
|---|---|
| `SSEStreamingHandler` | Wraps `Callable[[str, Any], None]` callback. Owns part lifecycle state machine, content accumulation, and tool call buffering. `_part_counter` is monotonic across handler lifetime. Accepts optional `allocate_part_id: Callable[[], int]` for delegating part ID assignment to a parent handler. Uses `_next_part_id()` helper to delegate or increment locally. |
| `SubAgentStreamingWrapper` | Wraps the root handler's emit callback for sub-agent streaming. Receives `emit`, `task_id`, and `allocate_part_id` (required) at construction. Namespaces all events with `subagent_` prefix, converts `agent_done`→`subagent_done`, `error`→`subagent_error`, and injects `task_id` into every event. No local part counter — uses the shared `allocate_part_id` callback for globally unique IDs. |

---

### 8. `engine/providers/` — LLM Provider Layer

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

#### `thinking_strategy.py` — Provider-Specific Thinking Extraction

Maps provider `base_url` domain to a thinking content extraction strategy. Each strategy is a class encapsulating its own state and extraction algorithm. `stream_chat()` calls `extractor.extract(delta)` per chunk and `extractor.flush()` when the stream ends.

**Common result type:**

| Dataclass | Description |
|---|---|
| `ThinkingResult` | Returned by every `extract()` / `flush()` call. Fields: `thinking_text`, `response_text`, `source` (strategy name or `None`) |

**Abstract base:**

| Class | Description |
|---|---|
| `ThinkingExtractor` | ABC requiring `extract(delta) → ThinkingResult` and `flush() → ThinkingResult` |

**Concrete strategies:**

| Class | Domain | Description |
|---|---|---|
| `ReasoningDetailsExtractor` | `api.minimaxi.com` | MiniMax. Reads `delta.reasoning_details[].text` as **incremental** text (each chunk contains only new characters). Strips `(think)\n` prefix that MiniMax sometimes leaks into the response content field. No state tracking needed. |
| `ReasoningContentExtractor` | `dashscope.aliyuncs.com` | Qwen/DashScope. Reads `delta.reasoning_content` as incremental text. No state tracking needed. |
| `TagParserExtractor` | default (DeepSeek, etc.) | Parses `<think/>` tags embedded in `delta.content`. Uses `ThinkingCapture` state machine internally to handle partial tags split across chunks. |

**Factory function:**

| Function | Description |
|---|---|
| `get_thinking_extractor(base_url)` | Parses the hostname from `base_url`, looks it up in the domain registry, and returns the matching `ThinkingExtractor` instance. Falls back to `TagParserExtractor` for unknown domains. |

**Domain registry:**

| Hostname | Extractor |
|---|---|
| `api.minimaxi.com` | `ReasoningDetailsExtractor` |
| `dashscope.aliyuncs.com` | `ReasoningContentExtractor` |
| *(any other)* | `TagParserExtractor` |

#### `thinking_capture.py` — Tag-Based Thinking Capture

Stateful tag parser for the `tag_parser` strategy (default/DeepSeek). Handles partial `<think/>` tags split across streaming chunks. Only used by `TagParserExtractor` — the other strategies (`ReasoningDetailsExtractor`, `ReasoningContentExtractor`) extract thinking content directly from structured delta fields without needing tag parsing.

**State machine:**

| Enum | State | Description |
|---|---|---|
| `ThinkState.OUTSIDE` | `outside` | Normal response text — everything goes to `response_text` |
| `ThinkState.INSIDE` | `inside` | Inside `<think...>` tag — everything goes to `thinking_text` |
| `ThinkState.MAYBE_OPEN` | `maybe_open` | Partial opening tag accumulated in buffer (e.g. `"<thi"`) |
| `ThinkState.MAYBE_CLOSE` | `maybe_close` | Partial closing tag accumulated in buffer (e.g. `"</thi"`) |

**Key class:**

| Class | Description |
|---|---|
| `ThinkingCapture` | Stateful tag parser. Tracks current `ThinkState` and an internal `_buffer` for partial tag text. |

**Methods:**

| Method | Description |
|---|---|
| `feed(content)` | Process a content chunk. Returns `CaptureResult(thinking_text, response_text)` with separated content. |
| `flush()` | Flush remaining buffered content when the stream ends. In `MAYBE_CLOSE` state, buffered text is thinking content; in `MAYBE_OPEN` state, it is response content. |

**Result type:**

| Dataclass | Description |
|---|---|
| `CaptureResult` | Fields: `thinking_text`, `response_text` |

#### `chunk_types.py` — LLM Streaming Chunk Types

Defines the unified chunk type yielded by `stream_chat()`.

| Dataclass | Description |
|---|---|
| `StreamChunk` | A single streaming chunk. Fields: `delta_text` (response text delta), `thinking_text` (thinking/reasoning text delta), `tool_calls` (optional list of partial/complete tool call deltas), `finish_reason` (optional string), `thinking_source` (strategy identifier: `"tag_parser"`, `"reasoning_content"`, `"reasoning_details"`, or `None`) |

---

### 9. `engine/subagent/` — Sub-Agent System

#### `manager.py` — SubAgentManager

Orchestrates the full child agent lifecycle. Created lazily by `SpawnTool` per agent (not owned by `Agent` directly). Receives `llm_provider` and `tool_pack` at construction and builds child agents directly. Prompt templates for sub-agent system prompts are defined in `engine/prompts.py` (Section 4).

**Constructor parameters:**

| Parameter | Description |
|---|---|
| `llm_provider` | Shared LLM provider for child agents |
| `tool_pack` | Parent agent's ToolPack |
| `root_streaming_handler` | Optional root-level `SSEStreamingHandler`. When set, depth-0 children receive a `SubAgentStreamingWrapper`; depth≥1 children receive `None`. Defaults to `None`. |

**Key methods:**

| Method | Description |
|---|---|
| `spawn()` | Create child session, register task, build system prompt, emit `subagent_start` event (if streaming), create `SubAgentStreamingWrapper` for depth-0 children, launch `asyncio.create_task` |
| `_run_child()` | Background execution with lane slot management |
| `_execute_child()` | Wraps `_run_child()`: emits `subagent_done`/`subagent_error` lifecycle events after child completes |
| `_on_child_complete()` | Gate-check handler: pending children? pending siblings? → collect results, notify parent |
| `_format_child_results()` | Format collected child results as JSON prompt |

**Sub-agent streaming logic:**

- When `root_streaming_handler` is provided AND `parent_session.depth == 0`: creates a `SubAgentStreamingWrapper` wrapping the root handler's emit callback and `_next_part_id()` method
- The wrapper is passed as `streaming_handler` to the child `Agent` constructor
- Lifecycle events (`subagent_start`, `subagent_done`, `subagent_error`) are emitted directly via the root callback, not through the wrapper

**Gate-check logic (`_on_child_complete()`):**

1. **Gate 1**: Still have pending children → return (wait)
2. **Gate 2**: Parent doesn't exist → return
3. **Gate 3**: Still have pending siblings → return (wait)
4. All gates passed → collect results, determine branch:
   - **Branch A**: Parent in `WAITING_FOR_CHILDREN` → direct resume via `run(trigger="children_settled")`
   - **Branch B**: Parent in `RUNNING` → enqueue `ChildCompletionEvent` for self-drain
   - **Branch C**: Parent already `COMPLETED` → re-propagate notification to grandparent

#### `spawn.py` — SpawnTool

`Tool` subclass that lazy-creates a `SubAgentManager` per agent on first `execute()` call. Uses `asyncio.Lock` for concurrency safety. The `SubAgentManager` receives `llm_provider`, `tool_pack`, and `root_streaming_handler` (from `parent_agent.streaming_handler`) from the parent agent and directly constructs child `Agent` instances. On agent completion, `release()` clears the cached manager.

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
| `AgentTask` | Task entry: task_id, session_id, description, parent references, child_task_ids, result |
| `CollectedChildResult` | Collected output: task_description + result string |

---

### 10. `engine/tools/` — Tool System

#### `base.py`

| Class | Description |
|---|---|
| `Tool` | Abstract base class with `name`, `description`, `parameters`, and async `execute()` |
| `FunctionTool` | Wraps a plain function (sync or async) as a Tool |
| `ToolRegistry` | Pure storage: `register()`, `register_many()`, `unregister()`, `get()`, `get_schemas()`, `all_tools()`. No spawn special-casing. |
| `ToolRegistrationError` | Raised on duplicate/empty tool names |

**Design notes:**

- `ToolRegistry` is pure storage with no business logic — all context-aware behavior lives in `ToolPack`
- Schemas follow OpenAI function calling format

#### `pack.py` — ToolPack

| Class | Description |
|---|---|
| `ToolPack` | Immutable view over `ToolRegistry` with context-aware schema filtering. Constructed with a list of `Tool` instances. |

**Key methods:**

| Method | Description |
|---|---|
| `get(name)` | Get a tool by name. Returns `None` if not found. |
| `get_schemas(session?)` | Get OpenAI function calling schemas. If session is provided and `depth >= config.max_depth`, the `spawn` schema is filtered out. |
| `release_spawn(agent_task_id)` | Forward `release()` to `SpawnTool` if present, cleaning up cached `SubAgentManager`. |
| `__len__` / `__contains__` | Standard container protocol. |

#### `custom/`

Auto-discovered custom tools directory. Place `Tool` subclasses here and they will be automatically loaded by `_discover_custom_tools()`. Currently contains:

- **`web_search`** (`web_search.py`) — Web search tool using the `ddgs` metasearch library. Aggregates results from multiple search engines (DuckDuckGo, Bing, Brave, Google, etc.) with automatic failover via `backend="auto"`. Uses `asyncio.to_thread()` to wrap the synchronous `DDGS.text()` call. Lazy singleton DDGS instance for connection reuse.
- **`web_fetch`** (`web_fetch.py`) — URL content fetching tool with configurable format (class variable `DEFAULT_FORMAT`, default: markdown), transient-error retry, Cloudflare handling, and response size limits.

---

### 11. `engine/logging/` — Logging

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

### 12. `tests/` — Test Suite

| File | Description |
|---|---|
| `test_easy_task.py` | Tests `delegate()` with a structured city-comparison research prompt |
| `test_multilayer_subagent.py` | Tests 3-child × 2-grandchild nesting with 3-level data provenance verification via JSONL logs |
| `test_session_reuse.py` | Unit tests for `delegate()` session reuse — 8 tests covering backward compat, ID preservation, env block refresh, warning logging (all mocked, no live LLM calls) |
| `test_subagent_streaming.py` | Unit tests for SubAgentStreamingWrapper, SSEStreamingHandler allocate_part_id, and spawn tool suppression |

Both integration tests use `pytest-asyncio` and call the real `delegate()` function (requires valid `engine.json`).

#### Frontend Tests

| File | Description |
|---|---|
| `web/tests/subagent-streaming.test.js` | Tests for sub-agent SSE event handling, panel rendering, and part lifecycle |

---

### 13. `app/` — FastAPI Web Application

A FastAPI application providing a chat UI and SSE-based streaming API. Static files are served from the `web/` directory.

#### `main.py` — Application Factory

Creates the FastAPI app, mounts static files from `web/`, and includes routers for chat, sessions, and health.

#### `_state.py` — Global Streaming Lock

Enforces single-request-at-a-time processing via a boolean flag (`set_streaming` / `is_streaming`). Returns HTTP 429 if a request arrives while another is streaming.

#### `session_store.py` — Session Persistence

In-memory store for `Session` objects. Provides `save(session)` and `load(session_id)` methods. Used by the chat endpoint to persist conversations across requests.

#### `models/sse_events.py` — Part-based SSE Event Dataclasses

Defines the wire-format SSE event types as dataclasses inheriting from `StreamEvent`:

| Dataclass | `type` Field | Data Fields |
|---|---|---|
| `StreamEvent` | (base) | `type`, `data` |
| `AgentStartEvent` | `agent_start` | — |
| `PartNewEvent` | `part_new` | `part_id`, `part_type`, `text` |
| `PartDeltaEvent` | `part_delta` | `part_id`, `text` |
| `PartCloseEvent` | `part_close` | `part_id` |
| `ToolCallStartEvent` | `tool_call_start` | `part_id`, `tool_name`, `arguments`, `call_id` |
| `ToolCallResultEvent` | `tool_call_result` | `part_id`, `tool_name`, `result`, `call_id` |
| `DoneEvent` | `done` | `success`, `session_id` |
| `ErrorEvent` | `error` | `message`, `session_id` |
| `SubAgentStartEvent` | `subagent_start` | `part_id`, `task_id`, `label`, `description` |
| `SubAgentPartNewEvent` | `subagent_part_new` | `part_id`, `task_id`, `part_type`, `text` |
| `SubAgentPartDeltaEvent` | `subagent_part_delta` | `part_id`, `task_id`, `text` |
| `SubAgentPartCloseEvent` | `subagent_part_close` | `part_id`, `task_id` |
| `SubAgentToolStartEvent` | `subagent_tool_start` | `part_id`, `task_id`, `tool_name`, `arguments`, `call_id` |
| `SubAgentToolResultEvent` | `subagent_tool_result` | `part_id`, `task_id`, `tool_name`, `result`, `call_id` |
| `SubAgentDoneEvent` | `subagent_done` | `task_id`, `success` |
| `SubAgentErrorEvent` | `subagent_error` | `task_id`, `message` |

17 dataclasses total (1 base `StreamEvent` + 8 root event types + 8 sub-agent event types). The `DoneEvent` does not include a `content` field — text content is delivered incrementally via Part events. Sub-agent events carry an additional `task_id` field to identify which sub-agent they belong to.

#### `routers/chat.py` — Chat SSE Endpoint

Exposes `POST /chat` which streams agent responses via Server-Sent Events.

**Event translation (`on_engine_event()`):**

The `on_engine_event()` callback maps internal engine events to SSE wire format:

| Engine Event | SSE Event | Notes |
|---|---|---|
| `part_new` | `part_new` | Direct pass-through |
| `part_delta` | `part_delta` | Direct pass-through |
| `part_close` | `part_close` | Direct pass-through |
| `tool_start` | `tool_call_start` | Renamed for clarity on wire |
| `tool_end` | `tool_call_result` | Renamed for clarity on wire |
| `agent_done` | `done` | Adds `session_id` |
| `error` | `error` | Adds `session_id` |
| `subagent_start` | `subagent_start` | Sub-agent lifecycle start |
| `subagent_part_new` | `subagent_part_new` | Sub-agent text/reasoning part |
| `subagent_part_delta` | `subagent_part_delta` | Sub-agent content delta |
| `subagent_part_close` | `subagent_part_close` | Sub-agent part close |
| `subagent_tool_start` | `subagent_tool_start` | Sub-agent tool call start |
| `subagent_tool_result` | `subagent_tool_result` | Sub-agent tool call result |
| `subagent_done` | `subagent_done` | Sub-agent completion (does NOT set `done_event`) |
| `subagent_error` | `subagent_error` | Sub-agent error (does NOT set `done_event`) |

Additionally, `agent_start` is emitted immediately when the SSE connection opens, before `delegate()` begins execution.

**Session management:**

- `ChatRequest` accepts optional `session_id` for conversation continuity
- `_truncate_session()` removes oldest complete turns when non-system messages exceed `MAX_MESSAGES` (20)
- Sessions are saved to `SessionStore` after streaming completes

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
  ├── is_tool_enabled filtering + SpawnTool injection
  └── ToolPack construction → Agent creation & registration
        │
        ▼
  Agent.run()
    ├── State: IDLE → RUNNING
    ├── _execute_cycle()
    │     ├── _process_tool_calls() ─── LLM chat loop (max 15 iterations)
    │     │     ├── _get_llm_response() ──→ FallbackLLMProvider.stream_chat()
    │     │     │     ├── StreamingHandler.on_chunk() (delegates all streaming state):
    │     │     │     │     ├── Part lifecycle (part_new/part_delta/part_close)
    │     │     │     │     ├── Content accumulation (get_content())
    │     │     │     │     └── Tool call buffering (get_tool_calls())
    │     │     │     └── FallbackLLMProvider.stream_chat()
    │     │     │           ├── APIKeyPool.acquire_key()
    │     │     │           ├── SlidingWindowRateLimiter.acquire() ← estimated_tokens capped to tpm_limit
    │     │     │           ├── AdaptivePacer.wait_if_needed()
    │     │     │           └── LLMProvider.stream_chat() (OpenAI SDK)
    │     │     ├── Tool execution (ToolPack → Tool.execute())
    │     │     │     └── per tool call → handler.on_tool_start() → execute → handler.on_tool_end()
    │     │     └── spawn tool → SpawnTool.execute()
    │     │                       ├── Lazy SubAgentManager init (per agent, asyncio.Lock)
    │     │                       │     └── passes root_streaming_handler from parent_agent.streaming_handler
    │     │                       ├── SubAgentManager.spawn()
    │     │                       │     ├── LaneConcurrencyQueue.acquire()
    │     │                       │     ├── Create child session + system prompt
    │     │                       │     ├── Build child Agent with shared llm_provider + tool_pack
    │     │                       │     │     └── if depth==0: create SubAgentStreamingWrapper(root handler's _next_part_id)
    │     │                       │     │         └── child Agent(streaming_handler=SubAgentStreamingWrapper)
    │     │                       │     ├── Emit subagent_start event (if streaming active)
    │     │                       │     ├── Register in AgentTaskRegistry
    │     │                       │     └── asyncio.create_task(_run_child)
    │     │
    │     ├── Drain ChildCompletionEvents
    │     └── State decision: WAITING_FOR_CHILDREN or COMPLETED
    │
    ├── _finish_and_notify() → ToolPack.release_spawn() + AgentTaskRegistry.complete()
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
| `httpx` | Async HTTP client (used by web fetch tool) |
| `markdownify` | HTML-to-Markdown conversion (used by web fetch tool) |
| `ddgs` | Metasearch library aggregating 9+ search engines with automatic failover (used by web search tool) |
| `pytest` + `pytest-asyncio` | Test framework with async support |

---

## SSE Protocol

The frontend communicates with the backend via Server-Sent Events (SSE) using a Part-based streaming model.

### Event Types

| SSE Event | Direction | Data |
|---|---|---|
| `agent_start` | Server → Client | `{session_id: str}` |
| `part_new` | Server → Client | `{part_id: int, part_type: "text"|"reasoning", text: str}` |
| `part_delta` | Server → Client | `{part_id: int, text: str}` |
| `part_close` | Server → Client | `{part_id: int}` |
| `tool_call_start` | Server → Client | `{part_id: int, tool_name: str, arguments: dict, call_id: str}` |
| `tool_call_result` | Server → Client | `{part_id: int, tool_name: str, result: str, call_id: str}` |
| `done` | Server → Client | `{success: bool, session_id: str}` |
| `error` | Server → Client | `{message: str, session_id: str}` |
| `subagent_start` | Server → Client | `{part_id: int, task_id: str, label: str, description: str}` |
| `subagent_part_new` | Server → Client | `{part_id: int, task_id: str, part_type: str, text: str}` |
| `subagent_part_delta` | Server → Client | `{part_id: int, task_id: str, text: str}` |
| `subagent_part_close` | Server → Client | `{part_id: int, task_id: str}` |
| `subagent_tool_start` | Server → Client | `{part_id: int, task_id: str, tool_name: str, arguments: dict, call_id: str}` |
| `subagent_tool_result` | Server → Client | `{part_id: int, task_id: str, tool_name: str, result: str, call_id: str}` |
| `subagent_done` | Server → Client | `{task_id: str, success: bool}` |
| `subagent_error` | Server → Client | `{task_id: str, message: str}` |

### Part Types

| Part Type | Description |
|---|---|
| `text` | LLM response text content |
| `reasoning` | LLM thinking/reasoning content |
| `tool` | Tool call execution (implicit via `tool_call_start`/`part_id`) |

### Part ID Assignment

Part IDs are simple incrementing integers assigned by `SSEStreamingHandler.on_tool_start()` and the part lifecycle logic in `on_chunk()`. The handler's `_part_counter` **never resets** — it increments monotonically across the handler's entire lifetime (including across multiple `_get_llm_response()` calls within a single agent run). This guarantees globally unique part IDs, preventing the frontend from misrouting events when the same agent produces multiple rounds of streaming output.

**Shared counter for sub-agents:** When sub-agent streaming is active, the root `SSEStreamingHandler` exposes its `_next_part_id()` method via the optional `allocate_part_id` callback. The `SubAgentStreamingWrapper` uses this callback (no local counter), ensuring globally unique part IDs across both root and sub-agent streams. Depth-0 children share the root counter; depth≥1 children do not stream (handler is `None`).

### Event Flow

```
Root Agent:
  StreamingHandler.on_chunk(chunk)
    ├── thinking_text → part_new(reasoning) → part_delta → ... → part_close
    ├── delta_text → part_new(text) → part_delta → ... → part_close
    └── finish_reason → close any active Parts

  StreamingHandler.on_tool_start() / on_tool_end()
    └── per tool call → counter++ → tool_start(part_id) → execute → tool_end(part_id)
                          (spawn tool calls silently return 0, no SSE event emitted)

Sub-Agent (depth-0 children only):
  SubAgentStreamingWrapper.on_chunk(chunk)
    ├── thinking_text → subagent_part_new(reasoning) → subagent_part_delta → ... → subagent_part_close
    ├── delta_text → subagent_part_new(text) → subagent_part_delta → ... → subagent_part_close
    └── finish_reason → close any active Parts

  SubAgentStreamingWrapper.on_tool_start() / on_tool_end()
    └── per tool call → allocate_part_id() → subagent_tool_start → execute → subagent_tool_result

  Lifecycle events (emitted by SubAgentManager, not wrapper):
    ├── subagent_start(task_id, label, description) — on spawn
    ├── subagent_done(task_id, success) — on completion
    └── subagent_error(task_id, message) — on error
```
