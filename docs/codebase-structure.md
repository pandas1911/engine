# Engine Codebase Structure

> A multi-agent orchestration framework that supports single-level sub-agent spawning (depth=1 architecture), multi-provider LLM routing with primary/fallback ordering, and per-provider rate limiting.

---

## Directory Overview

```
engine/
├── engine/                    # Core package
│   ├── __init__.py            # Thin re-export layer (re-exports from runner.py and submodules)
│   ├── runner.py              # Engine (singleton), Infrastructure, SessionManager — main entry point for the agent system
│   ├── config.py              # Configuration loading (engine.json)
│   ├── prompts.py             # Centralized prompt definitions (pure leaf module, zero engine.* imports)
│   ├── session_store.py       # Unified SessionStore — JSONL append persistence for root & child sessions
│   ├── time.py                # Timezone-aware time utilities
│   ├── safety/                # Rate limiting, concurrency, retry, pacing
│   │   ├── __init__.py        # Re-export layer for all safety classes
│   │   ├── concurrency.py     # LaneConcurrencyQueue, LaneSlot, LaneStatus
│   │   ├── rate_limit.py      # SlidingWindowRateLimiter
│   │   ├── context_truncation.py  # TPM-based context truncation for long conversations
│   │   ├── token_estimator.py # EmaTokenEstimator — adaptive chars→tokens estimator; ResultTruncator
│   │   ├── key_pool.py        # APIKeyPool
│   │   ├── retry.py           # RetryEngine
│   ├── streaming_handler.py   # BaseStreamingHandler + SSEStreamingHandler + SubAgentStreamingWrapper (streaming event handling)
│   ├── runtime/               # Agent execution core
│   │   ├── __init__.py
│   │   ├── agent.py           # Agent class — main execution loop (no SubAgentManager, uses ToolPack), streaming_handler is public attribute
│   │   ├── agent_models.py    # Data models (Session, Message, AgentResult, etc.)
│   │   ├── state.py           # Agent state machine
│   │   └── task_registry.py   # Task CRUD with handler-based notification
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
│   │   ├── manager.py         # SubAgentManager — spawn, per-child wake, notify; accepts root_streaming_handler for sub-agent streaming
│   │   ├── protocol.py        # Drainable protocol definition
│   │   ├── events.py          # Event types (ChildCompletionEvent)
│   │   └── subagent_models.py # AgentTask, ChildCompletionNotification
│   ├── tools/                 # Extensible tool system
│   │   ├── __init__.py
│   │   ├── base.py            # Tool ABC, FunctionTool, ToolRegistry (pure storage)
│   │   ├── pack.py            # ToolPack — immutable view over ToolRegistry with depth-aware schema filtering
│   │   ├── builtin/           # Built-in tools
│   │   │   ├── __init__.py    # BUILTIN_TOOLS list, re-exports
│   │   │   ├── spawn.py       # SpawnTool — lazy-caches SubAgentManager per agent, passes root_streaming_handler through
│   │   │   ├── read.py        # ReadTool — file/directory reading with pagination and binary detection
│   │   │   ├── grep.py        # GrepTool — regex content search with ripgrep/Python fallback
│   │   │   ├── glob_.py       # GlobTool — file pattern matching with ripgrep/Python fallback
│   │   │   └── _utils/        # Helper modules shared by builtin tools
│   │   │       ├── __init__.py
│   │   │       ├── security.py    # PathGuard — deny-list based file path security guard
│   │   │       ├── binary.py      # BinaryDetector — extension and content-based binary file detection
│   │   │       └── search.py      # SearchEngine ABC + RipgrepEngine + PythonEngine + get_search_engine()
│   │   └── custom/            # Auto-discovered custom tools (web search, web fetch)
│   │       ├── __init__.py
│   │       └── web_fetch.py   # URL content fetching with HTML→Markdown/Text conversion
│   └── logging/               # Structured logging
│       ├── __init__.py
│       └── sink.py            # Logger, formatters, async file handler
├── tests/                     # Test suite
│   ├── test_easy_task.py      # Simple delegation test
│   ├── test_state_machine.py  # State machine unit tests
│   ├── test_per_child_wake.py  # Per-child wake gate-check unit tests
│   ├── test_child_notification.py  # Child notification formatting unit tests
│   ├── test_child_notification_models.py  # ChildCompletionNotification model tests
│   ├── test_depth_one_enforcement.py  # depth=1 enforcement unit tests
│   ├── test_session_persistence.py  # SessionStore file persistence tests
│   ├── test_context_truncation.py  # TPM-based context truncation tests
│   ├── test_fallback_truncation.py  # Fallback provider truncation tests
│   ├── test_key_pool_sorting.py  # Key pool sorting priority tests
│   ├── test_rate_limiter.py  # Rate limiter unit tests
│   ├── test_file_security.py      # PathGuard deny-list unit tests
│   ├── test_binary_detector.py    # BinaryDetector extension/content detection tests
│   ├── test_search_engine.py      # SearchEngine abstraction layer tests
│   ├── test_read_tool.py          # ReadTool file/directory reading tests
│   ├── test_grep_tool.py          # GrepTool content search tests
│   ├── test_glob_tool.py          # GlobTool file pattern matching tests
├── app/                       # FastAPI web application
│   ├── main.py                # FastAPI app factory, static file mount
│   ├── _state.py              # Global streaming lock (single-request enforcement) + delegate_task storage
│   ├── models/
│   │   ├── __init__.py
│   │   └── sse_events.py      # Part-based SSE event dataclasses (StreamEvent + 8 root + 8 sub-agent event types)
│   └── routers/
│       ├── __init__.py
│       ├── chat.py            # POST /chat SSE endpoint + POST /chat/abort endpoint (Part-based event mapping for root + sub-agent events)
│       ├── sessions.py        # Session management endpoints (GET /sessions list, GET /sessions/{id}, DELETE /sessions/{id})
│       └── health.py          # Health check endpoint
├── web/                       # Frontend static files
│   ├── index.html             # HTML shell with sidebar (aside#sidebar) and main-content flex-row layout
│   ├── styles.css             # CSS styles (extracted from monolithic index.html, includes sub-agent panel styles + session sidebar with slide-in/out transition)
│   ├── app.js                 # Main JS: SSE handling, Part data model, UI logic (root + sub-agent event handling); session sidebar (toggle, fetch, render, switch, delete, hydration); input disabled during agent execution, stop button with SSE abort + POST /api/chat/abort
│   ├── parts.js               # Part rendering: create/update/close DOM elements (root + sub-agent parts)
│       └── tests/
│           ├── subagent-streaming.test.js  # Frontend tests for sub-agent SSE event handling and rendering
│           └── session-sidebar.test.js     # Frontend tests for session sidebar toggle, list rendering, hydration, and delete
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

A minimal re-export module (13 lines) that re-exports the public API from `runner.py` and submodules. All implementation logic was extracted to `runner.py`.

**Re-exports:**

| Symbol | Source |
|---|---|
| `Engine` | `engine.runner` |
| `DEFAULT_SYSTEM_PROMPT` | `engine.prompts` |
| `_discover_custom_tools` | `engine.runner` |
| `_refresh_custom_tools` | `engine.runner` |
| `Tool`, `FunctionTool` | `engine.tools.base` |
| `AgentResult` | `engine.runtime.agent_models` |
| `AgentTaskRegistry` | `engine.runtime.task_registry` |
| `init_logger`, `get_logger`, `stop_logger` | `engine.logging` |

---

### 2. `engine/runner.py` — Engine, Infrastructure & SessionManager

The main entry point containing the `Engine` class (singleton), `Infrastructure` (one-time setup), and `SessionManager` (per-conversation). Extracted from the original `engine/__init__.py`.

**Engine class (singleton via `Engine.get()`):**

| Method | Description |
|---|---|
| `Engine.get(config?)` | Singleton access. Returns existing Engine instance or creates one with the given config. |
| `Engine.reset()` | Clear singleton instance (for testing). |
| `Engine.delegate(task_description, ...)` | Main async entry point. Calls `create_session()` then `mgr.start()`. Returns `AgentResult`. On exception, returns `AgentResult(success=False)`. Finally unregisters the session manager. |
| `Engine.create_session(session?, event_callback?, system_prompt?)` | Factory method. Creates and returns a `SessionManager` for a conversation. |

**Infrastructure class (plain, owned by Engine):**

A plain class (no singleton methods) that holds all shared infrastructure: providers, rate limiters, key pool, retry engine, and tool pack. Created once by `Engine.__init__()`. All `SessionManager` instances share the same `Infrastructure` via the Engine that created them.

**Module-level helpers:**

| Function | Description |
|---|---|
| `_discover_custom_tools()` | Auto-discovers `Tool` subclasses from `engine/tools/custom/*.py` using `importlib` + `inspect`. Results are cached. |
| `_refresh_custom_tools()` | Clears the custom tools cache. |

**Startup flow (`Engine.get()` → `Engine.delegate()`):**

1. `Engine.get(config)` creates singleton on first call. `Engine.__init__()` creates `Infrastructure(config)`
2. `Infrastructure.__init__()` loads config, builds providers, rate limiters, key pool, fallback provider, and tool pack
3. `Engine.delegate()` calls `create_session()` which creates a `SessionManager` with the shared `Infrastructure`
4. `SessionManager.__init__()` sets up session, system prompt (with env block), event queue, streaming handler, task registry, root `Agent`, and `SessionStore`
5. `mgr.start(task_description)` registers agent in `AgentTaskRegistry`, runs the agent, waits for completion, returns `AgentResult`
6. Error handling: exception → `AgentResult(success=False)`, finally → `mgr.unregister()`

**SessionManager class:**

Per-conversation manager that creates and owns the root Agent. `get_active()` and `get_any_active()` have been removed. The module-level `_active_sessions` dict remains but is only used internally by `_register()` and `unregister()`.

**Key methods:**

| Method | Description |
|---|---|
| `start(message)` | Register agent, run agent loop, wait for completion, return `AgentResult` |
| `unregister()` | Remove session from the module-level `_active_sessions` dict |

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
| `spawn_timeout` | `30.0` | Seconds to wait for a concurrency slot before rejecting spawn |
| `max_result_length` | `3000` | Max chars for child agent results before truncation |
| `summary_warning_reserve` | `2` | Iterations before limit to inject summary warning |
| `emergency_summary_enabled` | `True` | Force text-only LLM call when iteration limit reached |
| `log_dir` | `None` | Directory for JSONL log files (defaults to `logs/`) |
| `llm_retry_max_attempts` | `3` | Max retry attempts per LLM call |
| `llm_retry_base_delay` | `1.0` | Base delay in seconds for exponential backoff |
| `subagent_lane_concurrency` | `5` | Max concurrent sub-agents in the SUBAGENT lane |
| `pacing_enabled` | `True` | Enable adaptive request pacing |
| `pacing_min_interval_ms` | `500.0` | Minimum interval between LLM calls |
| `key_rotation_enabled` | `True` | ⚠️ **Unused** — defined in config but not referenced in codebase |
| `fallback_enabled` | `True` | ⚠️ **Unused** — defined in config but not referenced in codebase |
| `cooldown_initial_ms` | `30000.0` | Initial key cooldown on rate limit |
| `cooldown_max_ms` | `300000.0` | Maximum key cooldown |
| `user_timezone` | `None` | Timezone override (env var `USER_TIMEZONE` takes precedence) |
| `tools` | `{}` | `Dict[str, bool]` — tool enable/disable mapping. Unlisted tools default to `True` (enabled). Use `config.is_tool_enabled(name)` to check. |
| `file_permissions` | `{}` | `Dict[str, Any]` — file tool security configuration. Contains `denied_patterns` list for PathGuard. |

**Note:** Sub-agent nesting depth is fixed at depth=1 (leaf workers only). Enforced at architecture level in `SpawnTool` and `SubAgentManager.spawn()`, not via config.

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

**Dynamic Functions:**

| Function | Description |
|---|---|
| `build_root_system_prompt(include_spawn)` | Assemble root agent prompt (BASE + optional SPAWN) |
| `get_subagent_system_prompt(parent_label, task_desc, depth, can_spawn, task_id, label)` | Build sub-agent system prompt. `can_spawn` is always `False` in depth=1 architecture. `depth` retained for logging. |
| `get_summary_warning(remaining_iterations)` | Iteration limit warning message |
| `get_emergency_summary_prompt()` | Emergency summary forcing final answer |
| `get_child_results_prompt(child_results_json)` | Format child results for parent consumption |
| `get_child_results_empty_warning()` | Warning when no child results collected |
| `get_spawn_confirmation(task_id, label)` | Spawn success confirmation message |
| `get_concurrency_timeout_rejection(task_desc, label, active, max_concurrent, timeout)` | Concurrency limit rejection (unified from two templates) |

**Derived values:**
- `DEFAULT_SYSTEM_PROMPT` = `build_root_system_prompt(include_spawn=True)` — backward-compatible alias

---

### 5. `engine/session_store.py` — Unified Session Persistence

A unified session persistence layer using JSONL append format. Replaces both the legacy `engine/subagent/session_store.py` (deprecated) and the deleted `app/session_store.py`. Supports both JSONL (new) and JSON (legacy) file formats for backward compatibility.

**Directory layout:**

```
sessions/{root_session_id}/
    main.jsonl           <- root agent session
    task_abc123.jsonl     <- child session (named by task_id)
```

**Classes:**

| Class | Description |
|---|---|
| `SessionStore` | Manages session persistence as JSONL files on disk. Constructor takes `root_dir` (default `"./sessions"`). JSONL format: line 1 = session header (`id`, `depth`, `parent_id`), lines 2+ = messages. Designed for single-process asyncio (no file locking). |

**Engine-facing API (requires `create_root()` first):**

| Method | Description |
|---|---|
| `create_root(root_session_id)` | Create the session directory for a root conversation. Returns `Path`. |
| `create_file(name, session)` | Create a JSONL file with header line. Atomic write (tmp + rename). Called once per session before `append_line()`. |
| `append_line(name, message)` | Append a single message as one JSON line. Called by `Session._on_message_added` callback for real-time persistence. |
| `rewrite_file(name, session)` | Full rewrite: header + all messages. Atomic write via tmp + rename. Used for final checkpoint/compaction. |
| `read_session_file(name)` | Read session from `.jsonl` or legacy `.json`. Returns `Session` object or `None`. |
| `read_child_session(task_id)` | Convenience wrapper: reads a child session by task_id. |

**App-facing API (manages its own paths):**

| Method | Description |
|---|---|
| `save(session)` | Save session to `sessions/{id}/main.jsonl` (full rewrite). Ensures directory exists. |
| `load(session_id)` | Load from `sessions/{id}/main.jsonl` or `main.json`. Returns `Session` or `None`. |
| `delete(session_id)` | Delete session directory. Returns `True` if deleted, `False` if not found. |
| `list_sessions()` | List all session IDs (directories with `main.jsonl` or `main.json`). |

**Real-time persistence via callback:**

The `Session` dataclass has an `_on_message_added: Any = field(default=None, repr=False)` field. When set to a callable, `Session.add_message()` invokes it after appending each message. The `SessionStore.append_line()` method is wired as this callback, enabling real-time per-message persistence without explicit save calls.

- **Root agent** (`engine/runner.py`): `session._on_message_added = lambda msg: session_store.append_line("main", msg)`
- **Child agents** (`engine/subagent/manager.py`): `child_session._on_message_added = lambda msg, tid=task_id: session_store.append_line(tid, msg)`

---

### 6. `engine/safety/` — Rate Limiting & Safety Guards

A package providing resource protection mechanisms for the agent system. Split into focused sub-modules, with `__init__.py` re-exporting all public classes for backward compatibility.

#### `__init__.py` — Re-export Layer

Re-exports all classes from sub-modules so that `from engine.safety import ...` continues to work without changes. Includes `TruncationResult` and `truncate_messages_for_tpm` from `context_truncation.py`.

#### `concurrency.py` — Concurrency Control

| Class | Description |
|---|---|
| `LaneConcurrencyQueue` | Per-lane (SUBAGENT) concurrency control with FIFO queuing via `asyncio.Condition` |
| `LaneSlot` | Async context manager representing a concurrency slot |
| `LaneStatus` | Data class for lane status queries |
| `_LaneState` | Internal state per lane |

Per-provider concurrency limiting adds a second safety dimension: when `max_concurrent_requests > 0` in a provider's config, an `asyncio.Semaphore` caps simultaneous in-flight LLM requests to that provider. This complements the lane-based concurrency (which limits sub-agent parallelism) with provider-level throttling.

#### `rate_limit.py` — Sliding Window Rate Limiter (with Adaptive Pacing)

| Class | Description |
|---|---|
| `SlidingWindowRateLimiter` | Dual RPM/TPM sliding window with event-driven scheduler (no busy waiting). Includes integrated adaptive pacing via `pacing_enabled` and `min_interval_ms` constructor params. |

**Key flow:** When pacing is enabled, `acquire()` first applies a pacing delay (minimum interval + pace-level extra delay), then follows the standard rate limit path: fast path (capacity available, no waiters) → immediate return. Slow path → enqueue Future, background `_scheduler` task wakes waiters when capacity frees up.

**Adaptive pacing (integrated):**

- Constructor accepts `pacing_enabled` (default `False`) and `min_interval_ms` (default `500.0`). The effective minimum interval is the larger of `min_interval_ms` and `60000/rpm_limit` (RPM-derived floor).
- `acquire()` calls `_wait_if_needed()` before the capacity check, which enforces the minimum interval between calls and adds pace-level delays.
- `record_usage()` calls `_update_pace_level()` after recording tokens, which adjusts the pace level based on remaining capacity fraction: HEALTHY (>50%) → 0ms, PRESSING (20-50%) → 200ms, CRITICAL (<20%) → 1000ms.

**Deadlock prevention:**

- `acquire()` caps `estimated_tokens` to `tpm_limit` so a single oversized request cannot block forever when estimated > capacity.
- `_scheduler()` includes deadlock detection: when the sliding window is empty but a waiter still cannot proceed (because its estimated request exceeds the full capacity), the scheduler force-releases the waiter to prevent permanent stall.
- `acquire()` wait is bounded by a configurable timeout derived from `2 * window_seconds`, raising `asyncio.TimeoutError` on expiry.
- Private helper `_remove_tpm_entry_by_rid()` consolidates TPM entry cleanup logic.

#### `context_truncation.py` — TPM-Based Context Truncation

| Class / Function | Description |
|---|---|
| `TruncationResult` | Dataclass: `messages` (truncated list), `rounds_removed` (int), `original_tokens` (int), `truncated_tokens` (int) |
| `truncate_messages_for_tpm(messages, tools, tpm_limit, token_estimator)` | Pure function that truncates conversation history by removing complete rounds (oldest first) to fit under a provider's TPM limit. Always preserves system prompt (messages[0] if role=="system") and the last round. Returns new list, never mutates input. |
| `_find_round_boundaries(messages)` | Private helper: returns indices of all user messages (round start positions) |

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
| `APIKeyPool` | Multi-key management with staircase cooldown (30s → 60s → 300s). Accepts `names: List[str]` (composite keys like `"provider/model"`). Selection returns first available key in insertion order (primary first). `consecutive_errors` only affects cooldown duration. |

**Key methods:**

| Method | Description |
|---|---|
| `acquire_key()` | Returns first available key in insertion order (primary first); `consecutive_errors` only affects cooldown duration |
| `report_rate_limited(name)` | Increments errors, applies staircase cooldown |
| `report_success(name)` | Resets error count and cooldown |
| `is_all_in_cooldown()` | Checks if all keys are in cooldown |
| `get_active_names()` | Returns key names not currently in cooldown |

#### `retry.py` — Retry Engine

| Class | Description |
|---|---|
| `RetryEngine` | Error classification (RATE_LIMITED/RETRYABLE/NON_RETRYABLE) with exponential backoff + jitter |
| `T` | TypeVar used for generic retry return type |

#### `token_estimator.py` — Token Estimation & Result Truncation

| Class | Description |
|---|---|
| `EmaTokenEstimator` | Adaptive chars→tokens estimator using exponential moving average |
| `ResultTruncator` | Static utility for truncating oversized results |

Adaptive pacing logic has been merged into `SlidingWindowRateLimiter` (see `rate_limit.py`). When `pacing_enabled=True`, the limiter applies a minimum interval delay and a pace-level-based extra delay (HEALTHY: 0ms, PRESSING: 200ms, CRITICAL: 1000ms) before each `acquire()` call. Pace level is updated after `record_usage()` based on the remaining capacity fraction.

---

### 7. `engine/time.py` — Time Utilities

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

### 8. `engine/runtime/` — Agent Execution Core

#### `agent.py` — Agent Class

The central execution engine. Each agent owns a session, tool pack, state machine, and event queue. Spawning is handled by `SpawnTool` within the `ToolPack`, which lazily creates a `SubAgentManager`.

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

**Execution flow (`run()` + `_execute_cycle()`):**

1. `run(message, trigger)` — if `message` is present, inject it as a user message and call `_process_tool_calls()` to process the initial user input (max 15 iterations)
2. `_execute_cycle()` — drain queued `ChildCompletionEvent`s one at a time — for each, inject the child's `ChildCompletionNotification.to_prompt()` as a user message and call `_process_tool_calls()`
3. If pending children remain → transition to `WAITING_FOR_CHILDREN`, emit `waiting_for_children` SSE event via `_emit()` with `{"session_id": self.session.id}`
4. If no pending children → finalize and notify parent

**Key features:**

- **ToolPack-based tools**: Agent receives a `ToolPack` (immutable tool view) at construction. Spawn tool is filtered out for sub-agents (depth ≥ 1) via `ToolPack.get_schemas()`. Tool context passes `agent` (the Agent instance), `session`, and `task_id`.
- **Properties**: `state`, `result`, `event_queue`, `tool_pack` — all read-only via properties
- **Streaming handler (public attribute)**: `streaming_handler` is a public attribute (renamed from `_streaming_handler`). The handler is passed through to `SubAgentManager` for sub-agent streaming via the `SpawnTool`.
- **Emergency summary**: When iteration limit is reached without a text response, makes one final LLM call WITHOUT tools to force a summary
- **Summary warning**: Injects a warning message N iterations before the limit
- **Timestamp injection**: All user messages get timezone-aware timestamps
- **Streaming via handler**: Agent delegates all streaming event emission to an optional `BaseStreamingHandler` (received as `streaming_handler: Optional[BaseStreamingHandler]` in constructor, imported from `engine.streaming_handler`). When a handler is present, `_get_llm_response()` uses `handler.reset()` → `handler.on_chunk(chunk)` loop → `handler.get_content()/get_thinking()/get_tool_calls()`. When no handler is present (sub-agents), the non-streaming `llm.chat()` path is used. The handler owns all part lifecycle state, content accumulation, and tool call buffering.
- **Thinking persistence**: All assistant messages store thinking content in metadata (via `session.add_message("assistant", content, thinking=...)`). Tool messages store `tool_name` and `tool_arguments` in metadata. These fields are NOT included in `to_dict()` output, preventing thinking leakage into LLM context.

#### `agent_models.py` — Data Models

| Model | Description |
|---|---|
| `AgentState` | Enum: `IDLE`, `RUNNING`, `WAITING_FOR_CHILDREN`, `COMPLETED`, `ERROR` |
| `ErrorCategory` | Enum: `LLM_ERROR`, `INTERNAL_ERROR` |
| `AgentError` | Structured error with category, message, and exception type |
| `Message` | Chat message with role, content, metadata, timestamp. Converts to dict for LLM API |
| `Session` | Conversation container with `add_message()` and `get_messages()`. Has `_on_message_added: Any = field(default=None, repr=False)` callback field — when set, `add_message()` invokes it after appending each message (used for real-time JSONL persistence). |
| `QueueEvent` | Internal event with trigger_task_id, child_results, error flag |
| `AgentResult` | Final output: content, session, success, optional error |

#### `state.py` — State Machine

`AgentStateMachine` with a static `TRANSITIONS` table mapping `(current_state, event)` → `next_state`. No re-awaken transition. Raises `InvalidTransitionError` on invalid transitions.

#### `task_registry.py` — Task Registry

CRUD for `AgentTask` entries with handler-based notification.
**Key operations:**

| Operation | Description |
|---|---|
| `register()` | Create a task with cycle detection |
| `store_result()` | Store result, return `CompleteInfo` (pending counts) |
| `complete()` | Store result + notify registered handler |
| `get_all_ancestors()` | BFS traversal up the task hierarchy |
| `register_handler()` | Map parent_task_id → completion callback |

---

### 9. `engine/streaming_handler.py` — Streaming Response Handler

Defines `BaseStreamingHandler` and two concrete implementations for handling streaming LLM output. Extracted from the Agent class to encapsulate all streaming-specific concerns.

**Base class:**

| Class | Description |
|---|---|
| `BaseStreamingHandler` | Base class with shared streaming logic: part lifecycle, content accumulation, tool call buffering. Subclasses must implement `emit(event_name, data)`. Part ID allocation is unified via `_next_part_id()`: delegates to `allocate_part_id` callback if provided, otherwise increments local `_part_counter`. |

**Base class methods:**

| Method | Description |
|---|---|
| `emit(event_name, data)` | Abstract — subclasses define event dispatch strategy |
| `on_chunk(chunk)` | Process a StreamChunk: manages part events + accumulates content/tool_calls |
| `get_content()` | Return accumulated text content |
| `get_thinking()` | Return accumulated thinking/reasoning text from LLM responses |
| `get_tool_calls()` | Return parsed ToolCall objects (after finish_reason) |
| `on_tool_start(tool_name, arguments, call_id)` | Emit tool_start, return part_id |
| `on_tool_end(tool_name, result, call_id, part_id)` | Emit tool_end |
| `reset()` | Reset per-call state (part IDs + data buffers). Does NOT reset counter. |

**Concrete implementations:**

| Class | Description |
|---|---|
| `SSEStreamingHandler` | Extends BaseStreamingHandler with callback-based emit. Wraps `Callable[[str, Any], None]` callback. Overrides `on_tool_start` to silently return 0 for `tool_name == "spawn"` (spawn suppression). `_part_counter` is monotonic across handler lifetime. Accepts optional `allocate_part_id: Callable[[], int]` for delegating part ID assignment. |
| `SubAgentStreamingWrapper` | Extends BaseStreamingHandler, receives `parent: BaseStreamingHandler` at construction. Uses parent's `_next_part_id()` via `allocate_part_id` callback for globally unique IDs. Namespaces events with `subagent_` prefix but passes through already-prefixed `subagent_*` events without re-prefixing (double-prefix prevention). Maps both `agent_done` and `subagent_done` to `subagent_done`; maps both `error` and `subagent_error` to `subagent_error`. Only injects `task_id` into event data when not already present. No spawn suppression — inherits base class behavior. |

---

### 10. `engine/providers/` — LLM Provider Layer

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

- Explicit 120s timeout on AsyncOpenAI client (prevents 600s default timeout black hole)
- Per-call retry with configurable max attempts and exponential backoff
- 300s cumulative retry timeout guard in `chat()` (abandons retries if total elapsed time exceeded)
- Thinking tag stripping (`<think/>` removal)
- Rate limit header extraction (`x-ratelimit-*`)
- Token usage tracking (prompt_tokens, completion_tokens)
- `model_params` merged into every `chat()` call (reserved keys `model`, `messages`, `tools` are forbidden)
- `chat()` extracts thinking content via regex (`<think/>` tags) BEFORE stripping tags, so `LLMResponse.thinking` preserves the reasoning text even when `strip_thinking=True`. For streaming, thinking extraction is handled by provider-specific strategies in `thinking_strategy.py`.

#### `provider_models.py`

| Model | Description |
|---|---|
| `ToolCall` | LLM tool call: name, arguments, call_id |
| `LLMResponse` | LLM response: content + optional thinking + optional tool_calls. The `thinking` field contains extracted thinking/reasoning content from LLM responses (DeepSeek `<think/>` tags, MiniMax `reasoning_details`, Qwen `reasoning_content`). |
| `PaceLevel` | Enum: `HEALTHY`, `PRESSING`, `CRITICAL` |
| `Lane` | Enum: `SUBAGENT` |
| `ErrorClass` | Enum: `RETRYABLE`, `NON_RETRYABLE`, `RATE_LIMITED` |
| `ProviderConfig` | Provider entry: name, api_key, base_url, rpm_limit (default 100), tpm_limit (default 100000), max_concurrent_requests (default 0, 0=no limit), models dict (model_name → model_params dict) |
| `ProviderParams` | Resolved call params: api_key, base_url, model |
| `resolve_model_ref()` | Splits `"provider/model"` string on first `/` into `(provider, model)` tuple |
| `ProviderHealth` | Per-key health: consecutive errors, cooldown |

#### `fallback_provider.py`

`FallbackLLMProvider` wraps multiple `LLMProvider` instances with automatic key rotation and sequential provider fallback.

**Constructor:** `FallbackLLMProvider(providers: Dict[str, LLMProvider], key_pool: APIKeyPool, rate_limiters: Dict[str, SlidingWindowRateLimiter], retry_engine: RetryEngine, concurrency_guards: Optional[Dict[str, asyncio.Semaphore]] = None)`

Providers are ordered by the insertion order of the `providers` dict (primary first, then fallbacks). No weight-based selection — ordering is deterministic from config.

**Token estimation:** `_estimate_tokens` proxies to an `EmaTokenEstimator` instance instead of using a fixed chars//3 formula, producing adaptive estimates. After each successful LLM call, `feedback()` updates the estimator with actual token usage so subsequent estimates self-correct.

**Flow:**

1. Acquire key from `APIKeyPool`
2. Apply rate limiting with adaptive pacing (sliding window) — estimated_tokens is capped to prevent deadlock
2.5. Apply per-provider concurrency guard (if configured) — limits simultaneous in-flight requests per provider via `asyncio.Semaphore`
3. Execute chat request
4. On success → record usage, report success, feed actual tokens back to estimator
5. On rate limit → report rate limited, rotate key, retry
6. On retryable error → release reservation, propagate
7. On non-retryable error → release reservation, raise

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

### 11. `engine/subagent/` — Sub-Agent System

#### `manager.py` — SubAgentManager

Orchestrates the full child agent lifecycle. Created lazily by `SpawnTool` per agent (not owned by `Agent` directly). Receives `llm_provider`, `tool_pack`, and `session_store` at construction and builds child agents directly. Prompt templates for sub-agent system prompts are defined in `engine/prompts.py` (Section 4).

**Constructor parameters:**

| Parameter | Description |
|---|---|
| `llm_provider` | Shared LLM provider for child agents |
| `tool_pack` | Parent agent's ToolPack |
| `root_streaming_handler` | Optional root-level `SSEStreamingHandler`. When set, depth-0 children receive a `SubAgentStreamingWrapper`; depth≥1 children receive `None`. Defaults to `None`. |
| `session_store` | Optional `SessionStore` for persisting child sessions to disk |

**Key methods:**

| Method | Description |
|---|---|
| `spawn()` | Create child session, register task, build system prompt, emit `subagent_start` event (if streaming), create `SubAgentStreamingWrapper` for depth-0 children, wire `child_session._on_message_added` callback to `session_store.append_line(task_id, msg)` for real-time persistence, persist session to `SessionStore`, launch `asyncio.create_task` |
| `_run_child()` | Background execution with lane slot management |
| `_execute_child()` | Wraps `_run_child()`: emits `subagent_done`/`subagent_error` lifecycle events after child completes |
| `_on_child_complete()` | Per-child immediate wake handler: build `ChildCompletionNotification`, resume or enqueue parent |
| `_build_child_notification()` | Extract label, status, summary from child task; set `child_task.agent = None` for memory cleanup; persist session via `rewrite_file()`; return `ChildCompletionNotification` |

**Sub-agent streaming logic:**

- When `root_streaming_handler` is provided AND `parent_session.depth == 0`: creates a `SubAgentStreamingWrapper` receiving `parent=root_streaming_handler`, which uses the parent's `_next_part_id()` for globally unique IDs
- The wrapper is passed as `streaming_handler` to the child `Agent` constructor
- Lifecycle events (`subagent_start`, `subagent_done`, `subagent_error`) are emitted directly via `root_streaming_handler.emit()`, not through the wrapper

**Per-child wake logic (`_on_child_complete()`):**

Each child independently triggers notification to the parent. No sibling gates, no batch collection.

1. **Gate**: Parent doesn't exist or not registered → return
2. Build `ChildCompletionNotification` for this child (label, status, summary, session_file)
3. Persist child session to `SessionStore`
4. Determine parent state and branch:
   - **Branch A**: Parent in `WAITING_FOR_CHILDREN` → direct resume via `run(formatted, trigger="children_settled")`
   - **Branch B**: Parent in `RUNNING` → enqueue `ChildCompletionEvent` for self-drain in `_execute_cycle()`
   - Parent in `COMPLETED`/`ERROR`/`IDLE` → skip (no Branch C re-awaken)

#### ~~`spawn.py`~~ — [DELETED]

> **Deleted.** The backward-compatibility shim that re-exported `SpawnTool` from `engine/tools/builtin/spawn.py` has been removed. Use `from engine.tools.builtin.spawn import SpawnTool` directly.

#### ~~`session_store.py`~~ — [DELETED]

> **Deleted.** The legacy JSON-based session persistence module has been removed. All session persistence is now handled by `engine/session_store.py` (Section 5), which provides a unified JSONL-based persistence layer for both root and child sessions with real-time callback-driven writes.

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
| `AgentTask` | Task entry: task_id, session_id, description, parent references, child_task_ids, result, agent reference |
| `ChildCompletionNotification` | Per-child notification: task_id, label, task description, status (completed/error), summary, session_file. Has `to_prompt()` method that formats a user message for the parent agent |

---

### 12. `engine/tools/` — Tool System

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
| `get_schemas(session?)` | Get OpenAI function calling schemas. If session is provided and `depth >= 1`, the `spawn` schema is filtered out (depth=1 enforcement). |
| `release_spawn(agent_task_id)` | Forward `release()` to `SpawnTool` if present, cleaning up cached `SubAgentManager`. |
| `__len__` / `__contains__` | Standard container protocol. |

#### `custom/`

Auto-discovered custom tools directory. Place `Tool` subclasses here and they will be automatically loaded by `_discover_custom_tools()`. Currently contains:

- **`web_search`** (`web_search.py`) — Web search tool using the `ddgs` metasearch library. Aggregates results from multiple search engines (DuckDuckGo, Bing, Brave, Google, etc.) with automatic failover via `backend="auto"`. Uses `asyncio.to_thread()` to wrap the synchronous `DDGS.text()` call. Lazy singleton DDGS instance for connection reuse.
- **`web_fetch`** (`web_fetch.py`) — URL content fetching tool with configurable format (class variable `DEFAULT_FORMAT`, default: markdown), transient-error retry, Cloudflare handling, and response size limits. Content is truncated to 15,000 characters (`_MAX_CONTENT_LENGTH`) to prevent LLM context overflow.

#### `builtin/` — Built-in Tools

Built-in tools registered by the engine at startup. Exported via `BUILTIN_TOOLS` list in `__init__.py`.

| Class | Name | Description |
|---|---|---|
| `SpawnTool` | `spawn` | Creates child agents via `SubAgentManager`. Lazy-creates manager per agent on first `execute()` call using `asyncio.Lock`. Owns the `LaneConcurrencyQueue` for SUBAGENT concurrency. Parameters: `task` (required), `label` (optional). |
| `ReadTool` | `read` | Reads file contents with line numbers and pagination, or lists directory entries. Uses `PathGuard` for security and `BinaryDetector` for binary rejection. Parameters: `path` (required), `offset`/`limit` (optional pagination). |
| `GrepTool` | `grep` | Searches file contents for regex patterns. Uses `SearchEngine` with ripgrep auto-fallback. Supports include glob filtering. Parameters: `pattern` (required), `include` (optional glob), `output_mode` (optional). |
| `GlobTool` | `glob` | Finds files matching glob patterns. Uses `SearchEngine` with ripgrep auto-fallback. Parameters: `pattern` (required). |

**Root-only tools:** `spawn` is filtered out for sub-agents by `ToolPack.get_schemas()` based on session depth. The other tools (`read`, `grep`, `glob`) are available to all agents.

#### Supporting modules (`_utils/`)

##### `_utils/security.py` — PathGuard

Deny-list based file path security guard. Used by `ReadTool`, `GrepTool`, and `GlobTool` to prevent access to sensitive files.

| Component | Description |
|---|---|
| `PathGuard` | Takes a `denied_patterns` list of glob patterns. Provides `is_path_allowed(path)` (returns bool) and `check_path(path)` (raises `PermissionError` on denied paths). |
| `DEFAULT_DENIED_PATTERNS` | Module-level list of default deny patterns (e.g. `.env`, `.git/`, `*.key`, `*.pem`). |

##### `_utils/binary.py` — BinaryDetector

Extension and content-based binary file detection. Used by `ReadTool` to reject binary files.

| Component | Description |
|---|---|
| `BinaryDetector` | Static utility class with `is_binary(path)`, `is_binary_extension(path)`, and `is_binary_content(data)` methods. |
| `BINARY_EXTENSIONS` | Module-level `frozenset` of file extensions considered binary (e.g. `.pyc`, `.so`, `.png`, `.zip`). |

##### `_utils/search.py` — SearchEngine Abstraction

Abstract base class and concrete implementations for file search operations. Used by `GrepTool` and `GlobTool` with automatic ripgrep detection and fallback.

| Component | Description |
|---|---|
| `SearchEngine` | ABC with abstract `search()` method. Defines the interface for file search backends. |
| `RipgrepEngine` | Concrete implementation using `ripgrep` (`rg`) subprocess. Requires `rg` on PATH. |
| `PythonEngine` | Pure-Python fallback using `pathlib` + `re`. Used when ripgrep is not available. |
| `SearchResult` | Dataclass for individual search results (file path, line number, matched text). |
| `get_search_engine()` | Factory function that returns `RipgrepEngine` if `rg` is available, otherwise `PythonEngine`. |

---

### 13. `engine/logging/` — Logging

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

### 14. `tests/` — Test Suite

| File | Description |
|---|---|
| `test_easy_task.py` | Tests `Engine.get().delegate()` with a structured research prompt |
| `test_state_machine.py` | Unit tests for `AgentStateMachine` transitions and `InvalidTransitionError` |
| `test_per_child_wake.py` | Unit tests for per-child wake gate-check logic (Branch A/B/skip) |
| `test_child_notification.py` | Unit tests for child notification formatting and handler invocation |
| `test_child_notification_models.py` | Unit tests for `ChildCompletionNotification.to_prompt()` formatting |
| `test_depth_one_enforcement.py` | Unit tests for depth=1 enforcement in SpawnTool and SubAgentManager |
| `test_session_persistence.py` | Unit tests for `SessionStore` file persistence and deserialization |
| `test_context_truncation.py` | Unit tests for TPM-based context truncation |
| `test_fallback_truncation.py` | Unit tests for fallback provider truncation |
| `test_key_pool_sorting.py` | Unit tests for key pool sorting priority (insertion order vs errors) |
| `test_rate_limiter.py` | Unit tests for `SlidingWindowRateLimiter` |
| `test_runner_infrastructure.py` | Unit tests for `Engine` singleton and `SessionManager` |
| `test_concurrency_guard.py` | Unit tests for per-provider concurrency guard |
| `test_file_security.py` | Unit tests for PathGuard deny-list access control |
| `test_binary_detector.py` | Unit tests for BinaryDetector extension and content detection |
| `test_search_engine.py` | Unit tests for SearchEngine abstraction layer (PythonEngine + RipgrepEngine) |
| `test_read_tool.py` | Unit tests for ReadTool (file/directory reading, pagination, truncation, binary/security rejection) |
| `test_grep_tool.py` | Unit tests for GrepTool (regex search, include filter, XML output) |
| `test_glob_tool.py` | Unit tests for GlobTool (file pattern matching, XML output; imports from `glob_` module) |

All tests use `pytest-asyncio` and are pure unit tests (mocked, no live LLM calls).

#### Frontend Tests

| File | Description |
|---|---|
| `web/tests/subagent-streaming.test.js` | Tests for sub-agent SSE event handling, panel rendering, and part lifecycle |

---

### 15. `app/` — FastAPI Web Application

A FastAPI application providing a chat UI and SSE-based streaming API. Static files are served from the `web/` directory.

#### `main.py` — Application Factory

Creates the FastAPI app, mounts static files from `web/`, and includes routers for chat, sessions, and health.

#### `_state.py` — Global Streaming Lock & Delegate Task Storage

Enforces single-request-at-a-time processing via a boolean flag (`set_streaming` / `is_streaming`). Returns HTTP 429 if a request arrives while another is streaming. Also stores a reference to the current `delegate_task` via `set_active_session()` / `get_active_session()`, enabling the `/chat/abort` endpoint to cancel the running agent.

#### `session_store.py` — ~~Session Persistence~~ [DELETED]

> **Deleted.** This module has been removed. Session persistence is now handled by `engine/session_store.py` (Section 5), imported directly in `app/routers/chat.py` as `from engine.session_store import SessionStore`. The unified `SessionStore` provides both engine-facing (JSONL append) and app-facing (save/load/delete/list) APIs.

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
| `SubAgentStartEvent` | `subagent_start` | `part_id`, `task_id`, `label`, `description`, `parent_task_id` |
| `SubAgentPartNewEvent` | `subagent_part_new` | `part_id`, `task_id`, `part_type`, `text` |
| `SubAgentPartDeltaEvent` | `subagent_part_delta` | `part_id`, `task_id`, `text` |
| `SubAgentPartCloseEvent` | `subagent_part_close` | `part_id`, `task_id` |
| `SubAgentToolStartEvent` | `subagent_tool_start` | `part_id`, `task_id`, `tool_name`, `arguments`, `call_id` |
| `SubAgentToolResultEvent` | `subagent_tool_result` | `part_id`, `task_id`, `tool_name`, `result`, `call_id` |
| `SubAgentDoneEvent` | `subagent_done` | `task_id`, `success` |
| `SubAgentErrorEvent` | `subagent_error` | `task_id`, `message` |

17 dataclasses total (1 base `StreamEvent` + 8 root event types + 8 sub-agent event types). Additionally, the `waiting_for_children` event is emitted as a raw SSE event (not a dataclass) by `_execute_cycle()` when the agent enters the `WAITING_FOR_CHILDREN` state. The `DoneEvent` does not include a `content` field — text content is delivered incrementally via Part events. Sub-agent events carry an additional `task_id` field to identify which sub-agent they belong to.

#### `routers/chat.py` — Chat SSE & Abort Endpoints

Exposes `POST /chat` which streams agent responses via Server-Sent Events, and `POST /chat/abort` which cancels a running agent session by closing the SSE connection and calling `agent.abort()` on the stored delegate task.

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
| `waiting_for_children` | `waiting_for_children` | Emitted when agent enters `WAITING_FOR_CHILDREN` state; passes through `session_id` |
| `subagent_start` | `subagent_start` | Sub-agent lifecycle start |
| `subagent_part_new` | `subagent_part_new` | Sub-agent text/reasoning part |
| `subagent_part_delta` | `subagent_part_delta` | Sub-agent content delta |
| `subagent_part_close` | `subagent_part_close` | Sub-agent part close |
| `subagent_tool_start` | `subagent_tool_start` | Sub-agent tool call start |
| `subagent_tool_result` | `subagent_tool_result` | Sub-agent tool call result |
| `subagent_done` | `subagent_done` | Sub-agent completion (does NOT set `done_event`) |
| `subagent_error` | `subagent_error` | Sub-agent error (does NOT set `done_event`) |

Additionally, `agent_start` is emitted immediately when the SSE connection opens, before `Engine.delegate()` begins execution.

**Session management:**

- `ChatRequest` accepts optional `session_id` for conversation continuity
- `_truncate_session()` removes oldest complete turns when non-system messages exceed `MAX_MESSAGES` (20)
- Sessions are saved to `SessionStore` (imported from `engine.session_store`) after streaming completes

---

## Data Flow

```
User
  │
  ▼
Engine.delegate() (engine/runner.py)
    ├── Engine.get() → singleton (first call: Infrastructure.__init__)
   ├── Provider initialization (providers dict → LLMProviders → primary+fallback ordering)
   ├── Lane queue setup (SUBAGENT only, owned by SpawnTool)
   ├── Tool discovery (custom tools auto-loaded)
   ├── is_tool_enabled filtering + SpawnTool injection
   ├── ToolPack construction → Agent creation & registration
   ├── SessionStore.create_root(session.id) → create main.jsonl
   └── Wire session._on_message_added → session_store.append_line("main", msg)
         │                                    (real-time JSONL persistence)
         ▼
  Agent.run()
    ├── State: IDLE → RUNNING
    ├── _execute_cycle()
    │     ├── _process_tool_calls() ─── LLM chat loop (max 15 iterations)
    │     │     ├── _get_llm_response() ──→ FallbackLLMProvider.stream_chat()
    │     │     │     ├── BaseStreamingHandler.on_chunk() (delegates all streaming state):
    │     │     │     │     ├── Part lifecycle (part_new/part_delta/part_close)
    │     │     │     │     ├── Content accumulation (get_content())
    │     │     │     │     └── Tool call buffering (get_tool_calls())
    │     │     │     └── FallbackLLMProvider.stream_chat()
    │     │     │           ├── APIKeyPool.acquire_key()
    │     │     │           ├── SlidingWindowRateLimiter.acquire() ← estimated_tokens capped to tpm_limit (includes adaptive pacing delay when enabled)
    │     │     │           ├── truncate_messages_for_tpm() ← removes oldest rounds when estimated tokens exceed provider TPM
    │     │     │           └── LLMProvider.stream_chat() (OpenAI SDK)
    │     │     ├── Tool execution (ToolPack → Tool.execute())
    │     │     │     └── per tool call → handler.on_tool_start() → execute → handler.on_tool_end()
    │     │     └── spawn tool → SpawnTool.execute()
    │     │                       ├── Lazy SubAgentManager init (per agent, asyncio.Lock)
    │     │                       │     └── passes root_streaming_handler from agent.streaming_handler (BaseStreamingHandler)
    │     │                       │     └── passes session_store for child session persistence
    │     │                       ├── SubAgentManager.spawn()
    │     │                       │     ├── LaneConcurrencyQueue.acquire()
    │     │                       │     ├── Create child session + system prompt
    │     │                       │     ├── Wire child_session._on_message_added → session_store.append_line(task_id, msg)
    │     │                       │     ├── Build child Agent with shared llm_provider + tool_pack
    │     │                       │     │     └── if depth==0: create SubAgentStreamingWrapper(parent=root_handler)
    │     │                       │     │         └── child Agent(streaming_handler=SubAgentStreamingWrapper)
    │     │                       │     ├── Emit subagent_start event (if streaming active)
    │     │                       │     ├── Register in AgentTaskRegistry
    │     │                       │     └── asyncio.create_task(_run_child)
    │     │
    │     ├── Drain ChildCompletionEvents (per-child notifications, one at a time)
    │     └── State decision: WAITING_FOR_CHILDREN (emit waiting_for_children SSE) or COMPLETED
    │
    ├── _finish_and_notify() → ToolPack.release_spawn() + AgentTaskRegistry.complete()
    └── Return AgentResult
```

---

## Key Design Patterns

1. **Lane-based concurrency**: Separate concurrency pool for sub-agents (lane=SUBAGENT) with an independent limit. The `LaneConcurrencyQueue` is owned by `SpawnTool`, not by the `Agent`.

2. **Push-based notification**: When a child completes, `AgentTaskRegistry.complete()` fires a registered handler on the parent's `SubAgentManager`, which handles gate-checks and parent notification — no polling required.

3. **Event-driven rate limiting**: `SlidingWindowRateLimiter` uses a background scheduler task that precisely calculates when capacity will free up, avoiding busy-waiting.

4. **Staircase cooldown**: `APIKeyPool` escalates cooldown (30s → 60s → 300s) on repeated rate limits, with automatic recovery on success.

5. **Per-child wake**: Each completing child independently notifies its parent via `_on_child_complete()`. No sibling gates or batch collection. The parent is woken immediately for every completing child, either via direct resume (Branch A: parent in `WAITING_FOR_CHILDREN`) or event queue enqueue (Branch B: parent still `RUNNING`).

6. **Real-time persistence via callback**: `Session._on_message_added` callback is wired to `SessionStore.append_line()` at agent creation time (root in `runner.py`, children in `manager.py`). Every message is persisted to JSONL immediately as it's added — no explicit save calls needed during execution. Final compaction via `rewrite_file()` happens on completion.

7. **Agent memory cleanup**: After a child agent completes, `child_task.agent = None` is set in `_build_child_notification()` to release the agent object (including its session and tool references) for garbage collection.

8. **Tool auto-discovery**: Custom tools in `engine/tools/custom/` are automatically discovered and registered by `engine/runner.py`.

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
| `waiting_for_children` | Server → Client | `{session_id: str}` |
| `subagent_start` | Server → Client | `{part_id: int, task_id: str, label: str, description: str, parent_task_id: str}` |
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

Part IDs are simple incrementing integers assigned by `BaseStreamingHandler.on_tool_start()` and the part lifecycle logic in `on_chunk()`. The handler's `_part_counter` **never resets** — it increments monotonically across the handler's entire lifetime (including across multiple `_get_llm_response()` calls within a single agent run). This guarantees globally unique part IDs, preventing the frontend from misrouting events when the same agent produces multiple rounds of streaming output.

**Shared counter for sub-agents:** When sub-agent streaming is active, the `SubAgentStreamingWrapper` receives `parent=root_handler` at construction. It delegates part ID allocation to the parent's `_next_part_id()` method via the `allocate_part_id` callback in `BaseStreamingHandler.__init__()`. This ensures globally unique part IDs across both root and sub-agent streams. Depth-0 children share the root counter; depth≥1 children do not stream (handler is `None`).

### Event Flow

```
Root Agent:
  BaseStreamingHandler.on_chunk(chunk)
    ├── thinking_text → part_new(reasoning) → part_delta → ... → part_close
    ├── delta_text → part_new(text) → part_delta → ... → part_close
    └── finish_reason → close any active Parts

  BaseStreamingHandler.on_tool_start() / on_tool_end()
    └── per tool call → counter++ → tool_start(part_id) → execute → tool_end(part_id)
                          (spawn tool calls silently return 0 in SSEStreamingHandler, no SSE event emitted)

Sub-Agent (depth-0 children only):
  SubAgentStreamingWrapper.on_chunk(chunk)
    ├── thinking_text → subagent_part_new(reasoning) → subagent_part_delta → ... → subagent_part_close
    ├── delta_text → subagent_part_new(text) → subagent_part_delta → ... → subagent_part_close
    └── finish_reason → close any active Parts

  SubAgentStreamingWrapper.on_tool_start() / on_tool_end()
    └── per tool call → parent._next_part_id() → subagent_tool_start → execute → subagent_tool_result

  Lifecycle events (emitted by SubAgentManager, not wrapper):
    ├── subagent_start(task_id, label, description) — on spawn
    ├── subagent_done(task_id, success) — on completion
    └── subagent_error(task_id, message) — on error
```

---

## Known Limitations

### Sub-agents cannot spawn further children

Depth=1 is enforced at architecture level. Sub-agents (depth ≥ 1) are leaf workers that cannot spawn children of their own. The spawn tool is filtered from their tool schemas, and `SubAgentManager.spawn()` rejects any spawn attempt with `depth >= 1`.

### SessionStore is single-process

`SessionStore` (in `engine/session_store.py`) is designed for single-process asyncio with no file locking. The JSONL append format is safe for concurrent async writes within a single process, but concurrent writes from multiple processes could cause data loss.
