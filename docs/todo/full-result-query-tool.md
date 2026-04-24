# Full Result Query Tool — Design Plan

## Background

After implementing result truncation in `_format_child_results()` (see `engine/subagent/manager.py`), child agent results are truncated to `Config.max_result_length` (default 4000 chars) before being injected into the parent agent's context. The original full results are lost after delivery.

This document outlines the design for a builtin tool that allows parent agents to retrieve the full (untruncated) result of a specific child agent by `task_id`.

## Why Not Now

Adding this tool introduces system complexity:
- A new result cache layer with eviction policy
- A new builtin tool registration
- Risk of LLM always querying full results, undermining truncation's token savings

Deferring until real usage data confirms the need.

---

## Core Problem

```
collect_and_cleanup()  →  child tasks deleted from task_registry
_format_child_results() →  results truncated
resume_from_children()  →  full data goes out of scope → GC
```

After the parent receives truncated results, the original full data no longer exists in any accessible storage.

## Reference Pattern

**LangGraph Async DeepAgents** (`check_async_subagent` tool) implements the same pattern:

| Tool | Purpose |
|------|---------|
| `launch_async_subagent` | Start background job, returns Job ID |
| `check_async_subagent` | Query status + full result by Job ID |

Key design insight: Job IDs are stored in a dedicated state channel that survives context compaction, not in message history.

Source: https://github.com/langchain-ai/async-deep-agents

---

## Implementation Plan

### 1. Result Cache — `engine/safety.py`

Add a `ResultCache` class alongside existing safety utilities:

```python
class ResultCache:
    """Stores full (untruncated) child results for on-demand retrieval."""

    def __init__(self, max_size: int = 100):
        self._cache: Dict[str, str] = {}  # task_id → full result
        self._max_size = max_size

    def put(self, task_id: str, result: str) -> None:
        """Store a full result. Evicts oldest entries if at capacity."""
        ...

    def get(self, task_id: str) -> Optional[str]:
        """Retrieve full result. Returns None if not found or already consumed."""
        ...

    def consume(self, task_id: str) -> Optional[str]:
        """Retrieve and remove (single-use)."""
        ...
```

Design decisions to make:
- **Single-use vs persistent**: `consume()` removes after query (prevents repeated pulls); `get()` allows multiple reads. Recommend `consume()`.
- **Eviction**: FIFO by insertion order, or use timestamps from `SubagentTask.ended_at`.
- **Size limit**: Reuse `Config.max_registry_size` or add a dedicated `Config.max_result_cache_size`.

### 2. Cache Population — `engine/subagent/manager.py`

Hook point: `_on_child_complete()`, between `collect_and_cleanup()` and `_format_child_results()`.

```python
async def _on_child_complete(self, task_id: str, info: CompleteInfo) -> None:
    ...
    child_results = await self._task_registry.collect_and_cleanup(self._agent_task_id)
    
    # NEW: cache full results before truncation
    for tid, info in child_results.items():
        self._result_cache.put(tid, info.result)
    
    formatted = self._format_child_results(child_results)
    ...
```

`SubAgentManager.__init__` needs to accept (or create) a `ResultCache` instance.

### 3. Query Tool — `engine/tools/builtin/query_result.py`

```python
class QueryFullResultTool(Tool):
    name = "query_child_result"
    description = (
        "Retrieve the full untruncated result of a child agent by its task_id. "
        "Use only when the truncated summary is insufficient for your task. "
        "Each result can only be queried once."
    )
    parameters = {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "The task_id of the child agent whose full result you want to retrieve.",
            },
        },
        "required": ["task_id"],
    }

    async def execute(self, arguments: Dict, context: Dict) -> str:
        task_id = arguments.get("task_id", "")
        # Lookup from result cache
        # Validate task_id is a child of the requesting agent
        # Return full result or error message
```

### 4. Tool Registration — `engine/runtime/agent.py`

Conditional registration (opt-in via config):

```python
if self.config.enable_full_result_query:
    query_tool = QueryFullResultTool(self._subagent_mgr._result_cache)
    self._tool_registry.register(query_tool)
```

### 5. Config Extension — `engine/config.py`

```python
@dataclass
class Config:
    ...
    enable_full_result_query: bool = False
    max_result_cache_size: int = 100
```

---

## Files to Modify

| File | Change |
|------|--------|
| `engine/safety.py` | Add `ResultCache` class |
| `engine/subagent/manager.py` | Accept `ResultCache`, populate in `_on_child_complete()` |
| `engine/tools/builtin/query_result.py` | **New file** — `QueryFullResultTool` |
| `engine/tools/builtin/__init__.py` | Export new tool |
| `engine/runtime/agent.py` | Conditional tool registration |
| `engine/config.py` | Add `enable_full_result_query`, `max_result_cache_size` |

---

## Open Design Questions

1. **Single-use or reusable?** `consume()` prevents token waste but means the parent can't re-read. `get()` is more flexible but risks the LLM repeatedly pulling the same large result.
2. **Depth restriction?** Should this tool only be available at certain depths, similar to how `SpawnTool` is gated by `max_depth`?
3. **Validation boundary?** Should the tool verify that the queried `task_id` belongs to the requesting agent's children, or allow querying any task in the task_registry?
4. **Cache shared or per-manager?** If shared across all SubAgentManagers (via the task_registry), deep nesting scenarios work naturally. If per-manager, simpler but only direct children are queryable.

---

## Simpler Alternative (No Tool Needed)

Before building the full query tool, consider adding per-spawn `max_result_length` override:

```python
# SpawnTool parameter
{
    "task": "...",
    "label": "...",
    "max_result_length": 8000  // optional, overrides global default
}
```

This lets the parent agent specify a larger limit for important tasks, with zero new infrastructure. Only proceed to the full query tool if this proves insufficient.
