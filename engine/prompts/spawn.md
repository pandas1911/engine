## Execution Strategy (Spawning)

1. **Decompose first** — Break the task into independent subtasks. Assign each to a child agent via `spawn`.
2. **Parallel over sequential** — If subtasks have no dependencies, dispatch them all in one turn.
3. **Handle simple tasks yourself** — If a task is trivial (single-step, no research needed), do it directly rather than spawning overhead.
4. **Iterate after synthesis** — After child agents report back, evaluate whether the results are sufficient to complete the task. If so, synthesize and respond. If not, plan and dispatch further work.

## Spawning Rules

- One `spawn` call = one focused subtask with clear completion criteria.
- Include sufficient context in the task description — the child agent starts isolated.
- Do NOT spawn a child for tasks that require a single tool call you can make yourself.

Each child returns a summary upon completion.
