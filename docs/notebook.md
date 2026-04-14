# NOTEBOOK

## 修复 `_drain_events()` 后 `spawned_any` 丢失导致的控制流 Bug

### 背景

`agent_core.py` 中 `run()` 和 `_resume_from_children()` 在调用 `_drain_events()` 之后，都需要根据"是否 spawn 了子代理"来决定下一步行为。但 `_drain_events()` 可能递归调用 `_resume_from_children()`，后者内部**也会 spawn 子代理、也会改变状态机**。

这导致外层的 `spawned_any` 局部变量在 drain 之后不再可信。

### Bug 复现场景

#### 场景：drain 内部递归 spawn 后，外层重复 trigger 导致 crash

```
run()
  spawned_any = True                     ← LLM spawn 了子代理 A
  _drain_events()
    pop_event → 拿到旧的完成事件
    _resume_from_children()
      _process_tool_calls() → spawn 了子代理 B
      _drain_events() → 无事件
      state == RUNNING, spawned_any=True
      → trigger("spawn_children")        ← 状态变为 WAITING_FOR_CHILDREN ✅
  回到 run()
    state != COMPLETED
    spawned_any == True                  ← 还是外层的旧值
    → trigger("spawn_children")          ← 💥 InvalidTransitionError!
                                         ← WAITING_FOR_CHILDREN 没有 spawn_children 转换
```

#### 状态机转换表（参考）

```python
TRANSITIONS = {
    (IDLE, "start"):                  RUNNING,
    (RUNNING, "spawn_children"):      WAITING_FOR_CHILDREN,
    (RUNNING, "finish"):              COMPLETED,
    (RUNNING, "error"):               ERROR,
    (WAITING_FOR_CHILDREN, "children_settled"): RUNNING,
}
```

注意：`WAITING_FOR_CHILDREN` 上只能触发 `children_settled`，没有 `spawn_children`。

### 问题本质

`run()` 和 `_resume_from_children()` 有**重复的 post-drain 分支逻辑**，且都只检查 `COMPLETED` 和 `spawned_any`，遗漏了 `WAITING_FOR_CHILDREN` 状态。当 drain 内部已经改变了状态机时，外层的分支逻辑就用过时的信息做了错误的决策。

### 方案：抽取 `_after_drain()` 消除重复

**核心原则**：`_drain_events()` 之后，状态机是唯一的真相来源。先查状态机，状态机没动才看本层的 `spawned_any`。

#### 新增方法 `_after_drain`

```python
async def _after_drain(self, spawned_any: bool) -> Optional[str]:
    """Post-drain branching: decide spawn/wait/finish based on current state.

    Args:
        spawned_any: Whether the current layer's _process_tool_calls spawned children.
                     Only consulted when state machine is still RUNNING after drain.

    Returns:
        str if the method has fully handled the outcome (caller should return it),
        None if the caller should proceed with its own return logic.
    """
    state = self.state_machine.current_state

    if state == AgentState.COMPLETED:
        # drain 触发了完整完成链（_resume_from_children → _finish_and_notify）
        return self._final_result or "[无回复]"

    if state == AgentState.WAITING_FOR_CHILDREN:
        # drain 内部已经 spawn 了子代理并处理了状态转换
        return "[等待子代理回调...]"

    # 状态还是 RUNNING → drain 什么都没做，由本层的 spawned_any 决定
    if spawned_any:
        self.state_machine.trigger("spawn_children")
        return "[等待子代理回调...]"
    else:
        await self._finish_and_notify()
        return None
```

#### 改动 1：`run()` 方法（L112-135）

```python
# Before
async def run(self, message: Optional[str] = None) -> str:
    if message:
        self.session.add_message("user", message)

    self.state_machine.trigger("start")
    print(f"[{self.label}|{self.task_id}] → Processing")

    spawned_any = await self._process_tool_calls()

    await self._drain_events()

    if self.state_machine.current_state == AgentState.COMPLETED:
        pass
    elif spawned_any:
        print(f"[{self.label}|{self.task_id}] → Waiting for subagents")
        self.state_machine.trigger("spawn_children")
        return "[等待子代理回调...]"
    else:
        print(f"[{self.label}|{self.task_id}] ✓ Completed")
        await self._finish_and_notify()

    return self._final_result or "[无回复]"

# After
async def run(self, message: Optional[str] = None) -> str:
    if message:
        self.session.add_message("user", message)

    self.state_machine.trigger("start")
    print(f"[{self.label}|{self.task_id}] → Processing")

    spawned_any = await self._process_tool_calls()

    await self._drain_events()

    result = await self._after_drain(spawned_any)
    if result is not None:
        return result

    return self._final_result or "[无回复]"
```

#### 改动 2：`_resume_from_children()` 方法（L248-260）

```python
# Before (L248-260)
        spawned_any = await self._process_tool_calls()

        await self._drain_events()

        if self.state_machine.current_state == AgentState.COMPLETED:
            return

        if spawned_any:
            print(f"{self.display_id} → Re-waiting for new subagents")
            self.state_machine.trigger("spawn_children")
        else:
            await self._finish_and_notify()

# After
        spawned_any = await self._process_tool_calls()

        await self._drain_events()

        await self._after_drain(spawned_any)
```

#### 改动 3：新增 `_after_drain` 方法（放在 `_drain_events` 之后）

```python
    async def _after_drain(self, spawned_any: bool) -> Optional[str]:
        """Post-drain branching: decide spawn/wait/finish based on current state.

        Called after _drain_events() in both run() and _resume_from_children().
        Uses state machine as the single source of truth — if drain already
        changed the state (to COMPLETED or WAITING_FOR_CHILDREN), we respect
        that. Only when state is still RUNNING do we consult the local
        spawned_any flag.

        Args:
            spawned_any: Whether the current layer's _process_tool_calls spawned children.

        Returns:
            str if fully handled (caller should return it), None otherwise.
        """
        state = self.state_machine.current_state

        if state == AgentState.COMPLETED:
            return self._final_result or "[无回复]"

        if state == AgentState.WAITING_FOR_CHILDREN:
            return "[等待子代理回调...]"

        if spawned_any:
            print(f"{self.display_id} → Waiting for subagents")
            self.state_machine.trigger("spawn_children")
            return "[等待子代理回调...]"
        else:
            print(f"{self.display_id} ✓ Completed")
            await self._finish_and_notify()
            return None
```

### 修改文件清单

| 文件 | 改动 |
|---|---|
| `src/agent_core.py` | 新增 `_after_drain()` 方法；修改 `run()` 和 `_resume_from_children()` 的 post-drain 分支 |

### 决策点

`_after_drain()` 返回 `None` vs 有值，用于区分"我已经处理完了，你直接返回"和"我走了 finish 路径，但你可能还有自己的返回逻辑"。目前 `run()` 需要这个区分（返回 `self._final_result or "[无回复]"`），`_resume_from_children()` 不需要（void 返回）。

如果后续觉得返回值语义不清晰，也可以统一让 `_after_drain` 只做分支、不返回值，让调用方自己处理返回——但那样 `run()` 就需要重新检查状态来决定返回内容，实质上是把相同逻辑又写了一遍。

### 测试场景覆盖

| # | 场景 | 预期行为 |
|---|---|---|
| 1 | `spawned_any=False`, drain 无事件 | → `_finish_and_notify()` → COMPLETED |
| 2 | `spawned_any=True`, drain 无事件 | → `spawn_children` → WAITING_FOR_CHILDREN |
| 3 | `spawned_any=True`, drain 内部走 finish | → COMPLETED，外层不重复 spawn |
| 4 | `spawned_any=True`, drain 内部又 spawn | → WAITING_FOR_CHILDREN，外层不重复 spawn |
| 5 | `spawned_any=False`, drain 内部 spawn | → WAITING_FOR_CHILDREN，外层 `spawned_any` 被忽略 |
| 6 | drain 递归多层，最深层 spawn | → WAITING_FOR_CHILDREN，中间层和外层都正确传递 |
| 7 | drain 递归多层，最深层 finish | → COMPLETED，中间层和外层都正确处理 |
