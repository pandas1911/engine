# NOTEBOOK

## 工具并行化运行

## 移除冗余函数 `_count_pending_descendants_locked`

### 背景

`SubagentRegistry` 中有两个功能高度重叠的函数：

- `_count_pending_descendants_locked(task_id)` — BFS 遍历**所有后代**（子、孙、曾孙…），统计 pending 数量
- `count_pending_for_parent(parent_task_id)` — 遍历全部 `_tasks`，统计**直接子节点**中 pending 的数量

两者唯一的调用点都在 `complete()` 方法中：

```python
# registry.py:152-155
pending_descendants = self._count_pending_descendants_locked(task_id)          # Gate 1
pending_siblings = self.count_pending_for_parent(parent_task_id) if parent_task_id else 0  # Gate 3
```

### 冗余原因：传播保证

`complete()` 的 Gate 1 机制保证了**自底向上**的完成顺序：一个任务的后代如果没有全部完成，该任务本身不可能越过 Gate 1 通知父节点。

这意味着：**不可能存在「直接子节点已完成，但孙节点仍 pending」的情况。**

推理链：

1. 任务 C 有子节点 D → D pending → C 的 `complete()` 被 Gate 1 拦住 → C 不可能完成
2. 因此，如果孙节点 pending，直接子节点必定也 pending
3. 只检查直接子节点就能得出与深遍历相同的结果

| 场景 | `_count_pending_descendants` | `count_pending_for_parent` | 行为一致？ |
|---|---|---|---|
| 所有后代都 done | 0 | 0 | ✅ |
| 直接子节点 pending | > 0 | > 0 | ✅ |
| 孙节点 pending、子节点 done | **不可能**（Gate 1 阻断） | — | — |

### 验证：无绕过路径

对全代码库的搜索确认传播保证是稳固的：

| 检查项 | 结果 |
|---|---|
| `_pending` 变异点 | 仅 `register()` (add) 和 `complete()` (discard) 两处 |
| `complete()` 调用点 | 仅 `agent_core.py:285`（正常完成）和 `spawn.py:135`（异常完成）两处 |
| 取消/超时/强制删除方法 | 不存在（config 定义了 timeout 但未实现） |
| `_tasks` 删除操作 | 不存在（任务注册后永不被移除） |
| 外部状态篡改 | `safety.py` 只读访问 `_tasks`，无写操作 |

无论正常完成还是异常退出，都经过 `complete()` → Gate 1，传播保证不被打破。

### 改动方案

#### 1. 注释掉 `_count_pending_descendants_locked` 方法（L196-227）

不删除，整体注释掉，方便回滚。

#### 2. `count_pending_for_parent` 改名为 `_count_pending_for_parent`

该方法唯一的调用点是 `complete()` 内部 L154，无外部调用，加下划线前缀表示私有。

#### 3. 替换调用点（`registry.py:152`）

```python
# Before
pending_descendants = self._count_pending_descendants_locked(task_id)

# After
pending_children = self._count_pending_for_parent(task_id)
```

#### 4. 优化 `_count_pending_for_parent` 实现（可选，降低复杂度）

当前实现是 O(n) 全表扫描，可改为 O(k) 集合查询（k = 子节点数）：

```python
# Before (O(n) — 遍历全部 tasks)
def _count_pending_for_parent(self, parent_task_id: str) -> int:
    count = 0
    for task_id, task in self._tasks.items():
        if task.parent_task_id == parent_task_id and task_id in self._pending:
            count += 1
    return count

# After (O(k) — 只查 child_task_ids)
def _count_pending_for_parent(self, parent_task_id: str) -> int:
    if parent_task_id not in self._tasks:
        return 0
    return sum(1 for cid in self._tasks[parent_task_id].child_task_ids if cid in self._pending)
```

#### 改动后 `complete()` 方法中的 Gate 逻辑

```python
parent_task_id = task.parent_task_id
pending_children = self._count_pending_for_parent(task_id)                        # Gate 1: 我的子节点
pending_siblings = self._count_pending_for_parent(parent_task_id) if parent_task_id else 0  # Gate 3: 兄弟节点

# [Gate 1] Still have pending children → return
if pending_children > 0:
    print(f"[Registry] {task_id} done, {pending_children} children pending")
    return
```

### 附带发现：`safety.py` 的 Bug

`safety.py:94` 引用了 `task.status`，但 `SubagentTask` 模型中没有 `status` 字段，会导致 `AttributeError`。建议修复为 `task.ended_at is not None`。这是独立问题，不在本次重构范围内。