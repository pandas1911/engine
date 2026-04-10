# Agent State 与 Task State 统一状态机合并

## TL;DR

> **Quick Summary**: 将 Agent 的 `AgentState` 枚举和 `SubagentTask.status` 字符串两套并行的状态系统，合并为一个自实现的统一状态机 `AgentStateMachine`，作为 Agent 和 Registry 共同的唯一状态真相源。
> 
> **Deliverables**:
> - 新增 `AgentStateMachine` 类（Swift 可移植，无外部依赖）
> - `AgentState` 枚举重新定义（5 个统一状态）
> - `SubagentTask` 去掉 `status` 字符串字段，改用状态机引用
> - `Agent` 和 `Registry` 的所有状态读写迁移到状态机
> - 现有测试全部更新并通过
> 
> **Estimated Effort**: Medium
> **Parallel Execution**: YES - 4 waves
> **Critical Path**: Task 1 (状态机类) → Task 3 (SubagentTask) → Task 4 (Registry) → Task 5 (Agent) → Task 6 (集成测试) → F1-F4

---

## Context

### Original Request
用户希望将系统中并行的 Agent State（`AgentState` 枚举）和 Task State（`SubagentTask.status` 字符串）合并为一套状态管理，减少复杂度。

### Interview Summary
**Key Discussions**:
- 选择路线 C：统一状态机
- 自己实现，不使用 `transitions` 库
- 代码需能移植到 Swift，避免 Python 独有写法
- ERROR 状态预留，不改动现有错误处理逻辑
- 先改代码后补测试

**Research Findings**:
- `AgentState.ENDED_WITH_PENDING_DESCENDANTS` 定义了但 Agent 从未使用过
- `CALLBACK_PENDING`（Agent 端）和 `"ended_with_pending_descendants"`（Task 端）描述的是同一个语义
- Registry 的 `complete()` 方法有 3 Gate + 2 Branch 逻辑，深度依赖 `task.status` 字符串比较
- 项目使用 pytest，已有 `tests/test_registry.py`（700 行）覆盖了核心流程

---

## Work Objectives

### Core Objective
用统一状态机替代当前两套并行状态系统，消除状态不一致的风险，并通过显式转换规则防止非法状态跳转。

### Concrete Deliverables
- `src/state_machine.py` — 新文件，包含 `AgentStateMachine` 类
- `src/models.py` — 重定义 `AgentState` 枚举，修改 `SubagentTask` dataclass
- `src/registry.py` — 所有 `task.status == "xxx"` 改为状态机枚举比较
- `src/agent_core.py` — 所有 `self.state = AgentState.XXX` 改为状态机 trigger
- `src/tools/builtin/spawn.py` — 如有状态引用则更新
- `main.py` — `agent.state` 检查改为通过状态机
- `tests/test_registry.py` — 所有 status 字符串断言改为枚举断言
- `tests/test_state_machine.py` — 新文件，状态机单元测试

### Definition of Done
- [ ] `AgentStateMachine` 类实现完成，包含 5 个状态和显式转换表
- [ ] `SubagentTask` 不再有 `status: str` 字段
- [ ] `Agent` 不再有 `self.state: AgentState` 字段，改用 `self._state_machine`
- [ ] Registry 的 Gate/Branch 逻辑全部使用 `AgentState` 枚举判断
- [ ] 所有现有测试通过（`pytest tests/`）
- [ ] 新增状态机测试通过
- [ ] 代码中不存在裸字符串状态比较

### Must Have
- 统一的 `AgentState` 枚举：IDLE, RUNNING, WAITING_FOR_CHILDREN, COMPLETED, ERROR
- `AgentStateMachine` 类：转换表 + 验证 + current_state 查询
- 所有状态转换通过状态机 `trigger()` 方法
- ERROR 状态在枚举中预留

### Must NOT Have (Guardrails)
- **不引入外部依赖** — 不用 transitions 等库
- **不使用 Python 独有写法** — 避免 asyncio 专用模式之外的 Python 特性（如元类、描述符、`__dunder__` 魔法方法过度使用），保持 Swift 可移植性
- **不改动错误处理逻辑** — `try/catch` 返回字符串的方式保持不变
- **不改动 QueueEvent** — QueueEvent 数据结构不变
- **不改动 Session/Message/ToolCall/LLMResponse** — 这些模型不动
- **不改动 `_event_queue` 机制** — 事件队列推拉逻辑不变，只是触发条件从字符串比较改为枚举比较
- **不过度抽象** — 状态机类保持简单，不要做成通用框架

---

## Verification Strategy

> **ZERO HUMAN INTERVENTION** - ALL verification is agent-executed. No exceptions.

### Test Decision
- **Infrastructure exists**: YES (pytest)
- **Automated tests**: Tests-after
- **Framework**: pytest

### QA Policy
Every task MUST include agent-executed QA scenarios.
Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

- **Core logic**: Use Bash (pytest) — run tests, verify pass/fail
- **State machine**: Use Bash (python REPL) — instantiate, trigger, assert state

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately - 独立基础模块):
├── Task 1: 实现 AgentStateMachine 类 [quick]
└── Task 2: 重新定义 AgentState 枚举 [quick]

Wave 2 (After Wave 1 - 数据模型 + 消费者迁移):
├── Task 3: 重构 SubagentTask，接入状态机 (depends: 1, 2) [quick]
├── Task 4: 重构 Registry 的状态判断逻辑 (depends: 1, 2) [unspecified-high]
├── Task 5: 重构 Agent 的状态转换逻辑 (depends: 1, 2, 3) [unspecified-high]
└── Task 6: 更新 main.py 的状态引用 (depends: 2, 5) [quick]

Wave 3 (After Wave 2 - 测试):
├── Task 7: 更新 tests/test_registry.py (depends: 3, 4) [unspecified-high]
└── Task 8: 新建 tests/test_state_machine.py (depends: 1) [quick]

Wave FINAL (After ALL tasks — 4 parallel reviews):
├── Task F1: Plan compliance audit (oracle)
├── Task F2: Code quality review (unspecified-high)
├── Task F3: Real manual QA (unspecified-high)
└── Task F4: Scope fidelity check (deep)
-> Present results -> Get explicit user okay

Critical Path: T1 → T3 → T5 → T6 → T7 → F1-F4
Parallel Speedup: ~50% faster than sequential
Max Concurrent: 4 (Wave 2)
```

### Dependency Matrix

| Task | Depends On | Blocks | Wave |
|------|-----------|--------|------|
| 1 | - | 3, 4, 5, 8 | 1 |
| 2 | - | 3, 4, 5, 6 | 1 |
| 3 | 1, 2 | 5, 7 | 2 |
| 4 | 1, 2 | 7 | 2 |
| 5 | 1, 2, 3 | 6 | 2 |
| 6 | 2, 5 | - | 2 |
| 7 | 3, 4 | F1-F4 | 3 |
| 8 | 1 | F1-F4 | 3 |

### Agent Dispatch Summary

- **Wave 1**: 2 tasks — T1 → `quick`, T2 → `quick`
- **Wave 2**: 4 tasks — T3 → `quick`, T4 → `unspecified-high`, T5 → `unspecified-high`, T6 → `quick`
- **Wave 3**: 2 tasks — T7 → `unspecified-high`, T8 → `quick`
- **FINAL**: 4 tasks — F1 → `oracle`, F2 → `unspecified-high`, F3 → `unspecified-high`, F4 → `deep`

---

## TODOs

- [ ] 1. 实现 AgentStateMachine 类

  **What to do**:
  - 在 `src/state_machine.py` 中新建 `AgentStateMachine` 类
  - 定义转换表 `TRANSITIONS: Dict[Tuple[AgentState, str], AgentState]`（key 为 `(from_state, trigger_name)`，value 为 `to_state`）
  - 实现 `trigger(event: str)` 方法：查找转换表，验证合法性，执行转换
  - 实现 `current_state` 属性（只读）
  - 实现 `can_trigger(event: str) -> bool` 方法
  - 非法转换时抛出明确异常（`InvalidTransitionError` 或类似）
  - **Swift 可移植约束**：使用简单的字典 + 类方法，不用 Python 特有语法（如 `@property` 可以用，但不用元类、`__init_subclass__`、dataclass field 高级特性等）

  **Must NOT do**:
  - 不引入外部依赖
  - 不做层级状态机、并行状态等复杂功能
  - 不过度设计 — 当前只需要 5 个状态和 ~7 条转换规则

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 单一文件，逻辑清晰，转换表是静态数据
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Task 2)
  - **Blocks**: Tasks 3, 4, 5, 8
  - **Blocked By**: None

  **References**:

  **Pattern References** (existing code to follow):
  - `src/models.py:13-20` — 当前的 `AgentState` 枚举定义，理解现有状态命名风格
  - `src/agent_core.py:48,103,116,214,247,277` — 当前所有 `self.state = AgentState.XXX` 赋值点，这些是要迁移到 `trigger()` 的调用点

  **API/Type References**:
  - 当前转换映射关系（从代码分析得出）：
    ```
    IDLE + "start"             → RUNNING
    RUNNING + "spawn_children" → WAITING_FOR_CHILDREN
    RUNNING + "finish"         → COMPLETED
    RUNNING + "error"          → ERROR
    WAITING_FOR_CHILDREN + "children_settled" → RUNNING
    ```

  **WHY Each Reference Matters**:
  - `src/models.py:13-20`: 状态枚举的命名风格需要保持一致
  - `src/agent_core.py` 的赋值点: 确定所有需要支持的 trigger 事件

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Happy path - 正常状态转换流程
    Tool: Bash (python -c)
    Preconditions: AgentStateMachine 类已创建
    Steps:
      1. from src.state_machine import AgentStateMachine; from src.models import AgentState
      2. sm = AgentStateMachine(AgentState.IDLE)
      3. sm.trigger("start"); assert sm.current_state == AgentState.RUNNING
      4. sm.trigger("finish"); assert sm.current_state == AgentState.COMPLETED
    Expected Result: 所有 assert 通过，无异常
    Failure Indicators: AssertionError 或 InvalidTransitionError
    Evidence: .sisyphus/evidence/task-1-happy-path.txt

  Scenario: 非法转换被拒绝
    Tool: Bash (python -c)
    Preconditions: AgentStateMachine 类已创建
    Steps:
      1. sm = AgentStateMachine(AgentState.IDLE)
      2. 尝试 sm.trigger("finish") (IDLE → COMPLETED 是非法的)
    Expected Result: 抛出异常，message 包含 "IDLE" 和 "finish"
    Failure Indicators: 没有抛异常（静默接受了非法转换）
    Evidence: .sisyphus/evidence/task-1-invalid-transition.txt

  Scenario: 循环转换 WAITING → RUNNING → WAITING
    Tool: Bash (python -c)
    Steps:
      1. sm = AgentStateMachine(AgentState.RUNNING)
      2. sm.trigger("spawn_children"); assert sm.current_state == AgentState.WAITING_FOR_CHILDREN
      3. sm.trigger("children_settled"); assert sm.current_state == AgentState.RUNNING
      4. sm.trigger("spawn_children"); assert sm.current_state == AgentState.WAITING_FOR_CHILDREN
      5. sm.trigger("children_settled"); assert sm.current_state == AgentState.RUNNING
      6. sm.trigger("finish"); assert sm.current_state == AgentState.COMPLETED
    Expected Result: 循环可重复，最终到达 COMPLETED
    Failure Indicators: 任何 assert 失败
    Evidence: .sisyphus/evidence/task-1-loop-transition.txt
  ```

  **Commit**: YES (groups with Task 2)
  - Message: `refactor(state): add unified state machine and redefine AgentState enum`
  - Files: `src/state_machine.py`
  - Pre-commit: `python -c "from src.state_machine import AgentStateMachine; print('OK')"`

- [ ] 2. 重新定义 AgentState 枚举

  **What to do**:
  - 修改 `src/models.py` 中的 `AgentState` 枚举：
    - 保留: `IDLE`, `RUNNING`, `COMPLETED`
    - 重命名: `CALLBACK_PENDING` → `WAITING_FOR_CHILDREN`
    - 删除: `ENDED_WITH_PENDING_DESCENDANTS`（这个状态 Agent 从未使用过，功能被 `WAITING_FOR_CHILDREN` 取代）
    - 新增: `ERROR`
  - 最终枚举值: `IDLE`, `RUNNING`, `WAITING_FOR_CHILDREN`, `COMPLETED`, `ERROR`

  **Must NOT do**:
  - 不改动枚举之外的任何模型（Session, Message, ToolCall 等）
  - 不删除 `SubagentTask`（那是 Task 3 的事）
  - 不改变枚举的底层类型（仍然是 `str` Enum）

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 只改一个枚举定义，几行代码
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Task 1)
  - **Blocks**: Tasks 3, 4, 5, 6
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `src/models.py:13-20` — 当前的 `AgentState` 枚举，这是要修改的目标

  **API/Type References**:
  - 枚举值映射：
    ```python
    class AgentState(Enum):
        IDLE = "idle"
        RUNNING = "running"
        WAITING_FOR_CHILDREN = "waiting_for_children"  # 替代 CALLBACK_PENDING + ended_with_pending_descendants
        COMPLETED = "completed"
        ERROR = "error"  # 新增预留
    ```

  **WHY Each Reference Matters**:
  - `src/models.py:13-20`: 这是唯一要修改的位置，需要精确替换

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: 枚举值正确性验证
    Tool: Bash (python -c)
    Steps:
      1. from src.models import AgentState
      2. assert hasattr(AgentState, 'IDLE')
      3. assert hasattr(AgentState, 'RUNNING')
      4. assert hasattr(AgentState, 'WAITING_FOR_CHILDREN')
      5. assert hasattr(AgentState, 'COMPLETED')
      6. assert hasattr(AgentState, 'ERROR')
      7. assert AgentState.WAITING_FOR_CHILDREN.value == "waiting_for_children"
      8. assert AgentState.ERROR.value == "error"
      9. assert not hasattr(AgentState, 'CALLBACK_PENDING')
      10. assert not hasattr(AgentState, 'ENDED_WITH_PENDING_DESCENDANTS')
    Expected Result: 所有 assert 通过
    Failure Indicators: AttributeError 或 value 不匹配
    Evidence: .sisyphus/evidence/task-2-enum-values.txt

  Scenario: 旧枚举值不存在验证
    Tool: Bash (python -c)
    Steps:
      1. from src.models import AgentState
      2. 确认 CALLBACK_PENDING 不存在: try getattr(AgentState, 'CALLBACK_PENDING'); except AttributeError: pass
      3. 确认 ENDED_WITH_PENDING_DESCENDANTS 不存在
    Expected Result: AttributeError 被正确抛出
    Evidence: .sisyphus/evidence/task-2-old-values-removed.txt
  ```

  **Commit**: YES (groups with Task 1)
  - Message: `refactor(state): add unified state machine and redefine AgentState enum`
  - Files: `src/models.py`

- [ ] 3. 重构 SubagentTask，接入状态机

  **What to do**:
  - 修改 `src/models.py` 中的 `SubagentTask` dataclass：
    - 删除 `status: str = "running"` 字段
    - 新增 `state_machine: AgentStateMachine` 字段（初始化为 `AgentStateMachine(AgentState.RUNNING)`，因为注册时就是 running 状态）
    - 保留所有其他字段不变：`task_id`, `session_id`, `task_description`, `parent_agent`, `parent_task_id`, `depth`, `child_task_ids`, `completed_event`, `wake_on_descendants_settle`, `ended_at`, `agent`
  - 注意 `state_machine` 字段不能用 `field(default_factory=...)` 因为需要 import AgentStateMachine，用 `field(default=None)` 加 `__post_init__` 初始化，或者直接在 Registry 注册时传入

  **Must NOT do**:
  - 不改动 `QueueEvent` 数据结构
  - 不改动 `Session`, `Message`, `ToolCall`, `LLMResponse`

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 单一 dataclass 修改，逻辑简单
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2 (sequential within wave, depends on 1+2)
  - **Blocks**: Tasks 5, 7
  - **Blocked By**: Tasks 1, 2

  **References**:

  **Pattern References**:
  - `src/models.py:82-98` — 当前 `SubagentTask` dataclass 定义
  - `src/state_machine.py` — Task 1 创建的 `AgentStateMachine` 类

  **API/Type References**:
  - `src/registry.py:64-81` — `register()` 方法中创建 `SubagentTask` 的位置，`status="running"` 是默认值，对应 `AgentState.RUNNING`

  **WHY Each Reference Matters**:
  - `src/models.py:82-98`: 这是修改目标
  - `src/registry.py:64-81`: 创建 SubagentTask 的地方，需要同步改为传入 `state_machine` 而非设置 `status`

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: SubagentTask 使用状态机
    Tool: Bash (python -c)
    Steps:
      1. from src.models import SubagentTask, AgentState
      2. from src.state_machine import AgentStateMachine
      3. sm = AgentStateMachine(AgentState.RUNNING)
      4. task = SubagentTask(task_id="t1", session_id="s1", task_description="test", parent_agent=None, state_machine=sm)
      5. assert task.state_machine.current_state == AgentState.RUNNING
      6. assert not hasattr(task, 'status') 或 task.status 不存在
    Expected Result: SubagentTask 通过 state_machine 管理状态，没有 status 字段
    Failure Indicators: AttributeError 或 status 字段仍存在
    Evidence: .sisyphus/evidence/task-3-subagent-task.txt
  ```

  **Commit**: YES (groups with Wave 2)
  - Message: `refactor(state): migrate SubagentTask, Registry, Agent, and main to state machine`
  - Files: `src/models.py`

- [ ] 4. 重构 Registry 的状态判断逻辑

  **What to do**:
  - 修改 `src/registry.py` 中所有 `task.status == "xxx"` 的比较，改为 `task.state_machine.current_state == AgentState.XXX`
  - 具体修改点：
    - `complete()` 方法 (line 140-188)：
      - `task.status = "error"` → `task.state_machine.trigger("error")`
      - `task.status = "completed"` → `task.state_machine.trigger("finish")`
      - `parent_task.status in ["completed", "ended_with_pending_descendants"]` → `parent_task.state_machine.current_state in [AgentState.COMPLETED, AgentState.WAITING_FOR_CHILDREN]`
      - `parent_task.status = "running"` → `parent_task.state_machine.trigger("children_settled")`
    - `mark_ended_with_pending_descendants()` (line 281-292)：
      - `task.status = "ended_with_pending_descendants"` → `task.state_machine.trigger("spawn_children")`（语义映射：结束当前处理，等待子代理）
    - `_pending` set 的逻辑：从基于 `task_id in self._pending` 改为基于 `task.state_machine.current_state` 判断是否仍在活跃状态（或保持 `_pending` set 不变，它与状态机是互补的 — `_pending` 跟踪注册/完成，状态机跟踪生命周期）
    - **决策**: `_pending` set 保持不变。它跟踪的是"注册但未完成"的任务集合，与状态机的生命周期状态是不同维度的关注点。`_pending` 更像是 Registry 的索引，不是状态。

  **Must NOT do**:
  - 不改变 Gate/Branch 的逻辑结构（3 Gate + 2 Branch 保持不变）
  - 不改变 `_pending` set 的管理逻辑
  - 不改变 `QueueEvent` 的创建和推送方式
  - 不改变 `_count_pending_descendants_locked()` 的算法

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Registry 的 Gate 逻辑复杂，需要仔细对照每个 status 比较，不能遗漏
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Task 5, 6)
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 7
  - **Blocked By**: Tasks 1, 2

  **References**:

  **Pattern References**:
  - `src/registry.py:140-188` — `complete()` 方法，包含所有 Gate/Branch 逻辑，这是核心修改区域
  - `src/registry.py:281-292` — `mark_ended_with_pending_descendants()` 方法

  **API/Type References**:
  - `src/state_machine.py` — `AgentStateMachine.trigger()` 和 `current_state` 的 API
  - 状态映射表：
    ```
    task.status = "error"                          → task.state_machine.trigger("error")
    task.status = "completed"                      → task.state_machine.trigger("finish")
    parent_task.status == "completed"              → parent_task.state_machine.current_state == AgentState.COMPLETED
    parent_task.status == "ended_with_pending_descendants" → parent_task.state_machine.current_state == AgentState.WAITING_FOR_CHILDREN
    parent_task.status = "running"                 → parent_task.state_machine.trigger("children_settled")
    task.status = "ended_with_pending_descendants" → task.state_machine.trigger("spawn_children")
    ```

  **Test References**:
  - `tests/test_registry.py` — 700 行测试，覆盖了所有 Gate/Branch 路径，修改后必须全部通过

  **WHY Each Reference Matters**:
  - `src/registry.py:140-188`: 这是最关键的区域，每个 status 比较都必须精确替换
  - `tests/test_registry.py`: 作为修改后的验证基准，确保行为不变

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Registry complete() 状态转换正确
    Tool: Bash (pytest)
    Preconditions: Tasks 1-4 完成
    Steps:
      1. pytest tests/test_registry.py -v
      2. 检查所有测试通过
    Expected Result: 所有 test_registry.py 测试 PASS（此时测试可能还没更新，先确认 import 正常，后续 Task 7 更新测试）
    Failure Indicators: ImportError 或 AttributeError
    Evidence: .sisyphus/evidence/task-4-registry-tests.txt

  Scenario: Registry 中无裸字符串状态比较
    Tool: Bash (grep)
    Steps:
      1. grep -n 'status ==' src/registry.py
      2. grep -n 'status = "' src/registry.py
    Expected Result: 0 行匹配（所有裸字符串比较已替换为枚举）
    Failure Indicators: 任何匹配行
    Evidence: .sisyphus/evidence/task-4-no-strings.txt
  ```

  **Commit**: YES (groups with Wave 2)
  - Message: `refactor(state): migrate SubagentTask, Registry, Agent, and main to state machine`
  - Files: `src/registry.py`

- [ ] 5. 重构 Agent 的状态转换逻辑

  **What to do**:
  - 修改 `src/agent_core.py` 中的 `Agent` 类：
    - 删除 `self.state = AgentState.IDLE` (line 48)
    - 新增 `self._state_machine = AgentStateMachine(AgentState.IDLE)`
    - 提供 `self.state` 属性作为 `self._state_machine.current_state` 的便捷访问（向后兼容 main.py 的 `agent.state` 用法）
    - 所有 `self.state = AgentState.XXX` 改为 `self._state_machine.trigger("event")`：
      - `self.state = AgentState.RUNNING` (line 103, 214) → `self._state_machine.trigger("start")`
      - `self.state = AgentState.CALLBACK_PENDING` (line 116, 247) → 不直接设状态，改为调用 `registry.mark_ended_with_pending_descendants()` 后状态机由 Registry 触发。**注意**: 这里需要仔细设计。Agent 的 CALLBACK_PENDING 对应 WAITING_FOR_CHILDREN，但触发者不同。当前流程是 Agent 自己设 CALLBACK_PENDING，然后调用 `registry.mark_ended_with_pending_descendants()`。合并后，Agent 应该通过 `self._state_machine.trigger("spawn_children")` 进入 WAITING_FOR_CHILDREN，同时 Registry 的 `mark_ended_with_pending_descendants()` 方法内部不需要再改 task status（因为 status 已经不存在了）。
      - `self.state = AgentState.COMPLETED` (line 277) → `self._state_machine.trigger("finish")`
    - 所有 `if self.state == AgentState.COMPLETED` 改为 `if self._state_machine.current_state == AgentState.COMPLETED`

  **Must NOT do**:
  - 不改变 `_process_tool_calls()` 的逻辑
  - 不改变 `_event_queue` 的机制
  - 不改变 `_resume_from_children()` 的业务逻辑（收集结果、构建 prompt），只改状态管理部分
  - 不改变 `_finish_and_notify()` 的通知逻辑，只改状态设置

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Agent 的状态转换点多，且与 Registry 有交互，需要仔细处理触发顺序
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Task 4, 6, but needs Task 3 first)
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 6
  - **Blocked By**: Tasks 1, 2, 3

  **References**:

  **Pattern References**:
  - `src/agent_core.py:48` — `self.state = AgentState.IDLE` 初始化
  - `src/agent_core.py:99-123` — `run()` 方法，包含 RUNNING 和 CALLBACK_PENDING 转换
  - `src/agent_core.py:212-250` — `_resume_from_children()` 方法，包含 RUNNING 和 CALLBACK_PENDING 转换
  - `src/agent_core.py:269-283` — `_finish_and_notify()` 方法，包含 COMPLETED 转换

  **API/Type References**:
  - `src/state_machine.py` — `AgentStateMachine` 类的 API
  - 转换映射：
    ```
    self.state = AgentState.RUNNING            → self._state_machine.trigger("start")
    self.state = AgentState.CALLBACK_PENDING   → self._state_machine.trigger("spawn_children")
    self.state = AgentState.COMPLETED          → self._state_machine.trigger("finish")
    self.state == AgentState.COMPLETED         → self._state_machine.current_state == AgentState.COMPLETED
    ```

  **WHY Each Reference Matters**:
  - `src/agent_core.py:48,103,116,214,247,277`: 每一个状态赋值点都需要精确替换
  - `src/agent_core.py:111,242,257`: 每一个状态读取点都需要改为通过状态机

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Agent 中无直接状态赋值
    Tool: Bash (grep)
    Steps:
      1. grep -n 'self\.state = ' src/agent_core.py
    Expected Result: 0 行匹配（所有状态赋值通过状态机）
    Failure Indicators: 任何匹配行
    Evidence: .sisyphus/evidence/task-5-no-direct-state.txt

  Scenario: Agent 状态属性可读
    Tool: Bash (python -c)
    Steps:
      1. from src.agent_core import Agent; from src.models import Session, AgentState; from src.config import Config
      2. session = Session(id="test", depth=0)
      3. config = Config()
      4. agent = Agent(session, config)
      5. assert agent.state == AgentState.IDLE
    Expected Result: agent.state 返回 AgentState.IDLE
    Failure Indicators: AttributeError 或值不正确
    Evidence: .sisyphus/evidence/task-5-agent-state-property.txt
  ```

  **Commit**: YES (groups with Wave 2)
  - Message: `refactor(state): migrate SubagentTask, Registry, Agent, and main to state machine`
  - Files: `src/agent_core.py`

- [ ] 6. 更新 main.py 的状态引用

  **What to do**:
  - 修改 `main.py` 第 79 行：
    - `if agent.state != AgentState.COMPLETED:` 保持不变（因为 Task 5 中 `agent.state` 作为属性保留了）
  - 修改 import 语句（如需要）：
    - 确保 `from src.models import AgentState` 仍然有效
  - 检查是否还有其他 `AgentState.CALLBACK_PENDING` 或 `AgentState.ENDED_WITH_PENDING_DESCENDANTS` 的引用需要更新

  **Must NOT do**:
  - 不改变 main.py 的业务逻辑流程

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 可能只需要确认 import 正确，改动极少
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Task 4, 5)
  - **Parallel Group**: Wave 2
  - **Blocks**: None
  - **Blocked By**: Tasks 2, 5

  **References**:

  **Pattern References**:
  - `main.py:79` — `if agent.state != AgentState.COMPLETED:` 唯一的状态检查

  **API/Type References**:
  - Task 5 中保留了 `agent.state` 属性，所以 main.py 的代码不需要改逻辑

  **WHY Each Reference Matters**:
  - `main.py:79`: 唯一需要确认的引用点

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: main.py import 和状态检查正常
    Tool: Bash (python -c)
    Steps:
      1. from src.models import AgentState
      2. assert hasattr(AgentState, 'COMPLETED')
      3. assert hasattr(AgentState, 'IDLE')
    Expected Result: import 成功，枚举值存在
    Evidence: .sisyphus/evidence/task-6-main-import.txt
  ```

  **Commit**: YES (groups with Wave 2)
  - Message: `refactor(state): migrate SubagentTask, Registry, Agent, and main to state machine`
  - Files: `main.py`

- [ ] 7. 更新 tests/test_registry.py

  **What to do**:
  - 将 `tests/test_registry.py` 中所有 `task.status == "xxx"` 断言改为 `task.state_machine.current_state == AgentState.XXX`
  - 具体修改：
    - `task.status == "running"` → `task.state_machine.current_state == AgentState.RUNNING`
    - `task.status == "completed"` → `task.state_machine.current_state == AgentState.COMPLETED`
    - `parent_task.status == "running"` → 同上
    - `MockAgent` 类保持不变（它不需要状态机，只模拟 `_resume_from_children` 和 `_event_queue`）
  - 确保 `from src.models import AgentState` import 正确
  - 移除任何对已删除的 `CALLBACK_PENDING` 或 `ENDED_WITH_PENDING_DESCENDANTS` 枚举值的引用

  **Must NOT do**:
  - 不改变测试的业务逻辑（不增加/删除测试用例）
  - 不改变 MockAgent 的行为
  - 不改变测试结构（类名、方法名保持不变）

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 700 行测试文件，需要逐一检查每个 status 引用
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO (needs both Registry and SubagentTask changes)
  - **Parallel Group**: Wave 3 (with Task 8)
  - **Blocks**: F1-F4
  - **Blocked By**: Tasks 3, 4

  **References**:

  **Pattern References**:
  - `tests/test_registry.py:186` — `assert parent_task.status == "running"` 典型断言
  - `tests/test_registry.py:656` — `assert task.status == "completed"` 典型断言

  **Test References**:
  - `tests/test_registry.py` — 整个文件都是修改目标

  **WHY Each Reference Matters**:
  - 这些断言是验证重构正确性的基准，必须精确替换

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: 所有 registry 测试通过
    Tool: Bash (pytest)
    Steps:
      1. pytest tests/test_registry.py -v
    Expected Result: 所有测试 PASS，0 failures
    Failure Indicators: 任何 FAILED 测试
    Evidence: .sisyphus/evidence/task-7-registry-tests-pass.txt

  Scenario: 测试中无裸字符串状态断言
    Tool: Bash (grep)
    Steps:
      1. grep -n 'status ==' tests/test_registry.py
      2. grep -n '\.status' tests/test_registry.py
    Expected Result: 0 行匹配
    Failure Indicators: 任何匹配行
    Evidence: .sisyphus/evidence/task-7-no-string-asserts.txt
  ```

  **Commit**: YES (groups with Task 8)
  - Message: `test(state): update registry tests and add state machine unit tests`
  - Files: `tests/test_registry.py`
  - Pre-commit: `pytest tests/test_registry.py -v`

- [ ] 8. 新建 tests/test_state_machine.py

  **What to do**:
  - 在 `tests/` 目录下新建 `test_state_machine.py`
  - 测试内容：
    - 正常转换流程：IDLE → RUNNING → COMPLETED
    - 正常转换流程：IDLE → RUNNING → WAITING_FOR_CHILDREN → RUNNING → COMPLETED
    - 循环流程：RUNNING → WAITING → RUNNING → WAITING → RUNNING → COMPLETED
    - 非法转换：COMPLETED → RUNNING (应报错)
    - 非法转换：IDLE → COMPLETED (应报错)
    - 非法转换：WAITING_FOR_CHILDREN → COMPLETED (应报错)
    - ERROR 状态不可转换
    - `can_trigger()` 方法正确返回 True/False
    - 初始状态为 IDLE
  - 测试风格遵循项目现有风格：使用 pytest + class 组织

  **Must NOT do**:
  - 不测试 Agent 或 Registry 的集成（那些在 test_registry.py 中）
  - 不使用 mock

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 状态机逻辑简单，测试用例是确定性的
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (only depends on Task 1)
  - **Parallel Group**: Wave 3 (with Task 7)
  - **Blocks**: F1-F4
  - **Blocked By**: Task 1

  **References**:

  **Pattern References**:
  - `tests/test_registry.py:1-33` — 测试文件的风格参考（import、helper 函数、class 组织）
  - `tests/test_tools.py` — 另一个风格参考

  **API/Type References**:
  - `src/state_machine.py` — `AgentStateMachine` 类的 API

  **WHY Each Reference Matters**:
  - `tests/test_registry.py:1-33`: 确保新测试文件风格一致

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: 状态机测试全部通过
    Tool: Bash (pytest)
    Steps:
      1. pytest tests/test_state_machine.py -v
    Expected Result: 所有测试 PASS
    Failure Indicators: 任何 FAILED
    Evidence: .sisyphus/evidence/task-8-state-machine-tests.txt

  Scenario: 完整测试套件通过
    Tool: Bash (pytest)
    Steps:
      1. pytest tests/ -v
    Expected Result: 所有测试 PASS（包括 test_registry.py 和 test_state_machine.py）
    Failure Indicators: 任何 FAILED
    Evidence: .sisyphus/evidence/task-8-full-test-suite.txt
  ```

  **Commit**: YES (groups with Task 7)
  - Message: `test(state): update registry tests and add state machine unit tests`
  - Files: `tests/test_state_machine.py`
  - Pre-commit: `pytest tests/test_state_machine.py -v`

---

## Final Verification Wave (MANDATORY — after ALL implementation tasks)

> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.

- [ ] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists (read file, run command). For each "Must NOT Have": search codebase for forbidden patterns — reject with file:line if found. Check evidence files exist in .sisyphus/evidence/. Compare deliverables against plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [ ] F2. **Code Quality Review** — `unspecified-high`
  Run `pytest tests/` and check all pass. Review all changed files for: bare string state comparisons (e.g., `== "running"`), inconsistent state access, commented-out code, unused imports. Check AI slop: excessive comments, over-abstraction.
  Output: `Tests [PASS/FAIL] | Files [N clean/N issues] | VERDICT`

- [ ] F3. **Real Manual QA** — `unspecified-high`
  Start from clean state. Run `python main.py` with default prompt. Verify agent system starts, spawns subagents, collects results, completes. Run `pytest tests/` and verify all pass. Save output to `.sisyphus/evidence/final-qa/`.
  Output: `Scenarios [N/N pass] | Integration [N/N] | VERDICT`

- [ ] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", read actual diff. Verify 1:1 — everything in spec was built, nothing beyond spec was built. Check "Must NOT do" compliance. Flag unaccounted changes.
  Output: `Tasks [N/N compliant] | Unaccounted [CLEAN/N files] | VERDICT`

---

## Commit Strategy

- **Wave 1**: `refactor(state): add unified state machine and redefine AgentState enum` - src/state_machine.py, src/models.py
- **Wave 2**: `refactor(state): migrate SubagentTask, Registry, Agent, and main to state machine` - src/models.py, src/registry.py, src/agent_core.py, main.py
- **Wave 3**: `test(state): update registry tests and add state machine unit tests` - tests/test_registry.py, tests/test_state_machine.py

---

## Success Criteria

### Verification Commands
```bash
pytest tests/ -v                   # Expected: all tests pass
python -c "from src.state_machine import AgentStateMachine; print('OK')"  # Expected: OK
grep -r 'status == "' src/         # Expected: no results (no bare string comparisons)
grep -r 'self\.state = ' src/      # Expected: no results (state set via state machine only)
```

### Final Checklist
- [ ] All "Must Have" present
- [ ] All "Must NOT Have" absent
- [ ] All tests pass (`pytest tests/`)
- [ ] No bare string state comparisons in `src/`
- [ ] No direct `self.state = ` assignment in Agent
- [ ] `SubagentTask` has no `status: str` field
