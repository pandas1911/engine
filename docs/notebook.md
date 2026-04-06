# 开发笔记 - 多层代理"提前汇报"重大Bug修复

## 问题发现过程

### 初始观察
在运行 `main.py` 测试多层代理系统时，发现偶发的"提前汇报"现象：
- **现象描述**：父代代理在所有子代代理完成之前就向上级汇报了结果
- **具体表现**：C有孙代理D，D还未完成，但C就已经向B汇报了

### 复现测试
创建了多个测试用例来复现和验证bug：

#### 测试1：并发孙子代理完成 (test_early_report.py)
```
场景：A → B → [C, D]
结果：正常，B正确等待C和D都完成才汇报
```

#### 测试2：子代理注册竞态 (test_registration_race.py) ⚠️
```
场景：A → B → C → [C1, C2, C3]
时序：
T1: C调用spawn(C1, C2, C3) → 创建3个异步任务
T2: C立即调用mark_ended_with_pending_descendants()
T3: C1注册并快速完成
T4: C被wake，检查count_pending_for_parent(C) = 0
    ❌ 因为C2和C3还没注册到registry！
T5: C向B汇报 ← 提前汇报！
T6: C2注册（太晚了）
T7: C3注册（太晚了）

结果：✗ BUG复现！C在C2/C3未完成时就向B汇报
```

**关键日志**：
```
[C] ← Woken by task_c1
[C] Pending children: 0
[C] ✗ ALL CHILDREN DONE - WOULD REPORT TO PARENT NOW!

Now C2 and C3 register (TOO LATE!):
  C's pending children: 2
```

---

## Bug根本原因分析

### 核心问题：时序竞态条件

**问题链条**：
1. `agent_core.py:94` - `await self._process_tool_calls()` 返回
   - spawn操作是异步的：`asyncio.create_task(_run_child_agent(...))`
   - 当方法返回时，子代理可能还没注册到registry！

2. `agent_core.py:98` - `await self.registry.mark_ended_with_pending_descendants(self.task_id)`
   - 此时父代理进入"等待子孙"状态
   - 但**不保证所有子代理都已注册**

3. 子代理C1快速注册并完成
4. `registry.py:171-178` - 触发wake父代理C
5. `agent_core.py:232` - C调用 `count_pending_for_parent(C)`
   - 只统计**已注册**的子代理
   - 正在注册中但未完成的子代理**被忽略**！

6. `agent_core.py:234-241` - C发现pending=0，调用 `_continue_processing()`
   - **提前汇报！**

### 代码路径分析

#### SpawnTool.execute() (tools.py:68-135)
```python
async def execute(self, arguments: Dict[str, Any], context: Dict[str, Any]) -> str:
    # ... 创建子session ...
    
    # 注册子代理
    await self.registry.register(
        task_id=task_id,
        parent_agent=context["parent_agent"],
        parent_task_id=self.parent_task_id,
        depth=child_session.depth,
    )
    
    # 异步启动子代理
    asyncio.create_task(  # ← 关键：异步启动！
        self._run_child_agent(child_session, task_desc, task_id, config)
    )
    
    # 立即返回，不等待子代理注册完成
    return f"━━━━ Spawned Task ━━━━\n..."
```

#### Agent.run() (agent_core.py:94-99)
```python
spawned_any = await self._process_tool_calls()  # ← spawn是异步的
if spawned_any:
    print(f"[{self.label}|{self.task_id}] → Waiting for subagents")
    self.state = AgentState.CALLBACK_PENDING
    await self.registry.mark_ended_with_pending_descendants(self.task_id)  # ← 问题点！
    return "[等待子代理回调...]"
```

**问题**：`_process_tool_calls()` 返回时，子代理可能还在注册中！

#### count_pending_for_parent() (registry.py:260-273)
```python
def count_pending_for_parent(self, parent_task_id: str) -> int:
    count = 0
    for task_id, task in self._tasks.items():
        if task.parent_task_id == parent_task_id and task_id in self._pending:
            # ← 只统计已注册且在pending中的任务
            count += 1
    return count
```

**问题**：无法检测正在注册中但未完成注册的任务！

---

## 修复方案对比

### 方案1：等待所有子代理注册完成

**思路**：在调用 `mark_ended_with_pending_descendants` 之前，确保所有子代理都已注册。

**实现**：
```python
# agent_core.py
spawned_any = await self._process_tool_calls()
if spawned_any:
    # 等待所有子代理注册完成
    await asyncio.sleep(0.1)  # 不优雅！
    await self.registry.mark_ended_with_pending_descendants(self.task_id)
```

**优点**：
- ✅ 简单直接

**缺点**：
- ❌ 阻塞异步执行
- ❌ sleep时间难以确定
- ❌ 不是真正的解决方案

---

### 方案2：在Agent中维护spawn计数 ⭐

**思路**：让Agent自己记录spawn的子任务数量，不依赖registry的 `_pending` 状态。

**实现**：
```python
class Agent:
    def __init__(self, ...):
        self._spawned_child_count = 0
        self._completed_child_count = 0
    
    # spawn时
    parent_agent._spawned_child_count += 1
    
    # 子任务完成时
    parent_agent._completed_child_count += 1
    if self._completed_child_count < self._spawned_child_count:
        return  # 还有子任务未完成
    # 所有子任务完成，继续处理
```

**优点**：
- ✅ 不依赖registry的 `_pending` 状态
- ✅ 不阻塞子代理的异步执行
- ✅ 即使子代理还在注册中，parent也知道有N个子任务在运行

**缺点**：
- ⚠️ Agent需要维护额外的状态
- ⚠️ 如果spawn失败，计数不准确
- ⚠️ 状态分散在各个Agent中，不易集中管理

---

### 方案3：在registry中维护"spawning"状态 ⭐⭐⭐ 推荐

**思路**：在spawn开始时就告诉registry"我准备spawn一个子任务"，注册完成时减少计数。

**核心思想**：
```
pending子任务数 = 已注册且在running的 + 正在注册中的
```

**实现**：
```python
# registry.py
class SubagentRegistry:
    def __init__(self):
        self._tasks: Dict[str, SubagentTask] = {}
        self._pending: Set[str] = set()
        self._spawning: Dict[str, int] = {}  # parent_task_id -> count
        self._lock = asyncio.Lock()

    async def begin_spawn(self, parent_task_id: str):
        """告诉registry：开始spawn一个子任务"""
        async with self._lock:
            self._spawning[parent_task_id] = \
                self._spawning.get(parent_task_id, 0) + 1
    
    async def register(self, task_id, ..., parent_task_id, ...):
        """注册子任务时，减少spawning计数"""
        async with self._lock:
            # ... 原有注册逻辑 ...
            
            # spawning计数减1
            if parent_task_id in self._spawning:
                self._spawning[parent_task_id] -= 1
                if self._spawning[parent_task_id] == 0:
                    del self._spawning[parent_task_id]
    
    def count_pending_for_parent(self, parent_task_id: str) -> int:
        """统计pending子任务数（包括正在注册的）"""
        count = 0
        # 已注册的pending子任务
        for task_id, task in self._tasks.items():
            if task.parent_task_id == parent_task_id and task_id in self._pending:
                count += 1
        
        # 正在spawn的子任务
        count += self._spawning.get(parent_task_id, 0)
        
        return count
```

**SpawnTool修改**：
```python
# tools.py
async def execute(self, arguments: Dict[str, Any], context: Dict[str, Any]) -> str:
    parent_task_id = self.parent_task_id
    
    # 告诉registry：开始spawn
    await self.registry.begin_spawn(parent_task_id)
    
    try:
        # ... 原有spawn逻辑 ...
        asyncio.create_task(
            self._run_child_agent(child_session, task_desc, task_id, config)
        )
    except Exception as e:
        # spawn失败，恢复计数
        await self.registry.cancel_spawn(parent_task_id)
        raise
```

**优点**：
- ✅ 不阻塞子代理的异步执行
- ✅ 语义清晰：`pending = 已注册的 + 正在注册的`
- ✅ 集中管理：状态在registry中，而不是分散在Agent中
- ✅ 最小改动：只需要修改registry和SpawnTool
- ✅ 准确计数：可以正确检测正在注册中的任务

**缺点**：
- ⚠️ registry增加额外状态 `_spawning`
- ⚠️ 需要确保spawn失败时正确减少计数
- ⚠️ 增加了API复杂度（新增 `begin_spawn` 方法）

---

### 方案4：使用Event/Signal机制

**思路**：子代理注册完成时发出信号，父代理等待所有信号。

**缺点**：
- ❌ 会让代码变复杂
- ❌ 需要维护多个Event对象
- ❌ 不推荐

---

## 最终选择：方案3（在registry中维护spawning状态）

### 选择理由

1. **不阻塞**：子代理仍然是异步启动的
2. **语义清晰**：`count_pending_for_parent` 返回的是"未完成的子任务数"，包括：
   - 已注册且在运行的
   - 正在注册中的
3. **最小改动**：只需要修改registry和SpawnTool
4. **集中管理**：状态管理在registry中，而不是分散在各个Agent中
5. **鲁棒性**：即使子代理注册有延迟，也能正确统计

### 关键优势

**对比方案2**：
- 方案2需要修改Agent类，增加实例变量
- 方案3的 `count_pending_for_parent` 接口不变，只是内部逻辑改变
- 方案3更容易测试和验证

---

## 实现草案

### 1. 修改 SubagentRegistry (src/registry.py)

#### 新增实例变量
```python
def __init__(self):
    self._tasks: Dict[str, SubagentTask] = {}
    self._pending: Set[str] = set()
    self._spawning: Dict[str, int] = {}  # 新增：parent_task_id -> spawning count
    self._lock = asyncio.Lock()
```

#### 新增方法：begin_spawn
```python
async def begin_spawn(self, parent_task_id: str):
    """告诉registry：开始spawn一个子任务
    
    Args:
        parent_task_id: 父任务的task_id
    
    Note:
        必须在spawn开始时调用，在register时计数会自动减少
    """
    async with self._lock:
        self._spawning[parent_task_id] = \
            self._spawning.get(parent_task_id, 0) + 1
```

#### 修改方法：register
```python
async def register(
    self,
    task_id: str,
    session_id: str,
    description: str,
    parent_agent: "Agent",
    parent_task_id: Optional[str] = None,
    depth: int = 0,
) -> SubagentTask:
    """Register a subagent - Similar to OpenClaw's registerSubagentRun.
    
    Args:
        ... (原有参数)
    
    Returns:
        The created SubagentTask
    
    Note:
        如果有parent_task_id，会自动减少spawning计数
    """
    # Check for cycles before registering
    if parent_task_id and self._would_create_cycle(task_id, parent_task_id):
        raise ValueError(
            f"Cycle detected: agent {task_id} already exists in ancestor chain"
        )

    task = SubagentTask(
        task_id=task_id,
        session_id=session_id,
        task_description=description,
        parent_agent=parent_agent,
        parent_task_id=parent_task_id,
        depth=depth,
    )

    async with self._lock:
        self._tasks[task_id] = task
        self._pending.add(task_id)

        if parent_task_id and parent_task_id in self._tasks:
            self._tasks[parent_task_id].child_task_ids.add(task_id)
        
        # 新增：减少spawning计数
        if parent_task_id and parent_task_id in self._spawning:
            self._spawning[parent_task_id] -= 1
            if self._spawning[parent_task_id] == 0:
                del self._spawning[parent_task_id]

    return task
```

#### 修改方法：count_pending_for_parent
```python
def count_pending_for_parent(self, parent_task_id: str) -> int:
    """Count pending subagents for a specific parent.
    
    Args:
        parent_task_id: The parent task ID to count for
    
    Returns:
        Number of pending child tasks for this parent, including:
        - Already registered and in _pending
        - Currently spawning (not yet registered)
    """
    count = 0
    
    # 统计已注册且在pending中的
    for task_id, task in self._tasks.items():
        if task.parent_task_id == parent_task_id and task_id in self._pending:
            count += 1
    
    # 统计正在spawn的（还未注册）
    count += self._spawning.get(parent_task_id, 0)
    
    return count
```

#### 新增方法：cancel_spawn（可选，用于错误处理）
```python
async def cancel_spawn(self, parent_task_id: str):
    """取消一个spawning计数（用于spawn失败时恢复）
    
    Args:
        parent_task_id: 父任务的task_id
    """
    async with self._lock:
        if parent_task_id in self._spawning:
            self._spawning[parent_task_id] -= 1
            if self._spawning[parent_task_id] == 0:
                del self._spawning[parent_task_id]
```

---

### 2. 修改 SpawnTool (src/tools.py)

#### 修改方法：execute
```python
async def execute(self, arguments: Dict[str, Any], context: Dict[str, Any]) -> str:
    """Execute spawn tool to create a child agent."""
    parent_session = context["session"]
    config = context["config"]

    if parent_session.depth >= config.max_depth:
        return f"[错误] 已达到最大深度限制 ({config.max_depth})"

    task_desc = arguments.get("task", "")
    label = arguments.get("label", "subagent")
    task_id = f"task_{uuid.uuid4().hex[:8]}"

    child_session = Session(
        id=f"sess_{uuid.uuid4().hex[:8]}",
        depth=parent_session.depth + 1,
        parent_id=parent_session.id,
    )

    parent_label = self.parent_label
    can_spawn = child_session.depth < config.max_depth

    print(
        f"[{self.parent_label}|{self.parent_task_id}] → Spawn: [{label}|{task_id}] (depth={child_session.depth})"
    )

    # 新增：告诉registry开始spawn
    await self.registry.begin_spawn(self.parent_task_id)

    system_prompt = f"""# Subagent Context

You are a **subagent** spawned by the {parent_label} for a specific task.

## Your Role
- You were created to handle: {task_desc}
- Complete this task. That's your entire purpose.
- You are NOT the {parent_label}. Don't try to be.

## Rules
1. **Stay focused** - Do your assigned task, nothing else
2. **Complete the task** - Your final message will be automatically reported to the {parent_label}
3. **Don't initiate** - No heartbeats, no proactive actions, no side quests
4. **Be ephemeral** - You may be terminated after task completion. That's fine.
5. **Trust push-based completion** - Descendant results are auto-announced back to you

{"## Sub-Agent Spawning\\nYou CAN spawn your own sub-agents." if can_spawn else "## Sub-Agent Spawning\\nYou are a leaf worker and CANNOT spawn further sub-agents."}

## Session Context
- Label: {label}
- Depth: {child_session.depth}/{config.max_depth}
- Your task ID: {task_id}"""

    child_session.add_message("system", system_prompt)

    # 注意：这里不再需要await register
    # register会在_run_child_agent中调用
    
    try:
        asyncio.create_task(
            self._run_child_agent(child_session, task_desc, task_id, config)
        )
    except Exception as e:
        # spawn失败，恢复spawning计数
        await self.registry.cancel_spawn(self.parent_task_id)
        print(f"[SpawnTool] Failed to spawn {task_id}: {e}")
        raise

    return f"""━━━━ Spawned Task ━━━━
Task ID: {task_id}
Agent: {label}

Sub-agent is now executing in the background. Upon completion, you will be automatically re-activated and receive a full result report. You may proceed with other independent tasks or simply end your current turn."""
```

#### 修改方法：_run_child_agent
```python
async def _run_child_agent(
    self, child_session: Session, task_desc: str, task_id: str, config: Config
):
    """Run child agent and handle completion.

    Args:
        child_session: Child agent's session
        task_desc: Task description for child
        task_id: Child's task ID
        config: Configuration object

    Note:
        On completion, registry.complete() will automatically notify parent.
        On error, registry.complete() is called with error=True.
    """
    try:
        # 创建agent
        agent = self.agent_factory(
            child_session, config, self.registry, self.parent_task_id, task_id
        )
        
        # 注册到registry（会自动减少spawning计数）
        await self.registry.register(
            task_id=task_id,
            session_id=child_session.id,
            description=task_desc,
            parent_agent=context["parent_agent"],
            parent_task_id=self.parent_task_id,
            depth=child_session.depth,
        )
        
        await self.registry.set_agent(task_id, agent)
        result = await agent.run(task_desc)
    except Exception as e:
        print(f"[Subagent|{task_id}] ✗ Failed: {e}")
        await self.registry.complete(task_id, f"[Error] {e}", error=True)
```

**注意**：`_run_child_agent` 的参数中缺少 `context`，需要从 `execute` 方法传递进来。

#### 完整修改：传递context
```python
async def execute(self, arguments: Dict[str, Any], context: Dict[str, Any]) -> str:
    # ... 前面的代码 ...
    
    try:
        asyncio.create_task(
            self._run_child_agent(
                child_session, task_desc, task_id, config, context  # 传递context
            )
        )
    except Exception as e:
        await self.registry.cancel_spawn(self.parent_task_id)
        print(f"[SpawnTool] Failed to spawn {task_id}: {e}")
        raise

    return f"""━━━━ Spawned Task ━━━━..."""

async def _run_child_agent(
    self, 
    child_session: Session, 
    task_desc: str, 
    task_id: str, 
    config: Config,
    context: Dict[str, Any]  # 新增参数
):
    """Run child agent and handle completion."""
    try:
        agent = self.agent_factory(
            child_session, config, self.registry, self.parent_task_id, task_id
        )
        
        await self.registry.register(
            task_id=task_id,
            session_id=child_session.id,
            description=task_desc,
            parent_agent=context["parent_agent"],
            parent_task_id=self.parent_task_id,
            depth=child_session.depth,
        )
        
        await self.registry.set_agent(task_id, agent)
        result = await agent.run(task_desc)
    except Exception as e:
        print(f"[Subagent|{task_id}] ✗ Failed: {e}")
        await self.registry.complete(task_id, f"[Error] {e}", error=True)
```

---

## 测试验证

### 测试用例1：复现原始bug（修复前应该失败，修复后应该通过）

```python
# test_registration_race_fixed.py
async def test_child_spawns_slowly_fixed():
    """修复后：C应该等待所有子代理注册完成才汇报"""
    registry = SubagentRegistry()
    
    # ... 创建A, B, C ...
    
    # C开始spawn
    await registry.begin_spawn("task_c")  # 开始spawn
    await registry.begin_spawn("task_c")  # 开始spawn
    await registry.begin_spawn("task_c")  # 开始spawn
    
    print(f"C's spawning count: {registry._spawning.get('task_c', 0)}")  # 应该是3
    print(f"C's pending children: {registry.count_pending_for_parent('task_c')}")  # 应该是3
    
    # C1注册并完成
    await registry.register("task_c1", "sess_c1", "C1", mock_c, "task_c", 3)
    print(f"After C1 registered, spawning: {registry._spawning.get('task_c', 0)}")  # 应该是2
    print(f"C's pending children: {registry.count_pending_for_parent('task_c')}")  # 应该是3 (1+2)
    
    await registry.complete("task_c1", "C1 result")
    await asyncio.sleep(0.1)
    
    # C被wake时，检查pending children
    print(f"C's pending children when woken: {registry.count_pending_for_parent('task_c')}")  # 应该是2 (0+2)
    
    # C不应该汇报，因为还有2个正在spawn的子任务
    assert mock_c.status != "completed", "C should NOT complete yet!"
```

### 测试用例2：验证spawning计数正确性

```python
async def test_spawning_count():
    """测试spawning计数的正确性"""
    registry = SubagentRegistry()
    
    # 初始状态
    assert registry.count_pending_for_parent("task_a") == 0
    
    # 开始spawn 3个子任务
    await registry.begin_spawn("task_a")
    await registry.begin_spawn("task_a")
    await registry.begin_spawn("task_a")
    
    assert registry._spawning["task_a"] == 3
    assert registry.count_pending_for_parent("task_a") == 3
    
    # 注册第1个子任务
    await registry.register("task_1", "sess_1", "Child1", mock_a, "task_a", 1)
    assert registry._spawning["task_a"] == 2
    assert registry.count_pending_for_parent("task_a") == 3  # 1 (registered) + 2 (spawning)
    
    # 注册第2个子任务
    await registry.register("task_2", "sess_2", "Child2", mock_a, "task_a", 1)
    assert registry._spawning["task_a"] == 1
    assert registry.count_pending_for_parent("task_a") == 3  # 2 (registered) + 1 (spawning)
    
    # 注册第3个子任务
    await registry.register("task_3", "sess_3", "Child3", mock_a, "task_a", 1)
    assert "task_a" not in registry._spawning  # spawning已清空
    assert registry.count_pending_for_parent("task_a") == 3  # 3 (registered) + 0 (spawning)
```

### 测试用例3：并发场景测试

```python
async def test_concurrent_spawns():
    """测试并发spawn的正确性"""
    registry = SubagentRegistry()
    
    # 并发spawn多个子任务
    tasks = []
    for i in range(10):
        tasks.append(registry.begin_spawn("task_a"))
    
    await asyncio.gather(*tasks)
    
    assert registry._spawning["task_a"] == 10
    
    # 并发注册
    register_tasks = []
    for i in range(10):
        register_tasks.append(
            registry.register(f"task_{i}", f"sess_{i}", f"Child{i}", mock_a, "task_a", 1)
        )
    
    await asyncio.gather(*register_tasks)
    
    assert "task_a" not in registry._spawning
    assert registry.count_pending_for_parent("task_a") == 10
```

---

## 边界情况和注意事项

### 1. spawn失败的情况

如果spawn过程中发生异常，必须恢复spawning计数：

```python
try:
    await self.registry.begin_spawn(self.parent_task_id)
    # spawn操作
except Exception as e:
    await self.registry.cancel_spawn(self.parent_task_id)
    raise
```

### 2. 重复register的情况

如果同一个task_id被register两次，应该抛出错误或者忽略第二次。当前的实现在 `self._tasks[task_id] = task` 会覆盖，需要考虑是否需要检查。

### 3. spawning计数为负的情况

理论上不应该出现，但为了防御性编程，可以添加断言：

```python
async def register(self, ...):
    async with self._lock:
        # ...
        if parent_task_id and parent_task_id in self._spawning:
            assert self._spawning[parent_task_id] > 0, "Spawning count should not be negative"
            self._spawning[parent_task_id] -= 1
            if self._spawning[parent_task_id] == 0:
                del self._spawning[parent_task_id]
```

### 4. 父任务不存在的情况

`begin_spawn` 和 `count_pending_for_parent` 都可以处理parent_task_id不存在的情况（返回0）。

### 5. 线程安全

所有修改都使用了 `async with self._lock`，确保线程安全。

---

## 后续改进建议

### 1. 添加日志和监控

在关键操作处添加详细日志：
```python
async def begin_spawn(self, parent_task_id: str):
    async with self._lock:
        old_count = self._spawning.get(parent_task_id, 0)
        self._spawning[parent_task_id] = old_count + 1
        print(f"[Registry] Begin spawn for {parent_task_id}: {old_count} → {old_count+1}")
```

### 2. 添加指标收集

可以收集以下指标用于监控：
- 平均spawning时间
- spawning计数峰值
- spawn失败率

### 3. 添加超时机制

如果一个任务长时间处于spawning状态，可能意味着spawn失败：

```python
async def begin_spawn(self, parent_task_id: str, timeout: float = 30.0):
    async with self._lock:
        self._spawning[parent_task_id] = self._spawning.get(parent_task_id, 0) + 1
        # 启动超时检查
        asyncio.create_task(
            self._check_spawn_timeout(parent_task_id, timeout)
        )

async def _check_spawn_timeout(self, parent_task_id: str, timeout: float):
    await asyncio.sleep(timeout)
    async with self._lock:
        if parent_task_id in self._spawning:
            print(f"[Registry] ⚠️ Spawn timeout for {parent_task_id}")
            # 可以选择自动清理或者发出告警
```

---

## 参考资料

### 相关代码文件
- `src/registry.py` - SubagentRegistry实现
- `src/tools.py` - SpawnTool实现
- `src/agent_core.py` - Agent核心逻辑
- `tests/test_registration_race.py` - Bug复现测试

### 相关概念
- Async/await 并发模型
- 竞态条件（Race Condition）
- 信号量/计数器模式

---

## 总结

### Bug本质
时序竞态条件：子代理的spawn是异步的，但 `count_pending_for_parent` 只统计已注册的任务，导致正在注册中的任务被忽略。

### 修复核心
在registry中维护 `_spawning` 计数器，`count_pending_for_parent` 返回：已注册的 + 正在注册的。

### 修复影响
- 最小改动：只需修改registry和SpawnTool
- 向后兼容：`count_pending_for_parent` 接口不变
- 鲁棒性增强：正确处理异步spawn的时序问题

### 验证方法
运行 `test_registration_race.py` 和 `main.py`，确保不再出现"提前汇报"现象。

---

*记录日期：2026-04-06*
*最后更新：待实现后更新测试结果*
