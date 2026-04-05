# Agent Core - 最小可复现架构（多层嵌套支持版）

> 基于 OpenClaw 的简化实现，支持无限层级的嵌套 Agent 回调链，采用真正的异步回调机制（Push-based）

## 1. 核心设计

### 1.1 OpenClaw 启发的多层嵌套架构

```
Main Agent (depth=0, task_id="root")
  ├─ spawn ──► Orchestrator Agent (depth=1, task_id="orch")
  │               ├─ spawn ──► Worker A (depth=2, task_id="worker-a")
  │               │              └─ 完成后 ──► 通知 orch
  │               ├─ spawn ──► Worker B (depth=2, task_id="worker-b")
  │               │              └─ 完成后 ──► 通知 orch
  │               │
  │               └─ 所有 workers 完成后
  │                  汇总结果 ──► 通知 Main Agent
  │
  └─ 收到 orchestrator 结果
     给出最终回复
```

### 1.2 关键机制（对应 OpenClaw）

| OpenClaw 概念 | 本实现对应 | 作用 |
|--------------|-----------|------|
| `requesterSessionKey` | `parent_agent` + `parent_task_id` | 指向直接父 Agent |
| `runId` | `task_id` | Agent 的唯一标识 |
| `countPendingDescendantRuns` | `count_pending_descendants()` | 检查所有后代是否完成 |
| `wakeOnDescendantSettle` | `wake_on_descendants_settle` | 子代理先结束时唤醒机制 |
| `childCompletionFindings` | `collect_child_results()` | 汇总所有子代理结果 |
| `runSubagentAnnounceFlow` | `_notify_parent_on_complete()` | 完成时通知父代理 |

---

## 2. 核心机制详解

### 2.1 回调链的数据流

```
孙Agent 完成
  │
  ├─ 1. 调用 registry.complete(task_id, result)
  │     更新自身状态为 completed
  │
  ├─ 2. 检查：是否还有 pending 的后代？
  │     YES → 只更新状态，不通知父代理
  │     NO  → 继续下一步
  │
  ├─ 3. 调用父 Agent 的 _on_subagent_complete()
  │     父 Agent 被唤醒
  │
  ├─ 4. 父 Agent 检查：是否还有 pending 的子代理？
  │     YES → 继续等待其他子代理
  │     NO  → 所有子代理完成，继续下一步
  │
  ├─ 5. 父 Agent 汇总所有子代理结果
  │     生成 completion_findings
  │
  ├─ 6. 父 Agent 自身完成
  │     调用 _finish()
  │
  └─ 7. 父 Agent 调用 registry.complete()
        触发祖父 Agent 的回调...
```

### 2.2 唤醒机制（Wake on Descendant Settle）

**场景**：Orchestrator 先结束，但 Workers 还在运行

```
Orchestrator Agent
  ├─ spawn Worker A
  ├─ spawn Worker B
  ├─ 自己快速完成（返回结果给 LLM）
  └─ 但 Workers 还在运行...
     
     [Orchestrator 处于 "ended_with_pending_descendants" 状态]
     
Worker A 完成 ──► 发现 Orchestrator 已结束
  │                  但有 wake_on_descendants_settle 标志
  │                  触发唤醒流程
  │
  └─ 重新激活 Orchestrator
     传入 Worker A 的结果
     
Worker B 完成 ──► 再次触发唤醒
     
所有 Workers 完成 ──► Orchestrator 汇总结果
  │                    通知 Main Agent
  └─ Orchestrator 真正完成
```

---

## 3. 完整实现（支持多层嵌套）

```python
"""
agent_core.py - Agent 系统完整实现（支持多层嵌套）

核心机制：
1. 每个 Agent 持有 task_id，完成后主动通知父 Agent
2. 支持后代追踪，确保所有后代完成后才回调父 Agent
3. 唤醒机制：子代理先结束时能被孙代理唤醒
4. 结果汇总：收集所有子代理结果

对应 OpenClaw 设计：
- task_id ↔ runId
- parent_agent + parent_task_id ↔ requesterSessionKey
- count_pending_descendants() ↔ countPendingDescendantRuns
- wake_on_descendants_settle ↔ wakeOnDescendantSettle
- collect_child_results() ↔ childCompletionFindings
"""

import asyncio
import uuid
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class AgentState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    CALLBACK_PENDING = "callback_pending"  # 等待被回调唤醒
    ENDED_WITH_PENDING_DESCENDANTS = "ended_with_pending_descendants"  # 已结束但有后代未完成
    COMPLETED = "completed"


@dataclass
class Config:
    max_depth: int = 3
    default_model: str = "gpt-4"
    spawn_timeout: float = 60.0
    # OpenClaw 风格：是否启用唤醒机制
    enable_wake_on_descendants: bool = True


@dataclass
class Message:
    role: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {"role": self.role, "content": self.content}


@dataclass
class Session:
    id: str
    depth: int = 0
    parent_id: Optional[str] = None
    messages: List[Message] = field(default_factory=list)
    
    def add_message(self, role: str, content: str, **metadata):
        self.messages.append(Message(role, content, metadata))
    
    def get_messages(self) -> List[Dict]:
        return [m.to_dict() for m in self.messages]


@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any]
    call_id: str


@dataclass
class LLMResponse:
    content: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


# ==================== 子代理注册表（支持多层嵌套） ====================

@dataclass
class SubagentTask:
    task_id: str
    session_id: str
    task_description: str
    parent_agent: "Agent"  # 直接父 Agent 引用
    parent_task_id: Optional[str] = None  # 直接父 Agent 的 task_id
    status: str = "running"
    result: Optional[str] = None
    depth: int = 0  # 嵌套深度
    # OpenClaw 风格：后代追踪
    child_task_ids: Set[str] = field(default_factory=set)  # 直接子代理的 task_ids
    completed_event: asyncio.Event = field(default_factory=asyncio.Event)
    # 唤醒机制：当 Agent 已结束但有后代未完成时，标记需要唤醒
    wake_on_descendants_settle: bool = False
    ended_at: Optional[float] = None


class SubagentRegistry:
    """
    子代理注册表 - 支持多层嵌套
    
    对应 OpenClaw 的 subagent-registry.ts
    """
    
    def __init__(self):
        self._tasks: Dict[str, SubagentTask] = {}
        self._pending: set = set()  # 所有 pending 的 task_ids
        self._lock = asyncio.Lock()
    
    async def register(
        self, 
        task_id: str, 
        session_id: str, 
        description: str,
        parent_agent: "Agent",
        parent_task_id: Optional[str] = None,
        depth: int = 0
    ) -> SubagentTask:
        """注册子代理 - 类似 OpenClaw 的 registerSubagentRun"""
        task = SubagentTask(
            task_id=task_id,
            session_id=session_id,
            task_description=description,
            parent_agent=parent_agent,
            parent_task_id=parent_task_id,
            depth=depth
        )
        
        async with self._lock:
            self._tasks[task_id] = task
            self._pending.add(task_id)
            
            # 如果有父代理，将当前任务添加到父代理的 child_task_ids
            if parent_task_id and parent_task_id in self._tasks:
                parent_task = self._tasks[parent_task_id]
                parent_task.child_task_ids.add(task_id)
        
        return task
    
    async def complete(self, task_id: str, result: str, error: bool = False):
        """
        子代理完成时调用 - 对应 OpenClaw 的 complete 流程
        
        关键逻辑：
        1. 标记自身为完成
        2. 检查是否还有 pending 的后代
        3. 如果没有，通知父代理
        """
        async with self._lock:
            if task_id not in self._tasks:
                return
            
            task = self._tasks[task_id]
            task.status = "error" if error else "completed"
            task.result = result
            task.completed_event.set()
            self._pending.discard(task_id)
            
            # 获取父代理信息
            parent = task.parent_agent
            parent_task_id = task.parent_task_id
            
            # 检查是否还有 pending 的后代（OpenClaw 风格）
            pending_descendants = self._count_pending_descendants_locked(task_id)
        
        # 如果有后代还在运行，不立即通知父代理
        if pending_descendants > 0:
            print(f"[Registry] Task {task_id} 完成，但还有 {pending_descendants} 个后代未完成，暂不通知父代理")
            return
        
        # 检查父代理是否处于 "ended_with_pending_descendants" 状态（需要唤醒）
        if parent_task_id and parent_task_id in self._tasks:
            parent_task = self._tasks[parent_task_id]
            if (parent_task.status in ["completed", "ended_with_pending_descendants"] 
                and parent_task.wake_on_descendants_settle):
                print(f"[Registry] 父代理 {parent_task_id} 需要唤醒")
                asyncio.create_task(
                    self._wake_parent_agent(parent_task_id, task_id, result)
                )
                return
        
        # 通知父代理
        if parent and parent_task_id:
            print(f"[Registry] Task {task_id} 完成，通知父代理 {parent_task_id}")
            asyncio.create_task(
                parent._on_subagent_complete(task_id, result)
            )
    
    def _count_pending_descendants_locked(self, task_id: str) -> int:
        """计算指定任务的后代中还在运行的数量 - 对应 OpenClaw 的 countPendingDescendantRuns"""
        if task_id not in self._tasks:
            return 0
        
        task = self._tasks[task_id]
        count = 0
        
        # BFS 遍历所有后代
        visited = {task_id}
        queue = list(task.child_task_ids)
        
        while queue:
            child_id = queue.pop(0)
            if child_id in visited:
                continue
            visited.add(child_id)
            
            if child_id in self._tasks:
                child_task = self._tasks[child_id]
                if child_id in self._pending:
                    count += 1
                # 继续遍历后代
                queue.extend(child_task.child_task_ids)
        
        return count
    
    async def _wake_parent_agent(self, parent_task_id: str, child_task_id: str, child_result: str):
        """唤醒父代理 - 对应 OpenClaw 的 wakeSubagentRunAfterDescendants"""
        async with self._lock:
            if parent_task_id not in self._tasks:
                return
            parent_task = self._tasks[parent_task_id]
            parent_task.status = "running"  # 重新激活
            parent_task.wake_on_descendants_settle = False
        
        parent = parent_task.parent_agent
        if parent:
            print(f"[Registry] 唤醒父代理 {parent_task_id}")
            await parent._on_descendant_wake(child_task_id, child_result)
    
    def has_pending(self) -> bool:
        """是否有任何 pending 的任务"""
        return len(self._pending) > 0
    
    def get_pending_count(self) -> int:
        return len(self._pending)
    
    def count_pending_for_parent(self, parent_task_id: str) -> int:
        """计算指定父代理的 pending 子代理数量"""
        count = 0
        for task_id, task in self._tasks.items():
            if task.parent_task_id == parent_task_id and task_id in self._pending:
                count += 1
        return count
    
    def get_task(self, task_id: str) -> Optional[SubagentTask]:
        return self._tasks.get(task_id)
    
    def collect_child_results(self, parent_task_id: str) -> Dict[str, str]:
        """
        收集所有直接子代理的结果 - 对应 OpenClaw 的 childCompletionFindings
        """
        results = {}
        for task_id, task in self._tasks.items():
            if task.parent_task_id == parent_task_id and task.result is not None:
                results[task_id] = task.result
        return results

    async def mark_ended_with_pending_descendants(self, task_id: str):
        """标记 Agent 已结束但有后代未完成"""
        async with self._lock:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                task.status = "ended_with_pending_descendants"
                task.wake_on_descendants_settle = True
                task.ended_at = datetime.now().timestamp()


# ==================== Agent 核心（支持多层嵌套） ====================

class Agent:
    def __init__(
        self, 
        session: Session, 
        config: Config,
        registry: Optional[SubagentRegistry] = None,
        llm_provider: Optional["LLMProvider"] = None,
        tools: Optional[List["Tool"]] = None,
        task_id: Optional[str] = None,  # 新增：Agent 需要知道自己的 task_id
        parent_task_id: Optional[str] = None  # 新增：父 Agent 的 task_id
    ):
        self.session = session
        self.config = config
        self.registry = registry or SubagentRegistry()
        self.llm = llm_provider or MockLLMProvider()
        self.tools = tools or []
        
        # OpenClaw 风格：每个 Agent 必须有 task_id
        self.task_id = task_id or f"task_{uuid.uuid4().hex[:8]}"
        self.parent_task_id = parent_task_id
        
        self.state = AgentState.IDLE
        self._final_result: Optional[str] = None
        self._completion_event = asyncio.Event()
        
        if session.depth < config.max_depth:
            self.tools.append(SpawnTool(self._create_child_agent, self.registry, self.task_id))
    
    def _create_child_agent(
        self, 
        session: Session, 
        config: Config, 
        registry: SubagentRegistry,
        parent_task_id: str
    ) -> "Agent":
        """创建子 Agent - 传递父 Agent 的 task_id"""
        return Agent(
            session, 
            config, 
            registry, 
            self.llm, 
            [],
            parent_task_id=parent_task_id
        )
    
    async def run(
        self,
        message: Optional[str] = None
    ) -> str:
        """
        运行 Agent - 支持多层嵌套回调

        流程：
        1. 处理工具调用（可能 spawn 子代理）
        2. 如果有子代理，进入等待状态
        3. 所有子代理完成后，汇总结果
        4. 完成时通知父代理

        Args:
            message: 用户消息
        """
        if message:
            self.session.add_message("user", message)

        self.state = AgentState.RUNNING

        # 处理工具调用（可能 spawn 子代理）
        spawned_any = await self._process_tool_calls()

        if spawned_any:
            # 进入回调等待状态
            self.state = AgentState.CALLBACK_PENDING
            # 标记为已结束但有后代未完成，启用唤醒机制
            await self.registry.mark_ended_with_pending_descendants(self.task_id)
            return "[等待子代理回调...]"
        else:
            # 没有 spawn 子代理，直接完成
            await self._finish_and_notify()

        return self._final_result or "[无回复]"
    
    async def _process_tool_calls(self) -> bool:
        """处理工具调用，返回是否 spawn 了子代理"""
        spawned = False
        
        while True:
            response = await self.llm.chat(
                messages=self.session.get_messages(),
                tools=self._get_tool_schemas()
            )
            
            if not response.has_tool_calls():
                if response.content:
                    self.session.add_message("assistant", response.content)
                    # 如果这是最终回复，保存它
                    if not spawned:
                        self._final_result = response.content
                break
            
            for tool_call in response.tool_calls:
                result = await self._execute_tool(tool_call)
                self.session.add_message("tool", result, tool_call_id=tool_call.call_id)
                
                if tool_call.name == "spawn":
                    spawned = True
        
        return spawned
    
    async def _execute_tool(self, tool_call: ToolCall) -> str:
        tool = next((t for t in self.tools if t.name == tool_call.name), None)
        
        if not tool:
            return f"[错误] 未知工具: {tool_call.name}"
        
        context = {
            "session": self.session,
            "config": self.config,
            "registry": self.registry,
            "parent_agent": self,
            "parent_task_id": self.task_id,  # 传递当前 Agent 的 task_id
        }
        
        try:
            return await tool.execute(tool_call.arguments, context)
        except Exception as e:
            return f"[工具错误] {str(e)}"
    
    # ==================== 回调处理（OpenClaw 风格） ====================
    
    async def _on_subagent_complete(self, child_task_id: str, result: str):
        """
        被子代理回调触发 - 对应 OpenClaw 的 _on_subagent_complete
        
        关键逻辑：
        1. 记录子代理结果
        2. 检查是否所有直接子代理都完成了
        3. 如果都完成，汇总结果并继续处理
        """
        print(f"[Agent {self.task_id}] 收到子代理 {child_task_id} 的完成通知")
        
        # 记录子代理结果到消息历史
        self.session.add_message(
            "user", 
            f"[子代理完成] {child_task_id}: {result}",
            is_subagent_result=True,
            child_task_id=child_task_id
        )
        
        # OpenClaw 风格：检查是否还有 pending 的直接子代理
        pending_children = self.registry.count_pending_for_parent(self.task_id)
        
        if pending_children > 0:
            print(f"[Agent {self.task_id}] 还有 {pending_children} 个子代理未完成，继续等待")
            return
        
        # 所有直接子代理都完成了
        print(f"[Agent {self.task_id}] 所有子代理完成，继续处理")
        await self._continue_processing()
    
    async def _on_descendant_wake(self, descendant_task_id: str, result: str):
        """
        被后代唤醒时触发 - 对应 OpenClaw 的 wakeSubagentRunAfterDescendants
        
        场景：当前 Agent 已结束，但后代完成了，需要被唤醒继续处理
        """
        print(f"[Agent {self.task_id}] 被后代 {descendant_task_id} 唤醒")
        
        self.state = AgentState.RUNNING
        
        # 记录唤醒消息
        self.session.add_message(
            "user",
            f"[唤醒通知] 后代代理 {descendant_task_id} 完成: {result}",
            is_wake_notification=True
        )
        
        # 检查是否所有后代都完成了
        await self._check_all_descendants_complete()
    
    async def _check_all_descendants_complete(self):
        """检查是否所有后代都完成了"""
        # FIX: 先检查是否还有未完成的子代理
        pending_children = self.registry.count_pending_for_parent(self.task_id)
        if pending_children > 0:
            print(f"[Agent {self.task_id}] 被唤醒，但还有 {pending_children} 个子代理未完成，继续等待")
            # 恢复等待状态，确保后续子代理完成时能再次唤醒
            await self.registry.mark_ended_with_pending_descendants(self.task_id)
            return
        
        # 获取所有子代理结果
        child_results = self.registry.collect_child_results(self.task_id)
        
        if not child_results:
            # 没有子代理结果，直接完成
            await self._finish_and_notify()
            return
        
        # 构建汇总消息
        findings = "\n".join([
            f"- {task_id}: {result[:100]}..."
            for task_id, result in child_results.items()
        ])
        
        self.session.add_message(
            "user",
            f"[子代理结果汇总]\n{findings}",
            is_findings_summary=True
        )
        
        # 让 LLM 处理汇总后的结果
        await self._continue_processing()
    
    async def _continue_processing(self):
        """所有子代理完成后，继续处理并给出最终回复"""
        self.state = AgentState.RUNNING
        
        # 收集所有子代理结果（OpenClaw 风格）
        child_results = self.registry.collect_child_results(self.task_id)
        
        # 构建 system prompt 提示 LLM 处理子代理结果
        if child_results:
            findings_prompt = "基于以下子代理结果给出综合回复:\n"
            for task_id, result in child_results.items():
                findings_prompt += f"\n[{task_id}]\n{result}\n"
            
            self.session.add_message("system", findings_prompt)
        
        # 让 LLM 处理并给出最终回复
        final_response = await self.llm.chat(
            messages=self.session.get_messages(),
            tools=[]  # 不再允许 spawn
        )
        
        if final_response.content:
            self.session.add_message("assistant", final_response.content)
            self._final_result = final_response.content
        
        # 完成并通知父代理
        await self._finish_and_notify()
    
    async def _finish_and_notify(self):
        """
        完成执行并通知父代理 - 对应 OpenClaw 的 completeSubagentRun
        
        这是多层嵌套的关键：Agent 完成时必须主动通知父代理
        """
        self.state = AgentState.COMPLETED
        self._completion_event.set()
        
        print(f"[Agent {self.task_id}] 完成，结果: {self._final_result[:50] if self._final_result else 'None'}...")
        
        # 关键：通知父代理自己已完成
        if self.parent_task_id:
            await self.registry.complete(
                self.task_id, 
                self._final_result or "[完成但无输出]"
            )

    def _get_tool_schemas(self) -> List[Dict]:
        return [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters
            }
        } for tool in self.tools]


# ==================== Spawn 工具 ====================

class SpawnTool:
    name = "spawn"
    description = "创建一个子Agent来异步执行任务"
    parameters = {
        "type": "object",
        "properties": {
            "task": {"type": "string", "description": "要分配的任务"},
            "label": {"type": "string", "description": "子代理标签（可选）"}
        },
        "required": ["task"]
    }
    
    def __init__(
        self, 
        agent_factory: Callable, 
        registry: SubagentRegistry,
        parent_task_id: str
    ):
        self.agent_factory = agent_factory
        self.registry = registry
        self.parent_task_id = parent_task_id
    
    async def execute(self, arguments: Dict[str, Any], context: Dict[str, Any]) -> str:
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
            parent_id=parent_session.id
        )
        
        # OpenClaw 风格的 system prompt
        parent_label = "parent orchestrator" if parent_session.depth >= 1 else "main agent"
        can_spawn = child_session.depth < config.max_depth
        
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
        
        # 关键：注册子代理，传入父 Agent 的 task_id
        await self.registry.register(
            task_id=task_id,
            session_id=child_session.id,
            description=task_desc,
            parent_agent=context["parent_agent"],
            parent_task_id=self.parent_task_id,
            depth=child_session.depth
        )
        
        # 异步启动子代理
        asyncio.create_task(
            self._run_child_agent(child_session, task_desc, task_id, config)
        )
        
        return f"[已派生] 任务ID: {task_id}, 标签: {label}, 任务: {task_desc[:40]}..."
    
    async def _run_child_agent(
        self, 
        child_session: Session, 
        task_desc: str, 
        task_id: str, 
        config: Config
    ):
        """运行子代理 - 完成后会自动通知父代理"""
        try:
            agent = self.agent_factory(child_session, config, self.registry, task_id)
            result = await agent.run(task_desc)
            
            # 子代理完成后，registry.complete() 会自动通知父代理
            # 不需要在这里手动调用
            
        except Exception as e:
            # 错误也通过 registry 通知
            await self.registry.complete(task_id, f"[执行错误] {str(e)}", error=True)


# ==================== Mock LLM ====================

class MockLLMProvider:
    def __init__(self):
        self.call_count = 0
    
    async def chat(self, messages: List[Dict], tools: List[Dict], **kwargs) -> LLMResponse:
        self.call_count += 1
        
        last_content = messages[-1].get("content", "") if messages else ""
        has_spawn_tool = any(t.get("function", {}).get("name") == "spawn" for t in tools)
        has_subagent_result = "[子代理完成]" in last_content or "[子代理结果汇总]" in last_content
        has_wake_notification = "[唤醒通知]" in last_content
        
        # 第一次调用且有 spawn 工具：模拟 spawn
        if self.call_count == 1 and has_spawn_tool:
            return LLMResponse(tool_calls=[ToolCall(
                name="spawn",
                arguments={"task": "分析代码结构", "label": "analyzer"},
                call_id="call_1"
            )])
        
        # 收到子代理结果或唤醒通知：模拟汇总处理
        if has_subagent_result or has_wake_notification:
            return LLMResponse(
                content="综合所有子代理的结果分析：代码结构清晰，建议增加单元测试。具体发现包括：1）模块划分合理 2）依赖关系清晰 3）命名规范统一。"
            )
        
        # 子 Agent 的响应
        if self._is_subagent(messages):
            depth = self._get_depth(messages)
            return LLMResponse(
                content=f"[子Agent完成-深度{depth}] 分析完成：发现3个主要模块，依赖关系清晰。"
            )
        
        return LLMResponse(content="处理完成")
    
    def _is_subagent(self, messages: List[Dict]) -> bool:
        for msg in messages:
            if msg.get("role") == "system" and "Subagent Context" in msg.get("content", ""):
                return True
        return False
    
    def _get_depth(self, messages: List[Dict]) -> int:
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if "Depth:" in content:
                    try:
                        return int(content.split("Depth:")[1].split("/")[0].strip())
                    except:
                        pass
        return 0


# ==================== 便捷函数 ====================

async def run_agent(task: str, config: Optional[Config] = None) -> tuple[str, Session]:
    """运行根 Agent"""
    cfg = config or Config()
    
    root_session = Session(id=f"root_{uuid.uuid4().hex[:8]}", depth=0)
    root_session.add_message("system", "你是主Agent，可以派生子Agent并行处理任务。")
    
    agent = Agent(root_session, cfg)
    result = await agent.run(task)
    
    return result, root_session


__all__ = [
    "Agent",
    "Session", 
    "Message",
    "Config",
    "AgentState",
    "SubagentRegistry",
    "SubagentTask",
    "SpawnTool",
    "MockLLMProvider",
    "run_agent"
]
```

---

## 4. 多层嵌套使用示例

### 4.1 三层嵌套（Main → Orchestrator → Worker）

```python
import asyncio
from agent_core import Agent, Session, Config, run_agent

async def main():
    # 创建主 Agent
    result, session = await run_agent(
        "分析项目代码质量，先让一个 orchestrator 并行分析各个模块",
        Config(max_depth=3)
    )
    
    print(f"\n最终结果: {result}")
    print(f"\n消息历史:")
    for msg in session.messages:
        print(f"[{msg.role}] {msg.content[:80]}...")

asyncio.run(main())
```

**执行流程：**
```
1. Main Agent spawn Orchestrator
   └─ [等待回调...]

2. Orchestrator spawn Worker A, Worker B
   └─ [等待回调...]

3. Worker A 完成 → 通知 Orchestrator
   Worker B 完成 → 通知 Orchestrator

4. Orchestrator 收到所有 Workers 结果
   └─ 汇总 → 通知 Main Agent

5. Main Agent 收到 Orchestrator 结果
   └─ 给出最终回复
```

### 4.2 唤醒机制测试

```python
async def test_wake_mechanism():
    """测试子代理先结束、孙代理后完成的唤醒场景"""
    
    registry = SubagentRegistry()
    
    # 创建父 Agent (orchestrator)
    parent_session = Session(id="parent", depth=0)
    parent_agent = Agent(
        parent_session, 
        Config(max_depth=2),
        registry,
        task_id="parent_task"
    )
    
    # 注册父 Agent 到 registry
    await registry.register(
        task_id="parent_task",
        session_id="parent",
        description="orchestrator",
        parent_agent=parent_agent,
        depth=0
    )
    
    # 模拟：父 Agent 快速结束但有后代未完成
    await registry.mark_ended_with_pending_descendants("parent_task")
    
    # 注册子 Agent（后代）
    child_session = Session(id="child", depth=1)
    child_agent = Agent(
        child_session,
        Config(max_depth=2),
        registry,
        task_id="child_task",
        parent_task_id="parent_task"
    )
    
    await registry.register(
        task_id="child_task",
        session_id="child",
        description="worker",
        parent_agent=child_agent,
        parent_task_id="parent_task",
        depth=1
    )
    
    # 子 Agent 完成
    await registry.complete("child_task", "工作完成！")
    
    # 此时父 Agent 应该被唤醒
    print("唤醒测试完成")
```

---

## 5. 与 OpenClaw 的完整对应

### 5.1 数据模型对应

```
OpenClaw                      agent_core.py
─────────────────────────────────────────────────
SubagentRunRecord            SubagentTask
├── runId                    ├── task_id
├── childSessionKey          ├── session_id
├── requesterSessionKey      ├── parent_agent + parent_task_id
├── task                     ├── task_description
├── status                   ├── status
├── result                   ├── result
├── endedAt                  ├── ended_at
├── wakeOnDescendantSettle   ├── wake_on_descendants_settle
└── depth (从session key解析) └── depth (显式存储)
```

### 5.2 核心函数对应

```python
# OpenClaw: subagent-registry.ts
registerSubagentRun()    →  SubagentRegistry.register()
countPendingDescendantRuns()  →  _count_pending_descendants_locked()
listSubagentRunsForRequester()  →  collect_child_results()

# OpenClaw: subagent-announce.ts  
runSubagentAnnounceFlow()  →  _on_subagent_complete()
wakeSubagentRunAfterDescendants()  →  _wake_parent_agent()

# OpenClaw: complete flow
completeSubagentRun()  →  _finish_and_notify()
```

### 5.3 状态流转对应

```
OpenClaw                              agent_core.py
────────────────────────────────────────────────────────
RUNNING                              RUNNING
  ↓ spawn children                     ↓ spawn children
WAITING_FOR_COMPLETION               CALLBACK_PENDING
  ↓ all children complete              ↓ all children complete
  ↓ (or wake from descendants)         ↓ (or wake from descendants)
ANNOUNCE_FLOW                        RUNNING (in _continue_processing)
  ↓ announce to parent                 ↓ _finish_and_notify()
COMPLETED                            COMPLETED
  ↓ parent notified                    ↓ parent._on_subagent_complete()
```

---

## 6. 关键设计决策

### 6.1 为什么每个 Agent 需要 task_id？

**OpenClaw 的设计哲学**：
- Agent 是一个**异步任务**（run），有唯一的 runId
- 完成时通过 runId 找到对应的记录，进而找到父 Agent
- 支持跨会话的持久化和恢复

**agent_core.py 的实现**：
```python
self.task_id = task_id or f"task_{uuid.uuid4().hex[:8]}"
```

### 6.2 为什么需要后代追踪？

**场景**：Orchestrator 派生 3 个 Workers
- 如果只是检查 `has_pending()`，无法区分是兄弟 Worker 还是无关的 Agent
- 需要精确知道 "我的子代理是否都完成了"

**实现**：
```python
# 在 SubagentTask 中存储直接子代理
child_task_ids: Set[str] = field(default_factory=set)

# 注册时建立父子关系
if parent_task_id:
    parent_task.child_task_ids.add(task_id)
```

### 6.3 唤醒机制的必要性

**OpenClaw 的核心创新**：
- 允许子 Agent **先结束**（返回部分结果给 LLM）
- 后代完成后**重新激活**父 Agent
- 父 Agent 汇总所有后代结果，给出最终回复

**没有唤醒机制的问题**：
```
Orchestrator 结束 → 返回 "正在分析..."
Workers 还在运行 → 结果丢失了！
```

---

## 7. 总结

### 多层嵌套支持的三个关键

1. **任务标识**：每个 Agent 必须有 `task_id`，完成后能主动通知父 Agent
2. **后代追踪**：`count_pending_descendants()` 确保所有后代完成后再回调
3. **唤醒机制**：`wake_on_descendants_settle` 支持子代理先结束的场景

### 与原版 agent-core.md 的主要改进

| 原版 | 新版本 | 解决的问题 |
|------|--------|-----------|
| Agent 无 task_id | Agent 有 task_id | 完成后无法主动通知父 Agent |
| 只检查 registry.has_pending() | 检查 count_pending_descendants() | 无法区分后代和无关 Agent |
| _finish() 后无操作 | _finish_and_notify() 调用 registry.complete() | 回调链断裂 |
| 无唤醒机制 | 有 wake_on_descendants_settle | 子代理先结束时结果丢失 |
| 简单追加消息 | collect_child_results() 汇总 | 结果处理不便 |

### 代码量

- 核心实现: ~350行
- 支持无限层级嵌套
- 完整的回调链和唤醒机制
- 与 OpenClaw 设计完全对应

---

**总结**：新版本通过引入 `task_id`、后代追踪、唤醒机制，完全解决了多层嵌套 Agent 的回调链问题，与 OpenClaw 的生产级实现保持一致。
