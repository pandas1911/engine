"""Spawn tool for creating child agents."""

import asyncio
import uuid
from typing import Any, Callable, Dict, TYPE_CHECKING

from src.config import Config
from src.models import Session
from src.tools.base import Tool

if TYPE_CHECKING:
    from src.registry import SubagentRegistry


class SpawnTool(Tool):
    """Tool for spawning child agents asynchronously."""

    name = "spawn"
    description = "创建一个子代理来异步执行任务，任务执行完毕后子代理将自动唤醒主代理并汇报任务结果"
    parameters = {
        "type": "object",
        "properties": {
            "task": {"type": "string", "description": "要分配的任务"},
            "label": {"type": "string", "description": "子代理标签（可选）"},
        },
        "required": ["task"],
    }

    def __init__(
        self,
        agent_factory: Callable,
        registry: "SubagentRegistry",
        parent_task_id: str,
        parent_label: str = "Root",
    ):
        self.agent_factory = agent_factory
        self.registry = registry
        self.parent_task_id = parent_task_id
        self.parent_label = parent_label

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

        await self.registry.register(
            task_id=task_id,
            session_id=child_session.id,
            description=task_desc,
            parent_agent=context["parent_agent"],
            parent_task_id=self.parent_task_id,
            depth=child_session.depth,
        )

        asyncio.create_task(
            self._run_child_agent(child_session, task_desc, task_id, config)
        )

        return f"""━━━━ Spawned Task ━━━━
Task ID: {task_id}
Agent Label: {label}

Sub-agent is now executing in the background. Upon completion, you will be automatically re-activated and receive a full result report. You may proceed with other independent tasks or simply end your current turn."""

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
            agent = self.agent_factory(
                child_session, config, self.registry, self.parent_task_id, task_id
            )
            await self.registry.set_agent(task_id, agent)
            result = await agent.run(task_desc)
        except Exception as e:
            print(f"[Subagent|{task_id}] ✗ Failed: {e}")
            await self.registry.complete(task_id, f"[Error] {e}", error=True)
