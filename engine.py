import uuid
from typing import List, Optional

from src.agent_core import Agent
from src.config import Config, ConfigLoader
from src.llm_provider import LLMProvider
from src.models import AgentResult, AgentState, Session
from src.registry import SubagentRegistry
from src.tools.base import Tool

DEFAULT_SYSTEM_PROMPT = (
    "你是主Agent，请尽可能构建子代理来并行处理任务。"
    "这不仅可以提升处理效率，还能减少无关信息对上下文的污染，帮助你做出更清晰的决策。"
)


async def delegate(
    task_description: str,
    system_prompt: Optional[str] = None,
    tools: Optional[List[Tool]] = None,
    config: Optional[Config] = None,
) -> AgentResult:
    session = Session(id=f"root_{uuid.uuid4().hex[:8]}", depth=0)
    session.add_message("system", system_prompt or DEFAULT_SYSTEM_PROMPT)

    try:
        if config is None:
            config = ConfigLoader.load_from_env()

        llm_provider = LLMProvider(config)
        registry = SubagentRegistry()

        agent = Agent(
            session=session,
            config=config,
            registry=registry,
            llm_provider=llm_provider,
            tools=tools,
        )

        await registry.register(
            task_id=agent.task_id,
            session_id=session.id,
            description="root task",
            parent_agent=None,
            agent=agent,
            depth=0,
        )

        await agent.run(task_description)

        if agent.state != AgentState.COMPLETED:
            await agent._completion_event.wait()

        return AgentResult(
            content=agent._final_result or "",
            session=session,
            success=True,
        )
    except Exception as e:
        return AgentResult(
            content="",
            session=session,
            success=False,
            error=str(e),
        )
