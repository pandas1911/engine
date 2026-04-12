#!/usr/bin/env python3
import asyncio
import sys
import uuid

from src.agent_core import Agent
from src.config import ConfigLoader
from src.llm_provider import LLMProvider
from src.models import AgentState, Session
from src.registry import SubagentRegistry

from src.tools.custom.mock import MockTool


DEFAULT_PROMPT = """ 
    现在对你的构建子代理的能力进行测试，请严格根据以下要求执行：
    - 有三个任务：1.写一首赞美春天的简短的诗；2.简单讲解一下金融中做多和多空的概念；3.查询一下上海的天气，以及适合的穿搭
    - 你可以构建子代理完成部分任务然后自己完成剩余任务，或者全部交由子代理完成，你进行汇总
    - 将最终结果进行汇报
  """


async def run_agent_system(prompt: str) -> str:
    print("=" * 60)
    print("Agent System - Starting")
    print("=" * 60)

    print("[1/4] Loading configuration...")
    try:
        config = ConfigLoader.load_from_env()
        print(f"      Model: {config.model}")
        print(f"      Max Depth: {config.max_depth}")
    except ValueError as e:
        print(f"[Error] Configuration error: {e}")
        sys.exit(1)

    print("[2/4] Initializing LLM provider...")
    llm_provider = LLMProvider(config)
    print(f"      Provider: {config.model}")

    print("[3/4] Creating subagent registry...")
    registry = SubagentRegistry()

    print("[4/4] Creating agent session...")
    session_id = f"root_{uuid.uuid4().hex[:8]}"
    session = Session(id=session_id, depth=0)
    session.add_message("system", "你是主Agent，尽可能构建子代理去处理任务，这样既可以并行处理任务，也可以减少不必要的信息污染上下文，干扰你的决策。")
    print(f"      Session ID: {session_id}")

    print("=" * 60)
    print("Processing Request")
    print("=" * 60)
    print(f"Prompt: {prompt}")
    print("-" * 60)

    agent = Agent(
        session=session,
        config=config,
        registry=registry,
        llm_provider=llm_provider,
        tools=[MockTool()],
    )
    print(f"      Agent: [{agent.label}|{agent.task_id}]")

    # Register root agent in SubagentRegistry
    await registry.register(
        task_id=agent.task_id,
        session_id=session.id,
        description="root task",
        parent_agent=None,
        parent_task_id=None,
        agent=agent,
        depth=0,
    )

    try:
        await agent.run(prompt)

        if agent.state != AgentState.COMPLETED:
            print("[System] Waiting for subagents to complete...")
            _ = await agent._completion_event.wait()

        final_result = agent._final_result or "[Agent 执行完毕但未生成回复]"

        print("=" * 60)
        print("Final Result")
        print("=" * 60)
        print(final_result)
        print("=" * 60)

        return final_result

    except Exception as e:
        print(f"[Error] Agent execution failed: {type(e).__name__}: {e}")
        raise


def main():
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
    else:
        prompt = DEFAULT_PROMPT
        print(f"No prompt provided, using default prompt.")
        print()

    try:
        _ = asyncio.run(run_agent_system(prompt))
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n[Interrupted] User cancelled the operation.")
        sys.exit(130)
    except Exception as e:
        print(f"\n[Fatal Error] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
