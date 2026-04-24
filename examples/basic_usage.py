#!/usr/bin/env python3
import asyncio
import sys
import uuid

from engine.runtime.agent import Agent
from engine.config import ConfigLoader
from engine.providers.llm_provider import LLMProvider
from engine.runtime.agent_models import AgentState, Session
from engine.runtime.task_registry import AgentTaskRegistry

from engine.tools.custom.mock import MockTool


DEFAULT_PROMPT = """ 
    现在对你的构建子代理的能力进行测试，请严格根据以下要求执行：
    - 请构建三个子代理
    - 然后要求每一个子代理再构建两个子代理（对于你来说是孙代理）
    - 要求每个孙代理随机生成一个数字，由子代理汇总
    - 最后你来汇总子代理的结果，并返回给用户
    - 如果出现任何问题，你需要在最后反馈给用户
  """


async def run_agent_system(prompt: str) -> str:
    print("=" * 60)
    print("Agent System - Starting")
    print("=" * 60)

    print("[1/4] Loading configuration...")
    try:
        config = ConfigLoader.load_from_json()
        primary = config.provider_profiles[0]
        print(f"      Model: {primary['model']}")
        print(f"      Max Depth: {config.max_depth}")
    except ValueError as e:
        print(f"[Error] Configuration error: {e}")
        sys.exit(1)

    print("[2/4] Initializing LLM provider...")
    llm_provider = LLMProvider(
        api_key=primary["api_key"],
        base_url=primary["base_url"],
        model=primary["model"],
        config=config,
    )
    print(f"      Provider: {primary['model']}")

    print("[3/4] Creating agent task registry...")
    task_registry = AgentTaskRegistry()

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
        task_registry=task_registry,
        llm_provider=llm_provider,
        tools=[MockTool()],
    )
    print(f"      Agent: [{agent.label}|{agent.task_id}]")

    # Register root agent in AgentTaskRegistry
    await task_registry.register(
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
