#!/usr/bin/env python3
"""Max depth enforcement integration test.

Tests that the spawn tool is removed at max_depth and that attempting to spawn
at max_depth returns an appropriate error message.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent_core import Agent
from src.config import get_config
from src.llm_provider import MiniMaxProvider
from src.models import Session
from src.registry import SubagentRegistry
from src.tools import SpawnTool
from src.config import Config


def print_section_header(title: str):
    print(f"\n--- {title} ---")


async def test_max_depth_enforcement():
    print("\n" + "=" * 60)
    print("MAX DEPTH ENFORCEMENT TEST")
    print("=" * 60)

    config = Config(
        openai_api_key="test-key",
        openai_base_url="https://api.test.com",
        openai_model="test-model",
        max_depth=2,
    )
    print(f"\n✓ Configured max_depth={config.max_depth}")

    registry = SubagentRegistry()
    try:
        llm_provider = MiniMaxProvider(get_config())
        print("✓ Using real MiniMax API")
    except Exception as e:
        print(f"⚠ Could not initialize MiniMax API: {e}")
        print("✗ Test requires real MiniMax API - ABORTING")
        sys.exit(1)

    print_section_header("Test 1: Depth 0 Agent")
    session_depth0 = Session(id="test_depth0", depth=0)
    agent_depth0 = Agent(
        session=session_depth0,
        config=config,
        registry=registry,
        llm_provider=llm_provider,
        tools=[],
        task_id="task_depth0",
    )

    has_spawn_depth0 = any(tool.name == "spawn" for tool in agent_depth0.tools)
    if has_spawn_depth0:
        print("✓ Spawn tool available at depth=0")
    else:
        print("✗ Spawn tool NOT available at depth=0 (EXPECTED to be available)")
        sys.exit(1)

    print_section_header("Test 2: Depth 1 Agent")
    session_depth1 = Session(id="test_depth1", depth=1, parent_id="test_depth0")
    agent_depth1 = Agent(
        session=session_depth1,
        config=config,
        registry=registry,
        llm_provider=llm_provider,
        tools=[],
        task_id="task_depth1",
        parent_task_id="task_depth0",
    )

    has_spawn_depth1 = any(tool.name == "spawn" for tool in agent_depth1.tools)
    if has_spawn_depth1:
        print("✓ Spawn tool available at depth=1")
    else:
        print("✗ Spawn tool NOT available at depth=1 (EXPECTED to be available)")
        sys.exit(1)

    print_section_header("Test 3: Depth 2 Agent (at max_depth)")
    session_depth2 = Session(id="test_depth2", depth=2, parent_id="test_depth1")
    agent_depth2 = Agent(
        session=session_depth2,
        config=config,
        registry=registry,
        llm_provider=llm_provider,
        tools=[],
        task_id="task_depth2",
        parent_task_id="task_depth1",
    )

    has_spawn_depth2 = any(tool.name == "spawn" for tool in agent_depth2.tools)
    if not has_spawn_depth2:
        print("✓ Spawn tool removed at depth=2 (max_depth)")
    else:
        print("✗ Spawn tool STILL available at depth=2 (should be removed)")
        sys.exit(1)

    print_section_header("Test 4: Spawn Attempt at Max Depth")

    def mock_agent_factory(session, config, registry, task_id):
        return Agent(session, config, registry, llm_provider, [], task_id=task_id)

    spawn_tool = SpawnTool(
        agent_factory=mock_agent_factory,
        registry=registry,
        parent_task_id="task_depth2",
    )

    context = {
        "session": session_depth2,
        "config": config,
        "registry": registry,
        "parent_agent": agent_depth2,
    }

    spawn_result = await spawn_tool.execute(
        arguments={"task": "test task", "label": "test_child"}, context=context
    )

    if "错误" in spawn_result and "最大深度限制" in spawn_result:
        print(f"✓ Spawn attempt at max_depth returned error: {spawn_result}")
    else:
        print(f"✗ Unexpected spawn result at max_depth: {spawn_result}")
        sys.exit(1)

    print_section_header("Test 5: Boundary Verification")

    session_depth1_alt = Session(id="test_depth1_alt", depth=1)
    spawn_tool_alt = SpawnTool(
        agent_factory=mock_agent_factory,
        registry=registry,
        parent_task_id="task_depth1_alt",
    )
    context_alt = {
        "session": session_depth1_alt,
        "config": config,
        "registry": registry,
        "parent_agent": None,
    }

    spawn_result_alt = await spawn_tool_alt.execute(
        arguments={"task": "test task", "label": "test_child"}, context=context_alt
    )

    if "已派生" in spawn_result_alt:
        print(
            f"✓ Spawn allowed at depth=1 (below max_depth): {spawn_result_alt[:60]}..."
        )
    else:
        print(f"✗ Spawn failed at depth=1: {spawn_result_alt}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("✓ Max depth test passed")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    try:
        asyncio.run(test_max_depth_enforcement())
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
