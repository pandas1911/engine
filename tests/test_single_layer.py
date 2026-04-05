#!/usr/bin/env python3
"""Single-layer integration test with real MiniMax API."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.utils import create_test_agent
from src.models import AgentState


async def test_single_layer():
    print("=" * 60)
    print("Single-Layer Integration Test")
    print("=" * 60)

    print("\n[1/4] Creating test agent...")
    agent = create_test_agent(task_id="test_single_layer_001", task="Say hello")
    print(f"  ✓ Agent created with task_id: {agent.task_id}")

    print("\n[2/4] Running agent with real MiniMax API...")
    try:
        result = await agent.run("Say hello")
        print(f"  ✓ Agent execution completed")
    except Exception as e:
        print(f"  ✗ Agent execution failed: {e}")
        return False

    print("\n[3/4] Verifying response...")
    if result is None:
        print("  ✗ Result is None")
        return False
    if not isinstance(result, str):
        print(f"  ✗ Result is not a string: {type(result)}")
        return False
    if len(result) == 0:
        print("  ✗ Result is empty")
        return False
    print(f"  ✓ Response received: {result[:100]}{'...' if len(result) > 100 else ''}")

    print("\n[4/4] Verifying agent state...")
    if agent.state != AgentState.COMPLETED:
        print(f"  ✗ Agent state is {agent.state}, expected COMPLETED")
        return False
    print(f"  ✓ Agent state: {agent.state.value}")

    print("\n" + "=" * 60)
    print("✓ Single-layer test passed")
    print("=" * 60)
    return True


if __name__ == "__main__":
    try:
        success = asyncio.run(test_single_layer())
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
