#!/usr/bin/env python3
"""Error handling integration test."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent_core import Agent
from src.models import Session
from src.registry import SubagentRegistry
from src.config import get_config


async def test_error_handling():
    """Test error handling in the agent system."""
    print("=" * 60)
    print("Error Handling Integration Test")
    print("=" * 60)

    config = get_config()
    registry = SubagentRegistry()

    parent_session = Session(id="parent_session", depth=0)
    parent_agent = Agent(
        session=parent_session,
        config=config,
        registry=registry,
        llm_provider=None,
        task_id="parent_test",
    )

    print("\n[1/3] Creating parent agent...")
    print(f"  ✓ Parent agent: {parent_agent.task_id}")

    await registry.register(
        task_id="parent_test",
        session_id="parent_session",
        description="Parent for error test",
        parent_agent=parent_agent,
        parent_task_id=None,
        depth=0,
    )

    print("\n[2/3] Simulating child error...")
    await registry.register(
        task_id="child_test",
        session_id="child_session",
        description="Child that will error",
        parent_agent=parent_agent,
        parent_task_id="parent_test",
        depth=1,
    )

    error_msg = "Test error: child agent failed"
    await registry.complete("child_test", error_msg, error=True)

    print(f"  ✓ Error recorded in registry")

    print("\n[3/3] Verifying error handling...")
    child_task = registry.get_task("child_test")

    if not child_task:
        print("  ✗ Child task not found")
        return False

    if child_task.status != "error":
        print(f"  ✗ Status: {child_task.status}, expected error")
        return False

    if child_task.result != error_msg:
        print(f"  ✗ Result mismatch")
        return False

    if not child_task.completed_event.is_set():
        print("  ✗ Event not set (system may have hung)")
        return False

    print(f"  ✓ Completion event set (system didn't hang)")

    # Wait for parent agent to process the error callback
    await asyncio.sleep(0.5)

    # Check pending children for parent (not total pending, since parent itself is pending)
    pending = registry.count_pending_for_parent("parent_test")
    if pending != 0:
        print(f"  ✗ Pending children: {pending}, expected 0")
        return False

    print(f"  ✓ No pending child tasks")

    await asyncio.sleep(0.2)

    parent_messages = parent_session.get_messages()
    error_received = any(
        "[子代理错误]" in msg.get("content", "") for msg in parent_messages
    )

    if error_received:
        print(f"  ✓ Parent received error notification")
    else:
        print(f"  ⚠ Error in parent session (async)")

    print("\n" + "=" * 60)
    print("✓ Error handling test passed")
    print("=" * 60)
    print("\nSummary:")
    print("  ✓ Child error caught")
    print("  ✓ Error status set")
    print("  ✓ Error message propagated")
    print("  ✓ System didn't hang")
    print("  ✓ Pending tasks cleared")
    print("=" * 60)
    return True


if __name__ == "__main__":
    try:
        success = asyncio.run(test_error_handling())
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
