#!/usr/bin/env python3
"""Wake mechanism integration test.

This test verifies that when a parent agent ends early with pending descendants,
it gets woken up when the child completes.
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent_core import Agent
from src.models import Session, AgentState
from src.registry import SubagentRegistry
from src.llm_provider import MiniMaxProvider
from src.config import get_config, Config


def create_agent_with_registry(
    task_id: str,
    task: str,
    registry: SubagentRegistry,
    parent_task_id: Optional[str] = None,
    depth: int = 0,
    config: Optional[Config] = None,
) -> Agent:
    """Create an agent with shared registry.

    Args:
        task_id: Unique identifier for this agent/task
        task: Task description for the agent
        registry: Shared registry instance
        parent_task_id: Optional parent task ID
        depth: Session depth (0 for root)
        config: Optional configuration

    Returns:
        Agent instance with all dependencies initialized
    """
    if config is None:
        try:
            config = get_config()
        except ValueError:
            config = Config(
                openai_api_key="test-key",
                openai_base_url="https://api.test.com",
                openai_model="test-model",
            )

    session = Session(
        id=f"test_session_{task_id}", depth=depth, parent_id=parent_task_id
    )

    session.add_message("system", f"You are a test agent. Task: {task}")

    llm_provider = MiniMaxProvider(config)

    agent = Agent(
        session=session,
        config=config,
        registry=registry,
        llm_provider=llm_provider,
        task_id=task_id,
        parent_task_id=parent_task_id,
    )

    return agent


async def test_wake_mechanism():
    """Test wake mechanism with parent ending early and child waking it up."""
    print("=" * 60)
    print("Wake Mechanism Integration Test")
    print("=" * 60)

    print("\n[1/8] Creating shared registry...")
    registry = SubagentRegistry()
    print("  ✓ Registry created")

    print("\n[2/8] Creating parent agent...")
    parent_agent = create_agent_with_registry(
        task_id="parent_task_001",
        task="You are the parent agent. Spawn a child to do work.",
        registry=registry,
        depth=0,
    )
    print(f"  ✓ Parent agent created: {parent_agent.task_id}")

    # Register parent agent
    await registry.register(
        task_id="parent_task_001",
        session_id=parent_agent.session.id,
        description="Parent orchestrator",
        parent_agent=parent_agent,
        depth=0,
    )
    print("  ✓ Parent agent registered")

    print("\n[3/8] Creating child agent (simulating spawn)...")
    child_agent = create_agent_with_registry(
        task_id="child_task_001",
        task="You are a child agent. Do some work and return a result.",
        registry=registry,
        parent_task_id="parent_task_001",
        depth=1,
    )
    print(f"  ✓ Child agent created: {child_agent.task_id}")

    # Register child agent
    await registry.register(
        task_id="child_task_001",
        session_id=child_agent.session.id,
        description="Child worker",
        parent_agent=child_agent,
        parent_task_id="parent_task_001",
        depth=1,
    )
    print("  ✓ Child agent registered")

    # Add child to parent's child list
    async with registry._lock:
        parent_task = registry._tasks["parent_task_001"]
        parent_task.child_task_ids.add("child_task_001")
    print("  ✓ Child added to parent's child list")

    print("\n[4/8] Simulating parent ending early...")
    parent_agent.state = AgentState.RUNNING
    await registry.mark_ended_with_pending_descendants("parent_task_001")

    async with registry._lock:
        parent_task = registry._tasks["parent_task_001"]
        if parent_task.status != "ended_with_pending_descendants":
            print(
                f"  ✗ Parent status is {parent_task.status}, expected ended_with_pending_descendants"
            )
            return False
        if not parent_task.wake_on_descendants_settle:
            print("  ✗ wake_on_descendants_settle flag not set")
            return False

    print("  ✓ Parent ended with pending descendants")
    print("  ✓ wake_on_descendants_settle flag is set")

    wake_received = asyncio.Event()

    original_on_descendant_wake = parent_agent._on_descendant_wake

    async def tracked_on_descendant_wake(descendant_task_id: str, result: str):
        print(f"  ✓ Parent woken up by descendant: {descendant_task_id}")
        wake_received.set()
        await original_on_descendant_wake(descendant_task_id, result)

    parent_agent._on_descendant_wake = tracked_on_descendant_wake

    print("\n[5/8] Completing child agent (this should wake parent)...")
    await registry.complete("child_task_001", "Child work completed successfully!")
    print("  ✓ Child agent completed")

    print("\n[6/8] Waiting for parent to be woken up...")
    try:
        await asyncio.wait_for(wake_received.wait(), timeout=5.0)
        print("  ✓ Parent received wake notification")
    except asyncio.TimeoutError:
        print("  ✗ Parent was not woken up within timeout")
        return False

    print("\n[7/8] Verifying parent state after wake...")
    if parent_agent.state != AgentState.RUNNING:
        print(f"  ✗ Parent state is {parent_agent.state}, expected RUNNING")
        return False
    print(f"  ✓ Parent state is RUNNING")

    async with registry._lock:
        parent_task = registry._tasks["parent_task_001"]
        if parent_task.wake_on_descendants_settle:
            print("  ✗ wake_on_descendants_settle flag should be cleared")
            return False
    print("  ✓ wake_on_descendants_settle flag was cleared")

    print("\n[8/8] Verifying parent received wake message...")
    wake_messages = [
        msg
        for msg in parent_agent.session.messages
        if msg.metadata.get("is_wake_notification")
    ]

    if not wake_messages:
        print("  ✗ Parent did not receive wake message")
        return False

    wake_msg = wake_messages[0]
    if "child_task_001" not in wake_msg.content:
        print(f"  ✗ Wake message doesn't mention child task: {wake_msg.content}")
        return False

    print(f"  ✓ Wake message received: {wake_msg.content[:80]}...")

    print("\n" + "=" * 60)
    print("✓ Wake mechanism test passed")
    print("=" * 60)
    return True


if __name__ == "__main__":
    try:
        success = asyncio.run(test_wake_mechanism())
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
