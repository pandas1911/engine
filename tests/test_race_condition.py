#!/usr/bin/env python3
"""Test for potential race condition in multi-layer agent completion."""

import asyncio
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.agent_core import Agent
from src.config import Config
from src.models import AgentState, Session
from src.registry import SubagentRegistry
from src.llm_provider import MockLLMProvider


@pytest.mark.asyncio
async def test_concurrent_grandchildren():
    print("=" * 60)
    print("Testing: Concurrent Grandchildren Completion")
    print("=" * 60)

    config = Config(api_key="test", base_url="http://test", model="test", max_depth=3)
    registry = SubagentRegistry()

    root_session = Session(id="root_test", depth=0)
    root_session.add_message("system", "Test root")
    llm = MockLLMProvider()

    root = Agent(
        session=root_session,
        config=config,
        registry=registry,
        llm_provider=llm,
    )

    print(f"Root: {root.task_id}")
    print(f"Pending tasks: {registry.get_pending_count()}")

    b_session = Session(id="sess_b", depth=1, parent_id=root_session.id)
    b = Agent(
        session=b_session,
        config=config,
        registry=registry,
        llm_provider=llm,
        parent_task_id=root.task_id,
        task_id="task_b",
    )
    await registry.register(
        task_id="task_b",
        session_id="sess_b",
        description="Child",
        parent_agent=None,
        agent=b,
        parent_task_id=root.task_id,
        depth=1,
    )

    print(f"\nB registered: {b.task_id}")
    print(f"Pending tasks: {registry.get_pending_count()}")

    c_session = Session(id="sess_c", depth=2, parent_id=b_session.id)
    c = Agent(
        session=c_session,
        config=config,
        registry=registry,
        llm_provider=llm,
        parent_task_id=b.task_id,
        task_id="task_c",
    )
    await registry.register(
        task_id="task_c",
        session_id="sess_c",
        description="Grandchild",
        parent_agent=None,
        agent=c,
        parent_task_id=b.task_id,
        depth=2,
    )

    print(f"C registered: {c.task_id}")
    print(f"Pending tasks: {registry.get_pending_count()}")

    await registry.mark_ended_with_pending_descendants("task_b")
    print(f"\nB marked as ended_with_pending_descendants")
    print(f"B status: {registry.get_task('task_b').state_machine.current_state}")
    print(
        f"B wake_on_descendants_settle: {registry.get_task('task_b').wake_on_descendants_settle}"
    )
    print(f"Pending tasks: {registry.get_pending_count()}")
    print(f"Is B in pending? {'task_b' in registry._pending}")

    await registry.complete("task_c", "C result")
    print(f"\nC completed")
    print(f"C status: {registry.get_task('task_c').state_machine.current_state}")
    print(f"Pending tasks: {registry.get_pending_count()}")
    print(f"Is C in pending? {'task_c' in registry._pending}")

    b_task = registry.get_task("task_b")
    print(f"\nAfter C completed:")
    print(f"B status: {b_task.state_machine.current_state}")
    print(f"B wake_on_descendants_settle: {b_task.wake_on_descendants_settle}")

    pending_b_children = registry.count_pending_for_parent("task_b")
    print(f"B's pending children: {pending_b_children}")

    pending_root_children = registry.count_pending_for_parent(root.task_id)
    print(f"Root's pending children: {pending_root_children}")

    print("\n" + "=" * 60)
    print("KEY FINDING:")
    print("=" * 60)
    if "task_b" in registry._pending:
        print(
            "✗ BUG: B is still in _pending even though it's in 'ended_with_pending_descendants' state!"
        )
        print("  This could cause count_pending_for_parent to return incorrect values.")
    else:
        print("✓ B is correctly removed from _pending")

    if b_task.state_machine.current_state == AgentState.RUNNING:
        print("✓ B was correctly woken to 'running' state")
    else:
        print(f"✗ B status is {b_task.state_machine.current_state}, expected 'running'")


if __name__ == "__main__":
    asyncio.run(test_concurrent_grandchildren())
