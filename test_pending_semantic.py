#!/usr/bin/env python3
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.registry import SubagentRegistry
from src.models import SubagentTask


class MockAgent:
    def __init__(self, task_id):
        self.task_id = task_id
        self.wake_count = 0
        self.complete_count = 0

    async def _on_subagent_complete(self, child_task_id, result):
        self.complete_count += 1
        print(f"  [{self.task_id}] ← Received complete from {child_task_id}")

    async def _on_descendant_wake(self, descendant_task_id, result):
        self.wake_count += 1
        print(f"  [{self.task_id}] ← Woken by {descendant_task_id}")
        print(f"  [{self.task_id}] Checking pending children...")
        pending = 0
        for tid, task in registry._tasks.items():
            if task.parent_task_id == self.task_id and tid in registry._pending:
                pending += 1
        print(f"  [{self.task_id}] Pending children count: {pending}")
        if pending == 0:
            print(
                f"  [{self.task_id}] All children done, would call _continue_processing()"
            )


async def test_scenario():
    print("=" * 60)
    print("SCENARIO: A → B → C")
    print("=" * 60)

    registry = SubagentRegistry()

    mock_a = MockAgent("task_a")
    mock_b = MockAgent("task_b")
    mock_c = MockAgent("task_c")

    await registry.register(
        task_id="task_a",
        session_id="sess_a",
        description="Root",
        parent_agent=None,
        parent_task_id=None,
        depth=0,
    )
    await registry.set_agent("task_a", mock_a)

    await registry.register(
        task_id="task_b",
        session_id="sess_b",
        description="Child",
        parent_agent=mock_a,
        parent_task_id="task_a",
        depth=1,
    )
    await registry.set_agent("task_b", mock_b)

    await registry.register(
        task_id="task_c",
        session_id="sess_c",
        description="Grandchild",
        parent_agent=mock_b,
        parent_task_id="task_b",
        depth=2,
    )
    await registry.set_agent("task_c", mock_c)

    print("\n[Step 1] Initial state:")
    print(f"  Pending: {registry._pending}")
    print(f"  A's pending children: {registry.count_pending_for_parent('task_a')}")
    print(f"  B's pending children: {registry.count_pending_for_parent('task_b')}")

    print("\n[Step 2] B enters 'ended_with_pending_descendants':")
    await registry.mark_ended_with_pending_descendants("task_b")
    print(f"  B status: {registry.get_task('task_b').status}")
    print(f"  B in pending? {'task_b' in registry._pending}")
    print(f"  Pending: {registry._pending}")

    print("\n[Step 3] C completes:")
    await registry.complete("task_c", "C done")
    await asyncio.sleep(0.1)

    print(f"  C status: {registry.get_task('task_c').status}")
    print(f"  C in pending? {'task_c' in registry._pending}")
    print(f"  B status: {registry.get_task('task_b').status}")
    print(
        f"  B wake_on_descendants_settle: {registry.get_task('task_b').wake_on_descendants_settle}"
    )
    print(f"  B's pending children: {registry.count_pending_for_parent('task_b')}")
    print(f"  B wake_count: {mock_b.wake_count}")
    print(f"  B in pending? {'task_b' in registry._pending}")

    print(f"\n[Step 4] Check if B was removed from pending after wake:")
    print(f"  B status should be 'running': {registry.get_task('task_b').status}")
    print(f"  B should NOT be in pending: {'task_b' not in registry._pending}")

    print("\n" + "=" * 60)
    print("KEY QUESTION:")
    print("=" * 60)
    print("When C completes:")
    print(
        "  1. Should B be woken? YES (because B is in 'ended_with_pending_descendants')"
    )
    print(f"  2. Was B woken? {mock_b.wake_count > 0}")
    print("  3. Should B report to A? Only after B finishes processing")
    print(f"  4. Did A receive any report? {mock_a.complete_count}")


if __name__ == "__main__":
    asyncio.run(test_scenario())
