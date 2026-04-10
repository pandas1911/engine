#!/usr/bin/env python3
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.registry import SubagentRegistry


class MockAgent:
    def __init__(self, task_id, name, registry):
        self.task_id = task_id
        self.name = name
        self.registry = registry
        self._event_queue = []
        self.status = "running"
        self.resumed = False

    async def _resume_from_children(self):
        self.resumed = True
        print(f"  [{self.name}] ← Resumed from children")

        pending = self.registry.count_pending_for_parent(self.task_id)
        print(f"  [{self.name}] Pending children: {pending}")

        if pending == 0:
            print(f"  [{self.name}] All children done - would report to parent now")
            self.status = "completed"


async def test_child_spawns_slowly():
    print("=" * 60)
    print("SCENARIO: Race Condition in Child Registration")
    print("=" * 60)
    print("A → B → C")
    print("C spawns children C1, C2, C3 ASYNCHRONOUSLY")
    print("C1 completes BEFORE C2 and C3 are registered")
    print("=" * 60)

    registry = SubagentRegistry()

    mock_a = MockAgent("task_a", "A", registry)
    mock_b = MockAgent("task_b", "B", registry)
    mock_c = MockAgent("task_c", "C", registry)

    await registry.register(
        task_id="task_a",
        session_id="sess_a",
        description="Root",
        parent_agent=None,
        agent=mock_a,
        parent_task_id=None,
        depth=0,
    )

    await registry.register(
        task_id="task_b",
        session_id="sess_b",
        description="Child",
        parent_agent=None,
        agent=mock_b,
        parent_task_id="task_a",
        depth=1,
    )

    await registry.register(
        task_id="task_c",
        session_id="sess_c",
        description="Grandchild",
        parent_agent=None,
        agent=mock_c,
        parent_task_id="task_b",
        depth=2,
    )

    print("\n[Step 1] C enters 'ended_with_pending_descendants':")
    await registry.mark_ended_with_pending_descendants("task_c")
    print(f"  C status: {registry.get_task('task_c').status}")
    print(f"  C in pending: {'task_c' in registry._pending}")

    print("\n[Step 2] C1 registers and completes IMMEDIATELY:")
    await registry.register(
        task_id="task_c1",
        session_id="sess_c1",
        description="GreatGrandchild1",
        parent_agent=None,
        agent=MockAgent("task_c1", "C1", registry),
        parent_task_id="task_c",
        depth=3,
    )
    print(f"  C1 registered")
    print(f"  Pending: {[t[:8] for t in registry._pending]}")

    await registry.complete("task_c1", "C1 result")
    await asyncio.sleep(0.1)

    print(f"\n  C's pending children: {registry.count_pending_for_parent('task_c')}")
    print(f"  C status: {registry.get_task('task_c').status}")
    print(f"  C event queue: {mock_c._event_queue}")
    print(f"  C resumed: {mock_c.resumed}")

    print("\n[Step 3] Now C2 and C3 register (TOO LATE!):")
    await registry.register(
        task_id="task_c2",
        session_id="sess_c2",
        description="GreatGrandchild2",
        parent_agent=None,
        agent=MockAgent("task_c2", "C2", registry),
        parent_task_id="task_c",
        depth=3,
    )
    await registry.register(
        task_id="task_c3",
        session_id="sess_c3",
        description="GreatGrandchild3",
        parent_agent=None,
        agent=MockAgent("task_c3", "C3", registry),
        parent_task_id="task_c",
        depth=3,
    )
    print(f"  C2 and C3 registered")
    print(f"  Pending: {[t[:8] for t in registry._pending]}")
    print(f"  C's pending children: {registry.count_pending_for_parent('task_c')}")

    print("\n" + "=" * 60)
    print("CRITICAL ANALYSIS:")
    print("=" * 60)

    if mock_c.resumed:
        print("✗ BUG DETECTED:")
        print("  C was resumed/woken BEFORE C2 and C3 completed!")
        print("  This is the '提前汇报' race condition!")
        print(
            f"  C's pending children when C1 completed: {registry.count_pending_for_parent('task_c')}"
        )
        print("  But C2 and C3 were registered AFTER C was woken!")
    else:
        print("✓ No bug detected in this scenario")


if __name__ == "__main__":
    asyncio.run(test_child_spawns_slowly())
