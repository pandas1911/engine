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
        self.results_received = []
        self.status = "running"

    async def _on_subagent_complete(self, child_task_id, result):
        self.results_received.append(("complete", child_task_id, result))
        print(f"  [{self.name}] ← Complete from {child_task_id[:8]}")

        pending = self.registry.count_pending_for_parent(self.task_id)
        print(f"  [{self.name}] Pending children: {pending}")

        if pending == 0:
            print(f"  [{self.name}] ✗ ALL CHILDREN DONE - WOULD REPORT TO PARENT NOW!")
            print(f"  [{self.name}] ✗ But maybe some grandchildren are still running?!")
            self.status = "completed"

    async def _on_descendant_wake(self, descendant_task_id, result):
        self.results_received.append(("wake", descendant_task_id, result))
        print(f"  [{self.name}] ← Woken by {descendant_task_id[:8]}")

        pending = self.registry.count_pending_for_parent(self.task_id)
        print(f"  [{self.name}] Pending children: {pending}")

        if pending == 0:
            print(f"  [{self.name}] ✗ ALL CHILDREN DONE - WOULD REPORT TO PARENT NOW!")
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

    await registry.register("task_a", "sess_a", "Root", None, None, 0)
    await registry.set_agent("task_a", mock_a)

    await registry.register("task_b", "sess_b", "Child", mock_a, "task_a", 1)
    await registry.set_agent("task_b", mock_b)

    await registry.register("task_c", "sess_c", "Grandchild", mock_b, "task_b", 2)
    await registry.set_agent("task_c", mock_c)

    print("\n[Step 1] C enters 'ended_with_pending_descendants':")
    await registry.mark_ended_with_pending_descendants("task_c")
    print(f"  C status: {registry.get_task('task_c').status}")
    print(f"  C in pending: {'task_c' in registry._pending}")

    print("\n[Step 2] C1 registers and completes IMMEDIATELY:")
    await registry.register(
        "task_c1", "sess_c1", "GreatGrandchild1", mock_c, "task_c", 3
    )
    await registry.set_agent("task_c1", MockAgent("task_c1", "C1", registry))
    print(f"  C1 registered")
    print(f"  Pending: {[t[:8] for t in registry._pending]}")

    await registry.complete("task_c1", "C1 result")
    await asyncio.sleep(0.1)

    print(f"\n  C's pending children: {registry.count_pending_for_parent('task_c')}")
    print(f"  C status: {registry.get_task('task_c').status}")
    print(f"  C received: {mock_c.results_received}")
    print(f"  C mock status: {mock_c.status}")

    print("\n[Step 3] Now C2 and C3 register (TOO LATE!):")
    await registry.register(
        "task_c2", "sess_c2", "GreatGrandchild2", mock_c, "task_c", 3
    )
    await registry.register(
        "task_c3", "sess_c3", "GreatGrandchild3", mock_c, "task_c", 3
    )
    print(f"  C2 and C3 registered")
    print(f"  Pending: {[t[:8] for t in registry._pending]}")
    print(f"  C's pending children: {registry.count_pending_for_parent('task_c')}")

    print("\n" + "=" * 60)
    print("CRITICAL ANALYSIS:")
    print("=" * 60)

    if mock_c.status == "completed":
        print("✗ BUG DETECTED:")
        print("  C reported to B BEFORE C2 and C3 completed!")
        print("  This is the '提前汇报' bug you mentioned!")
        print(
            f"  C's pending children when C1 completed: {registry.count_pending_for_parent('task_c')}"
        )
        print("  But C2 and C3 were registered AFTER C reported!")
    else:
        print("✓ No bug detected in this scenario")


if __name__ == "__main__":
    asyncio.run(test_child_spawns_slowly())
