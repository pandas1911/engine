#!/usr/bin/env python3
import asyncio
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.registry import SubagentRegistry
from src.models import AgentState


class MockAgent:
    def __init__(self, task_id, name):
        self.task_id = task_id
        self.name = name
        self.results_received = []

    async def _on_subagent_complete(self, child_task_id, result):
        self.results_received.append(("complete", child_task_id, result))
        print(f"  [{self.name}] ← Received complete from {child_task_id[:8]}")

    async def _on_descendant_wake(self, descendant_task_id, result):
        self.results_received.append(("wake", descendant_task_id, result))
        print(f"  [{self.name}] ← Woken by {descendant_task_id[:8]}")


@pytest.mark.asyncio
async def test_b_has_multiple_children():
    print("=" * 60)
    print("SCENARIO: A → B → [C, D]")
    print("=" * 60)
    print("B has TWO children C and D")
    print("C completes FIRST, then D completes")
    print("=" * 60)

    registry = SubagentRegistry()

    mock_a = MockAgent("task_a", "A")
    mock_b = MockAgent("task_b", "B")
    mock_c = MockAgent("task_c", "C")
    mock_d = MockAgent("task_d", "D")

    await registry.register("task_a", "sess_a", "Root", None, None, 0)
    await registry.set_agent("task_a", mock_a)

    await registry.register("task_b", "sess_b", "Child", mock_a, "task_a", 1)
    await registry.set_agent("task_b", mock_b)

    await registry.register("task_c", "sess_c", "Grandchild1", mock_b, "task_b", 2)
    await registry.set_agent("task_c", mock_c)

    await registry.register("task_d", "sess_d", "Grandchild2", mock_b, "task_b", 2)
    await registry.set_agent("task_d", mock_d)

    print("\n[Step 1] Initial state:")
    print(f"  Pending: {[t[:8] for t in registry._pending]}")
    print(
        f"  B's children: {[t[:8] for t in registry.get_task('task_b').child_task_ids]}"
    )

    print("\n[Step 2] B enters 'ended_with_pending_descendants':")
    await registry.mark_ended_with_pending_descendants("task_b")
    print(f"  B status: {registry.get_task('task_b').state_machine.current_state}")
    print(f"  B in pending? {'task_b' in registry._pending}")

    print("\n[Step 3] C completes:")
    await registry.complete("task_c", "C result")
    await asyncio.sleep(0.1)

    print(f"  B's pending children: {registry.count_pending_for_parent('task_b')}")
    print(f"  B status: {registry.get_task('task_b').state_machine.current_state}")
    print(f"  B in pending? {'task_b' in registry._pending}")
    print(f"  B received: {mock_b.results_received}")

    print("\n[Step 4] D completes:")
    await registry.complete("task_d", "D result")
    await asyncio.sleep(0.1)

    print(f"  B's pending children: {registry.count_pending_for_parent('task_b')}")
    print(f"  B status: {registry.get_task('task_b').state_machine.current_state}")
    print(f"  B in pending? {'task_b' in registry._pending}")
    print(f"  B received: {mock_b.results_received}")
    print(f"  A received: {mock_a.results_received}")

    print("\n" + "=" * 60)
    print("ANALYSIS:")
    print("=" * 60)
    print(
        f"B was woken {len([r for r in mock_b.results_received if r[0] == 'wake'])} times"
    )
    print(
        f"B received {len([r for r in mock_b.results_received if r[0] == 'complete'])} complete callbacks"
    )
    print(f"A received {len(mock_a.results_received)} callbacks from B")


@pytest.mark.asyncio
async def test_b_has_grandchild():
    print("\n" + "=" * 60)
    print("SCENARIO: A → B → C → D")
    print("=" * 60)
    print("B has child C, C has child D")
    print("D completes FIRST")
    print("=" * 60)

    registry = SubagentRegistry()

    mock_a = MockAgent("task_a", "A")
    mock_b = MockAgent("task_b", "B")
    mock_c = MockAgent("task_c", "C")
    mock_d = MockAgent("task_d", "D")

    await registry.register("task_a", "sess_a", "Root", None, None, 0)
    await registry.set_agent("task_a", mock_a)

    await registry.register("task_b", "sess_b", "Child", mock_a, "task_a", 1)
    await registry.set_agent("task_b", mock_b)

    await registry.register("task_c", "sess_c", "Grandchild", mock_b, "task_b", 2)
    await registry.set_agent("task_c", mock_c)

    await registry.register("task_d", "sess_d", "GreatGrandchild", mock_c, "task_c", 3)
    await registry.set_agent("task_d", mock_d)

    print("\n[Step 1] Initial state:")
    print(f"  Pending: {[t[:8] for t in registry._pending]}")

    print("\n[Step 2] C enters 'ended_with_pending_descendants':")
    await registry.mark_ended_with_pending_descendants("task_c")
    print(f"  C in pending? {'task_c' in registry._pending}")

    print("\n[Step 3] B enters 'ended_with_pending_descendants':")
    await registry.mark_ended_with_pending_descendants("task_b")
    print(f"  B in pending? {'task_b' in registry._pending}")

    print("\n[Step 4] D completes:")
    await registry.complete("task_d", "D result")
    await asyncio.sleep(0.1)

    print(f"  C's pending children: {registry.count_pending_for_parent('task_c')}")
    print(f"  C status: {registry.get_task('task_c').state_machine.current_state}")
    print(f"  C in pending? {'task_c' in registry._pending}")
    print(f"  C received: {mock_c.results_received}")

    print(f"\n  B's pending children: {registry.count_pending_for_parent('task_b')}")
    print(f"  B status: {registry.get_task('task_b').state_machine.current_state}")
    print(f"  B in pending? {'task_b' in registry._pending}")
    print(f"  B received: {mock_b.results_received}")

    print(f"\n  A received: {mock_a.results_received}")

    print("\n" + "=" * 60)
    print("KEY FINDING:")
    print("=" * 60)
    print("When D completes:")
    print("  1. C should be woken")
    print(f"  2. Was C woken? {len(mock_c.results_received) > 0}")
    print(f"  3. Was B notified? {len(mock_b.results_received) > 0}")
    print(f"  4. Was A notified? {len(mock_a.results_received) > 0}")
    print("\n  Expected: Only C should be woken, B and A should NOT be notified yet")
    print(
        f"  Actual: C woken={len(mock_c.results_received) > 0}, B notified={len(mock_b.results_received) > 0}, A notified={len(mock_a.results_received) > 0}"
    )


if __name__ == "__main__":
    asyncio.run(test_b_has_multiple_children())
    asyncio.run(test_b_has_grandchild())
