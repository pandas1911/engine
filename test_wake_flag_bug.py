#!/usr/bin/env python3
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.registry import SubagentRegistry


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
        print(f"  [{self.name}] ERROR: I'm already completed! Should not be woken!")


async def test_parent_completed_but_wake_flag_still_true():
    print("=" * 60)
    print("POTENTIAL BUG SCENARIO:")
    print("=" * 60)
    print("A → B → C → D")
    print("")
    print("Scenario:")
    print("1. B spawns C, C spawns D")
    print("2. B enters 'ended_with_pending_descendants', wake_flag=True")
    print("3. C completes")
    print("4. B is woken, continues processing, COMPLETES")
    print("5. B status='completed' BUT wake_flag STILL=True")
    print("6. D completes")
    print("7. Should B be woken again? NO! But if wake_flag is still True...")
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

    print("\n[Step 1] B and C enter 'ended_with_pending_descendants':")
    await registry.mark_ended_with_pending_descendants("task_c")
    await registry.mark_ended_with_pending_descendants("task_b")

    b_task = registry.get_task("task_b")
    print(f"  B status: {b_task.status}")
    print(f"  B wake_flag: {b_task.wake_on_descendants_settle}")

    print("\n[Step 2] C completes:")
    await registry.complete("task_c", "C result")
    await asyncio.sleep(0.1)

    print(f"  B status after C complete: {registry.get_task('task_b').status}")
    print(
        f"  B wake_flag after C complete: {registry.get_task('task_b').wake_on_descendants_settle}"
    )
    print(f"  B received: {mock_b.results_received}")

    print("\n[Step 3] Simulate B completing (without clearing wake_flag):")
    b_task = registry.get_task("task_b")
    async with registry._lock:
        b_task.status = "completed"
        registry._pending.discard("task_b")

    print(f"  B status: {b_task.status}")
    print(f"  B wake_flag: {b_task.wake_on_descendants_settle}")
    print(f"  B in pending? {'task_b' in registry._pending}")

    print("\n[Step 4] D completes:")
    await registry.complete("task_d", "D result")
    await asyncio.sleep(0.1)

    print(f"  B status after D complete: {registry.get_task('task_b').status}")
    print(
        f"  B wake_flag after D complete: {registry.get_task('task_b').wake_on_descendants_settle}"
    )
    print(f"  B received (TOTAL): {mock_b.results_received}")

    print("\n" + "=" * 60)
    print("CRITICAL ANALYSIS:")
    print("=" * 60)

    wake_count = len([r for r in mock_b.results_received if r[0] == "wake"])
    print(f"B was woken {wake_count} time(s)")

    if wake_count > 1:
        print("✗ BUG DETECTED: B was woken multiple times!")
        print("  This happens when:")
        print("  1. B completes but wake_flag is NOT cleared")
        print("  2. D completes and sees B.status='completed' + wake_flag=True")
        print("  3. B is incorrectly woken again!")
    elif wake_count == 1:
        print("✓ B was woken once (correct)")
        b_final = registry.get_task("task_b")
        if b_final.wake_on_descendants_settle:
            print("✗ WARNING: B's wake_flag is still True even though B is completed!")
            print("  This is a potential bug waiting to happen!")
        else:
            print("✓ B's wake_flag was correctly cleared")


if __name__ == "__main__":
    asyncio.run(test_parent_completed_but_wake_flag_still_true())
