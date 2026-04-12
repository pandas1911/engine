#!/usr/bin/env python3
import asyncio
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.registry import SubagentRegistry
from src.models import AgentState
from src.state_machine import AgentStateMachine


class MockAgent:
    def __init__(self, task_id, name):
        self.task_id = task_id
        self.name = name
        self.results_received = []
        self.state_machine = AgentStateMachine(AgentState.IDLE)

    async def _on_subagent_complete(self, child_task_id, result):
        self.results_received.append(("complete", child_task_id, result))
        print(f"  [{self.name}] ← Received complete from {child_task_id[:8]}")

    async def _on_descendant_wake(self, descendant_task_id, result):
        self.results_received.append(("wake", descendant_task_id, result))
        print(f"  [{self.name}] ← Woken by {descendant_task_id[:8]}")
        print(f"  [{self.name}] ERROR: I'm already completed! Should not be woken!")

    async def _resume_from_children(self, child_results):
        self.results_received.append(("resume", "children", str(child_results)))
        print(f"  [{self.name}] ← Resumed from children")
        if self.state_machine.current_state == AgentState.WAITING_FOR_CHILDREN:
            self.state_machine.trigger("children_settled")


@pytest.mark.asyncio
async def test_parent_completed_but_wake_flag_still_true():
    print("=" * 60)
    print("WAKE FLAG BUG SCENARIO (unified state machine):")
    print("=" * 60)
    print("A → B → C → D")
    print("")
    print("Scenario:")
    print("1. B spawns C, C spawns D")
    print("2. B and C enter WAITING_FOR_CHILDREN via state machine")
    print("3. C completes")
    print("4. B resumes, continues processing, COMPLETES")
    print("5. D completes — B should NOT be woken again")
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

    # Put B and C into WAITING_FOR_CHILDREN via state machine triggers
    print("\n[Step 1] B and C enter WAITING_FOR_CHILDREN:")
    mock_c.state_machine.trigger("start")
    mock_c.state_machine.trigger("spawn_children")
    mock_b.state_machine.trigger("start")
    mock_b.state_machine.trigger("spawn_children")

    print(f"  B status: {mock_b.state_machine.current_state}")

    print("\n[Step 2] C completes:")
    mock_c.state_machine.trigger("children_settled")
    await registry.complete("task_c", "C result")
    await asyncio.sleep(0.1)

    print(f"  B status after C complete: {mock_b.state_machine.current_state}")
    print(f"  B received: {mock_b.results_received}")

    print("\n[Step 3] Simulate B completing:")
    # Drive B through children_settled → RUNNING → COMPLETED
    mock_b.state_machine.trigger("children_settled")
    mock_b.state_machine.trigger("finish")
    registry._pending.discard("task_b")

    print(f"  B status: {mock_b.state_machine.current_state}")
    print(f"  B in pending? {'task_b' in registry._pending}")

    print("\n[Step 4] D completes:")
    await registry.complete("task_d", "D result")
    await asyncio.sleep(0.1)

    print(f"  B status after D complete: {mock_b.state_machine.current_state}")
    print(f"  B received (TOTAL): {mock_b.results_received}")

    print("\n" + "=" * 60)
    print("CRITICAL ANALYSIS:")
    print("=" * 60)

    wake_count = len([r for r in mock_b.results_received if r[0] == "wake"])
    print(f"B was woken {wake_count} time(s)")

    if wake_count > 1:
        print("✗ BUG DETECTED: B was woken multiple times!")
        print("  This happens when:")
        print("  1. B completes but is somehow still reachable for wake")
        print("  2. D completes and triggers B again")
        print("  3. B is incorrectly woken again!")
    elif wake_count == 1:
        print("✓ B was woken once (correct)")
    else:
        print("✓ B was woken 0 times — state machine is the single authority")
        print("✓ B's wake_flag no longer exists — state machine prevents double-wake")


if __name__ == "__main__":
    asyncio.run(test_parent_completed_but_wake_flag_still_true())
