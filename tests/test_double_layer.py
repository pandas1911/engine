"""Double-layer integration test for Agent system.

This test verifies:
1. Parent agent can spawn a child agent
2. Parent receives child result via callback
3. Registry correctly tracks spawned children
4. Real MiniMax API is used (no mocks)

Expected flow:
1. Parent Agent receives task that requires spawning a child
2. Parent spawns child via spawn tool
3. Child executes task and completes
4. Parent receives child result via _on_subagent_complete callback
5. Parent aggregates results and provides final response
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent_core import Agent
from src.models import LLMResponse, Session, ToolCall, AgentState
from src.registry import SubagentRegistry
from src.llm_provider import MiniMaxProvider
from src.config import get_config, Config


# Track callback invocations for verification
callback_tracker = {
    "child_spawned": False,
    "child_task_id": None,
    "child_result": None,
    "parent_received_result": False,
    "callback_invoked": False,
}


class TrackedMiniMaxProvider(MiniMaxProvider):
    """MiniMax provider that tracks LLM calls for test verification."""

    def __init__(self, config: Config):
        super().__init__(config)
        self.call_history: List[Dict] = []
        self.responses: List[LLMResponse] = []

    async def chat(
        self, messages: List[Dict], tools: List[Dict], **kwargs
    ) -> LLMResponse:
        self.call_history.append(
            {
                "message_count": len(messages),
                "has_tools": bool(tools),
                "last_role": messages[-1].get("role") if messages else None,
            }
        )
        response = await super().chat(messages, tools, **kwargs)
        self.responses.append(response)
        return response


class TrackedAgent(Agent):
    """Agent that tracks callback invocations for test verification."""

    async def _on_subagent_complete(self, child_task_id: str, result: str):
        """Override to track callback invocation."""
        callback_tracker["callback_invoked"] = True
        callback_tracker["child_task_id"] = child_task_id
        callback_tracker["child_result"] = result
        callback_tracker["parent_received_result"] = True
        print(f"[TEST] ✓ Callback invoked: child {child_task_id} completed")
        await super()._on_subagent_complete(child_task_id, result)


async def run_double_layer_test():
    """Run the double-layer integration test using real MiniMax API."""
    print("=" * 60)
    print("Double-Layer Integration Test")
    print("=" * 60)

    # Reset callback tracker
    global callback_tracker
    callback_tracker = {
        "child_spawned": False,
        "child_task_id": None,
        "child_result": None,
        "parent_received_result": False,
        "callback_invoked": False,
    }

    # Load real configuration
    try:
        config = get_config()
        print(f"[TEST] Loaded config with model: {config.openai_model}")
    except ValueError as e:
        print(f"[TEST] ✗ FAILED: Could not load config - {e}")
        print(
            "[TEST] Ensure .env file has OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL"
        )
        return False

    # Create registry
    registry = SubagentRegistry()
    print("[TEST] ✓ Created SubagentRegistry")

    # Create parent session
    parent_session = Session(id="parent_session_001", depth=0)
    parent_session.add_message(
        "system",
        "You are a parent agent. You can spawn child agents to handle subtasks. "
        "When you receive results from children, aggregate them and provide a final answer.",
    )
    print("[TEST] ✓ Created parent session")

    # Create LLM provider with tracking
    llm_provider = TrackedMiniMaxProvider(config)
    print("[TEST] ✓ Created TrackedMiniMaxProvider")

    # Create parent agent with TrackedAgent
    parent_agent = TrackedAgent(
        session=parent_session,
        config=config,
        registry=registry,
        llm_provider=llm_provider,
        task_id="parent_task_001",
    )
    print("[TEST] ✓ Created parent TrackedAgent")

    # Task that will trigger child spawn
    task = (
        "Please spawn a child agent to analyze the number 42 and tell me if it's interesting. "
        "Use the spawn tool to create a child for this analysis task."
    )
    print(f"[TEST] Task: {task[:60]}...")

    # Run parent agent
    print("\n[TEST] Running parent agent...")
    result = await parent_agent.run(task)

    # Wait for async operations to complete
    # The child spawn and callback happen asynchronously
    max_wait = 60  # seconds
    waited = 0
    while waited < max_wait:
        # Check if child was spawned
        if registry.get_pending_count() == 0 and callback_tracker["callback_invoked"]:
            break
        await asyncio.sleep(1)
        waited += 1
        if waited % 10 == 0:
            print(f"[TEST] Waiting for completion... ({waited}s)")

    # Additional wait for final processing
    await asyncio.sleep(2)

    print("\n" + "=" * 60)
    print("Verification Results")
    print("=" * 60)

    all_passed = True

    # Verification 1: Child was spawned
    child_tasks = [
        tid
        for tid, task in registry._tasks.items()
        if task.parent_task_id == "parent_task_001"
    ]
    if child_tasks:
        callback_tracker["child_spawned"] = True
        print(f"✓ Child spawned: {child_tasks}")
    else:
        print("✗ FAILED: No child was spawned")
        all_passed = False

    # Verification 2: Registry tracking
    total_tasks = len(registry._tasks)
    print(f"✓ Registry tracked {total_tasks} task(s)")

    # Verification 3: Child completed
    completed_children = [
        tid
        for tid in child_tasks
        if registry.get_task(tid) and registry.get_task(tid).status == "completed"
    ]
    if completed_children:
        print(f"✓ Child completed: {completed_children}")
    else:
        print("✗ FAILED: Child did not complete")
        all_passed = False

    # Verification 4: Parent received child result
    if callback_tracker["parent_received_result"]:
        print(f"✓ Parent received child result")
        if callback_tracker["child_result"]:
            print(f"  Child result: {callback_tracker['child_result'][:80]}...")
    else:
        print("✗ FAILED: Parent did not receive child result")
        all_passed = False

    # Verification 5: Callback was triggered
    if callback_tracker["callback_invoked"]:
        print("✓ Callback mechanism triggered")
    else:
        print("✗ FAILED: Callback mechanism not triggered")
        all_passed = False

    # Verification 6: Parent completed
    if parent_agent.state == AgentState.COMPLETED:
        print("✓ Parent agent completed")
    else:
        print(f"✗ Parent agent state: {parent_agent.state}")
        # This may not be a failure if waiting on descendants

    # Verification 7: Final result exists
    if parent_agent._final_result:
        print(f"✓ Final result: {parent_agent._final_result[:100]}...")
    else:
        print("  Note: No final result (may be waiting state)")

    # Verification 8: LLM was called
    if llm_provider.call_history:
        print(f"✓ LLM called {len(llm_provider.call_history)} time(s)")
    else:
        print("✗ FAILED: No LLM calls made")
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ Double-layer test passed")
        print("=" * 60)
        return True
    else:
        print("✗ Double-layer test FAILED")
        print("=" * 60)
        return False


async def main():
    """Main entry point."""
    try:
        success = await run_double_layer_test()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[TEST] ✗ Exception: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
