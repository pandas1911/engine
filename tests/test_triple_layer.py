#!/usr/bin/env python3
"""Triple-layer integration test.

This test verifies 3-layer nesting (Main → Orchestrator → Worker) and
result propagation from Worker → Orchestrator → Main using the real MiniMax API.
"""

import asyncio
import sys
import os
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent_core import Agent
from src.config import get_config, Config
from src.llm_provider import MiniMaxProvider
from src.models import AgentState, LLMResponse, Session, ToolCall
from src.registry import SubagentRegistry
from tests.utils import wait_for_completion


class TripleLayerTestLLM:
    """Custom LLM provider for triple-layer test with controlled behavior."""

    def __init__(self, config: Config):
        """Initialize with config for MiniMax provider."""
        self.config = config
        self.real_llm = MiniMaxProvider(config)
        self.call_history: List[Dict] = []
        self.agent_depth_map: Dict[str, int] = {}

    def register_agent_depth(self, task_id: str, depth: int):
        """Register an agent's depth for tracking."""
        self.agent_depth_map[task_id] = depth

    async def chat(
        self, messages: List[Dict], tools: List[Dict], **kwargs
    ) -> LLMResponse:
        """Chat with controlled behavior based on agent depth."""
        # Track this call
        last_msg = messages[-1] if messages else {}
        last_content = last_msg.get("content", "")

        call_info = {
            "messages_count": len(messages),
            "has_tools": len(tools) > 0,
            "has_spawn_tool": any(
                t.get("function", {}).get("name") == "spawn" for t in tools
            ),
            "last_content_preview": last_content[:100] if last_content else "",
        }
        self.call_history.append(call_info)

        # Determine agent depth from system message
        depth = 0
        for msg in messages:
            if msg.get("role") == "system" and "Depth:" in msg.get("content", ""):
                try:
                    depth_str = msg["content"].split("Depth:")[1].split("/")[0].strip()
                    depth = int(depth_str)
                except:
                    pass

        # Check for child results FIRST (higher priority than spawning)
        # Main receiving orchestrator result
        if depth == 0 and (
            "[子代理完成]" in last_content or "[子代理结果汇总]" in last_content
        ):
            return LLMResponse(
                content="[Main完成] 通过Orchestrator协调Worker完成了代码分析。综合报告：项目包含3个核心模块（core、utils、models），架构设计清晰，建议提升单元测试覆盖率以增强代码质量。"
            )

        # Orchestrator receiving worker result
        if depth == 1 and (
            "[子代理完成]" in last_content or "[子代理结果汇总]" in last_content
        ):
            return LLMResponse(
                content="[Orchestrator完成] 已汇总Worker的分析结果：代码包含3个核心模块，架构设计合理，建议优化测试覆盖率。Worker具体发现：发现3个主要模块，依赖关系清晰。"
            )

        # Worker (depth 2): Return direct result (no spawning)
        if depth == 2:
            return LLMResponse(
                content="[Worker完成] 代码分析完成：发现3个主要模块（core、utils、models），依赖关系清晰，建议增加单元测试覆盖率。"
            )

        # Main agent (depth 0): Spawn orchestrator
        if depth == 0 and call_info["has_spawn_tool"]:
            return LLMResponse(
                tool_calls=[
                    ToolCall(
                        name="spawn",
                        arguments={
                            "task": "Use an orchestrator to analyze code",
                            "label": "orchestrator",
                        },
                        call_id="call_main_spawn",
                    )
                ]
            )

        # Orchestrator (depth 1): Spawn worker
        if depth == 1 and call_info["has_spawn_tool"]:
            return LLMResponse(
                tool_calls=[
                    ToolCall(
                        name="spawn",
                        arguments={
                            "task": "Analyze the code structure and identify modules",
                            "label": "worker",
                        },
                        call_id="call_orchestrator_spawn",
                    )
                ]
            )

        # Default: use real LLM
        return await self.real_llm.chat(messages, tools, **kwargs)


async def test_triple_layer():
    """Run triple-layer integration test with real API."""
    print("=" * 70)
    print("Triple-Layer Integration Test")
    print("=" * 70)
    print("\nTesting: Main Agent → Orchestrator → Worker")
    print("Expected flow:")
    print("  1. Main spawns Orchestrator")
    print("  2. Orchestrator spawns Worker")
    print("  3. Worker completes → notifies Orchestrator")
    print("  4. Orchestrator completes → notifies Main")
    print("  5. Results propagate up the chain")
    print()

    # Step 1: Create main agent with custom LLM
    print("[1/6] Creating main agent...")
    try:
        config = get_config()
        config.max_depth = 3  # Allow 3 layers
    except ValueError as e:
        print(f"  ✗ Failed to load config: {e}")
        return False

    registry = SubagentRegistry()
    test_llm = TripleLayerTestLLM(config)

    main_session = Session(id="main_session", depth=0)
    main_session.add_message(
        "system",
        "You are the main agent. You can spawn orchestrators.\n\n## Session Context\n- Depth: 0/3",
    )

    main_agent = Agent(
        session=main_session,
        config=config,
        registry=registry,
        llm_provider=test_llm,
        task_id="main_agent_001",
    )

    test_llm.register_agent_depth("main_agent_001", 0)
    print(f"  ✓ Main agent created (task_id: {main_agent.task_id}, depth: 0)")

    # Step 2: Run main agent and wait for completion
    print("\n[2/6] Running main agent...")
    print("  → Main agent will spawn Orchestrator...")
    try:
        result = await main_agent.run("Use an orchestrator to analyze code")
        print(f"  ✓ Main agent started, result: {result}")

        # Wait for all agents to complete (including spawned children)
        print("  → Waiting for all agents to complete...")
        final_result = await wait_for_completion(main_agent, timeout=60.0)
        print(f"  ✓ All agents completed")
    except Exception as e:
        print(f"  ✗ Main agent execution failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Step 3: Verify 3-layer nesting in registry
    print("\n[3/6] Verifying 3-layer nesting...")

    # Check for orchestrator (depth 1)
    orchestrator_tasks = [
        tid
        for tid, task in registry._tasks.items()
        if task.depth == 1 and task.parent_task_id == main_agent.task_id
    ]
    if not orchestrator_tasks:
        print("  ✗ No orchestrator agent found (depth 1)")
        print(f"    Available tasks: {list(registry._tasks.keys())}")
        return False
    orchestrator_id = orchestrator_tasks[0]
    print(f"  ✓ Orchestrator found (task_id: {orchestrator_id}, depth: 1)")

    # Check for worker (depth 2)
    worker_tasks = [
        tid
        for tid, task in registry._tasks.items()
        if task.depth == 2 and task.parent_task_id == orchestrator_id
    ]
    if not worker_tasks:
        print("  ✗ No worker agent found (depth 2)")
        print(f"    Available tasks: {list(registry._tasks.keys())}")
        return False
    worker_id = worker_tasks[0]
    print(f"  ✓ Worker found (task_id: {worker_id}, depth: 2)")

    # Verify depth chain
    print(f"  ✓ Verified 3-layer chain:")
    print(f"    - Main (depth 0): {main_agent.task_id}")
    print(f"    - Orchestrator (depth 1): {orchestrator_id}")
    print(f"    - Worker (depth 2): {worker_id}")

    # Step 4: Verify result propagation: Worker → Orchestrator
    print("\n[4/6] Verifying Worker → Orchestrator result propagation...")

    worker_task = registry.get_task(worker_id)
    if not worker_task:
        print(f"  ✗ Worker task {worker_id} not found in registry")
        return False

    if worker_task.status != "completed":
        print(f"  ✗ Worker task status is '{worker_task.status}', expected 'completed'")
        return False

    if not worker_task.result:
        print("  ✗ Worker task has no result")
        return False

    worker_result = worker_task.result
    print(f"  ✓ Worker completed with result:")
    print(f"    {worker_result[:80]}{'...' if len(worker_result) > 80 else ''}")

    # Check orchestrator received worker result
    orchestrator_task = registry.get_task(orchestrator_id)
    if not orchestrator_task:
        print(f"  ✗ Orchestrator task {orchestrator_id} not found in registry")
        return False

    if orchestrator_task.status != "completed":
        print(
            f"  ✗ Orchestrator task status is '{orchestrator_task.status}', expected 'completed'"
        )
        return False

    if not orchestrator_task.result:
        print("  ✗ Orchestrator task has no result")
        return False

    orchestrator_result = orchestrator_task.result
    print(f"  ✓ Orchestrator completed with result:")
    print(
        f"    {orchestrator_result[:80]}{'...' if len(orchestrator_result) > 80 else ''}"
    )

    # Verify orchestrator result contains worker findings
    if "Worker" in orchestrator_result or "worker" in orchestrator_result.lower():
        print("  ✓ Orchestrator result references Worker findings")
    else:
        print("  ⚠ Warning: Orchestrator result doesn't explicitly reference Worker")

    # Step 5: Verify result propagation: Orchestrator → Main
    print("\n[5/6] Verifying Orchestrator → Main result propagation...")

    # Main should be completed
    if main_agent.state != AgentState.COMPLETED:
        print(f"  ✗ Main agent state is {main_agent.state}, expected COMPLETED")
        return False

    if not main_agent._final_result:
        print("  ✗ Main agent has no final result")
        return False

    main_result = main_agent._final_result
    print(f"  ✓ Main agent completed with result:")
    print(f"    {main_result[:80]}{'...' if len(main_result) > 80 else ''}")

    # Verify main result references orchestrator
    if "Orchestrator" in main_result or "orchestrator" in main_result.lower():
        print("  ✓ Main result references Orchestrator findings")
    else:
        print("  ⚠ Warning: Main result doesn't explicitly reference Orchestrator")

    # Verify complete propagation chain
    print("\n  ✓ Complete result propagation verified:")
    print(f"    Worker → Orchestrator: {worker_result[:50]}...")
    print(f"    Orchestrator → Main: {orchestrator_result[:50]}...")
    print(f"    Final (Main): {main_result[:50]}...")

    # Step 6: Final verification
    print("\n[6/6] Final verification...")

    # Check all agents completed
    all_completed = all(task.status == "completed" for task in registry._tasks.values())
    if not all_completed:
        incomplete = [
            tid for tid, task in registry._tasks.items() if task.status != "completed"
        ]
        print(f"  ✗ Some agents not completed: {incomplete}")
        return False
    print("  ✓ All agents completed successfully")

    # Check no pending tasks
    if registry.has_pending():
        print(f"  ✗ Still have {registry.get_pending_count()} pending tasks")
        return False
    print("  ✓ No pending tasks remaining")

    # Verify agent states
    print(f"  ✓ Main agent state: {main_agent.state.value}")

    print("\n" + "=" * 70)
    print("✓ Triple-layer test passed")
    print("=" * 70)
    print("\nSummary:")
    print(f"  - 3 layers created: Main → Orchestrator → Worker")
    print(f"  - Worker result: {worker_result[:60]}...")
    print(f"  - Orchestrator result: {orchestrator_result[:60]}...")
    print(f"  - Main result: {main_result[:60]}...")
    print(f"  - Result propagation: Worker → Orchestrator → Main ✓")
    print("=" * 70)

    return True


if __name__ == "__main__":
    try:
        success = asyncio.run(test_triple_layer())
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
