#!/usr/bin/env python3
"""Test callback deferral mechanism for race condition fix.

This file verifies that the event queue mechanism works correctly
to prevent re-entrant callback race conditions.
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent_core import Agent
from src.config import Config
from src.models import Session, AgentState, LLMResponse, ToolCall
from src.registry import SubagentRegistry
from src.llm_provider import MockLLMProvider


async def test_basic_race_fix():
    """Test 1: Basic race condition fix - callback deferred during tool loop."""
    print("Test 1: Basic race fix...")
    
    session = Session(id='test', depth=0)
    config = Config(api_key='test', base_url='test', model='test', max_depth=3)
    registry = SubagentRegistry()
    llm = MockLLMProvider()
    agent = Agent(session=session, config=config, registry=registry, llm_provider=llm)
    
    # Simulate being in tool loop
    agent._in_tool_loop = True
    
    # Call callback - should be deferred
    await agent._on_subagent_complete('child_1', 'child result')
    
    # Assert event was queued
    assert len(agent._event_queue) == 1, f"Expected 1 event in queue, got {len(agent._event_queue)}"
    assert agent._event_queue[0][0] == 'complete', f"Expected 'complete', got {agent._event_queue[0][0]}"
    assert agent._event_queue[0][1] == 'child_1', f"Expected 'child_1', got {agent._event_queue[0][1]}"
    
    # Reset flag and drain
    agent._in_tool_loop = False
    await agent._drain_events()
    
    # Assert queue is empty
    assert len(agent._event_queue) == 0, f"Expected empty queue after drain, got {len(agent._event_queue)}"
    
    print("  PASS: Basic race fix works correctly")


async def test_multiple_children_deferred():
    """Test 2: Multiple children complete during tool loop."""
    print("Test 2: Multiple children deferred...")
    
    session = Session(id='test', depth=0)
    config = Config(api_key='test', base_url='test', model='test', max_depth=3)
    registry = SubagentRegistry()
    llm = MockLLMProvider()
    agent = Agent(session=session, config=config, registry=registry, llm_provider=llm)
    
    # Simulate being in tool loop
    agent._in_tool_loop = True
    
    # Call callbacks for two children
    await agent._on_subagent_complete('child_1', 'result 1')
    await agent._on_subagent_complete('child_2', 'result 2')
    
    # Assert queue has 2 events
    assert len(agent._event_queue) == 2, f"Expected 2 events in queue, got {len(agent._event_queue)}"
    assert agent._event_queue[0][1] == 'child_1', f"Expected first event from child_1"
    assert agent._event_queue[1][1] == 'child_2', f"Expected second event from child_2"
    
    # Reset flag and drain
    agent._in_tool_loop = False
    await agent._drain_events()
    
    # Assert queue is empty
    assert len(agent._event_queue) == 0, f"Expected empty queue after drain, got {len(agent._event_queue)}"
    
    print("  PASS: Multiple children deferred correctly")


async def test_error_callback_deferred():
    """Test 3: Error callback is also deferred."""
    print("Test 3: Error callback deferred...")
    
    session = Session(id='test', depth=0)
    config = Config(api_key='test', base_url='test', model='test', max_depth=3)
    registry = SubagentRegistry()
    llm = MockLLMProvider()
    agent = Agent(session=session, config=config, registry=registry, llm_provider=llm)
    
    # Simulate being in tool loop
    agent._in_tool_loop = True
    
    # Call error callback
    await agent._on_subagent_error('child_err', 'something failed')
    
    # Assert event was queued with type 'error'
    assert len(agent._event_queue) == 1, f"Expected 1 event in queue, got {len(agent._event_queue)}"
    assert agent._event_queue[0][0] == 'error', f"Expected 'error', got {agent._event_queue[0][0]}"
    assert agent._event_queue[0][1] == 'child_err', f"Expected 'child_err', got {agent._event_queue[0][1]}"
    
    # Reset flag and drain
    agent._in_tool_loop = False
    await agent._drain_events()
    
    # Assert queue is empty
    assert len(agent._event_queue) == 0, f"Expected empty queue after drain, got {len(agent._event_queue)}"
    
    print("  PASS: Error callback deferred correctly")


async def test_wake_callback_deferred():
    """Test 4: Wake callback is also deferred."""
    print("Test 4: Wake callback deferred...")
    
    session = Session(id='test', depth=0)
    config = Config(api_key='test', base_url='test', model='test', max_depth=3)
    registry = SubagentRegistry()
    llm = MockLLMProvider()
    agent = Agent(session=session, config=config, registry=registry, llm_provider=llm)
    
    # Simulate being in tool loop
    agent._in_tool_loop = True
    
    # Call wake callback
    await agent._on_descendant_wake('desc_1', 'descendant result')
    
    # Assert event was queued with type 'wake'
    assert len(agent._event_queue) == 1, f"Expected 1 event in queue, got {len(agent._event_queue)}"
    assert agent._event_queue[0][0] == 'wake', f"Expected 'wake', got {agent._event_queue[0][0]}"
    assert agent._event_queue[0][1] == 'desc_1', f"Expected 'desc_1', got {agent._event_queue[0][1]}"
    
    # Reset flag and drain
    agent._in_tool_loop = False
    await agent._drain_events()
    
    # Assert queue is empty
    assert len(agent._event_queue) == 0, f"Expected empty queue after drain, got {len(agent._event_queue)}"
    
    print("  PASS: Wake callback deferred correctly")


async def test_non_deferred_path_unchanged():
    """Test 5: When flag is False, callbacks execute normally (not deferred)."""
    print("Test 5: Non-deferred path unchanged...")
    
    session = Session(id='test', depth=0)
    config = Config(api_key='test', base_url='test', model='test', max_depth=3)
    registry = SubagentRegistry()
    llm = MockLLMProvider()
    agent = Agent(session=session, config=config, registry=registry, llm_provider=llm)
    
    # Flag is False by default - no need to set it
    assert agent._in_tool_loop == False, "Flag should be False by default"
    
    # Call callback - should NOT be deferred
    await agent._on_subagent_complete('child_1', 'result')
    
    # Assert queue is STILL empty (event was not queued, callback ran normally)
    assert len(agent._event_queue) == 0, f"Expected empty queue (callback ran normally), got {len(agent._event_queue)}"
    
    print("  PASS: Non-deferred path unchanged")


async def test_flag_managed_by_try_finally():
    """Test 6: Flag properly managed in _process_tool_calls with try/finally."""
    print("Test 6: Flag managed by try/finally...")
    
    session = Session(id='test', depth=0)
    config = Config(api_key='test', base_url='test', model='test', max_depth=3)
    registry = SubagentRegistry()
    llm = MockLLMProvider()
    agent = Agent(session=session, config=config, registry=registry, llm_provider=llm)
    
    # Track flag state during execution
    flag_during_execution = None
    
    async def patched_chat(*args, **kwargs):
        nonlocal flag_during_execution
        flag_during_execution = agent._in_tool_loop
        # Return empty response to end processing quickly
        return LLMResponse(content="test response")
    
    # Patch llm.chat to capture flag state
    original_chat = agent.llm.chat
    agent.llm.chat = patched_chat
    
    try:
        # Call _process_tool_calls
        await agent._process_tool_calls()
        
        # Assert flag was True during execution
        assert flag_during_execution == True, f"Flag should be True during _process_tool_calls, was {flag_during_execution}"
        
        # Assert flag is False after return
        assert agent._in_tool_loop == False, f"Flag should be False after _process_tool_calls returns, was {agent._in_tool_loop}"
        
    finally:
        # Restore original chat method
        agent.llm.chat = original_chat
    
    print("  PASS: Flag correctly managed by try/finally")


if __name__ == "__main__":
    tests = [
        ("Basic race fix", test_basic_race_fix),
        ("Multiple children deferred", test_multiple_children_deferred),
        ("Error callback deferred", test_error_callback_deferred),
        ("Wake callback deferred", test_wake_callback_deferred),
        ("Non-deferred path unchanged", test_non_deferred_path_unchanged),
        ("Flag managed by try/finally", test_flag_managed_by_try_finally),
    ]
    
    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            asyncio.run(test_fn())
            passed += 1
        except Exception as e:
            print(f"FAIL: {name}: {e}")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")
