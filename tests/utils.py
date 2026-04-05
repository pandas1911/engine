"""Test utilities for integration tests.

This module provides helper functions for creating and managing test agents.
"""

import asyncio
import sys
from io import StringIO
from contextlib import contextmanager
from typing import Optional

from src.agent_core import Agent
from src.models import Session
from src.registry import SubagentRegistry
from src.llm_provider import MiniMaxProvider
from src.config import get_config, Config


def create_test_agent(
    task_id: str, task: str, config: Optional[Config] = None
) -> Agent:
    """Create a test agent with all dependencies.

    Args:
        task_id: Unique identifier for this agent/task
        task: Task description for the agent
        config: Optional configuration. If not provided, a test config will be created.

    Returns:
        Agent instance with all dependencies initialized
    """
    if config is None:
        try:
            config = get_config()
        except ValueError:
            config = Config(
                openai_api_key="test-key",
                openai_base_url="https://api.test.com",
                openai_model="test-model",
            )

    session = Session(id=f"test_session_{task_id}", depth=0, parent_id=None)

    session.add_message("system", f"You are a test agent. Task: {task}")

    registry = SubagentRegistry()

    llm_provider = MiniMaxProvider(config)

    agent = Agent(
        session=session,
        config=config,
        registry=registry,
        llm_provider=llm_provider,
        task_id=task_id,
    )

    return agent


async def wait_for_completion(agent: Agent, timeout: float = 30.0) -> str:
    """Wait for agent completion with timeout.

    Args:
        agent: The agent to wait for
        timeout: Maximum time to wait in seconds (default: 30.0)

    Returns:
        The final result from agent._final_result

    Raises:
        TimeoutError: If the agent doesn't complete within the timeout period
    """
    try:
        await asyncio.wait_for(agent._completion_event.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        raise TimeoutError(
            f"Agent {agent.task_id} did not complete within {timeout} seconds"
        )

    return agent._final_result


def assert_result_contains(result: str, expected_text: str) -> None:
    """Assert that result contains expected text.

    Args:
        result: The actual result string
        expected_text: The text expected to be contained in result

    Raises:
        AssertionError: If expected_text is not found in result
    """
    if expected_text not in result:
        raise AssertionError(
            f"Expected result to contain '{expected_text}', but got:\n{result}"
        )


@contextmanager
def capture_print_output():
    """Context manager to capture stdout during function execution.

    Usage:
        with capture_print_output() as output:
            some_function_that_prints()
        print(f"Captured: {output.getvalue()}")

    Yields:
        StringIO buffer containing captured output
    """
    old_stdout = sys.stdout
    buffer = StringIO()
    sys.stdout = buffer
    try:
        yield buffer
    finally:
        sys.stdout = old_stdout


__all__ = [
    "create_test_agent",
    "wait_for_completion",
    "assert_result_contains",
    "capture_print_output",
]
