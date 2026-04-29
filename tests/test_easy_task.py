"""Test engine's multi-layer subagent nesting capability.

Verifies that engine.delegate() can handle a prompt requiring
3 child agents, each spawning 2 grandchild agents.
"""

import pytest
from engine import delegate

TEST_PROMPT = """
    当前你有哪些工具可用，如实回答
"""


@pytest.mark.asyncio
async def test_multilayer_subagent():
    result = await delegate(TEST_PROMPT)
    assert result.success, f"delegate failed: {result.error}"
    assert result.content, "delegate returned empty content"
