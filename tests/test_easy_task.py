"""Test engine's basic delegation capability.

Verifies that engine.delegate() can handle a structured research task
(comparing cities for a remote hiring base) and return a complete result.
"""

import pytest
from engine import delegate

TEST_PROMPT = """
    帮我查一下五一节上海的天气怎么样
"""


@pytest.mark.asyncio
async def test_multilayer_subagent():
    result = await delegate(TEST_PROMPT)
    assert result.success, f"delegate failed: {result.error}"
    assert result.content, "delegate returned empty content"
