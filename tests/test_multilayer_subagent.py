"""Test engine's multi-layer subagent nesting capability.

Verifies that engine.delegate() can handle a prompt requiring
3 child agents, each spawning 2 grandchild agents.
"""

import pytest
from engine import delegate

TEST_PROMPT = """
    现在对你的构建子代理的能力进行测试，请严格根据以下要求执行：
    - 请构建三个子代理
    - 然后要求每一个子代理再构建两个子代理（对于你来说是孙代理）
    - 要求每个孙代理随机生成一个数字，由子代理汇总并汇报给你
    - 最后你来汇总子代理的结果，并返回给用户
    - 如果出现任何问题，你需要在最后反馈给用户
"""


@pytest.mark.asyncio
async def test_multilayer_subagent():
    result = await delegate(TEST_PROMPT)
    assert result.success, f"delegate failed: {result.error}"
    assert result.content, "delegate returned empty content"
