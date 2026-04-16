"""Test engine's multi-layer subagent nesting capability.

Verifies that engine.delegate() can handle a prompt requiring
3 child agents, each spawning 2 grandchild agents.
"""

import pytest
from engine import delegate

TEST_PROMPT = """
    现在对你的构建子代理的能力进行测试，请严格根据以下要求执行：
    - 现在有两个任务：1.查询一下上海的天气；2.写一首赞美春天的简短的诗
    - 你可以构建子代理来完成全部任务，最后你来汇总；或者你完成一个任务，另一个任务分配给子代理来完成，最后你汇总
"""


@pytest.mark.asyncio
async def test_multilayer_subagent():
    result = await delegate(TEST_PROMPT)
    assert result.success, f"delegate failed: {result.error}"
    assert result.content, "delegate returned empty content"
