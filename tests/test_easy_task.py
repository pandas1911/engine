"""Test engine's multi-layer subagent nesting capability.

Verifies that engine.delegate() can handle a prompt requiring
3 child agents, each spawning 2 grandchild agents.
"""

import pytest
from engine import delegate

TEST_PROMPT = """
    周末想去上海周边的地方玩一下，你有什么好的推荐吗，帮我写一份出行计划，大概周五晚上六点出发周日下午四点返程

    *要求*
     - 离上海尽可能的近，考虑自驾或者高铁出行
     - 希望在民宿住宿
     - 避开热门景点，想去一些小众的地方
     - 计划中要有美食推荐
     - 预算在3000左右，两个人出行
"""


@pytest.mark.asyncio
async def test_multilayer_subagent():
    result = await delegate(TEST_PROMPT)
    assert result.success, f"delegate failed: {result.error}"
    assert result.content, "delegate returned empty content"
