"""Test engine's multi-layer subagent nesting capability.

Verifies that engine.delegate() can handle a prompt requiring
3 child agents, each spawning 2 grandchild agents.
"""

import pytest
from engine import delegate

TEST_PROMPT = """
    我最近想买车了，请你帮我调研一下
    *要求*
    - 最好是新能源的车
    - 价格不要超过20万
    - 续航里程至少400公里
    - 车的外形要帅气，适合年轻人
    - 把你推荐的车型列表排名

    你可以构建几个不同的代理去调研，最后把几个代理推荐最多的车型汇总给我
"""


@pytest.mark.asyncio
async def test_multilayer_subagent():
    result = await delegate(TEST_PROMPT)
    assert result.success, f"delegate failed: {result.error}"
    assert result.content, "delegate returned empty content"
