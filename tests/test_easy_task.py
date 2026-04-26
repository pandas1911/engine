"""Test engine's multi-layer subagent nesting capability.

Verifies that engine.delegate() can handle a prompt requiring
3 child agents, each spawning 2 grandchild agents.
"""

import pytest
from engine import delegate

TEST_PROMPT = """
    用户需要在下周一至周五期间，从北京出发完成一次商务出差，并最终返回北京。

已知会议地点与时间约束：

上海会议1：上海环球金融中心（周二 10:00–12:00）
上海会议2：张江高科技园区（周三 14:00–16:00）
杭州会议：阿里巴巴西溪园区（周四 10:00–12:00）

要求：

- 出发地/返回地：北京
- 出发时间：下周一任意时间
- 必须在周五 20:00 前返回北京
- 城市间交通：高铁优先，其次飞机
- 每场会议至少提前30分钟到达
- 酒店要求：距离会议地点通勤时间 ≤30分钟
- 总成本最低（交通+住宿）
"""


@pytest.mark.asyncio
async def test_multilayer_subagent():
    result = await delegate(TEST_PROMPT)
    assert result.success, f"delegate failed: {result.error}"
    assert result.content, "delegate returned empty content"
