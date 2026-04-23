"""Test engine's multi-layer subagent nesting capability.

Verifies that engine.delegate() can handle a prompt requiring
3 child agents, each spawning 2 grandchild agents.
"""

import pytest
from engine import delegate

TEST_PROMPT = """
    有一道题目帮我完成一下：
    题目名称：2026年全球前沿科技趋势深度调研报告
    - 任务背景与目标
    假设你是一家顶级风险投资机构的AI首席分析师。合伙人需要一份关于“2026年全球最具颠覆潜力的三大前沿科技领域”的深度调研报告，用于指导下一季度的投资方向。

    *要求* 请生成一份结构化的深度报告，报告必须包含以下三个部分：
    1.  领域识别：精准选出三个2026年最具潜力的领域（需基于2026年最新数据）。
    2.  深度分析：对每个领域进行独立且深度的分析（包含技术成熟度、市场规模、关键玩家）。
    3.  综合建议：基于上述分析，给出投资优先级排序及理由。
"""


@pytest.mark.asyncio
async def test_multilayer_subagent():
    result = await delegate(TEST_PROMPT)
    assert result.success, f"delegate failed: {result.error}"
    assert result.content, "delegate returned empty content"
