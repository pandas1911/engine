"""Test engine's basic delegation capability.

Verifies that engine.delegate() can handle a structured research task
(comparing cities for a remote hiring base) and return a complete result.
"""

import pytest
from engine import delegate

TEST_PROMPT = """
    你需要为一家中国软件公司选择一个海外远程办公招聘基地，在以下三个城市中做出决策：

    - 新加坡（Singapore）
    - 柏林（Germany）
    - 班加罗尔（India）

    ---

    ## 任务目标

    输出一份结构化对比报告，用于决策“在哪一个城市建立远程招聘中心”。

    ---

    ## 任务要求

    必须严格完成以下 5 个部分，所有数据需基于公开可获取的信息：

    ### 1. 人才供给（必须量化）
    - 软件工程师数量（或开发者规模）
    - 英语能力水平（使用指数或排名）
    - 主流技术栈分布（至少列3类）

    ---

    ### 2. 成本分析（必须给出数值）
    - 初级软件工程师平均月薪（USD）
    - 中级软件工程师平均月薪（USD）
    - 共享办公月租（USD/工位）

    ---

    ### 3. 政策环境（必须可比较）
    - 是否支持远程办公/数字游民签证（是/否）
    - 企业雇佣合规难度（低/中/高）

    ---

    ### 4. 基础设施（必须客观指标）
    - 平均网络速度（Mbps）
    - 与中国的时差（小时）

    ---

    ### 5. 风险评估（必须列举）
    - 至少列出2个具体风险（如政策变化、人才流失等）

    ---

    ## 最终输出要求

    必须包含以下内容：

    1. 三城市对比表（覆盖所有指标）
    2. 综合评分（满分100，需说明评分权重）
    3. 最优城市选择（只能选择1个）
    4. 选择理由（不超过150字）

    ---

    ## 约束条件

    - 所有5个模块必须完成
    - 各模块可独立拆解并行执行
    - 不允许使用模糊描述（如“较高”“较低”），必须尽量量化
    - 若存在数据冲突，需自行判断并说明依据
"""


@pytest.mark.asyncio
async def test_multilayer_subagent():
    result = await delegate(TEST_PROMPT)
    assert result.success, f"delegate failed: {result.error}"
    assert result.content, "delegate returned empty content"
