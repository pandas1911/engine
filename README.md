# Engine

基于多 Agent 协作的 AI 自动化框架，支持嵌套子 Agent 调度、多 Provider LLM 路由和自适应限流。

## 环境要求

- Python >= 3.10

## 安装

```bash
# 基础安装
uv sync

# 开发环境（含测试依赖）
uv sync --extra dev
```

## 运行测试

```bash
# 方式一：直接运行
uv run pytest

# 方式二：激活虚拟环境后运行
source .venv/bin/activate
pytest
```

## 快速开始

```python
from engine import delegate

result = delegate("你的任务描述")
print(result.content)
```

## 配置

在项目根目录创建 `engine.json` 配置 LLM Provider 和模型参数，参考 `engine.json.example`。
