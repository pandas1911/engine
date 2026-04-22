"""engine - AI Agent automation package."""

import importlib
import inspect
import uuid
from pathlib import Path
from typing import List, Optional

from engine.runtime.agent import Agent
from engine.config import Config, get_config
from engine.providers.llm_provider import LLMProvider
from engine.logging import init_logger, get_logger, stop_logger
from engine.runtime.agent_models import AgentError, AgentResult, AgentState, ErrorCategory, Session
from engine.runtime.task_registry import AgentTaskRegistry
from engine.tools.base import Tool, FunctionTool

__all__ = ["delegate", "Tool", "FunctionTool", "AgentResult", "AgentTaskRegistry", "init_logger", "get_logger", "stop_logger"]

DEFAULT_SYSTEM_PROMPT = (
    "你是主Agent，请尽可能构建子代理来并行处理任务。"
    "这不仅可以提升处理效率，还能减少无关信息对上下文的污染，帮助你做出更清晰的决策。"
)

_custom_tools_cache: Optional[List] = None


def _discover_custom_tools() -> List:
    """Discover and cache custom tools from engine/tools/custom/."""
    global _custom_tools_cache
    if _custom_tools_cache is not None:
        return _custom_tools_cache

    tools = []
    custom_dir = Path(__file__).parent / "tools" / "custom"

    if not custom_dir.exists():
        _custom_tools_cache = tools
        return tools

    for py_file in custom_dir.glob("*.py"):
        if py_file.name.startswith("_") or py_file.name == "__init__.py":
            continue

        module_name = f"engine.tools.custom.{py_file.stem}"
        try:
            module = importlib.import_module(module_name)
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, Tool) and obj is not Tool:
                    try:
                        tools.append(obj())
                    except Exception:
                        pass
        except Exception:
            pass

    _custom_tools_cache = tools
    return tools


def _refresh_custom_tools() -> None:
    """Clear the custom tools cache, forcing re-discovery on next call."""
    global _custom_tools_cache
    _custom_tools_cache = None


async def delegate(
    task_description: str,
    system_prompt: Optional[str] = None,
    tools: Optional[List] = None,
    config: Optional[Config] = None,
) -> AgentResult:
    """Delegate a task to the agent system."""
    session = Session(id=f"root_{uuid.uuid4().hex[:8]}", depth=0)
    session.add_message("system", system_prompt or DEFAULT_SYSTEM_PROMPT)

    try:
        if config is None:
            config = get_config()

        init_logger(log_dir=config.log_dir)

        llm_provider = LLMProvider(config)
        task_registry = AgentTaskRegistry()

        custom_tools = _discover_custom_tools()
        all_tools = custom_tools + (tools or [])

        agent = Agent(
            session=session,
            config=config,
            task_registry=task_registry,
            llm_provider=llm_provider,
            tools=all_tools,
        )

        await task_registry.register(
            task_id=agent.task_id,
            session_id=session.id,
            description="root task",
            parent_agent=None,
            agent=agent,
            depth=0,
        )

        await agent.run(task_description)

        if agent.state not in (AgentState.COMPLETED, AgentState.ERROR):
            await agent._completion_event.wait()

        success = agent.state == AgentState.COMPLETED
        return AgentResult(
            content=agent._final_result or "",
            session=session,
            success=success,
            error=None if success else agent._error_info,
        )
    except Exception as e:
        return AgentResult(
            content="",
            session=session,
            success=False,
            error=AgentError(
                category=ErrorCategory.INTERNAL_ERROR,
                message=str(e),
                exception_type=type(e).__name__,
            ),
        )
    finally:
        await stop_logger()
