"""engine - AI Agent automation package."""

# Core API — extracted to runner.py
from engine.runner import delegate, DEFAULT_SYSTEM_PROMPT, _discover_custom_tools, _refresh_custom_tools

# Public API re-exports
from engine.tools.base import Tool, FunctionTool
from engine.runtime.agent_models import AgentResult
from engine.runtime.task_registry import AgentTaskRegistry
from engine.logging import init_logger, get_logger, stop_logger

__all__ = ["delegate", "Tool", "FunctionTool", "AgentResult", "AgentTaskRegistry", "init_logger", "get_logger", "stop_logger"]
